import json
import time
import importlib
import librosa
import numpy as np
import whisperx
import os
import threading
from loguru import logger
import torch
from dotenv import load_dotenv

from .utils import save_wav, resolve_torch_device
load_dotenv()

_whisper_models = {}
_align_models = {}
_diarize_models = {}
_registry_lock = threading.RLock()
_infer_locks = {}
_ALIGN_FALLBACK_LANGUAGE = os.getenv('WHISPERX_ALIGN_FALLBACK_LANGUAGE', 'en')


def _get_registry_lock():
    return _registry_lock


def _supports_diarization_pipeline():
    return _get_diarization_pipeline_class() is not None


def _get_diarization_pipeline_class():
    # Newer WhisperX versions may not export DiarizationPipeline at top-level.
    pipeline_cls = getattr(whisperx, 'DiarizationPipeline', None)
    if pipeline_cls is not None:
        return pipeline_cls
    try:
        diarize_module = importlib.import_module('whisperx.diarize')
        return getattr(diarize_module, 'DiarizationPipeline', None)
    except Exception:
        return None


def _get_hf_token():
    return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')


def _get_diarization_model_name():
    return os.getenv('WHISPERX_DIARIZATION_MODEL')


def _normalize_whisperx_device(device):
    if isinstance(device, str) and device.startswith('cuda:'):
        # whisperx/faster-whisper expects "cuda" rather than "cuda:{idx}".
        # Keep GPU selection by setting current CUDA device explicitly.
        try:
            torch.cuda.set_device(int(device.split(':', 1)[1]))
        except Exception:
            pass
        return 'cuda'
    return device


def _resolve_whisperx_device(device: str):
    resolved = str(resolve_torch_device(device))
    api_device = _normalize_whisperx_device(resolved)
    return api_device, resolved


def _normalize_source_language(source_language):
    if source_language is None:
        return None
    src = str(source_language).strip().lower()
    if not src or src in {'auto', 'none'}:
        return None
    aliases = {
        'english': 'en',
        'en-us': 'en',
        'en-gb': 'en',
        'chinese': 'zh',
        'zh-cn': 'zh',
        'zh-hans': 'zh',
    }
    return aliases.get(src, src)


def init_whisperx():
    load_whisper_model()
    load_align_model()
    load_diarize_model()
    
def load_whisper_model(model_name: str = 'large-v3', download_root = 'models/ASR/whisper', device='auto'):
    if model_name == 'large':
        model_name = 'large-v3'
    api_device, device_key = _resolve_whisperx_device(device)
    key = (model_name, device_key)
    lock = _get_registry_lock()
    with lock:
        model = _whisper_models.get(key)
        if model is not None:
            return model
    logger.info(f'Loading WhisperX model: {model_name} on {device_key}')
    t_start = time.time()
    model = whisperx.load_model(model_name, download_root=download_root, device=api_device)
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_name} on {device_key} in {t_end - t_start:.2f}s')
    with lock:
        _whisper_models[key] = model
        _infer_locks.setdefault(key, threading.Lock())
    return model

def load_align_model(language='en', device='auto'):
    api_device, device_key = _resolve_whisperx_device(device)
    key = (language, device_key)
    lock = _get_registry_lock()
    with lock:
        cached = _align_models.get(key)
        if cached is not None:
            return cached
    t_start = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=api_device)
    t_end = time.time()
    logger.info(f'Loaded alignment model: {language} on {device_key} in {t_end - t_start:.2f}s')
    with lock:
        _align_models[key] = (align_model, align_metadata)
    return align_model, align_metadata
    
def load_diarize_model(device='auto'):
    if not _supports_diarization_pipeline():
        logger.warning('Current whisperx version does not support DiarizationPipeline, diarization disabled.')
        return None
    hf_token = _get_hf_token()
    if not hf_token:
        logger.warning('HF_TOKEN/HUGGINGFACE_TOKEN is missing, diarization disabled.')
        return None
    diarization_pipeline_cls = _get_diarization_pipeline_class()
    if diarization_pipeline_cls is None:
        logger.warning('Cannot import whisperx diarization pipeline, diarization disabled.')
        return None
    api_device, device_key = _resolve_whisperx_device(device)
    key = device_key
    lock = _get_registry_lock()
    with lock:
        existing = _diarize_models.get(key)
        if existing is not None:
            return existing
    t_start = time.time()
    model_name = _get_diarization_model_name()
    init_kwargs = {'device': api_device}
    if model_name:
        init_kwargs['model_name'] = model_name
    try:
        # Prefer new signature (token=...), and fallback to legacy signature.
        diarize_model = diarization_pipeline_cls(token=hf_token, **init_kwargs)
    except TypeError:
        diarize_model = diarization_pipeline_cls(use_auth_token=hf_token, **init_kwargs)
    except Exception as e:
        logger.exception(
            f'Failed to load diarization model: {e}. '
            'Please verify HF token permission and accepted pyannote model terms.'
        )
        return None
    t_end = time.time()
    logger.info(f'Loaded diarization model on {device_key} in {t_end - t_start:.2f}s')
    with lock:
        _diarize_models[key] = diarize_model
    return diarize_model


def _ends_with_terminal_punctuation(text: str) -> bool:
    text = (text or '').strip()
    if not text:
        return False
    return text[-1] in {'。', '！', '？', '.', '!', '?'}


def merge_segments(transcript, max_gap=0.5):
    if not transcript:
        return []

    merged_transcription = [dict(transcript[0])]
    for segment in transcript[1:]:
        current = merged_transcription[-1]
        gap = max(0.0, float(segment['start']) - float(current['end']))
        same_speaker = current.get('speaker') == segment.get('speaker')
        should_merge = (
            gap < max_gap and
            not _ends_with_terminal_punctuation(current.get('text', '')) and
            same_speaker
        )
        if should_merge:
            spacer = '' if current.get('text', '').endswith(' ') else ' '
            current['text'] = f"{current.get('text', '')}{spacer}{segment.get('text', '').strip()}".strip()
            current['end'] = max(float(current['end']), float(segment['end']))
        else:
            merged_transcription.append(dict(segment))
    return merged_transcription

def transcribe_audio(
    folder,
    model_name: str = 'large',
    download_root='models/ASR/whisper',
    device='auto',
    batch_size=32,
    diarization=True,
    min_speakers=None,
    max_speakers=None,
    source_language='auto',
):
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    active_device, active_device_key = _resolve_whisperx_device(device)
    model = load_whisper_model(model_name, download_root, active_device_key)
    model_key = ('large-v3' if model_name == 'large' else model_name, active_device_key)
    infer_lock = _infer_locks.setdefault(model_key, threading.Lock())
    batch_try = max(1, int(batch_size))
    forced_language = _normalize_source_language(source_language)
    rec_result = None
    while batch_try >= 1 and rec_result is None:
        try:
            with infer_lock:
                kwargs = {'batch_size': batch_try}
                if forced_language:
                    kwargs['language'] = forced_language
                rec_result = model.transcribe(wav_path, **kwargs)
        except Exception as e:
            error_message = str(e).lower()
            too_large = ('out of memory' in error_message) or ('batch_size' in error_message and 'too large' in error_message)
            if too_large and str(active_device).startswith('cuda') and batch_try > 1:
                next_batch = max(1, batch_try // 2)
                logger.warning(f'WhisperX CUDA batch too large ({batch_try}), retry with batch_size={next_batch}')
                batch_try = next_batch
                torch.cuda.empty_cache()
                continue
            if too_large and str(active_device).startswith('cuda'):
                logger.warning('WhisperX CUDA still OOM/batch-too-large, fallback to CPU...')
                active_device = 'cpu'
                active_device_key = 'cpu'
                model = load_whisper_model(model_name, download_root, active_device_key)
                model_key = ('large-v3' if model_name == 'large' else model_name, active_device_key)
                infer_lock = _infer_locks.setdefault(model_key, threading.Lock())
                batch_try = min(4, max(1, batch_try))
                continue
            raise
    if rec_result is None:
        raise RuntimeError('WhisperX transcription failed after batch-size fallback and CPU retry.')
    
    if rec_result['language'] == 'nn':
        logger.warning(f'No language detected in {wav_path}')
        return False
    
    detected_language = str(rec_result.get('language', '')).strip().lower()
    align_language = forced_language or detected_language
    try:
        align_model, align_metadata = load_align_model(align_language, active_device_key)
    except Exception:
        if forced_language:
            raise
        fallback_lang = _normalize_source_language(_ALIGN_FALLBACK_LANGUAGE) or 'en'
        logger.warning(
            f'No default align model for "{align_language}", retry transcription with fallback language "{fallback_lang}".'
        )
        with infer_lock:
            rec_result = model.transcribe(wav_path, batch_size=batch_try, language=fallback_lang)
        align_language = fallback_lang
        align_model, align_metadata = load_align_model(align_language, active_device_key)
    rec_result = whisperx.align(rec_result['segments'], align_model, align_metadata,
                                wav_path, active_device, return_char_alignments=False)
    
    if diarization:
        diarize_model = load_diarize_model(active_device_key)
        if diarize_model is not None:
            diarize_segments = diarize_model(wav_path,min_speakers=min_speakers, max_speakers=max_speakers)
            rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)
        else:
            logger.warning('Diarization requested but unavailable, continue without speaker diarization.')
        
    transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(), 'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
    transcript = merge_segments(transcript)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return True

def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end']+delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0, ))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)
            

