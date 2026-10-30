from contextlib import contextmanager
import importlib
import json
import os
import pickle
import threading
import time

from dotenv import load_dotenv
import librosa
from loguru import logger
import numpy as np
import torch
import whisperx

from .utils import resolve_torch_device, save_wav

load_dotenv()

_whisper_models = {}
_align_models = {}
_diarize_models = {}
_whisper_load_locks = {}
_align_load_locks = {}
_diarize_load_locks = {}
_registry_lock = threading.RLock()
_infer_locks = {}
_align_locks = {}
_diarize_locks = {}
_torch_load_lock = threading.Lock()
_ALIGN_FALLBACK_LANGUAGE = os.getenv('WHISPERX_ALIGN_FALLBACK_LANGUAGE', 'en')


@contextmanager
def _pyannote_torch_load_compat():
    """Scoped fallback for pyannote/lightning checkpoints on PyTorch >= 2.6.
    Uses setdefault so explicit weights_only=True calls remain intact."""
    with _torch_load_lock:
        original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load


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


def _resolve_whisperx_device(device: str) -> tuple[str, int, str]:
    """
    Resolve device into:
      - api_device: 'cuda' or 'cpu' (for faster-whisper / ctranslate2)
      - device_index: GPU integer index (for faster-whisper multi-GPU selection)
      - device_key: exact torch device string e.g. 'cuda:1' or 'cpu'
        (for PyTorch alignment & diarization models, avoiding process-global
        torch.cuda.set_device races across threads).
    """
    resolved = str(resolve_torch_device(device))
    if resolved.startswith('cuda:'):
        try:
            idx = int(resolved.split(':', 1)[1])
        except ValueError:
            idx = 0
        return 'cuda', idx, f'cuda:{idx}'
    if resolved == 'cuda':
        return 'cuda', 0, 'cuda:0'
    return resolved, 0, resolved


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


def init_whisperx(
    model_name: str = 'large-v3',
    download_root: str = 'models/ASR/whisper',
    device: str = 'auto',
    diarization: bool = True,
):
    load_whisper_model(model_name=model_name, download_root=download_root, device=device)
    if diarization:
        load_diarize_model(device=device)


def load_whisper_model(model_name: str = 'large-v3', download_root='models/ASR/whisper', device='auto'):
    if model_name == 'large':
        model_name = 'large-v3'
    api_device, device_index, device_key = _resolve_whisperx_device(device)
    key = (model_name, device_key)
    lock = _get_registry_lock()
    with lock:
        model = _whisper_models.get(key)
        if model is not None:
            return model
        key_load_lock = _whisper_load_locks.setdefault(key, threading.Lock())

    with key_load_lock:
        with lock:
            model = _whisper_models.get(key)
            if model is not None:
                return model
        logger.info(f'Loading WhisperX model: {model_name} on {device_key}')
        t_start = time.time()
        load_kwargs = {'download_root': download_root, 'device': api_device}
        if api_device == 'cuda':
            load_kwargs['device_index'] = device_index
        try:
            model = whisperx.load_model(model_name, **load_kwargs)
        except TypeError:
            load_kwargs.pop('device_index', None)
            model = whisperx.load_model(model_name, **load_kwargs)
        t_end = time.time()
        logger.info(f'Loaded WhisperX model: {model_name} on {device_key} in {t_end - t_start:.2f}s')
        with lock:
            _whisper_models[key] = model
            _infer_locks.setdefault(key, threading.Lock())
        return model


def load_align_model(language='en', device='auto'):
    _, _, device_key = _resolve_whisperx_device(device)
    key = (language, device_key)
    lock = _get_registry_lock()
    with lock:
        cached = _align_models.get(key)
        if cached is not None:
            return cached
        key_load_lock = _align_load_locks.setdefault(key, threading.Lock())

    with key_load_lock:
        with lock:
            cached = _align_models.get(key)
            if cached is not None:
                return cached
        t_start = time.time()
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language, device=device_key
            )
        except pickle.UnpicklingError:
            with _pyannote_torch_load_compat():
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language, device=device_key
                )
        t_end = time.time()
        logger.info(f'Loaded alignment model: {language} on {device_key} in {t_end - t_start:.2f}s')
        with lock:
            _align_models[key] = (align_model, align_metadata)
            _align_locks.setdefault(key, threading.Lock())
        return align_model, align_metadata


def _instantiate_diarize_pipeline(diarization_pipeline_cls, hf_token: str, init_kwargs: dict):
    try:
        return diarization_pipeline_cls(token=hf_token, **init_kwargs)
    except TypeError:
        return diarization_pipeline_cls(use_auth_token=hf_token, **init_kwargs)


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
    _, _, device_key = _resolve_whisperx_device(device)
    key = device_key
    lock = _get_registry_lock()
    with lock:
        existing = _diarize_models.get(key)
        if existing is not None:
            return existing
        key_load_lock = _diarize_load_locks.setdefault(key, threading.Lock())

    with key_load_lock:
        with lock:
            existing = _diarize_models.get(key)
            if existing is not None:
                return existing
        t_start = time.time()
        model_name = _get_diarization_model_name()
        init_kwargs = {'device': device_key}
        if model_name:
            init_kwargs['model_name'] = model_name
        try:
            try:
                diarize_model = _instantiate_diarize_pipeline(diarization_pipeline_cls, hf_token, init_kwargs)
            except pickle.UnpicklingError:
                logger.info('Retrying pyannote diarization load with PyTorch 2.6+ compatibility mode...')
                with _pyannote_torch_load_compat():
                    diarize_model = _instantiate_diarize_pipeline(diarization_pipeline_cls, hf_token, init_kwargs)
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
            _diarize_locks.setdefault(key, threading.Lock())
        return diarize_model


def _ends_with_terminal_punctuation(text: str) -> bool:
    text = (text or '').strip()
    if not text:
        return False
    return text[-1] in {'。', '！', '？', '.', '!', '?'}


def _sanitize_aligned_segments(raw_segments: list[dict]) -> list[dict]:
    """Repair or skip segments whose 'start'/'end' keys are missing or None after whisperx.align."""
    sanitized = []
    cursor = 0.0
    for seg in raw_segments or []:
        if not isinstance(seg, dict):
            continue
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        start = seg.get('start')
        end = seg.get('end')
        if start is None or end is None:
            words = [w for w in (seg.get('words') or []) if isinstance(w, dict)]
            word_starts = [float(w['start']) for w in words if w.get('start') is not None]
            word_ends = [float(w['end']) for w in words if w.get('end') is not None]
            if start is None and word_starts:
                start = min(word_starts)
            if end is None and word_ends:
                end = max(word_ends)
        if start is None and end is not None:
            start = max(cursor, float(end) - 0.5)
        elif start is not None and end is None:
            end = float(start) + max(0.5, min(4.0, len(text) * 0.15))
        elif start is None and end is None:
            start = cursor
            end = cursor + max(0.5, min(4.0, len(text) * 0.15))

        start_f = max(0.0, float(start))
        end_f = max(start_f + 0.05, float(end))
        cursor = end_f
        sanitized.append({
            'start': round(start_f, 3),
            'end': round(end_f, 3),
            'text': text,
            'speaker': seg.get('speaker') or 'SPEAKER_00',
        })
    return sanitized


def merge_segments(transcript, max_gap=0.5):
    if not transcript:
        return []

    first = dict(transcript[0])
    first['speaker'] = first.get('speaker') or 'SPEAKER_00'
    merged_transcription = [first]
    for segment in transcript[1:]:
        current = merged_transcription[-1]
        seg_speaker = segment.get('speaker') or 'SPEAKER_00'
        gap = max(0.0, float(segment['start']) - float(current['end']))
        same_speaker = current.get('speaker') == seg_speaker
        should_merge = (
            gap < max_gap
            and not _ends_with_terminal_punctuation(current.get('text', ''))
            and same_speaker
        )
        if should_merge:
            spacer = '' if current.get('text', '').endswith(' ') else ' '
            current['text'] = f"{current.get('text', '')}{spacer}{(segment.get('text') or '').strip()}".strip()
            current['end'] = max(float(current['end']), float(segment['end']))
        else:
            entry = dict(segment)
            entry['speaker'] = seg_speaker
            merged_transcription.append(entry)
    return merged_transcription


def _has_valid_speaker_wavs(folder: str) -> bool:
    spk_dir = os.path.join(folder, 'SPEAKER')
    if not os.path.isdir(spk_dir):
        return False
    return any(
        fn.endswith('.wav') and os.path.getsize(os.path.join(spk_dir, fn)) > 44
        for fn in os.listdir(spk_dir)
    )


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
    transcript_path = os.path.join(folder, 'transcript.json')
    if (
        os.path.isfile(transcript_path)
        and os.path.getsize(transcript_path) > 2
        and _has_valid_speaker_wavs(folder)
    ):
        logger.info(f'Transcript already exists in {folder}')
        return True

    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f'Vocals wav not found for transcription: {wav_path}')

    logger.info(f'Transcribing {wav_path}')
    active_api_device, _, active_device_key = _resolve_whisperx_device(device)
    normalized_model_name = 'large-v3' if model_name == 'large' else model_name
    model = load_whisper_model(normalized_model_name, download_root, active_device_key)
    model_key = (normalized_model_name, active_device_key)
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
            too_large = ('out of memory' in error_message) or (
                'batch_size' in error_message and 'too large' in error_message
            )
            if too_large and str(active_api_device).startswith('cuda') and batch_try > 1:
                next_batch = max(1, batch_try // 2)
                logger.warning(f'WhisperX CUDA batch too large ({batch_try}), retry with batch_size={next_batch}')
                batch_try = next_batch
                torch.cuda.empty_cache()
                continue
            if too_large and str(active_api_device).startswith('cuda'):
                logger.warning('WhisperX CUDA still OOM/batch-too-large, fallback to CPU...')
                active_api_device = 'cpu'
                active_device_key = 'cpu'
                model = load_whisper_model(normalized_model_name, download_root, active_device_key)
                model_key = (normalized_model_name, active_device_key)
                infer_lock = _infer_locks.setdefault(model_key, threading.Lock())
                batch_try = min(4, max(1, batch_try))
                continue
            raise
    if rec_result is None:
        raise RuntimeError('WhisperX transcription failed after batch-size fallback and CPU retry.')

    if rec_result.get('language') == 'nn':
        raise RuntimeError(f'No recognizable spoken language detected in {wav_path}')

    if not rec_result.get('segments'):
        raise RuntimeError(f'No speech segments detected in {wav_path}')

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

    align_lock = _align_locks.setdefault((align_language, active_device_key), threading.Lock())
    with align_lock:
        rec_result = whisperx.align(
            rec_result['segments'],
            align_model,
            align_metadata,
            wav_path,
            active_device_key,
            return_char_alignments=False,
        )

    if diarization:
        diarize_model = load_diarize_model(active_device_key)
        if diarize_model is not None:
            diarize_lock = _diarize_locks.setdefault(active_device_key, threading.Lock())
            with diarize_lock:
                diarize_segments = diarize_model(
                    wav_path, min_speakers=min_speakers, max_speakers=max_speakers
                )
            rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)
        else:
            logger.warning('Diarization requested but unavailable, continue without speaker diarization.')

    transcript = _sanitize_aligned_segments(rec_result.get('segments') or [])
    transcript = merge_segments(transcript)
    if not transcript:
        raise RuntimeError(
            f'Transcription produced no non-empty speech segments for {wav_path}. '
            'The audio may contain only music, noise, or silence.'
        )

    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {transcript_path}')
    generate_speaker_audio(folder, transcript)
    return True


def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = {}
    speaker_prompts = {}
    max_prompt_samples = int(29.0 * samplerate)
    length = len(audio_data)
    delay = 0.05

    for segment in transcript:
        speaker = segment.get('speaker') or 'SPEAKER_00'
        start = max(0, int((float(segment['start']) - delay) * samplerate))
        end = min(int((float(segment['end']) + delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end] if end > start else np.zeros((0,))
        prev_audio = speaker_dict.get(speaker, np.zeros((0,)))

        # Record verbatim transcript matching the <=29s reference prefix of SPEAKER/{speaker}.wav
        # so same-language zero-shot TTS has an exact audio-text pair.
        if len(speaker_segment_audio) > 0 and len(prev_audio) + len(speaker_segment_audio) <= max_prompt_samples:
            seg_text = (segment.get('text') or '').strip()
            if seg_text:
                speaker_prompts.setdefault(speaker, []).append(seg_text)

        speaker_dict[speaker] = np.concatenate((prev_audio, speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    os.makedirs(speaker_folder, exist_ok=True)

    fallback_clip = (
        audio_data[: min(length, int(10.0 * samplerate))]
        if length > 0
        else np.zeros((int(0.5 * samplerate),))
    )

    for speaker, audio in speaker_dict.items():
        if len(audio) == 0:
            audio = fallback_clip
        speaker_file_path = os.path.join(speaker_folder, f'{speaker}.wav')
        save_wav(audio, speaker_file_path)
        prompt_lines = speaker_prompts.get(speaker)
        if prompt_lines:
            prompt_txt_path = os.path.join(speaker_folder, f'{speaker}.txt')
            with open(prompt_txt_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(prompt_lines))
