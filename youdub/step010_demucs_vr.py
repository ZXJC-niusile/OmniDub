import os
import subprocess
import threading
import time
import wave

from demucs.api import Separator
from loguru import logger

from .utils import resolve_torch_device, save_wav

_separators = {}
_separator_locks = {}
_separator_load_locks = {}
_separator_registry_lock = threading.Lock()


def init_demucs(model_name: str = "htdemucs_ft", device: str = "auto", shifts: int = 5):
    return load_model(model_name=model_name, device=device, shifts=shifts)


def _resolve_device(device: str) -> str:
    return str(resolve_torch_device(device))


def load_model(
    model_name: str = "htdemucs_ft",
    device: str = 'auto',
    progress: bool = True,
    shifts: int = 5,
) -> Separator:
    resolved_device = _resolve_device(device)
    cache_key = (model_name, resolved_device, int(shifts))
    with _separator_registry_lock:
        existing = _separators.get(cache_key)
        if existing is not None:
            logger.info(f'Demucs model already loaded on {resolved_device}')
            return existing
        _separator_locks.setdefault(resolved_device, threading.Lock())
        key_load_lock = _separator_load_locks.setdefault(cache_key, threading.Lock())

    with key_load_lock:
        with _separator_registry_lock:
            existing = _separators.get(cache_key)
            if existing is not None:
                return existing
        logger.info(f'Loading Demucs model: {model_name} on {resolved_device}')
        t_start = time.time()
        separator = Separator(model_name, device=resolved_device, progress=progress, shifts=shifts)
        t_end = time.time()
        logger.info(f'Demucs model loaded on {resolved_device} in {t_end - t_start:.2f} seconds')
        with _separator_registry_lock:
            _separators[cache_key] = separator
        return separator


def separate_audio(
    folder: str,
    model_name: str = "htdemucs_ft",
    device: str = 'auto',
    progress: bool = True,
    shifts: int = 5,
) -> None:
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Input audio not found for vocal separation: {audio_path}')

    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if _has_valid_audio_content(vocal_output_path) and _has_valid_audio_content(instruments_output_path):
        logger.info(f'Audio already separated in {folder}')
        return

    # Demucs cannot process empty wav; fail fast with a clear error.
    if not _has_valid_audio_content(audio_path):
        raise RuntimeError(f'Invalid/empty audio extracted from {folder}: {audio_path}')

    logger.info(f'Separating audio from {folder}')
    sep = load_model(model_name, device, progress, shifts)
    resolved_device = _resolve_device(device)
    sep_lock = _separator_locks.setdefault(resolved_device, threading.Lock())
    t_start = time.time()
    # Separator is a shared global object; protect inference in parallel mode.
    with sep_lock:
        try:
            origin, separated = sep.separate_audio_file(audio_path)
        except Exception as e:
            logger.exception(f'Demucs failed on {folder}: {e}')
            should_fallback_cpu = str(resolved_device).startswith('cuda') or device == 'auto'
            if should_fallback_cpu:
                logger.warning('Retrying Demucs on CPU...')
                try:
                    cpu_sep = load_model(model_name, device='cpu', progress=progress, shifts=shifts)
                    cpu_lock = _separator_locks.setdefault('cpu', threading.Lock())
                    with cpu_lock:
                        origin, separated = cpu_sep.separate_audio_file(audio_path)
                    logger.warning('Demucs CPU fallback succeeded.')
                except Exception as cpu_e:
                    logger.exception(f'Demucs CPU fallback failed on {folder}: {cpu_e}')
                    raise RuntimeError(f'Error separating audio from {folder}') from cpu_e
            else:
                raise RuntimeError(f'Error separating audio from {folder}') from e
    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds')

    vocals = separated['vocals'].numpy().T
    instruments = None
    for k, v in separated.items():
        if k == 'vocals':
            continue
        if instruments is None:
            instruments = v
        else:
            instruments = instruments + v
    instruments = instruments.numpy().T

    save_wav(vocals, vocal_output_path, sample_rate=44100)
    logger.info(f'Vocals saved to {vocal_output_path}')

    save_wav(instruments, instruments_output_path, sample_rate=44100)
    logger.info(f'Instruments saved to {instruments_output_path}')


def extract_audio_from_video(folder: str) -> bool:
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Input video not found: {video_path}')
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        if _has_valid_audio_content(audio_path):
            logger.info(f'Audio already extracted in {folder}')
            return True
        logger.warning(f'Existing audio is invalid, re-extracting: {audio_path}')
        os.remove(audio_path)
    logger.info(f'Extracting audio from {folder}')
    ffmpeg_command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        audio_path,
        '-y',
    ]
    subprocess.run(ffmpeg_command, check=True)
    if not _has_valid_audio_content(audio_path):
        raise RuntimeError(
            f'Extracted audio is empty for {folder}. '
            f'This video may have no audio track or a broken audio stream.'
        )
    logger.info(f'Audio extracted from {folder}')
    return True


def _has_valid_audio_content(audio_path: str) -> bool:
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) <= 44:
        return False
    try:
        with wave.open(audio_path, 'rb') as wf:
            return wf.getnframes() > 0
    except Exception:
        return False
