import os
import re
import importlib
import pickle
import threading
from contextlib import contextmanager
from TTS.api import TTS
from loguru import logger
import numpy as np
import torch
import time
from .utils import save_wav, resolve_torch_device

# CPML 合规：
# OmniDub 不主动从 HuggingFace / Coqui 自动下载 XTTS v2 权重（CPML 禁止再分发）。
# 启用 XTTS 必须由调用方自行：
#   1) 自行从 HuggingFace / Coqui 下载权重并同意 CPML；
#   2) 通过设置环境变量 OMNIDUB_XTTS_LOCAL_PATH=<本地权重目录>，
#      或显式传入 model_path=<本地路径>。
_XTTS_LOCAL_PATH_ENV = "OMNIDUB_XTTS_LOCAL_PATH"
_models = {}
_tts_locks = {}
_models_lock = threading.Lock()
_cpu_model = None
_cpu_lock = threading.Lock()


@contextmanager
def _torch_load_weights_compat():
    """
    Some TTS/XTTS checkpoints fail on torch>=2.6 default weights_only=True.
    Temporarily force weights_only=False during model loading.
    """
    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _allowlist_unsupported_global_from_error(error_message: str) -> bool:
    """
    PyTorch 2.6+ uses weights_only=True by default in torch.load.
    Old XTTS checkpoints may reference config classes that need allowlisting.
    """
    pattern = r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)"
    match = re.search(pattern, error_message or "")
    if not match:
        return False
    full_name = match.group(1)
    if "." not in full_name:
        return False
    module_name, attr_name = full_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        torch.serialization.add_safe_globals([attr])
        logger.warning(f"Allowlisted torch global for XTTS checkpoint: {full_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to allowlist torch global: {full_name}, err={e}")
        return False

def init_TTS():
    load_model()
    
def load_model(model_path=None, device='auto'):
    device = str(resolve_torch_device(device))
    if model_path is None:
        model_path = os.getenv(_XTTS_LOCAL_PATH_ENV)
    if not model_path:
        logger.info(
            "XTTS v2 weights not configured: env %s unset and no model_path passed. "
            "XTTS engine unavailable; to opt in see CPML note in step042_tts_xtts.py.",
            _XTTS_LOCAL_PATH_ENV,
        )
        return None
    with _models_lock:
        existing = _models.get(device)
        if existing is not None:
            return existing
        _tts_locks.setdefault(device, threading.Lock())

    logger.info(f'Loading TTS model from {model_path} on {device}')
    t_start = time.time()
    last_error = None
    allowlisted = []
    max_attempts = 12
    loaded = None
    with _torch_load_weights_compat():
        for _ in range(max_attempts):
            try:
                loaded = TTS(model_path).to(device)
                break
            except pickle.UnpicklingError as e:
                last_error = e
                # Auto-fix the common torch>=2.6 checkpoint allowlist issue.
                if not _allowlist_unsupported_global_from_error(str(e)):
                    raise
                match = re.search(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)", str(e) or "")
                if match:
                    allowlisted.append(match.group(1))
            except Exception as e:
                last_error = e
                raise
    if loaded is None and last_error is not None:
        raise RuntimeError(
            f'Failed to load XTTS after {max_attempts} attempts. '
            f'allowlisted={allowlisted}. last_error={last_error}'
        )
    with _models_lock:
        _models[device] = loaded
    t_end = time.time()
    logger.info(f'TTS model loaded on {device} in {t_end - t_start:.2f}s')
    return loaded
    

def _is_cuda_runtime_error(error: Exception) -> bool:
    msg = str(error).lower()
    cuda_markers = [
        'device-side assert triggered',
        'indexselectsmallindex',
        'cuda error',
        'cublas',
        'cudnn',
    ]
    return any(marker in msg for marker in cuda_markers)


def _get_cpu_model(model_name=None):
    global _cpu_model
    if model_name is None:
        model_name = os.getenv(_XTTS_LOCAL_PATH_ENV)
    if not model_name:
        raise RuntimeError(
            "XTTS v2 CPU fallback unavailable: no model configured. "
            f"Set env {_XTTS_LOCAL_PATH_ENV} or pass model_name explicitly."
        )
    if _cpu_model is None:
        logger.warning('Initializing XTTS CPU fallback model...')
        with _torch_load_weights_compat():
            _cpu_model = TTS(model_name).to('cpu')
    return _cpu_model


def tts(text, output_path, speaker_wav, model_name=None, device='auto', language='zh-cn'):
    if os.path.exists(output_path):
        logger.info(f'TTS output already exists: {text}')
        return
    if model_name is None:
        model_name = os.getenv(_XTTS_LOCAL_PATH_ENV)
    device = str(resolve_torch_device(device))
    model = load_model(model_name, device)
    if model is None:
        raise RuntimeError(
            "XTTS v2 模型未配置——OmniDub 不会自动下载 CPML 协议下的 XTTS 权重。\n"
            "如需启用 XTTS：\n"
            "  1. 自行从 https://huggingface.co/coqui/XTTS-v2 或 Coqui TTS 模型库下载权重，\n"
            "     并由您本人同意 CPML 条款；\n"
            "  2. 设置环境变量 OMNIDUB_XTTS_LOCAL_PATH=<权重本地目录>，或\n"
            "     在调用 load_model / tts 时显式传入 model_path=<本地路径>。\n"
            "如需商用，建议改用 tts_engine='cosyvoice'（Apache 2.0），或自行联系 Coqui "
            "获取 XTTS v2 的商业授权。详见 https://coqui.ai/cpml"
        )
    tts_lock = _tts_locks.setdefault(device, threading.Lock())

    last_error = None
    for retry in range(3):
        try:
            # XTTS model is shared across pipeline workers; inference must be serialized.
            with tts_lock:
                wav = model.tts(text, speaker_wav=speaker_wav, language=language)
            wav = np.array(wav)
            save_wav(wav, output_path)
            logger.info(f'TTS {text}')
            return
        except Exception as e:
            last_error = e
            logger.warning(f'TTS failed: {text}')
            logger.warning(e)
            if _is_cuda_runtime_error(e):
                logger.warning('XTTS CUDA runtime error detected, retrying on CPU fallback...')
                try:
                    cpu_model = _get_cpu_model(model_name)
                    with _cpu_lock:
                        wav = cpu_model.tts(text, speaker_wav=speaker_wav, language=language)
                    wav = np.array(wav)
                    save_wav(wav, output_path)
                    logger.warning(f'TTS CPU fallback succeeded: {text}')
                    return
                except Exception as cpu_e:
                    last_error = cpu_e
                    logger.warning(f'TTS CPU fallback failed: {cpu_e}')

    raise RuntimeError(f'XTTS failed after retries, text={text}, err={last_error}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <speaker_wav_path>")
        sys.exit(1)
    speaker_wav = sys.argv[1]
    while True:
        text = input('Enter text: ')
        tts(text, f'playground/{text}.wav', speaker_wav)
        
