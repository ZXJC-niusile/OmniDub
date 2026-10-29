import os
import sys
import threading

from loguru import logger
import torch
import torchaudio

from .utils import resolve_torch_device

# 找到 CosyVoice 的根目录
cosyvoice_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CosyVoice")
if cosyvoice_dir not in sys.path:
    sys.path.append(cosyvoice_dir)

# Only add local third_party/Matcha-TTS when an actual Matcha-TTS checkout is present
# (contains matcha/models), preventing the empty stub package from shadowing pip wheels.
matcha_dir = os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS")
if os.path.isdir(os.path.join(matcha_dir, "matcha", "models")) and matcha_dir not in sys.path:
    sys.path.append(matcha_dir)

_cosyvoice_models = {}
_cosyvoice_locks = {}
_registry_lock = threading.Lock()
_trim_lock = threading.Lock()


def _normalize_device_key(device: str) -> str:
    resolved = str(resolve_torch_device(device))
    if resolved == 'cuda':
        return 'cuda:0'
    return resolved


def load_cosyvoice_model(device: str = 'auto'):
    from cosyvoice.cli.cosyvoice import CosyVoice

    device_key = _normalize_device_key(device)
    with _registry_lock:
        existing = _cosyvoice_models.get(device_key)
        if existing is not None:
            return existing
        _cosyvoice_locks.setdefault(device_key, threading.Lock())

    model_candidates = ["iic/CosyVoice-300M"]
    last_error = None
    loaded = None
    for model_name in model_candidates:
        try:
            logger.info(f"Loading CosyVoice model: {model_name} on {device_key}")
            if device_key.startswith('cuda:') and torch.cuda.is_available():
                gpu_idx = int(device_key.split(':', 1)[1])
                with torch.cuda.device(gpu_idx):
                    loaded = CosyVoice(model_name)
            else:
                loaded = CosyVoice(model_name)
            logger.info(f"CosyVoice model loaded from: {model_name} on {device_key}")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load CosyVoice model from {model_name}: {e}")

    if loaded is None:
        raise RuntimeError(f"Unable to load CosyVoice model. last_error={last_error}")

    with _registry_lock:
        existing = _cosyvoice_models.get(device_key)
        if existing is not None:
            return existing
        _cosyvoice_models[device_key] = loaded
    return loaded


def _ensure_under_30s(wav_path: str) -> str:
    """Return a path to a wav trimmed to <=29.0s (safe margin for 16kHz resampling).
    Caches the trimmed file on disk."""
    out = wav_path + "_30s.wav"
    if os.path.isfile(out) and os.path.getsize(out) > 44:
        return out

    with _trim_lock:
        if os.path.isfile(out) and os.path.getsize(out) > 44:
            return out
        wav, sr = torchaudio.load(wav_path)
        max_samples = int(29.0 * sr)
        if wav.shape[-1] <= max_samples:
            return wav_path
        trimmed = wav[:, :max_samples]
        torchaudio.save(out, trimmed, sr)
        logger.info(f"Trimmed reference audio from {wav.shape[-1]/sr:.1f}s to 29.0s: {out}")
        return out


def _collect_tts_chunks(result_generator):
    chunks = []
    for result in result_generator:
        if isinstance(result, dict) and "tts_speech" in result:
            chunks.append(result["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice returned no tts_speech chunks.")
    return chunks


def _has_cjk(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in (text or ''))


def _same_script_family(text_a: str, text_b: str) -> bool:
    """Return True when prompt_text and target text share CJK vs non-CJK script."""
    if not text_a or not text_b:
        return False
    return _has_cjk(text_a) == _has_cjk(text_b)


def generate_cosyvoice_audio(
    text,
    prompt_speech_path,
    output_path,
    prompt_text="",
    device="auto",
):
    """
    Generate TTS audio with CosyVoice voice cloning.

    Uses zero-shot mode ONLY when a genuine verbatim transcript matching the
    reference speaker audio is available in the same language family; otherwise
    uses cross-lingual mode directly to avoid audio-prompt text mismatch artifacts.
    """
    device_key = _normalize_device_key(device)
    model = load_cosyvoice_model(device=device_key)
    infer_lock = _cosyvoice_locks.setdefault(device_key, threading.Lock())
    prompt_path = _ensure_under_30s(prompt_speech_path)
    prompt_text = (prompt_text or "").strip()

    use_zero_shot = bool(prompt_text) and _same_script_family(prompt_text, text)

    try:
        with infer_lock:
            if device_key.startswith('cuda:') and torch.cuda.is_available():
                gpu_idx = int(device_key.split(':', 1)[1])
                dev_ctx = torch.cuda.device(gpu_idx)
            else:
                from contextlib import nullcontext
                dev_ctx = nullcontext()

            with dev_ctx:
                if use_zero_shot:
                    try:
                        result_generator = model.inference_zero_shot(text, prompt_text, prompt_path)
                        chunks = _collect_tts_chunks(result_generator)
                    except Exception as zero_shot_err:
                        logger.warning(
                            f"CosyVoice zero-shot failed, fallback to cross-lingual. err={zero_shot_err}"
                        )
                        result_generator = model.inference_cross_lingual(text, prompt_path)
                        chunks = _collect_tts_chunks(result_generator)
                else:
                    result_generator = model.inference_cross_lingual(text, prompt_path)
                    chunks = _collect_tts_chunks(result_generator)

        merged = torch.cat(chunks, dim=-1)
        if merged.dim() == 1:
            merged = merged.unsqueeze(0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, merged.cpu(), model.sample_rate)
        logger.info(f"CosyVoice synthesis succeeded: {output_path}")
        return output_path
    except Exception as e:
        logger.exception(f"CosyVoice synthesis failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("step043_tts_cosyvoice module loaded successfully.")
