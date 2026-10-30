from contextlib import nullcontext
import os
import sys
import threading

from loguru import logger
import numpy as np
import torch
import torchaudio

from .utils import resolve_torch_device, save_wav

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
_cosyvoice_load_locks = {}
_registry_lock = threading.Lock()
_trim_lock = threading.Lock()


def _normalize_device_key(device: str) -> str:
    resolved = str(resolve_torch_device(device))
    if resolved == 'cuda':
        return 'cuda:0'
    return resolved


def _bind_cosyvoice_device(loaded, device_key: str) -> None:
    """Bind CosyVoice frontend & model explicitly to device_key ('cpu' or 'cuda:idx')
    so background threads (llm_job) and explicit CPU mode do not drift onto cuda:0."""
    if not hasattr(torch, 'device'):
        return
    target_dev = torch.device(device_key)
    frontend = getattr(loaded, 'frontend', None)
    if frontend is not None:
        frontend.device = target_dev
        spk2info = getattr(frontend, 'spk2info', None)
        if isinstance(spk2info, dict):
            for spk_entry in spk2info.values():
                if isinstance(spk_entry, dict):
                    for k, v in list(spk_entry.items()):
                        if hasattr(v, 'to'):
                            spk_entry[k] = v.to(target_dev)

    model = getattr(loaded, 'model', None)
    if model is not None:
        model.device = target_dev
        for sub_name in ('llm', 'flow', 'hift'):
            sub_mod = getattr(model, sub_name, None)
            if sub_mod is not None and hasattr(sub_mod, 'to'):
                sub_mod.to(target_dev)
        if target_dev.type == 'cpu':
            model.llm_context = nullcontext()
        elif target_dev.type == 'cuda' and torch.cuda.is_available():
            model.llm_context = torch.cuda.stream(torch.cuda.Stream(target_dev))


def load_cosyvoice_model(device: str = 'auto'):
    from cosyvoice.cli.cosyvoice import CosyVoice

    device_key = _normalize_device_key(device)
    with _registry_lock:
        existing = _cosyvoice_models.get(device_key)
        if existing is not None:
            return existing
        _cosyvoice_locks.setdefault(device_key, threading.Lock())
        dev_load_lock = _cosyvoice_load_locks.setdefault(device_key, threading.Lock())

    with dev_load_lock:
        with _registry_lock:
            existing = _cosyvoice_models.get(device_key)
            if existing is not None:
                return existing

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
                _bind_cosyvoice_device(loaded, device_key)
                logger.info(f"CosyVoice model loaded from: {model_name} on {device_key}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load CosyVoice model from {model_name}: {e}")

        if loaded is None:
            raise RuntimeError(f"Unable to load CosyVoice model. last_error={last_error}")

        with _registry_lock:
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
    text_clean = (text or "").strip()
    if not any(ch.isalnum() for ch in text_clean):
        logger.warning(f"Skipping non-speakable text in CosyVoice (writing 0.1s silence): {text!r}")
        save_wav(np.zeros((int(0.1 * 24000),)), output_path, sample_rate=24000)
        return output_path

    device_key = _normalize_device_key(device)
    model = load_cosyvoice_model(device=device_key)
    infer_lock = _cosyvoice_locks.setdefault(device_key, threading.Lock())
    prompt_path = _ensure_under_30s(prompt_speech_path)
    prompt_text = (prompt_text or "").strip()

    use_zero_shot = bool(prompt_text) and _same_script_family(prompt_text, text_clean)

    try:
        with infer_lock:
            if device_key.startswith('cuda:') and torch.cuda.is_available():
                gpu_idx = int(device_key.split(':', 1)[1])
                dev_ctx = torch.cuda.device(gpu_idx)
            else:
                dev_ctx = nullcontext()

            with dev_ctx:
                if use_zero_shot:
                    try:
                        result_generator = model.inference_zero_shot(text_clean, prompt_text, prompt_path)
                        chunks = _collect_tts_chunks(result_generator)
                        if not chunks:
                            raise RuntimeError("CosyVoice zero-shot yielded 0 chunks.")
                    except Exception as zero_shot_err:
                        logger.warning(
                            f"CosyVoice zero-shot failed, fallback to cross-lingual. err={zero_shot_err}"
                        )
                        result_generator = model.inference_cross_lingual(text_clean, prompt_path)
                        chunks = _collect_tts_chunks(result_generator)
                else:
                    result_generator = model.inference_cross_lingual(text_clean, prompt_path)
                    chunks = _collect_tts_chunks(result_generator)

        if not chunks:
            logger.warning(
                f"CosyVoice produced 0 speech chunks after text normalization for {text_clean!r}; "
                "writing 0.1s silence fallback."
            )
            save_wav(np.zeros((int(0.1 * getattr(model, 'sample_rate', 22050)),)), output_path, sample_rate=getattr(model, 'sample_rate', 22050))
            return output_path

        merged = torch.cat(chunks, dim=-1)
        if merged.dim() == 1:
            merged = merged.unsqueeze(0)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torchaudio.save(output_path, merged.cpu(), model.sample_rate)
        logger.info(f"CosyVoice synthesis succeeded: {output_path}")
        return output_path
    except Exception as e:
        logger.exception(f"CosyVoice synthesis failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("step043_tts_cosyvoice module loaded successfully.")
