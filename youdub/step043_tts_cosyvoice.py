import os
import sys

import torch
import torchaudio
from loguru import logger

# 找到 CosyVoice 的根目录
cosyvoice_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CosyVoice")

# Add CosyVoice root to sys.path
sys.path.append(cosyvoice_dir)

# Add Matcha-TTS dependency to sys.path
sys.path.append(os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice


cosyvoice = None
DEFAULT_MANDARIN_PROMPT_TEXT = "今天我们用标准普通话进行配音测试，语速平稳，发音清晰。"


def load_cosyvoice_model():
    global cosyvoice
    if cosyvoice is not None:
        return cosyvoice
    model_candidates = ["iic/CosyVoice-300M"]
    last_error = None
    for model_name in model_candidates:
        try:
            logger.info(f"Loading CosyVoice model: {model_name}")
            cosyvoice = CosyVoice(model_name)
            logger.info(f"CosyVoice model loaded from: {model_name}")
            return cosyvoice
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load CosyVoice model from {model_name}: {e}")
    raise RuntimeError(f"Unable to load CosyVoice model. last_error={last_error}")


def _ensure_under_30s(wav_path):
    """Return a path to a wav trimmed to ≤30s. Caches the trimmed file."""
    wav, sr = torchaudio.load(wav_path)
    max_samples = 30 * sr
    if wav.shape[-1] <= max_samples:
        return wav_path
    trimmed = wav[:, :max_samples]
    out = wav_path + "_30s.wav"
    torchaudio.save(out, trimmed, sr)
    logger.info(f"Trimmed reference audio from {wav.shape[-1]/sr:.1f}s to 30s: {out}")
    return out


def _collect_tts_chunks(result_generator):
    chunks = []
    for result in result_generator:
        if isinstance(result, dict) and "tts_speech" in result:
            chunks.append(result["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice returned no tts_speech chunks.")
    return chunks


def _is_probably_chinese_text(text: str) -> bool:
    if not text:
        return False
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def generate_cosyvoice_audio(text, prompt_speech_path, output_path, prompt_text=""):
    """
    Generate TTS audio with CosyVoice voice cloning.

    Args:
        text: Target text to synthesize.
        prompt_speech_path: Reference speaker wav path.
        output_path: Path to save synthesized wav.
        prompt_text: Transcript of the reference audio (improves pronunciation).
                     If empty, falls back to cross-lingual mode.
    """
    try:
        model = load_cosyvoice_model()

        prompt_path = _ensure_under_30s(prompt_speech_path)

        # 默认强制 zero-shot，优先保证普通话稳定性。
        # 若 prompt_text 缺失或明显不是中文，使用固定普通话锚文本。
        prompt_text = (prompt_text or "").strip()
        if not _is_probably_chinese_text(prompt_text):
            prompt_text = DEFAULT_MANDARIN_PROMPT_TEXT

        try:
            result_generator = model.inference_zero_shot(text, prompt_text, prompt_path)
            chunks = _collect_tts_chunks(result_generator)
        except Exception as zero_shot_err:
            logger.warning(f"CosyVoice zero-shot failed, fallback to cross-lingual. err={zero_shot_err}")
            result_generator = model.inference_cross_lingual(text, prompt_path)
            chunks = _collect_tts_chunks(result_generator)

        merged = torch.cat(chunks, dim=-1)
        if merged.dim() == 1:
            merged = merged.unsqueeze(0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, merged.cpu(), 22050)
        logger.info(f"CosyVoice synthesis succeeded: {output_path}")
        return output_path
    except Exception as e:
        logger.exception(f"CosyVoice synthesis failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("step043_tts_cosyvoice module loaded successfully.")
