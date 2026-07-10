import json
import os
import re
import librosa

from loguru import logger
import numpy as np

from .utils import save_wav, save_wav_norm
from .step042_tts_xtts import tts as xtts_tts
from .step043_tts_cosyvoice import generate_cosyvoice_audio
from .cn_tx import TextNorm
from audiostretchy.stretch import stretch_audio
normalizer = TextNorm()

# Timing controls tuned for natural Chinese dubbing:
# - avoid aggressive speed-up when translated text is longer
# - prefer mild slow-down to absorb short silent gaps
MIN_STRETCH_RATIO = 0.82
# Keep Chinese speech natural: never slow down for timing alignment.
MAX_STRETCH_RATIO = 1.0
SHORT_GAP_NO_SILENCE_SEC = 0.6
NEXT_SLOT_BORROW_SEC = 0.35
FORMATION_TTS_STYLE = os.getenv('FORMATION_TTS_STYLE', 'chinese_compact')
_DIGIT_TO_CN = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}


def normalize_name_connector_notation(text: str) -> str:
    # Keep foreign names fluent in TTS:
    # - remove Chinese middle dots used as name connectors (e.g., "马丁·厄德高")
    # - collapse latin initials/segments like "A.B" / "A. B"
    text = re.sub(r'(?<=[A-Za-z\u4e00-\u9fff])[·•・･](?=[A-Za-z\u4e00-\u9fff])', '', text)
    text = re.sub(r'(?<=[A-Za-z])\.\s*(?=[A-Za-z])', '', text)
    return text


def normalize_score_notation(text: str) -> str:
    # Score: 2-1 / 2:1 -> 2比1 (avoid touching formations like 4-4-2).
    pattern = r'(?<!\d)(\d{1,2})\s*[-–—－:：]\s*(\d{1,2})(?!\s*[-–—－]\s*\d)'

    def _replace(match):
        # Skip chained forms like 4-4-2 / 3-5-2 where this pair is only a tail.
        prefix = text[max(0, match.start() - 12):match.start()]
        if re.search(r'\d{1,2}\s*[-–—－]\s*$', prefix):
            return match.group(0)
        return f"{match.group(1)}比{match.group(2)}"

    return re.sub(pattern, _replace, text)


def normalize_year_list_notation(text: str) -> str:
    # Prevent "6、7年" from being read as "六十七年".
    # Convert list-like year expression into explicit range/readable form.
    # Also normalize "50、60、70年" -> "50年、60年、70年".
    text = re.sub(
        r'(?<!\d)(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*年(?!\d)',
        r'\1年、\2年、\3年',
        text
    )
    text = re.sub(r'(?<!\d)(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*年(?!\d)', r'\1到\2年', text)
    return text


def normalize_formation_notation(text: str) -> str:
    # Football formations like 4-4-2 are often read with pauses by TTS.
    # Convert them to speech-friendly form while keeping subtitles unchanged.
    # Require at least 3 groups to avoid accidental conversion of regular scores like 2-1.
    pattern = r'(?<!\d)(\d{1,2}(?:\s*[-–—－]\s*\d{1,2}){2,3})(?!\d)'

    def _replace(match):
        nums = re.findall(r'\d{1,2}', match.group(1))
        cn_nums = [''.join(_DIGIT_TO_CN.get(ch, ch) for ch in n) for n in nums]
        if FORMATION_TTS_STYLE in {'chinese_compact', 'cn_compact'}:
            # 4-4-2 -> 四四二
            return ''.join(cn_nums)
        if FORMATION_TTS_STYLE in {'chinese_split', 'cn_split'}:
            # 4-4-2 -> 四、四、二
            return '、'.join(cn_nums)
        if FORMATION_TTS_STYLE in {'split_dunhao', 'dunhao'}:
            return '、'.join(nums)
        if FORMATION_TTS_STYLE in {'split_comma', 'comma'}:
            return '，'.join(nums)
        if FORMATION_TTS_STYLE in {'compact', 'concat'}:
            return ''.join(nums)
        return '、'.join(nums)

    return re.sub(pattern, _replace, text)


def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = normalize_name_connector_notation(text)
    text = normalize_year_list_notation(text)
    text = normalize_formation_notation(text)
    text = normalize_score_notation(text)
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text
    
    
def adjust_audio_length(
    wav_path,
    desired_length,
    sample_rate=24000,
    min_speed_factor=MIN_STRETCH_RATIO,
    max_speed_factor=MAX_STRETCH_RATIO
):
    wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    current_length = len(wav) / sample_rate
    desired_length = max(0.05, desired_length)
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    adjusted_length = current_length * speed_factor
    target_path = wav_path.replace('.wav', f'_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[:int(adjusted_length * sample_rate)], adjusted_length, speed_factor, current_length

def _load_prompt_text(folder):
    """Try to load original transcript to use as prompt_text for CosyVoice."""
    for name in ('transcript.json',):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                # Collect the first ~15s of source text as pronunciation anchor
                lines = []
                for seg in segments:
                    if float(seg.get('end', 0)) > 15:
                        break
                    t = seg.get('text', '').strip()
                    if t:
                        lines.append(t)
                if lines:
                    return ' '.join(lines)
            except Exception:
                pass
    return ''


def _select_tts_func(tts_engine, num_speakers, prompt_text='', tts_device='auto'):
    """Return a callable(text, output_path, speaker_wav) for the chosen engine."""
    if tts_engine == 'cosyvoice':
        _pt = prompt_text  # capture for closure
        def _cosyvoice(text, output_path, speaker_wav):
            generate_cosyvoice_audio(text, speaker_wav, output_path, prompt_text=_pt)
        return _cosyvoice
    # default: xtts
    def _xtts(text, output_path, speaker_wav):
        xtts_tts(text, output_path, speaker_wav, device=tts_device)
    return _xtts


def generate_wavs(folder, tts_engine=None, tts_device='auto'):
    if tts_engine is None:
        tts_engine = 'xtts'
    if tts_engine not in {'xtts', 'cosyvoice'}:
        logger.warning(f'Unsupported tts_engine={tts_engine}, fallback to xtts.')
        tts_engine = 'xtts'

    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    speakers = set()

    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers, tts_engine={tts_engine}')

    prompt_text = _load_prompt_text(folder) if tts_engine == 'cosyvoice' else ''
    if prompt_text:
        logger.info(f'CosyVoice prompt_text: "{prompt_text[:80]}..."')
    else:
        logger.info('CosyVoice: no transcript found, using cross-lingual mode')

    tts_func = _select_tts_func(tts_engine, num_speakers, prompt_text, tts_device=tts_device)

    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        tts_func(text, output_path, speaker_wav)
        if not os.path.isfile(output_path):
            raise RuntimeError(f'TTS output missing after synthesis: {output_path}')
        start = float(line['start'])
        end = float(line['end'])
        original_length = max(0.05, end - start)
        last_end = len(full_wav) / 24000
        pre_gap = max(0.0, start - last_end)

        # Keep only long intentional pauses. Short gaps are absorbed by stretching
        # the current sentence, so audio remains continuous and natural.
        if pre_gap > SHORT_GAP_NO_SILENCE_SEC:
            full_wav = np.concatenate((full_wav, np.zeros((int(pre_gap * 24000), ))))

        start = len(full_wav) / 24000
        line['start'] = start

        desired_end = start + original_length
        next_start = None
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            next_start = float(next_line['start'])

        wav, length, speed_factor, current_length = adjust_audio_length(
            output_path,
            desired_end - start,
            min_speed_factor=MIN_STRETCH_RATIO,
            max_speed_factor=MAX_STRETCH_RATIO
        )

        if speed_factor <= MIN_STRETCH_RATIO + 1e-6 and current_length > length:
            logger.info(
                f'TTS line {i} is timing-limited; keep near-natural pace with minor spillover '
                f'(raw={current_length:.2f}s, out={length:.2f}s)'
            )

        full_wav = np.concatenate((full_wav, wav))
        line_end = start + length

        # If speech is still shorter than source slot, fill with silence instead of
        # stretching voice too much (prevents "dragging" artifacts).
        pad_end = desired_end
        if next_start is not None:
            pad_end = min(pad_end, next_start)
        post_pad = max(0.0, pad_end - line_end)
        if post_pad > 1e-3:
            full_wav = np.concatenate((full_wav, np.zeros((int(post_pad * 24000), ))))
            line_end += post_pad

        line['end'] = line_end
        
    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    max_full = np.max(np.abs(full_wav))
    max_vocal = np.max(np.abs(vocal_wav))
    if max_full > 0 and max_vocal > 0:
        full_wav = full_wav / max_full * max_vocal
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)
    
    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    combined_wav = full_wav + instruments_wav
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
        

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_folder> [tts_engine]")
        print(f"  tts_engine: xtts (default) | cosyvoice")
        sys.exit(1)
    folder = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else 'xtts'
    generate_wavs(folder, tts_engine=engine)
