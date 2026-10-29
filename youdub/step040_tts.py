import copy
import json
import os
import re

from audiostretchy.stretch import stretch_audio
import librosa
from loguru import logger
import numpy as np

from .cn_tx import TextNorm
from .utils import compute_fingerprint, save_wav, save_wav_norm

normalizer = TextNorm()

# Timing controls tuned for natural dubbing:
# - avoid aggressive speed-up when translated text is longer
# - never slow speech below natural pace
MIN_STRETCH_RATIO = 0.82
MAX_STRETCH_RATIO = 1.0
SHORT_GAP_NO_SILENCE_SEC = 0.6
FORMATION_TTS_STYLE = os.getenv('FORMATION_TTS_STYLE', 'chinese_compact')
_DIGIT_TO_CN = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}
_TARGET_TO_XTTS_LANG = {
    '简体中文': 'zh-cn',
    '繁体中文': 'zh-cn',
    'zh': 'zh-cn',
    'zh-cn': 'zh-cn',
    'english': 'en',
    'en': 'en',
    'deutsch': 'de',
    'de': 'de',
    'français': 'fr',
    'francais': 'fr',
    'fr': 'fr',
    'русский': 'ru',
    'ru': 'ru',
    'español': 'es',
    'es': 'es',
    'ja': 'ja',
    'ko': 'ko',
}


def _resolve_xtts_language(target_language: str) -> str:
    key = str(target_language or '简体中文').strip().lower()
    return _TARGET_TO_XTTS_LANG.get(key, 'zh-cn')


def _is_chinese_text_or_target(text: str, target_language: str = '简体中文') -> bool:
    if str(target_language or '').strip() in {'简体中文', '繁体中文', 'zh', 'zh-cn'}:
        return True
    return any('\u4e00' <= ch <= '\u9fff' for ch in (text or ''))


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
        prefix = text[max(0, match.start() - 12):match.start()]
        if re.search(r'\d{1,2}\s*[-–—－]\s*$', prefix):
            return match.group(0)
        return f"{match.group(1)}比{match.group(2)}"

    return re.sub(pattern, _replace, text)


def normalize_year_list_notation(text: str) -> str:
    # Prevent "6、7年" from being read as "六十七年".
    text = re.sub(
        r'(?<!\d)(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*年(?!\d)',
        r'\1年、\2年、\3年',
        text,
    )
    text = re.sub(r'(?<!\d)(\d{1,2})\s*[、,，]\s*(\d{1,2})\s*年(?!\d)', r'\1到\2年', text)
    return text


def normalize_formation_notation(text: str) -> str:
    # Football formations like 4-4-2 -> 四四二 (or configured style).
    style = os.getenv('FORMATION_TTS_STYLE', FORMATION_TTS_STYLE)
    pattern = r'(?<!\d)(\d{1,2}(?:\s*[-–—－]\s*\d{1,2}){2,3})(?!\d)'

    def _replace(match):
        nums = re.findall(r'\d{1,2}', match.group(1))
        cn_nums = [''.join(_DIGIT_TO_CN.get(ch, ch) for ch in n) for n in nums]
        if style in {'chinese_compact', 'cn_compact'}:
            return ''.join(cn_nums)
        if style in {'chinese_split', 'cn_split'}:
            return '、'.join(cn_nums)
        if style in {'split_dunhao', 'dunhao'}:
            return '、'.join(nums)
        if style in {'split_comma', 'comma'}:
            return '，'.join(nums)
        if style in {'compact', 'concat'}:
            return ''.join(nums)
        return '、'.join(nums)

    return re.sub(pattern, _replace, text)


def preprocess_text(text, target_language='简体中文'):
    text = (text or '').strip()
    if not _is_chinese_text_or_target(text, target_language):
        return text

    text = re.sub(r'(?<![A-Za-z0-9])AI(?![A-Za-z0-9])', '人工智能', text)
    text = normalize_name_connector_notation(text)
    text = normalize_year_list_notation(text)
    text = normalize_formation_notation(text)
    text = normalize_score_notation(text)
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text.strip()


def adjust_audio_length(
    wav_path,
    desired_length,
    sample_rate=24000,
    min_speed_factor=MIN_STRETCH_RATIO,
    max_speed_factor=MAX_STRETCH_RATIO,
):
    current_length = float(librosa.get_duration(path=wav_path, sr=sample_rate))
    if current_length <= 1e-4:
        return np.zeros((int(0.05 * sample_rate),)), 0.05, 1.0, 0.0
    desired_length = max(0.05, desired_length)
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor
    )
    adjusted_length = current_length * speed_factor
    target_path = wav_path.replace('.wav', '_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[: int(adjusted_length * sample_rate)], adjusted_length, speed_factor, current_length


def _load_speaker_prompt_texts(folder: str, speakers: set[str]) -> dict[str, str]:
    """Load verbatim reference prompt text per speaker (saved alongside SPEAKER/{spk}.wav)."""
    prompts = {}
    speaker_dir = os.path.join(folder, 'SPEAKER')
    for spk in speakers:
        txt_path = os.path.join(speaker_dir, f'{spk}.txt')
        if os.path.isfile(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    prompts[spk] = content
            except OSError:
                pass
    return prompts


def _prepare_segment_wav_cache(output_folder: str, cache_params: dict) -> None:
    """Ensure wavs/ holds segments only for the active TTS engine & parameters.
    Purges stale segment wavs if engine/language/text changed, while keeping
    valid completed segments when resuming an interrupted run of the same config."""
    os.makedirs(output_folder, exist_ok=True)
    fp_path = os.path.join(output_folder, '.segments.fingerprint')
    current_fp = compute_fingerprint(cache_params)
    cached_fp = None
    if os.path.isfile(fp_path):
        try:
            with open(fp_path, 'r', encoding='utf-8') as f:
                cached_fp = f.read().strip()
        except OSError:
            cached_fp = None

    if cached_fp == current_fp:
        return

    for entry in os.listdir(output_folder):
        if entry.endswith('.wav'):
            try:
                os.remove(os.path.join(output_folder, entry))
            except OSError:
                pass
    with open(fp_path, 'w', encoding='utf-8') as f:
        f.write(current_fp)


def _select_tts_func(
    tts_engine,
    speaker_prompts: dict[str, str],
    tts_device='auto',
    target_language='简体中文',
):
    """Return a callable(text, output_path, speaker_wav, speaker_id) for the chosen engine."""
    if tts_engine == 'cosyvoice':
        from .step043_tts_cosyvoice import generate_cosyvoice_audio

        def _cosyvoice(text, output_path, speaker_wav, speaker_id):
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 44:
                return
            prompt_text = speaker_prompts.get(speaker_id, '')
            generate_cosyvoice_audio(
                text,
                speaker_wav,
                output_path,
                prompt_text=prompt_text,
                device=tts_device,
            )
        return _cosyvoice

    from .step042_tts_xtts import tts as xtts_tts
    xtts_lang = _resolve_xtts_language(target_language)

    def _xtts(text, output_path, speaker_wav, speaker_id):
        xtts_tts(
            text,
            output_path,
            speaker_wav,
            device=tts_device,
            language=xtts_lang,
        )

    return _xtts


def generate_wavs(folder, tts_engine=None, tts_device='auto', target_language='简体中文'):
    if tts_engine is None:
        tts_engine = 'xtts'
    if tts_engine not in {'xtts', 'cosyvoice'}:
        logger.warning(f'Unsupported tts_engine={tts_engine}, fallback to xtts.')
        tts_engine = 'xtts'

    source_translation_path = os.path.join(folder, 'translation.json')
    aligned_translation_path = os.path.join(folder, 'translation_tts.json')
    output_folder = os.path.join(folder, 'wavs')

    with open(source_translation_path, 'r', encoding='utf-8') as f:
        # Work on a deep copy so Step 3's translation.json stays immutable across reruns.
        transcript = copy.deepcopy(json.load(f))

    speakers = {line.get('speaker', 'SPEAKER_00') for line in transcript}
    logger.info(f'Found {len(speakers)} speakers, tts_engine={tts_engine}')

    _prepare_segment_wav_cache(
        output_folder,
        {
            'tts_engine': tts_engine,
            'target_language': target_language,
            'formation_tts_style': os.getenv('FORMATION_TTS_STYLE', FORMATION_TTS_STYLE),
            'segments': [
                (line.get('speaker', 'SPEAKER_00'), line.get('translation', ''))
                for line in transcript
            ],
        },
    )

    speaker_prompts = _load_speaker_prompt_texts(folder, speakers) if tts_engine == 'cosyvoice' else {}
    tts_func = _select_tts_func(
        tts_engine,
        speaker_prompts,
        tts_device=tts_device,
        target_language=target_language,
    )

    # Pass 1: synthesize every segment up-front and record original intended bounds.
    original_starts = [float(line['start']) for line in transcript]
    original_ends = [float(line['end']) for line in transcript]

    for i, line in enumerate(transcript):
        speaker = line.get('speaker', 'SPEAKER_00')
        text = preprocess_text(line.get('translation', ''), target_language=target_language)
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        if not text:
            save_wav(np.zeros((int(0.1 * 24000),)), output_path)
            continue
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        tts_func(text, output_path, speaker_wav, speaker)
        if not os.path.isfile(output_path):
            raise RuntimeError(f'TTS output missing after synthesis: {output_path}')

    # Pass 2: assemble full_wav with drift-recovery scheduling.
    full_wav = np.zeros((0,))
    n = len(transcript)
    for i, line in enumerate(transcript):
        original_start = original_starts[i]
        original_end = original_ends[i]
        original_length = max(0.05, original_end - original_start)
        next_start = original_starts[i + 1] if i < n - 1 else None

        last_end = len(full_wav) / 24000
        lag = original_start - last_end

        if lag >= 0.0:
            if lag > 1e-3:
                full_wav = np.concatenate((full_wav, np.zeros((int(lag * 24000),))))
            start = len(full_wav) / 24000
        else:
            start = last_end

        line['start'] = start

        desired_end = start + original_length
        if next_start is not None:
            desired_end = min(desired_end, next_start)

        wav, length, speed_factor, current_length = adjust_audio_length(
            os.path.join(output_folder, f'{str(i).zfill(4)}.wav'),
            desired_end - start,
            min_speed_factor=MIN_STRETCH_RATIO,
            max_speed_factor=MAX_STRETCH_RATIO,
        )

        if speed_factor <= MIN_STRETCH_RATIO + 1e-6 and current_length > length:
            logger.info(
                f'TTS line {i} is timing-limited; keep near-natural pace with minor spillover '
                f'(raw={current_length:.2f}s, out={length:.2f}s)'
            )

        full_wav = np.concatenate((full_wav, wav))
        line_end = start + length

        if next_start is not None:
            slack_to_next = next_start - line_end
            if slack_to_next >= SHORT_GAP_NO_SILENCE_SEC:
                post_pad = slack_to_next
            else:
                post_pad = 0.0
        else:
            post_pad = max(0.0, original_end - line_end)

        if post_pad > 1e-3:
            full_wav = np.concatenate((full_wav, np.zeros((int(post_pad * 24000),))))
            line_end += post_pad

        line['end'] = line_end

    vocal_wav, _ = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    max_full = np.max(np.abs(full_wav)) if len(full_wav) > 0 else 0.0
    max_vocal = np.max(np.abs(vocal_wav)) if len(vocal_wav) > 0 else 0.0
    if max_full > 0 and max_vocal > 0:
        full_wav = full_wav / max_full * max_vocal
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))

    # Save aligned timings separately instead of mutating Step 3's translation.json.
    with open(aligned_translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    instruments_wav, _ = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    if len_full_wav > len_instruments_wav:
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant'
        )
    elif len_instruments_wav > len_full_wav:
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant'
        )
    combined_wav = full_wav + instruments_wav
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_folder> [tts_engine]")
        print("  tts_engine: xtts (default) | cosyvoice")
        sys.exit(1)
    folder = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else 'xtts'
    generate_wavs(folder, tts_engine=engine)
