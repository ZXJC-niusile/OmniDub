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

_STAGE_BRACKETS_RE = re.compile(r'[\(\（\[\【][^)\）\]\】]*[\)\）\]\】]')
_SCORE_CONTEXT_BEFORE_RE = re.compile(r'(?:比分|战成|踢成|打成|领先|落后|以)\s*$')
_SCORE_CONTEXT_AFTER_RE = re.compile(r'^\s*(?:获胜|取胜|胜|负|平|战平|领先|落后|击败|小胜|大胜|险胜|惜败)')
_TIME_CONTEXT_BEFORE_RE = re.compile(
    r'(?:早上|上午|中午|下午|晚上|凌晨|傍晚|午夜|今[晚夜早]|明[晚早天]|昨[晚天]|'
    r'时间|当地|北京|伦敦|东京|纽约|于|在|从|到|至)\s*$'
)
_TIME_CONTEXT_AFTER_RE = re.compile(
    r'^\s*(?:开球|开赛|开始|开场|播出|直播|举行|出发|抵达|到达|结束|截止|整|左右|前|后)'
)
_RANGE_UNITS_PATTERN = (
    r'(?:分钟|秒钟?|小时|个?月|星期|周|天|日|年|岁|世纪|代|人|名|位|个|次|回|遍|倍|'
    r'米|千米|公里|厘米|毫米|克|千克|公斤|吨|元|块|美元|欧元|英镑|日元|人民币|万|亿|'
    r'%|％|度|摄氏度|华氏度|成|轮|场|局|集|季|章|节|页|行|条|项|种|座|辆|台|部|支|只|件)'
)


def _resolve_xtts_language(target_language: str) -> str:
    key = str(target_language or '简体中文').strip().lower()
    return _TARGET_TO_XTTS_LANG.get(key, 'zh-cn')


def _is_chinese_text_or_target(text: str, target_language: str = '简体中文') -> bool:
    if str(target_language or '').strip() in {'简体中文', '繁体中文', 'zh', 'zh-cn'}:
        return True
    return any('\u4e00' <= ch <= '\u9fff' for ch in (text or ''))


def _has_speakable_content(text: str) -> bool:
    """Return False for empty, punctuation-only, or bracketed stage-direction-only lines."""
    if not text:
        return False
    stripped = _STAGE_BRACKETS_RE.sub('', text).strip()
    return any(ch.isalnum() for ch in stripped)


def normalize_name_connector_notation(text: str) -> str:
    # Keep foreign names fluent in TTS:
    # - remove Chinese middle dots used as name connectors (e.g., "马丁·厄德高")
    # - collapse latin initials/segments like "A.B" / "A. B"
    text = re.sub(r'(?<=[A-Za-z\u4e00-\u9fff])[·•・･](?=[A-Za-z\u4e00-\u9fff])', '', text)
    text = re.sub(r'(?<=[A-Za-z])\.\s*(?=[A-Za-z])', '', text)
    return text


def normalize_iso_date_notation(text: str) -> str:
    """Convert YYYY-MM-DD / YYYY/MM/DD dates into Chinese YYYY年M月D日 before score/fraction rules."""
    pattern = r'(?<!\d)((?:19|20)\d{2})\s*[-/.]\s*(0?[1-9]|1[0-2])\s*[-/.]\s*(0?[1-9]|[12]\d|3[01])(?!\d)'
    return re.sub(
        pattern,
        lambda m: f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日",
        text,
    )


def normalize_clock_time_notation(text: str) -> str:
    """Convert HH:MM(:SS) clock expressions into Chinese 点/分/秒 so ':' is not read as score '比'."""
    text = re.sub(
        r'(?<!\d)([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)(?!\d)',
        lambda m: f"{int(m.group(1))}点{int(m.group(2))}分{int(m.group(3))}秒",
        text,
    )

    pattern = r'(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)(?!\d)'

    def _replace_hhmm(match):
        h_str, m_str = match.group(1), match.group(2)
        before = text[max(0, match.start() - 10):match.start()]
        after = text[match.end():match.end() + 8]
        if _SCORE_CONTEXT_BEFORE_RE.search(before) or _SCORE_CONTEXT_AFTER_RE.search(after):
            return match.group(0)
        is_clock = (
            bool(_TIME_CONTEXT_BEFORE_RE.search(before))
            or bool(_TIME_CONTEXT_AFTER_RE.search(after))
            or (len(h_str) == 2 and h_str.startswith('0'))
            or m_str in {'00', '15', '30', '45'}
        )
        if not is_clock:
            return match.group(0)
        h_val = int(h_str)
        m_val = int(m_str)
        if m_val == 0:
            return f"{h_val}点整"
        if m_val < 10:
            return f"{h_val}点零{m_val}分"
        return f"{h_val}点{m_val}分"

    return re.sub(pattern, _replace_hhmm, text)


def normalize_range_notation(text: str) -> str:
    """Convert numeric ranges (e.g. '3-5分钟', '18-25岁', '第10-15分钟', '5~10') into '到'."""
    text = re.sub(
        rf'(?<!\d)(\d{{1,4}}(?:\.\d+)?)\s*[-–—－~～]\s*(\d{{1,4}}(?:\.\d+)?)(?=\s*{_RANGE_UNITS_PATTERN})',
        r'\1到\2',
        text,
    )
    text = re.sub(
        r'((?:第|约|大约|大概|近|长达|耗时|历时|时速)\s*)(\d{1,4}(?:\.\d+)?)\s*[-–—－~～]\s*(\d{1,4}(?:\.\d+)?)(?!\s*[-–—－]\s*\d)',
        r'\1\2到\3',
        text,
    )
    text = re.sub(
        r'(?<!\d)(\d{1,4}(?:\.\d+)?)\s*[~～]\s*(\d{1,4}(?:\.\d+)?)(?!\d)',
        r'\1到\2',
        text,
    )
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
    if not _has_speakable_content(text):
        return ''
    text = _STAGE_BRACKETS_RE.sub('', text).strip()
    if not _has_speakable_content(text):
        return ''
    if not _is_chinese_text_or_target(text, target_language):
        return text

    text = re.sub(r'(?<![A-Za-z0-9])AI(?![A-Za-z0-9])', '人工智能', text)
    text = normalize_name_connector_notation(text)
    text = normalize_iso_date_notation(text)
    text = normalize_year_list_notation(text)
    text = normalize_clock_time_notation(text)
    text = normalize_formation_notation(text)
    text = normalize_range_notation(text)
    text = normalize_score_notation(text)
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text.strip() if _has_speakable_content(text) else ''


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
    if abs(speed_factor - 1.0) <= 1e-4:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
        return wav, len(wav) / float(sample_rate), 1.0, current_length

    stem, ext = os.path.splitext(wav_path)
    target_path = f"{stem}_adjusted{ext or '.wav'}"
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    sliced = wav[: int(adjusted_length * sample_rate)]
    actual_len = len(sliced) / float(sample_rate) if len(sliced) > 0 else adjusted_length
    return sliced, actual_len, speed_factor, current_length


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


def _resolve_speaker_wav(folder: str, speaker: str) -> str:
    """Resolve a valid non-empty reference WAV for speaker, falling back gracefully."""
    speaker_dir = os.path.join(folder, 'SPEAKER')
    primary = os.path.join(speaker_dir, f'{speaker}.wav')
    if os.path.isfile(primary) and os.path.getsize(primary) > 44:
        return primary
    if os.path.isdir(speaker_dir):
        for fn in sorted(os.listdir(speaker_dir)):
            if fn.endswith('.wav') and not fn.endswith('_30s.wav'):
                cand = os.path.join(speaker_dir, fn)
                if os.path.isfile(cand) and os.path.getsize(cand) > 44:
                    logger.warning(f'Speaker wav missing for {speaker}, falling back to {cand}')
                    return cand
    for fallback_name in ('audio_vocals.wav', 'audio.wav'):
        cand = os.path.join(folder, fallback_name)
        if os.path.isfile(cand) and os.path.getsize(cand) > 44:
            logger.warning(f'Speaker wav missing for {speaker}, falling back to {cand}')
            return cand
    return primary


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

    for line in transcript:
        line['speaker'] = line.get('speaker') or 'SPEAKER_00'

    speakers = {line['speaker'] for line in transcript}
    logger.info(f'Found {len(speakers)} speakers, tts_engine={tts_engine}')

    _prepare_segment_wav_cache(
        output_folder,
        {
            'tts_engine': tts_engine,
            'target_language': target_language,
            'formation_tts_style': os.getenv('FORMATION_TTS_STYLE', FORMATION_TTS_STYLE),
            'segments': [
                (line['speaker'], line.get('translation', ''))
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
        speaker = line['speaker']
        text = preprocess_text(line.get('translation', ''), target_language=target_language)
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        if not text:
            save_wav(np.zeros((int(0.1 * 24000),)), output_path)
            continue
        speaker_wav = _resolve_speaker_wav(folder, speaker)
        tts_func(text, output_path, speaker_wav, speaker)
        if not os.path.isfile(output_path):
            raise RuntimeError(f'TTS output missing after synthesis: {output_path}')

    # Pass 2: assemble full_wav with drift-recovery scheduling.
    # Importantly, line['end'] records ONLY the end of spoken audio (not trailing silence padding),
    # preventing subtitles from staying frozen across long instrumental/silent gaps.
    full_wav = np.zeros((0,))
    n = len(transcript)
    for i, line in enumerate(transcript):
        original_start = original_starts[i]
        original_end = original_ends[i]
        original_length = max(0.05, original_end - original_start)
        next_start = original_starts[i + 1] if i < n - 1 else None

        last_end = len(full_wav) / 24000.0
        lag = original_start - last_end

        if lag >= 0.0:
            if lag > 1e-3:
                full_wav = np.concatenate((full_wav, np.zeros((int(lag * 24000),))))
            start = len(full_wav) / 24000.0
        else:
            start = last_end

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
        speech_end = len(full_wav) / 24000.0
        line['start'] = round(start, 3)
        line['end'] = round(max(start + 0.05, speech_end), 3)

        if next_start is not None:
            slack_to_next = next_start - speech_end
            if slack_to_next >= SHORT_GAP_NO_SILENCE_SEC:
                post_pad = slack_to_next
            else:
                post_pad = 0.0
        else:
            post_pad = max(0.0, original_end - speech_end)

        if post_pad > 1e-3:
            full_wav = np.concatenate((full_wav, np.zeros((int(post_pad * 24000),))))

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
