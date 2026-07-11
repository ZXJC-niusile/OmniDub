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
    # Word-boundary-aware "AI" -> "人工智能" replacement.
    # The blind .replace('AI', ...) mis-fires on substrings like "TRAIN", "AIR",
    # "Shanghai", etc. We only substitute when "AI" is a standalone uppercase
    # token (not glued to other letters/digits on either side).
    text = re.sub(r'(?<![A-Za-z0-9])AI(?![A-Za-z0-9])', '人工智能', text)
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
    # Use the cheap duration probe instead of decoding the full wav just to
    # compute its length. The actual stretched audio is decoded below.
    current_length = float(librosa.get_duration(path=wav_path, sr=sample_rate))
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

    # Pass 1: synthesize every segment up-front and record the ORIGINAL intended
    # start times. We need the natural (un-stretched) durations so pass 2 can
    # decide how much speed-up is available before an overrun becomes inevitable,
    # and to detect which segments are timing-limited.
    original_starts = [float(line['start']) for line in transcript]
    original_ends = [float(line['end']) for line in transcript]

    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        tts_func(text, output_path, speaker_wav)
        if not os.path.isfile(output_path):
            raise RuntimeError(f'TTS output missing after synthesis: {output_path}')
        # Pass 2 will read the natural duration from disk via adjust_audio_length;
        # no need to cache it here.

    # Pass 2: assemble full_wav with drift-recovery scheduling.
    #
    # The original single-pass loop appended whatever the synthesizer produced
    # and let the next segment start from the growing write position. When one
    # segment was too long even after max ~22% speed-up, the overrun pushed
    # every later segment later, causing cumulative one-directional time drift
    # (audio progressively desynced from the video and the whole track grew
    # longer than the source).
    #
    # Drift-recovery strategy (forward O(n) pass):
    #   * We track `lag` = original_starts[i] - current_write_pos.
    #     Positive lag means we're behind the original timeline (good - we
    #     have room to grow into the source). Negative lag means we've already
    #     overrun the original slot.
    #   * Long intentional silences (> SHORT_GAP_NO_SILENCE_SEC) are kept as
    #     real silence. Short gap slacks are SPENDABLE: when we are behind
    #     the original timeline, we use them to re-sync instead of padding
    #     after overruns, which is what accumulated drift in the first place.
    #   * We never slow speech below natural (MAX_STRETCH_RATIO stays 1.0);
    #     if a clip would still overrun, we absorb the spill but immediately
    #     claw back from the upcoming gap on the NEXT iteration rather than
    #     letting the overrun compound.
    full_wav = np.zeros((0, ))
    n = len(transcript)
    for i, line in enumerate(transcript):
        original_start = original_starts[i]
        original_end = original_ends[i]
        original_length = max(0.05, original_end - original_start)
        next_start = original_starts[i + 1] if i < n - 1 else None

        last_end = len(full_wav) / 24000
        # How far the current write position is from the original timeline
        # for this segment. Positive = we're behind (have slack). Negative =
        # we've already drifted past the original start.
        lag = original_start - last_end

        # Determine the target placement time for this segment. We prefer
        # the original timeline when we haven't already overrun it; otherwise
        # we start immediately (no extra pre-silence would only worsen drift).
        if lag >= 0.0:
            # We have room. Pre-segment silence is naturally "spendable" for
            # drift recovery: long intentional pauses (> SHORT_GAP_NO_SILENCE_SEC)
            # are preserved by clamping desired_end in Pass 2 below; short gaps
            # are also kept as-is here and the post-pad logic recovers from any
            # accumulated overrun.
            pre_gap = lag
            if pre_gap > 1e-3:
                full_wav = np.concatenate(
                    (full_wav, np.zeros((int(pre_gap * 24000), ))))
            start = len(full_wav) / 24000
        else:
            # Already past the original start due to accumulated overrun.
            # Start immediately - do NOT add compensating pre-silence, that
            # would worsen drift. (Resetting at original_start is worse: it
            # would chop the previous clip and shift the absolute timeline.)
            start = last_end

        line['start'] = start

        # Desired end: original slot end, but never past the next segment's
        # original start (that's the primary drift source in the old loop).
        desired_end = start + original_length
        if next_start is not None:
            desired_end = min(desired_end, next_start)

        # Stretch the clip toward desired_end - start, capped at natural
        # length (MAX_STRETCH_RATIO = 1.0 means we never slow down).
        wav, length, speed_factor, current_length = adjust_audio_length(
            os.path.join(output_folder, f'{str(i).zfill(4)}.wav'),
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

        # Post-segment pad. KEY drift-recovery point: the inter-segment
        # silence up to the next segment's original start is a BUFFER we can
        # spend to re-sync. Instead of always padding up to pad_end (which
        # was the source of monotonic growth), we shrink the post-pad to
        # claw back accumulated lag. lag_after = original_next_start -
        # line_end; if it's already negative (we're past the next slot) we
        # add zero post-pad; if it's between 0 and SHORT_GAP_NO_SILENCE_SEC
        # we collapse it to zero so the next segment starts sooner and we
        # re-approach the original timeline; only keep long intentional gaps
        # (> SHORT_GAP_NO_SILENCE_SEC) as real silence.
        if next_start is not None:
            slack_to_next = next_start - line_end
            if slack_to_next >= SHORT_GAP_NO_SILENCE_SEC:
                # Long intentional pause - preserve it.
                post_pad = slack_to_next
            elif slack_to_next > 0.0:
                # Short / spendable gap. Collapse it to recover from drift
                # instead of letting small overruns compound.
                post_pad = 0.0
            else:
                # Already overrun into the next slot; no padding, the next
                # iteration will start immediately from line_end.
                post_pad = 0.0
        else:
            # Last segment: pad to original_end for clean wrap-up.
            post_pad = max(0.0, original_end - line_end)

        if post_pad > 1e-3:
            full_wav = np.concatenate(
                (full_wav, np.zeros((int(post_pad * 24000), ))))
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
