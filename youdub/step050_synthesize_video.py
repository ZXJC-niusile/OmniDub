# -*- coding: utf-8 -*-
import json
import os
import subprocess

from loguru import logger


def _snap_word_boundary(text: str, idx: int, total_len: int) -> int:
    """Snap character index to nearest whitespace boundary for space-delimited text,
    guaranteeing that words are never truncated mid-token."""
    if idx <= 0:
        return 0
    if idx >= total_len:
        return total_len
    if ' ' not in text:
        return idx
    if text[idx - 1].isspace() or (idx < total_len and text[idx].isspace()):
        return idx
    left = text.rfind(' ', 0, idx)
    right = text.find(' ', idx)
    if left == -1 and right == -1:
        return idx
    if left == -1:
        return right + 1
    if right == -1:
        return left
    return left if (idx - left) <= (right - idx) else (right + 1)


def split_text(
    input_data,
    punctuations=('，', '；', '：', '。', '？', '！', '\n', '”', ',', ';', '.', '?', '!'),
):
    def is_punctuation(char):
        return char in punctuations

    output_data = []
    for item in input_data:
        start = float(item["start"])
        end = float(item["end"])
        speaker = item.get("speaker") or "SPEAKER_00"
        original_text = (item.get("text", "") or "").strip()
        text = (item.get("translation", "") or "").strip()
        if not text:
            text = original_text
        if not text:
            continue

        sentence_start = 0
        src_total_len = max(1, len(original_text))
        tgt_total_len = max(1, len(text))
        duration_per_char = max(0.0, end - start) / tgt_total_len

        for i, char in enumerate(text):
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i + 1]):
                continue
            sentence = text[sentence_start : i + 1].strip()
            if not sentence:
                sentence_start = i + 1
                continue
            sentence_end = start + duration_per_char * (i + 1 - sentence_start)
            src_start_raw = int(round((sentence_start / tgt_total_len) * src_total_len))
            src_end_raw = int(round(((i + 1) / tgt_total_len) * src_total_len))
            src_start = _snap_word_boundary(original_text, src_start_raw, src_total_len)
            src_end = _snap_word_boundary(original_text, src_end_raw, src_total_len)
            if src_end <= src_start:
                src_end = min(src_total_len, max(src_start + 1, src_end_raw))
            sentence_original = original_text[src_start:src_end].strip() or original_text

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": sentence_original,
                "translation": sentence,
                "speaker": speaker,
            })

            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    total_sec, millisec = divmod(total_ms, 1000)
    hours, rem_sec = divmod(total_sec, 3600)
    minutes, sec = divmod(rem_sec, 60)
    return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"


def _escape_subtitles_path_for_filter(path: str) -> str:
    # ffmpeg filter parser treats these characters specially.
    escaped = path.replace('\\', '/')
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(':', r'\:')
    escaped = escaped.replace('[', r'\[').replace(']', r'\]')
    escaped = escaped.replace(',', r'\,').replace(';', r'\;')
    return escaped


def _build_atempo_filter(speed_up: float) -> str:
    """Build chained ffmpeg atempo filters supporting any positive speed factor
    (each individual atempo stage in ffmpeg must lie in [0.5, 2.0])."""
    speed = float(speed_up)
    if speed <= 0:
        raise ValueError(f'speed_up must be positive, got {speed_up}')
    stages = []
    while speed > 2.0:
        stages.append('atempo=2.0')
        speed /= 2.0
    while speed < 0.5:
        stages.append('atempo=0.5')
        speed /= 0.5
    stages.append(f'atempo={speed:.6g}')
    return ','.join(stages)


def wrap_text(text, max_line_char=30):
    """Wrap subtitle text cleanly without dropping trailing characters or slicing words mid-token."""
    text = ' '.join((text or '').replace('\n', ' ').split()).strip()
    if not text:
        return ''
    max_line_char = max(4, int(max_line_char))
    if len(text) <= max_line_char:
        return text

    num_lines = (len(text) + max_line_char - 1) // max_line_char
    target_width = min(max_line_char, (len(text) + num_lines - 1) // num_lines)

    # Space-delimited languages (EN/DE/FR/RU/ES): wrap on word boundaries.
    if ' ' in text:
        words = text.split(' ')
        lines = []
        current = ''
        for word in words:
            while len(word) > max_line_char:
                if current:
                    lines.append(current)
                    current = ''
                lines.append(word[:max_line_char])
                word = word[max_line_char:]
            if not word:
                continue
            if not current:
                current = word
            else:
                candidate = f'{current} {word}'
                if len(candidate) <= target_width or (
                    len(candidate) <= max_line_char and len(lines) >= num_lines - 1
                ):
                    current = candidate
                else:
                    lines.append(current)
                    current = word
        if current:
            lines.append(current)
        return '\n'.join(lines)

    # CJK / non-space-delimited text: slice completely using ceiling step so 0 chars are ever lost.
    return '\n'.join(text[k : k + target_width] for k in range(0, len(text), target_width))


def generate_srt(
    translation,
    srt_path,
    speed_up=1,
    max_line_char=30,
    orig_max_line_char=None,
    split_subtitles: bool = False,
):
    speed_up = float(speed_up) if float(speed_up) > 0 else 1.0
    if orig_max_line_char is None:
        orig_max_line_char = max(max_line_char + 4, int(round(max_line_char * 1.4)))
    if split_subtitles:
        translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(translation):
            start = format_timestamp(item['start'] / speed_up)
            end = format_timestamp(item['end'] / speed_up)
            original_text = wrap_text(item.get('text', ''), max_line_char=orig_max_line_char)
            translated_text = wrap_text(item.get('translation', ''), max_line_char=max_line_char)
            bilingual_text = (
                translated_text if not original_text else f'{translated_text}\n{original_text}'
            )
            f.write(f'{i + 1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{bilingual_text}\n\n')


def get_aspect_ratio(video_path):
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get('streams') or []
    if not streams:
        raise RuntimeError(f'No video stream found in {video_path}')
    dimensions = streams[0]
    height = int(dimensions['height'])
    if height <= 0:
        raise RuntimeError(f'Invalid video height={height} in {video_path}')
    return int(dimensions['width']) / height


def convert_resolution(aspect_ratio, resolution='1080p'):
    base_px = int(str(resolution).rstrip('pP'))
    if aspect_ratio < 1:
        width = base_px
        height = int(width / aspect_ratio)
    else:
        height = base_px
        width = int(height * aspect_ratio)
    # make sure width and height are divisible by 2
    width = max(2, width - width % 2)
    height = max(2, height - height % 2)
    return width, height


def _subtitle_layout_for_aspect(aspect_ratio: float) -> tuple[int, int, int]:
    """Return (max_line_char, orig_max_line_char, ass_font_size) adapted to landscape vs portrait."""
    if aspect_ratio < 1.0:
        # Portrait / 9:16 video (e.g. Shorts, Reels, Douyin): narrower lines + compact font
        max_line_char = max(12, int(round(28 * aspect_ratio)))
        orig_max_line_char = max(18, int(round(40 * aspect_ratio)))
        font_size = max(9, min(12, int(round(11 * min(1.0, aspect_ratio / 0.5625)))))
        return max_line_char, orig_max_line_char, font_size
    return 30, 42, 15


def synthesize_video(folder, subtitles=True, speed_up=1.05, fps=30, resolution='1080p'):
    output_video = os.path.join(folder, 'video.mp4')
    if os.path.isfile(output_video) and os.path.getsize(output_video) > 0:
        logger.info(f'Video already synthesized in {folder}')
        return

    # Prefer TTS-aligned subtitle timings produced by Step 4; fall back to Step 3 translation.json.
    aligned_translation_path = os.path.join(folder, 'translation_tts.json')
    source_translation_path = os.path.join(folder, 'translation.json')
    translation_path = (
        aligned_translation_path
        if os.path.isfile(aligned_translation_path)
        else source_translation_path
    )
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')

    for required in (translation_path, input_audio, input_video):
        if not os.path.isfile(required):
            raise FileNotFoundError(f'Required input missing for video synthesis: {required}')

    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    max_line_char, orig_max_line_char, font_size = _subtitle_layout_for_aspect(aspect_ratio)

    srt_path = os.path.join(folder, 'subtitles.srt')
    generate_srt(
        translation,
        srt_path,
        speed_up=speed_up,
        max_line_char=max_line_char,
        orig_max_line_char=orig_max_line_char,
    )

    # Run ffmpeg with cwd=folder so filter args use safe local filenames.
    input_video_name = os.path.basename(input_video)
    input_audio_name = os.path.basename(input_audio)
    output_video_name = os.path.basename(output_video)
    srt_path_ffmpeg = _escape_subtitles_path_for_filter(os.path.basename(srt_path))
    outline = max(1, int(round(font_size / 8)))
    font_name = os.getenv('SUBTITLE_FONT_NAME', 'sans-serif')
    video_speed_and_scale = f"setpts=PTS/{speed_up},scale={width}:{height}"
    audio_speed_filter = _build_atempo_filter(speed_up)
    subtitle_filter = (
        f"subtitles='{srt_path_ffmpeg}':"
        f"force_style='FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline={outline},WrapStyle=0'"
    )

    if subtitles and os.path.getsize(srt_path) > 0:
        filter_complex = f"[0:v]{video_speed_and_scale},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_and_scale}[v];[1:a]{audio_speed_filter}[a]"

    ffmpeg_command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', input_video_name,
        '-i', input_audio_name,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-r', str(fps),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        output_video_name,
        '-y',
    ]
    subprocess.run(ffmpeg_command, check=True, cwd=folder)
