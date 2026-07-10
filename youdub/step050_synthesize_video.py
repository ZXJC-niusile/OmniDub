# -*- coding: utf-8 -*-
import json
import os
import subprocess
import time

from loguru import logger


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    # Chinese punctuation marks for sentence ending

    # Function to check if a character is a Chinese ending punctuation
    def is_punctuation(char):
        return char in punctuations

    # Process each item in the input data
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = (item.get("text", "") or "").strip()
        sentence_start = 0
        src_total_len = max(1, len(original_text))
        tgt_total_len = max(1, len(text))

        # Calculate the duration for each character
        duration_per_char = (item["end"] - item["start"]) / len(text)
        for i, char in enumerate(text):
            # If the character is a punctuation, split the sentence
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)
            src_start = int(round((sentence_start / tgt_total_len) * src_total_len))
            src_end = int(round(((i + 1) / tgt_total_len) * src_total_len))
            src_end = min(src_total_len, max(src_start + 1, src_end))
            sentence_original = original_text[src_start:src_end].strip() or original_text

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": sentence_original,
                "translation": sentence,
                "speaker": speaker
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start = i + 1

    return output_data
    
def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def _escape_subtitles_path_for_filter(path: str) -> str:
    # ffmpeg filter parser treats these characters specially.
    escaped = path.replace('\\', '/')
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(':', r'\:')
    escaped = escaped.replace('[', r'\[').replace(']', r'\]')
    escaped = escaped.replace(',', r'\,').replace(';', r'\;')
    return escaped

def wrap_text(text, max_line_char=30):
    text = ' '.join((text or '').replace('\n', ' ').split()).strip()
    if not text:
        return ''
    lines = len(text) // (max_line_char + 1) + 1
    avg = min(round(len(text) / lines), max_line_char)
    avg = max(1, avg)
    return '\n'.join([text[i*avg:(i+1)*avg] for i in range(lines)])

def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(translation):
            start = format_timestamp(item['start']/speed_up)
            end = format_timestamp(item['end']/speed_up)
            original_text = wrap_text(item.get('text', ''), max_line_char=max_line_char + 12)
            translated_text = wrap_text(item.get('translation', ''), max_line_char=max_line_char)
            bilingual_text = translated_text if not original_text else f'{translated_text}\n{original_text}'
            f.write(f'{i+1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{bilingual_text}\n\n')


def get_aspect_ratio(video_path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'json', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    dimensions = json.loads(result.stdout)['streams'][0]
    return dimensions['width'] / dimensions['height']


def convert_resolution(aspect_ratio, resolution='1080p'):
    if aspect_ratio < 1:
        width = int(resolution[:-1])
        height = int(width / aspect_ratio)
    else:
        height = int(resolution[:-1])
        width = int(height * aspect_ratio)
    # make sure width and height are divisibal by 2
    width = width - width % 2
    height = height - height % 2
    
    # return f'{width}x{height}'
    return width, height
    
def synthesize_video(folder, subtitles=True, speed_up=1.05, fps=30, resolution='1080p'):
    if os.path.exists(os.path.join(folder, 'video.mp4')):
        logger.info(f'Video already synthesized in {folder}')
        return
    
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    if not os.path.exists(translation_path) or not os.path.exists(input_audio):
        return
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)
        
    srt_path = os.path.join(folder, 'subtitles.srt')
    output_video = os.path.join(folder, 'video.mp4')
    generate_srt(translation, srt_path, speed_up)
    # Run ffmpeg with cwd=folder so filter args use safe local filenames.
    input_video_name = os.path.basename(input_video)
    input_audio_name = os.path.basename(input_audio)
    output_video_name = os.path.basename(output_video)
    srt_path_ffmpeg = _escape_subtitles_path_for_filter(os.path.basename(srt_path))
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution = f'{width}x{height}'
    font_size = int(width/128)
    outline = int(round(font_size/8))
    video_speed_filter = f"setpts=PTS/{speed_up}"
    audio_speed_filter = f"atempo={speed_up}"
    subtitle_filter = (
        f"subtitles='{srt_path_ffmpeg}':"
        f"force_style='FontName=Arial,FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline={outline},WrapStyle=2'"
    )
    
    if subtitles:
        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video_name,
        '-i', input_audio_name,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-r', str(fps),
        '-s', resolution,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_video_name,
        '-y'
    ]
    subprocess.run(ffmpeg_command, check=True, cwd=folder)
    time.sleep(1)
    

