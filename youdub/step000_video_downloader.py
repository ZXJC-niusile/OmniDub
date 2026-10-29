import os
import re
import subprocess
from loguru import logger
import yt_dlp


def sanitize_title(title):
    title = str(title or '')
    # Only keep numbers, letters, Chinese characters, hyphens, underscores, and spaces
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def _resolve_upload_date(info: dict) -> str:
    for key in ('upload_date', 'release_date'):
        val = info.get(key)
        if val and val != 'Unknown':
            cleaned = re.sub(r'[^\w-]', '', str(val)).strip()
            if cleaned:
                return cleaned
    return '00000000'


def get_target_folder(info, folder_path):
    if not isinstance(info, dict):
        return None
    raw_title = info.get('title') or info.get('id') or 'Untitled'
    sanitized_title = sanitize_title(raw_title) or sanitize_title(info.get('id', '')) or 'video'
    sanitized_uploader = sanitize_title(info.get('uploader') or 'Unknown') or 'Unknown'
    upload_date = _resolve_upload_date(info)

    return os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}'
    )


def _ensure_mp4_output(output_folder: str) -> bool:
    target_mp4 = os.path.join(output_folder, 'download.mp4')
    if os.path.isfile(target_mp4) and os.path.getsize(target_mp4) > 0:
        return True

    if not os.path.isdir(output_folder):
        return False

    for ext in ('.mkv', '.webm', '.mov', '.flv', '.m4v', ''):
        candidate = os.path.join(output_folder, f'download{ext}')
        if candidate != target_mp4 and os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            logger.info(f'Remuxing {candidate} -> {target_mp4}')
            try:
                subprocess.run(
                    ['ffmpeg', '-loglevel', 'error', '-i', candidate, '-c', 'copy', target_mp4, '-y'],
                    check=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ['ffmpeg', '-loglevel', 'error', '-i', candidate, '-c:v', 'libx264', '-c:a', 'aac', target_mp4, '-y'],
                    check=True,
                )
            if os.path.isfile(target_mp4) and os.path.getsize(target_mp4) > 0:
                try:
                    os.remove(candidate)
                except OSError:
                    pass
                return True
    return False


def download_single_video(info, folder_path, resolution='1080p'):
    output_folder = get_target_folder(info, folder_path)
    if output_folder is None:
        return None

    target_mp4 = os.path.join(output_folder, 'download.mp4')
    if os.path.isfile(target_mp4) and os.path.getsize(target_mp4) > 0:
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder

    webpage_url = info.get('webpage_url') or info.get('url')
    if not webpage_url:
        logger.error(f'Missing webpage_url in video info: {info.get("title", "?")}')
        return None

    res_height = str(resolution).replace('p', '')
    ydl_opts = {
        'format': f'bestvideo[ext=mp4][height<={res_height}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'writeinfojson': True,
        'writethumbnail': True,
        'outtmpl': os.path.join(output_folder, 'download.%(ext)s'),
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([webpage_url])

    if not _ensure_mp4_output(output_folder):
        logger.error(f'Video download failed or produced no valid video file in {output_folder}')
        return None

    logger.info(f'Video downloaded in {output_folder}')
    return output_folder


def get_info_list_from_url(url, num_videos):
    if isinstance(url, str):
        url = [url]

    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            result = ydl.extract_info(u, download=False)
            if not result:
                logger.warning(f'Failed to extract video info from URL: {u}')
                continue
            if 'entries' in result and result['entries'] is not None:
                for video_info in result['entries']:
                    if video_info is None:
                        continue  # 跳过私有或已下架视频
                    yield video_info
            else:
                yield result
