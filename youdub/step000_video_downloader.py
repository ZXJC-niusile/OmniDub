import os
import re
from loguru import logger
import yt_dlp

def sanitize_title(title):
    # Only keep numbers, letters, Chinese characters, and spaces
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title)
    return title


def get_target_folder(info, folder_path):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None

    output_folder = os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')

    return output_folder

def download_single_video(info, folder_path, resolution='1080p'):
    sanitized_title = sanitize_title(info['title'])
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None
    
    output_folder = os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}')
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder
    
    resolution = resolution.replace('p', '')
    ydl_opts = {
        'format': f'bestvideo[ext=mp4][height<={resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'writeinfojson': True,
        'writethumbnail': True,
        'outtmpl': os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}', 'download'),
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([info['webpage_url']])
    logger.info(f'Video downloaded in {output_folder}')
    return output_folder

def get_info_list_from_url(url, num_videos):
    if isinstance(url, str):
        url = [url]

    # Download JSON information first
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            result = ydl.extract_info(u, download=False)
            if 'entries' in result:
                for video_info in result['entries']:
                    if video_info is None:
                        continue  # 跳过私有或已下架视频
                    yield video_info
            else:
                if result is not None:
                    yield result
