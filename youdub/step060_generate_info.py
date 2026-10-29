import json
import os
from loguru import logger
from PIL import Image


def resize_thumbnail(folder, size=(1280, 960)):
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_path = None
    for suffix in image_suffix:
        candidate = os.path.join(folder, f'download{suffix}')
        if os.path.exists(candidate):
            image_path = candidate
            break
    if image_path is None:
        return None

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_ratio = img.width / max(1, img.height)
        target_ratio = size[0] / size[1]

        if img_ratio < target_ratio:
            new_height = size[1]
            new_width = max(1, int(new_height * img_ratio))
        else:
            new_width = size[0]
            new_height = max(1, int(new_width / img_ratio))

        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', size, 'black')
        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2
        canvas.paste(resized, (x_offset, y_offset))

        new_img_path = os.path.join(folder, 'video.png')
        canvas.save(new_img_path)
        return new_img_path


def generate_summary_txt(folder):
    summary_path = os.path.join(folder, 'summary.json')
    info_path = os.path.join(folder, 'download.info.json')

    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        title = f'{summary_data.get("title", "Untitled")} - {summary_data.get("author", "Unknown")}'
        body = summary_data.get('summary', '')
    elif os.path.isfile(info_path):
        logger.warning(f'summary.json not found in {folder}, falling back to download.info.json')
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        title = f'{info.get("title", "Untitled")} - {info.get("uploader", "Unknown")}'
        body = info.get('description', '')
    else:
        title = os.path.basename(folder) or 'Video'
        body = ''

    txt = f'{title}\n\n{body}'.strip() + '\n'
    out_path = os.path.join(folder, 'video.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    return out_path


def generate_info(folder):
    generate_summary_txt(folder)
    resize_thumbnail(folder)
