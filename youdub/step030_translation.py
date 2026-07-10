# -*- coding: utf-8 -*-
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import time
from loguru import logger

load_dotenv()

model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
logger.info(f'using model {model_name}')
extra_body = {
    'repetition_penalty': 1.1,
}
def get_necessary_info(info: dict):
    return {
        'title': info.get('title', 'Local Video'),
        'uploader': info.get('uploader', 'Local File'),
        'description': info.get('description', ''),
        'upload_date': info.get('upload_date', time.strftime('%Y%m%d')),
        'categories': info.get('categories', []),
        'tags': info.get('tags', []),
    }


def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript)//2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length//2
    return before[:length] + after[-length:]


def _extract_json_object(text: str):
    if not text:
        raise ValueError('Empty model response')
    cleaned = text.strip()
    # Try direct JSON object first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: extract first {...} block.
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        raise ValueError(f'No JSON object found in response: {cleaned[:200]}')
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError(f'JSON is not an object: {type(obj).__name__}')
    return obj


def _response_snapshot(response):
    try:
        if hasattr(response, 'model_dump'):
            dumped = response.model_dump()
            dumped_str = json.dumps(dumped, ensure_ascii=False)
            return dumped_str[:1000]
    except Exception:
        pass

    snapshot = {
        'type': type(response).__name__,
        'id': getattr(response, 'id', None),
        'object': getattr(response, 'object', None),
        'model': getattr(response, 'model', None),
        'choices_len': len(getattr(response, 'choices', []) or []),
        'usage': getattr(response, 'usage', None),
    }
    return json.dumps(snapshot, ensure_ascii=False)[:1000]


def _format_api_exception(e: Exception):
    parts = [f'{type(e).__name__}: {e}']
    status_code = getattr(e, 'status_code', None)
    if status_code is not None:
        parts.append(f'status_code={status_code}')

    body = getattr(e, 'body', None)
    if body:
        try:
            body_text = json.dumps(body, ensure_ascii=False)
        except Exception:
            body_text = str(body)
        parts.append(f'body={body_text[:800]}')

    response = getattr(e, 'response', None)
    if response is not None:
        text = getattr(response, 'text', None)
        if text:
            parts.append(f'response_text={str(text)[:800]}')

    return ' | '.join(parts)


def _get_response_text(response):
    """Safely extract text content from an OpenAI-compatible response object.

    Defensive: never raises ``NoneType is not subscriptable`` — every level
    of the response hierarchy is checked before access.
    """
    if response is None:
        raise ValueError('Model response is None (API returned nothing).')

    choices = getattr(response, 'choices', None)
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(
            f'Model response has no choices. snapshot={_response_snapshot(response)}'
        )

    first_choice = choices[0]
    message = getattr(first_choice, 'message', None)
    if message is None:
        raise ValueError(
            f'Model response first choice has no message. snapshot={_response_snapshot(response)}'
        )

    content = getattr(message, 'content', None)
    if content is None:
        raise ValueError(
            f'Model response content is empty. snapshot={_response_snapshot(response)}'
        )
    return str(content).replace('\n', '')


def summarize(info, transcript, target_language='简体中文'):
    client = OpenAI(
    # This is the default and can be omitted
    base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
    api_key=os.getenv('OPENAI_API_KEY')
)
    transcript = ' '.join(line['text'] for line in transcript)
    transcript = ensure_transcript_length(transcript, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '
    
    full_description = f'The following is the full content of the video:\n{info_message}\n{transcript}\n{info_message}\nAccording to the above content, detailedly Summarize the video in JSON format:\n```json\n{{"title": "", "summary": ""}}\n```'
    
    messages = [
        {'role': 'system',
            'content': f'You are a expert in the field of this video. Please detailedly summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary": "the summary of the video"}}\n```'},
        {'role': 'user', 'content': full_description},
    ]
    retry_message=''
    success = False
    for retry in range(5):
        try:
            messages = [
                {'role': 'system', 'content': f'You are a expert in the field of this video. Please summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary": "the summary of the video"}}\n```'},
                {'role': 'user', 'content': full_description+retry_message},
            ]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=240,
                    extra_body=extra_body
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            summary = _get_response_text(response)
            if '视频标题' in summary:
                raise Exception("Response contains meta-word '视频标题'")
            logger.info(summary)
            summary = _extract_json_object(summary)
            if 'title' not in summary or 'summary' not in summary:
                raise ValueError(f'Invalid summary json keys: {list(summary.keys())}')
            summary = {
                'title': str(summary['title']).replace('title:', '').strip(),
                'summary': str(summary['summary']).replace('summary:', '').strip()
            }
            if 'title' in summary['title']:
                raise Exception('Invalid summary')
            success = True
            break
        except Exception as e:
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'Summary generation failed\n{e}')
            time.sleep(1)
    if not success:
        raise Exception('Summary generation failed')
        
    title = summary['title']
    summary = summary['summary']
    tags = info['tags']
    messages = [
        {'role': 'system',
            'content': f'You are a native speaker of {target_language}. Please translate the title and summary into {target_language} in JSON format. ```json\n{{"title": "the {target_language} title of the video", "summary": "the {target_language} summary of the video", "tags": [list of tags in {target_language}]}}\n```.'},
        {'role': 'user',
            'content': f'The title of the video is "{title}". The summary of the video is "{summary}". Tags: {tags}.\nPlease translate the above title and summary and tags into {target_language} in JSON format. ```json\n{{"title": "", "summary": "", "tags": []}}\n```. Remember to translate the title and the summary and tags into {target_language} in JSON.'},
    ]
    for retry in range(10):
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=240,
                    extra_body=extra_body
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            summary = _get_response_text(response)
            logger.info(summary)
            summary = _extract_json_object(summary)
            if 'title' not in summary or 'summary' not in summary or 'tags' not in summary:
                raise ValueError(f'Invalid translation json keys: {list(summary.keys())}')
            if target_language in summary['title'] or target_language in summary['summary']:
                raise Exception('Invalid translation')
            title = str(summary['title']).strip()
            if (title.startswith('"') and title.endswith('"')) or (title.startswith('“') and title.endswith('”')) or (title.startswith('‘') and title.endswith('’')) or (title.startswith("'") and title.endswith("'")) or (title.startswith('《') and title.endswith('》')):
                title = title[1:-1]
            result = {
                'title': title,
                'author': info['uploader'],
                'summary': str(summary['summary']),
                'tags': summary['tags'] if isinstance(summary['tags'], list) else [str(summary['tags'])],
                'language': target_language
            }
            return result
        except Exception as e:
            logger.warning(f'Translation summary failed\n{e}')
            time.sleep(1)
    raise Exception('Translation summary failed: max retries exceeded')


def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result

def valid_translation(text, translation):
    
    if (translation.startswith('```') and translation.endswith('```')):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)
    
    if (translation.startswith('“') and translation.endswith('”')) or (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)
    
    if '翻译' in translation and '：“' in translation and '”' in translation:
        translation = translation.split('：“')[-1].split('”')[0]
        return True, translation_postprocess(translation)
    
    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split(':"')[-1].split('"')[0]
        return True, translation_postprocess(translation)
    
    if len(text) <= 10:
        if len(translation) > 15:
            return False, f'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text)*0.75:
        return False, f'The translation is too long. Only translate the following sentence and give me the result.'
    
    forbidden = ['翻译', '这句', '\n', '简体中文', '中文', 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."
    
    return True, translation_postprocess(translation)
def split_text_into_sentences(para):
    para = re.sub(r'([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def split_sentences(translation):
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item['translation']
        sentences = split_text_into_sentences(translation_text)
        duration_per_char = (item['end'] - item['start']
                             ) / len(translation_text)
        sentence_start = 0
        for sentence in sentences:
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start += len(sentence)
    return output_data
    
def _translate(summary, transcript, target_language='简体中文'):
    client = OpenAI(
        # This is the default and can be omitted
        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=os.getenv('OPENAI_API_KEY')
    )
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation = []
    fixed_message = [
        {'role': 'system', 'content': f'You are a expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}.下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。'},
        {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
        {'role': 'user', 'content': '使用地道的中文Translate:"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},]
    
    history = []
    for line in transcript:
        text = line['text']

        retry_message = 'Only translate the quoted sentence and give me the final translation.'
        translation = None  # will be set on success or left as None
        for retry in range(30):
            messages = fixed_message + \
                history[-30:] + [{'role': 'user',
                                  'content': f'使用地道的中文Translate:"{text}"'}]

            try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        timeout=240,
                        extra_body=extra_body
                    )
                except Exception as api_e:
                    raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
                translation = _get_response_text(response)
                logger.info(f'原文：{text}')
                logger.info(f'译文：{translation}')
                success, translation = valid_translation(text, translation)
                if not success:
                    retry_message += translation
                    raise Exception('Invalid translation')
                break
            except Exception as e:
                logger.error(e)
                if str(e) == 'Internal Server Error':
                    client = OpenAI(
                        base_url=os.getenv(
                            'OPENAI_API_BASE', 'https://api.openai.com/v1'),
                        api_key=os.getenv('OPENAI_API_KEY')
                    )
                time.sleep(1)

        # Safety net: if all 30 retries exhausted without a valid translation,
        # fall back to the original text so the pipeline never crashes.
        if translation is None:
            logger.warning(f'Translation exhausted retries for: {text[:60]}... — using original text.')
            translation = text

        full_translation.append(translation)
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'翻译：“{translation}”'})
        time.sleep(0.1)

    return full_translation

def translate(folder, target_language='简体中文'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True

    info_path = os.path.join(folder, 'download.info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        info = get_necessary_info(info)
    else:
        local_title = os.path.basename(folder).strip() or 'Local Video'
        info = get_necessary_info({
            'title': local_title,
            'uploader': 'Local File',
            'description': 'Generated from local video file.',
            'upload_date': time.strftime('%Y%m%d'),
            'categories': ['Local'],
            'tags': ['local', 'test'],
        })
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.warning(f'{info_path} not found, generated local metadata automatically.')

    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    if not transcript or not isinstance(transcript, list):
        logger.error(f'Invalid or empty transcript in {folder}')
        return False

    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            if not isinstance(summary, dict) or 'title' not in summary:
                raise ValueError('summary.json missing required keys')
        except Exception as e:
            logger.warning(f'Corrupt summary.json, regenerating: {e}')
            summary = summarize(info, transcript, target_language)
    else:
        summary = summarize(info, transcript, target_language)

    if summary is None:
        logger.error(f'Failed to summarize {folder}')
        return False
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(summary, transcript, target_language)

    # Defensive: pad or trim translation list to match transcript length.
    if len(translation) < len(transcript):
        logger.warning(
            f'Translation has {len(translation)} items but transcript has {len(transcript)}. '
            f'Padding missing entries with original text.'
        )
        for i in range(len(translation), len(transcript)):
            translation.append(transcript[i].get('text', ''))
    elif len(translation) > len(transcript):
        logger.warning(
            f'Translation has more items ({len(translation)}) than transcript ({len(transcript)}). Truncating.'
        )
        translation = translation[:len(transcript)]

    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    transcript = split_sentences(transcript)
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return True

