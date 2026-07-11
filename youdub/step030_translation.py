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
# Lines translated per API call. Batching cuts request count ~10x vs the old
# one-call-per-line loop while preserving order via index tags. Env-overridable.
try:
    TRANSLATION_BATCH_SIZE = max(1, int(os.getenv('TRANSLATION_BATCH_SIZE', '10')))
except ValueError:
    TRANSLATION_BATCH_SIZE = 10


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
    client = _make_client()
    transcript_text = ensure_transcript_length(
        ' '.join(line['text'] for line in transcript),
        max_length=2000,
    )
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '

    full_description = (
        f'The following is the full content of the video:\n{info_message}\n'
        f'{transcript_text}\n{info_message}\n'
        f'According to the above content, detailedly Summarize the video in JSON format:\n'
        f'```json\n{{"title": "", "summary": ""}}\n```'
    )

    summary_system = (
        'You are a expert in the field of this video. Please summarize the video '
        'in JSON format.\n```json\n{"title": "the title of the video", '
        '"summary": "the summary of the video"}\n```'
    )
    # Constant format reminder reused on every retry. The old code appended the
    # same template to `retry_message` on each failure, which grew the prompt
    # unboundedly without adding any new information to the model.
    format_hint = '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'

    # Phase 1: produce source-language summary as a JSON object with stable keys.
    summary_obj = None
    for retry in range(5):
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': summary_system},
                        {'role': 'user', 'content': full_description + format_hint},
                    ],
                    timeout=240,
                    extra_body=extra_body,
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            raw = _get_response_text(response)
            logger.info(raw)
            parsed = _extract_json_object(raw)
            if 'title' not in parsed or 'summary' not in parsed:
                raise ValueError(f'Invalid summary json keys: {list(parsed.keys())}')
            cleaned_title = str(parsed['title']).replace('title:', '').strip()
            cleaned_summary = str(parsed['summary']).replace('summary:', '').strip()
            if 'title' in cleaned_title:
                raise Exception('Invalid summary')
            summary_obj = {'title': cleaned_title, 'summary': cleaned_summary}
            break
        except Exception as e:
            logger.warning(f'Summary generation failed (attempt {retry + 1}/5): {e}')
            time.sleep(1)

    if summary_obj is None:
        raise Exception('Summary generation failed: max retries exceeded')

    # Phase 2: translate the (title, summary, tags) tuple into target_language.
    title_en = summary_obj['title']
    summary_en = summary_obj['summary']
    tags = info['tags']

    translate_system = (
        f'You are a native speaker of {target_language}. Please translate the title '
        f'and summary into {target_language} in JSON format. '
        f'```json\n{{"title": "the {target_language} title of the video", '
        f'"summary": "the {target_language} summary of the video", '
        f'"tags": [list of tags in {target_language}]}}\n```.'
    )
    translate_user = (
        f'The title of the video is "{title_en}". The summary of the video is '
        f'"{summary_en}". Tags: {tags}.\nPlease translate the above title and '
        f'summary and tags into {target_language} in JSON format. '
        f'```json\n{{"title": "", "summary": "", "tags": []}}\n```. '
        f'Remember to translate the title and the summary and tags into '
        f'{target_language} in JSON.'
    )

    for retry in range(10):
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': translate_system},
                        {'role': 'user', 'content': translate_user},
                    ],
                    timeout=240,
                    extra_body=extra_body,
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            raw = _get_response_text(response)
            logger.info(raw)
            translation = _extract_json_object(raw)
            if 'title' not in translation or 'summary' not in translation or 'tags' not in translation:
                raise ValueError(f'Invalid translation json keys: {list(translation.keys())}')
            if target_language in translation['title'] or target_language in translation['summary']:
                raise Exception('Invalid translation')
            translated_title = str(translation['title']).strip()
            if (
                (translated_title.startswith('"') and translated_title.endswith('"'))
                or (translated_title.startswith('“') and translated_title.endswith('”'))
                or (translated_title.startswith('‘') and translated_title.endswith('’'))
                or (translated_title.startswith("'") and translated_title.endswith("'"))
                or (translated_title.startswith('《') and translated_title.endswith('》'))
            ):
                translated_title = translated_title[1:-1]
            return {
                'title': translated_title,
                'author': info['uploader'],
                'summary': str(translation['summary']),
                'tags': translation['tags'] if isinstance(translation['tags'], list) else [str(translation['tags'])],
                'language': target_language,
            }
        except Exception as e:
            logger.warning(f'Translation summary failed (attempt {retry + 1}/10): {e}')
            time.sleep(1)
    raise Exception('Translation summary failed: max retries exceeded')


def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    # Word-boundary-aware "AI" -> 人工智能: the old blind replace corrupted any
    # surviving English token containing "AI" (TRAIN/AIR/product names).
    result = re.sub(r'(?<![A-Za-z0-9])AI(?![A-Za-z0-9])', '人工智能', result)
    # NOTE: removed the blind '变压器'->'Transformer' swap — it corrupted genuine
    # mentions of electrical transformers. ML-term fidelity is now steered by the
    # translation prompt glossary instead of post-hoc string surgery.
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

            # Advance start for the next sentence in this item
            start = sentence_end
    return output_data
    
def _make_client():
    return OpenAI(
        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=os.getenv('OPENAI_API_KEY'),
    )


def _translation_system_content(info, target_language):
    # Shared guidance for batch / single-line / rescue paths. The glossary keeps
    # ML terms intact (e.g. "Transformer" must NOT become the electrical "变压器"),
    # replacing the old blind post-hoc string swaps.
    return (
        f'You are a expert in the field of this video.\n{info}\n'
        f'Translate into {target_language}. 请充当翻译家，把内容翻译得自然、流畅、地道，'
        f'不要翻译腔，使用优美高雅的表达。术语要求：人工智能中的"agent"译为"智能体"；'
        f'强化学习中是`Q-Learning`而非`Queue Learning`；深度学习语境下的"Transformer"是'
        f'模型架构名，保持英文"Transformer"，不要译成"变压器"；专有名词、产品名、代码标识符'
        f'保持原文。数学公式写成 plain text，不要用 latex。确保准确、简洁，注意信达雅。'
    )


def _get_response_text_multiline(response):
    """Like _get_response_text but preserves newlines (needed to parse the
    numbered batch format, which _get_response_text would flatten)."""
    if response is None:
        raise ValueError('Model response is None (API returned nothing).')
    choices = getattr(response, 'choices', None)
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(f'Model response has no choices. snapshot={_response_snapshot(response)}')
    message = getattr(choices[0], 'message', None)
    if message is None:
        raise ValueError(f'Model response first choice has no message. snapshot={_response_snapshot(response)}')
    content = getattr(message, 'content', None)
    if content is None:
        raise ValueError(f'Model response content is empty. snapshot={_response_snapshot(response)}')
    return str(content)


_BATCH_LINE_RE = re.compile(r'^\s*[\[\(]?(\d+)[\]\)]?\s*[\.、:：\)）]\s*(.*\S)\s*$')


def _parse_numbered_batch(reply, expected):
    """Parse a numbered reply into a list of length ``expected``; each slot is
    the translation for that 1-based index or ``None`` if missing/unparseable.
    Tolerant of extra prose lines and of quote wrapping."""
    parsed = {}
    for raw in reply.splitlines():
        m = _BATCH_LINE_RE.match(raw)
        if not m:
            continue
        i = int(m.group(1))
        if 1 <= i <= expected and i not in parsed:
            val = m.group(2).strip()
            for lq, rq in (('“', '”'), ('"', '"'), ('‘', '’'), ("'", "'")):
                if len(val) >= 2 and val.startswith(lq) and val.endswith(rq):
                    val = val[1:-1].strip()
                    break
            if val:
                parsed[i] = translation_postprocess(val)
    return [parsed.get(i + 1) for i in range(expected)]


def _translate_batch(client, system_content, context_pairs, batch_texts, target_language, max_retries=3):
    """Translate a batch of lines in ONE call. Returns a list aligned to
    ``batch_texts`` with ``None`` for any line not confidently obtained. Retries
    the whole batch up to ``max_retries`` times, merging partial results.
    On retries 2+ the user prompt adds a format emphasis to break out of the
    case where the model locked onto a malformed output pattern."""
    n = len(batch_texts)
    merged = [None] * n

    context_block = ''
    if context_pairs:
        refs = '\n'.join(f'- {s[:40]} => {d[:40]}' for s, d in context_pairs[-5:])
        context_block = f'已翻译片段（仅供术语一致性参考，不要重复输出）：\n{refs}\n\n'

    numbered = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(batch_texts))
    base_user_content = (
        f'{context_block}把下面 {n} 行分别翻译成{target_language}，保持行的顺序与数量。'
        f'只输出 {n} 行，每行格式为 `序号. 译文`，不要合并或拆分行，不要额外解释。\n\n{numbered}'
    )
    format_emphasis = (
        '\n注意：严格按 `数字 + 半角点 + 空格 + 译文` 一行一条输出，不要使用项目符号、'
        '不要加粗、不要写说明文字、不要代码块包裹。'
    )

    for attempt in range(max_retries):
        # Vary the prompt on retries: identical prompts keep producing identical
        # failures, so add format reinforcement starting from the 2nd try.
        user_content = base_user_content if attempt == 0 else base_user_content + format_emphasis
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content},
        ]
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, timeout=240, extra_body=extra_body
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            got = _parse_numbered_batch(_get_response_text_multiline(response), n)
            for j in range(n):
                if merged[j] is None and got[j]:
                    merged[j] = got[j]
            if all(v is not None for v in merged):
                break
        except Exception as e:
            logger.error(f'batch translate failed (attempt {attempt + 1}/{max_retries}): {e}')
            time.sleep(1)
    return merged


def _translate_single(client, system_content, context_pairs, text, target_language, max_retries=10):
    """Per-line translation with strict validation. Returns str or None."""
    history = []
    for s, d in context_pairs[-10:]:
        history.append({'role': 'user', 'content': f'Translate:"{s}"'})
        history.append({'role': 'assistant', 'content': f'翻译：“{d}”'})
    few_shot = [
        {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
    ]
    for _ in range(max_retries):
        messages = [{'role': 'system', 'content': system_content}] + few_shot + history + [
            {'role': 'user', 'content': f'使用地道的中文Translate:"{text}"'}
        ]
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, timeout=240, extra_body=extra_body
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            ok, translation = valid_translation(text, _get_response_text(response))
            if ok:
                return translation
        except Exception as e:
            logger.error(e)
            time.sleep(1)
    return None


def _rescue_translate(client, text, target_language):
    """Last-ditch minimal-prompt translation (no history) so an exhausted line
    is rescued into the target language instead of leaking the source text.
    Returns str or None."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user',
                 'content': f'Translate the following into {target_language}. '
                            f'Output only the translation, nothing else:\n"{text}"'}
            ],
            timeout=240,
            extra_body=extra_body,
        )
        out = _get_response_text(response).strip()
        for lq, rq in (('“', '”'), ('"', '"')):
            if len(out) >= 2 and out.startswith(lq) and out.endswith(rq):
                out = out[1:-1].strip()
                break
        return translation_postprocess(out) if out else None
    except Exception as e:
        logger.error(f'rescue translate failed: {e}')
        return None


def _translate(summary, transcript, target_language='简体中文'):
    client = _make_client()
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    system_content = _translation_system_content(info, target_language)
    n = len(transcript)
    texts = [line['text'] for line in transcript]
    full_translation = [None] * n  # index-aligned; guarantees order + length

    # Rolling (src, dst) context so terminology stays consistent across batches.
    context_pairs = []

    # Pass 1: batched translation (the ~10x request-count win).
    idx = 0
    while idx < n:
        batch_texts = texts[idx: idx + TRANSLATION_BATCH_SIZE]
        got = _translate_batch(client, system_content, context_pairs, batch_texts, target_language)
        for j, tr in enumerate(got):
            if tr:
                full_translation[idx + j] = tr
                context_pairs.append((batch_texts[j], tr))
        context_pairs = context_pairs[-30:]
        idx += len(batch_texts)

    # Pass 2: fill any gap (unparseable/failed line) WITHOUT breaking order.
    # Per-line strict retry -> minimal rescue prompt -> original text as last resort.
    for pos in range(n):
        if full_translation[pos] is not None:
            continue
        text = texts[pos]
        tr = _translate_single(client, system_content, context_pairs, text, target_language)
        if tr is None:
            tr = _rescue_translate(client, text, target_language)
        if tr is None:
            logger.warning(
                f'Translation failed for line {pos}: "{text[:60]}..." — using original text.'
            )
            tr = text
        full_translation[pos] = tr

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

