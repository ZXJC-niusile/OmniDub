# -*- coding: utf-8 -*-
import json
import os
import re
import time

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

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

_CJK_TARGET_LANGUAGES = {'简体中文', '繁体中文', 'zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant'}


def _is_chinese_target(target_language: str) -> bool:
    return str(target_language or '').strip().lower() in {x.lower() for x in _CJK_TARGET_LANGUAGES}


def _contains_cjk_or_kana(text: str) -> bool:
    return any(
        ('\u4e00' <= ch <= '\u9fff') or ('\u3040' <= ch <= '\u30ff') or ('\uac00' <= ch <= '\ud7af')
        for ch in (text or '')
    )


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
    mid = len(transcript) // 2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length // 2
    return before[:length] + after[-length:]


def _extract_json_object(text: str):
    """Extract a JSON dict from an LLM reply without failing when multiple {...} blocks exist."""
    if not text:
        raise ValueError('Empty model response')
    cleaned = text.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    candidates = []
    pos = 0
    while True:
        brace = cleaned.find('{', pos)
        if brace == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(cleaned, brace)
            if isinstance(obj, dict):
                candidates.append(obj)
                pos = end_idx
                continue
        except json.JSONDecodeError:
            pass
        pos = brace + 1

    if not candidates:
        raise ValueError(f'No JSON object found in response: {cleaned[:200]}')

    # Prefer the last non-empty-value dict in case the prompt template was echoed first.
    for cand in reversed(candidates):
        if any(v not in ('', [], None) for v in cand.values()):
            return cand
    return candidates[-1]


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
    """Safely extract single-line text content from an OpenAI-compatible response object."""
    return _get_response_text_multiline(response).replace('\n', '')


def _get_response_text_multiline(response):
    """Safely extract text content while preserving newlines (needed for numbered batches)."""
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
            if 'title' in cleaned_title.lower() and cleaned_title.lower().startswith('the title'):
                raise ValueError('Invalid placeholder summary')
            summary_obj = {'title': cleaned_title, 'summary': cleaned_summary}
            break
        except Exception as e:
            logger.warning(f'Summary generation failed (attempt {retry + 1}/5): {e}')
            time.sleep(1)

    if summary_obj is None:
        raise RuntimeError('Summary generation failed: max retries exceeded')

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
            if f'the {target_language} title' in str(translation['title']):
                raise ValueError('Invalid placeholder translation')
            translated_title = str(translation['title']).strip()
            for lq, rq in (('"', '"'), ('“', '”'), ('‘', '’'), ("'", "'"), ('《', '》')):
                if len(translated_title) >= 2 and translated_title.startswith(lq) and translated_title.endswith(rq):
                    translated_title = translated_title[1:-1]
                    break
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
    raise RuntimeError('Translation summary failed: max retries exceeded')


def translation_postprocess(result, target_language='简体中文'):
    result = (result or '').strip()
    if not _is_chinese_target(target_language):
        return re.sub(r'(?<=\d),(?=\d)', '', result)

    result = re.sub(r'[（\(][^）\)]*[）\)]', '', result).strip() or result
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = (
        result.replace('²', '的平方')
        .replace('————', '：')
        .replace('——', '：')
        .replace('°', '度')
    )
    # Word-boundary-aware "AI" -> 人工智能: avoids corrupting tokens like TRAIN/AIR.
    result = re.sub(r'(?<![A-Za-z0-9])AI(?![A-Za-z0-9])', '人工智能', result)
    return result.strip()


def _strip_translation_wrappers(translation: str) -> str:
    translation = (translation or '').strip()
    if translation.startswith('```') and translation.endswith('```') and len(translation) >= 6:
        translation = translation[3:-3].strip()

    for prefix in ('翻译：“', '翻译："', '翻译:"', 'Translation: "', 'Translation:"', '译文：“', '译文："'):
        if prefix in translation:
            tail = translation.split(prefix, 1)[-1]
            closing = '”' if prefix.endswith('“') else '"'
            if closing in tail:
                extracted = tail.split(closing, 1)[0].strip()
                if extracted:
                    return extracted

    for lq, rq in (('“', '”'), ('"', '"'), ('‘', '’'), ("'", "'")):
        if len(translation) >= 2 and translation.startswith(lq) and translation.endswith(rq):
            inner = translation[1:-1].strip()
            if inner:
                return inner
    return translation


def valid_translation(text, translation, target_language='简体中文'):
    cleaned = _strip_translation_wrappers(translation)
    if not cleaned:
        return False, 'Empty translation. Output only the translated sentence.'

    # Language-aware length sanity check to reject verbose model chatter without
    # penalizing valid translations into longer alphabetic languages (EN/DE/FR/RU).
    if text is not None:
        if _is_chinese_target(target_language) and not _contains_cjk_or_kana(text):
            max_allowed = 20 if len(text) <= 10 else max(25, int(len(text) * 1.2))
        else:
            max_allowed = max(80, len(text) * 6)
        if len(cleaned) > max_allowed:
            return False, 'The translation is too long. Only translate the sentence and output nothing else.'

    if '\n' in cleaned:
        return False, 'Do not include newlines. Output only the translated sentence.'

    meta_markers = ['翻译：', '译文：', '这句', '简体中文', '繁体中文', 'Translation:']
    src_lower = (text or '').lower()
    for marker in meta_markers:
        if marker in cleaned and marker.lower() not in src_lower:
            return False, f"Don't include `{marker}` in the output. Output only the translation."

    return True, translation_postprocess(cleaned, target_language)


_ALPHA_SENTENCE_SPLIT_RE = re.compile(
    r'(?<![A-Z])(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bVs)(?<!\bvs)(?<!\bSt)(?<!\bNo)(?<!\bJr)(?<!\bSr)'
    r'([.!?]["”’\']?)\s+(?=[A-ZА-ЯЁ¡¿"“‘\'\u00C0-\u017F])'
)


def split_text_into_sentences(para):
    if not para:
        return []
    # CJK sentence boundaries
    para = re.sub(r'([。！？\?])([^，。！？\?”’》\s])', r"\1\n\2", para)
    para = re.sub(r'(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub(r'(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    # Alphabetic (EN/DE/FR/RU/ES) sentence boundaries with abbreviation guard
    para = _ALPHA_SENTENCE_SPLIT_RE.sub(r'\1\n', para)
    para = para.rstrip()
    return [s.strip() for s in para.split("\n") if s and s.strip()]


def split_sentences(translation):
    output_data = []
    for item in translation:
        start = float(item['start'])
        end = float(item['end'])
        text = item.get('text', '')
        speaker = item.get('speaker') or 'SPEAKER_00'
        translation_text = (item.get('translation') or '').strip()
        if not translation_text:
            translation_text = (text or '').strip()
        if not translation_text:
            continue

        sentences = split_text_into_sentences(translation_text)
        if not sentences:
            sentences = [translation_text]

        total_chars = max(1, sum(len(s) for s in sentences))
        duration_per_char = max(0.0, end - start) / total_chars
        for sentence in sentences:
            sentence_end = start + duration_per_char * len(sentence)
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence,
            })
            start = sentence_end
    return output_data


def _make_client():
    return OpenAI(
        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=os.getenv('OPENAI_API_KEY'),
    )


def _translation_system_content(info, target_language):
    if _is_chinese_target(target_language):
        return (
            f'You are a expert in the field of this video.\n{info}\n'
            f'Translate into {target_language}. 请充当翻译家，把内容翻译得自然、流畅、地道，'
            f'不要翻译腔，使用优美高雅的表达。术语要求：人工智能中的"agent"译为"智能体"；'
            f'强化学习中是`Q-Learning`而非`Queue Learning`；深度学习语境下的"Transformer"是'
            f'模型架构名，保持英文"Transformer"，不要译成"变压器"；专有名词、产品名、代码标识符'
            f'保持原文。数学公式写成 plain text，不要用 latex。确保准确、简洁，注意信达雅。'
        )
    return (
        f'You are an expert translator in the domain of this video.\n{info}\n'
        f'Translate accurately, naturally, and idiomatically into {target_language}. '
        f'Keep proper nouns, product names, code identifiers, and technical acronyms '
        f'(such as Transformer, Q-Learning) intact where appropriate. '
        f'Write math expressions as plain text without LaTeX.'
    )


_BATCH_LINE_RE = re.compile(r'^\s*[\[\(]?(\d+)[\]\)]?\s*[\.、:：\)）]\s*(.*\S)\s*$')


def _parse_numbered_batch(reply, expected, target_language='简体中文', batch_texts=None):
    """Parse a numbered reply into a list of length ``expected``; each slot is
    the validated translation for that 1-based index or ``None`` if missing/contaminated."""
    parsed = {}
    for raw in reply.splitlines():
        m = _BATCH_LINE_RE.match(raw)
        if not m:
            continue
        i = int(m.group(1))
        if 1 <= i <= expected and i not in parsed:
            val = m.group(2).strip()
            src_text = batch_texts[i - 1] if batch_texts and (i - 1) < len(batch_texts) else None
            ok, cleaned = valid_translation(src_text, val, target_language=target_language)
            if ok and cleaned:
                parsed[i] = cleaned
    return [parsed.get(i + 1) for i in range(expected)]


def _translate_batch(client, system_content, context_pairs, batch_texts, target_language, max_retries=3):
    n = len(batch_texts)
    merged = [None] * n

    context_block = ''
    if context_pairs:
        refs = '\n'.join(f'- {s[:40]} => {d[:40]}' for s, d in context_pairs[-5:])
        context_block = f'Previous translations (for terminology consistency only, do NOT repeat):\n{refs}\n\n'

    numbered = '\n'.join(f'{i + 1}. {t.replace(chr(10), " ").strip()}' for i, t in enumerate(batch_texts))
    base_user_content = (
        f'{context_block}Translate each of the {n} numbered lines below into {target_language}, '
        f'preserving exact line order and line count. Output ONLY {n} lines formatted as `index. translation` '
        f'(do not merge or split lines, no extra commentary).\n\n{numbered}'
    )
    format_emphasis = (
        f'\nIMPORTANT: Strictly output `{n}` lines as `number. translation` (ASCII period + space), '
        'no bullet points, no bold formatting, no code fences, and no explanations.'
    )

    for attempt in range(max_retries):
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
            got = _parse_numbered_batch(
                _get_response_text_multiline(response),
                n,
                target_language=target_language,
                batch_texts=batch_texts,
            )
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
    is_zh = _is_chinese_target(target_language)
    history = []
    for s, d in context_pairs[-10:]:
        history.append({'role': 'user', 'content': f'Translate:"{s}"'})
        if is_zh:
            history.append({'role': 'assistant', 'content': f'翻译：“{d}”'})
        else:
            history.append({'role': 'assistant', 'content': f'Translation: "{d}"'})

    if is_zh:
        few_shot = [
            {'role': 'user', 'content': f'使用地道的{target_language}Translate:"Knowledge is power."'},
            {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
        ]
        user_prompt = f'使用地道的{target_language}Translate:"{text}"'
    else:
        few_shot = [
            {'role': 'user', 'content': f'Translate into {target_language}: "Knowledge is power."'},
            {'role': 'assistant', 'content': 'Translation: "Knowledge is power."'},
        ]
        user_prompt = f'Translate into {target_language} (output ONLY `Translation: "..."`): "{text}"'

    for _ in range(max_retries):
        messages = [{'role': 'system', 'content': system_content}] + few_shot + history + [
            {'role': 'user', 'content': user_prompt}
        ]
        try:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, timeout=240, extra_body=extra_body
                )
            except Exception as api_e:
                raise RuntimeError(f'chat.completions.create failed: {_format_api_exception(api_e)}')
            ok, translation = valid_translation(
                text, _get_response_text(response), target_language=target_language
            )
            if ok:
                return translation
        except Exception as e:
            logger.error(e)
            time.sleep(1)
    return None


def _rescue_translate(client, text, target_language):
    """Last-ditch minimal-prompt translation (no history)."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f'Translate the following into {target_language}. '
                        f'Output only the translation, nothing else:\n"{text}"'
                    ),
                }
            ],
            timeout=240,
            extra_body=extra_body,
        )
        out = _strip_translation_wrappers(_get_response_text(response))
        return translation_postprocess(out, target_language) if out else None
    except Exception as e:
        logger.error(f'rescue translate failed: {e}')
        return None


def _translate(summary, transcript, target_language='简体中文'):
    client = _make_client()
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    system_content = _translation_system_content(info, target_language)
    n = len(transcript)
    texts = [line['text'] for line in transcript]
    full_translation = [None] * n

    context_pairs = []

    # Pass 1: batched translation (~10x fewer API requests).
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

    # Pass 2: fill any gap without breaking order.
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
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.isfile(translation_path) and os.path.getsize(translation_path) > 0:
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
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f'Transcript not found: {transcript_path}')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    if not transcript or not isinstance(transcript, list):
        raise RuntimeError(f'Invalid or empty transcript in {folder}')

    summary_path = os.path.join(folder, 'summary.json')
    summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                cached_summary = json.load(f)
            if (
                isinstance(cached_summary, dict)
                and 'title' in cached_summary
                and cached_summary.get('language', target_language) == target_language
            ):
                summary = cached_summary
        except Exception as e:
            logger.warning(f'Corrupt summary.json, regenerating: {e}')

    if summary is None:
        summary = summarize(info, transcript, target_language)
        if summary is None:
            raise RuntimeError(f'Failed to summarize {folder}')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation = _translate(summary, transcript, target_language)

    # Defensive: pad or trim translation list to match transcript length without mutating originals.
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

    translated_items = []
    for i, line in enumerate(transcript):
        entry = dict(line)
        entry['translation'] = translation[i]
        translated_items.append(entry)

    split_items = split_sentences(translated_items)
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(split_items, f, indent=2, ensure_ascii=False)
    return True
