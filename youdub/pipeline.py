# -*- coding: utf-8 -*-
"""
OmniDub — Production Pipeline

Exactly two public entry points:
    process_online_video(url, ...)
    process_local_video(file_path, ...)

Each step is guarded by a parameter- and upstream-input-aware fingerprint check:
if the expected artefacts already exist on disk and match both current parameters
and upstream artefact digests, the step is skipped; any upstream change cascades
automatically through all downstream steps.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

from loguru import logger

from .text_frontend import normalization_identity

# ── step imports ──────────────────────────────────────────────────────
from .step000_video_downloader import (
    download_single_video,
    get_info_list_from_url,
    get_target_folder,
)
from .step010_demucs_vr import (
    extract_audio_from_video,
    init_demucs,
    separate_audio,
)
from .step020_whisperx import (
    init_whisperx,
    transcribe_audio,
)
from .step030_translation import translate
from .step040_tts import generate_wavs
from .step050_synthesize_video import synthesize_video
from .step060_generate_info import generate_info
from .utils import (
    mark_step_done,
    reserved_cuda_devices,
    should_skip_step,
)

_LOCAL_PREP_LOCK = threading.Lock()
_COSYVOICE_UNSUPPORTED_TARGETS = {
    'deutsch', 'de',
    'français', 'francais', 'fr',
    'русский', 'ru',
    'español', 'espanol', 'es',
}


# ── configuration ─────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Immutable pipeline settings with sensible production defaults."""

    root_folder: str = "videos"

    # Step 0 — download
    resolution: str = "1080p"
    num_videos: int = 20

    # Step 1 — demucs
    demucs_model: str = "htdemucs_ft"
    demucs_device: str = "auto"
    demucs_shifts: int = 5

    # Step 2 — whisperx
    whisper_model: str = "large"
    whisper_download_root: str = "models/ASR/whisper"
    whisper_batch_size: int = 32
    whisper_device: str = "auto"
    source_language: str = "auto"
    whisper_diarization: bool = True
    whisper_min_speakers: Optional[int] = None
    whisper_max_speakers: Optional[int] = None

    # Step 3 — translation
    translation_target_language: str = "简体中文"

    # Step 4 — TTS
    tts_engine: str = "cosyvoice"  # "cosyvoice" | "xtts"
    tts_device: str = "auto"

    # Step 5 — video compositing
    subtitles: bool = True
    speed_up: float = 1.05
    fps: int = 30
    target_resolution: str = "1080p"

    # Misc
    max_retries: int = 3
    max_workers: int = 1


# ── helpers ───────────────────────────────────────────────────────────
def _sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return cleaned or f"local_video_{int(time.time())}"


def _file_input_token(path: str) -> str:
    """Compute a deterministic content/stat token for an upstream artefact.
    Small JSON/text files (<= 2 MB) are hashed by content; large media files
    are tracked via (size, mtime_ns) for sub-millisecond checks."""
    if not os.path.isfile(path):
        return 'missing'
    try:
        st = os.stat(path)
        if st.st_size <= 2 * 1024 * 1024:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            return f'sha256:{h.hexdigest()}'
        return f'stat:{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return 'unreadable'


def _copy_or_remux_to_mp4(src_path: str, target_mp4: str) -> None:
    ext = os.path.splitext(src_path)[1].lower()
    if ext == '.mp4':
        shutil.copy2(src_path, target_mp4)
        return
    logger.info(f"Remuxing non-MP4 local video ({ext or 'unknown'}) -> {target_mp4}")
    try:
        subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', src_path, '-c', 'copy', target_mp4, '-y'],
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                'ffmpeg', '-loglevel', 'error', '-i', src_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                target_mp4, '-y',
            ],
            check=True,
        )


def _prepare_local_video_folder(local_video_path: str, root_folder: str) -> str:
    """Copy/remux a local file into the standard folder layout without basename collisions or stale reuse."""
    if not local_video_path:
        raise ValueError("Local video path is empty.")
    if not os.path.isfile(local_video_path):
        raise FileNotFoundError(f"Local video not found: {local_video_path}")

    abs_src = os.path.abspath(local_video_path)
    st = os.stat(abs_src)
    src_meta = {
        "abspath": abs_src,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

    base_name = os.path.splitext(os.path.basename(abs_src))[0]
    sanitized_base = _sanitize_folder_name(base_name)

    with _LOCAL_PREP_LOCK:
        folder = os.path.join(root_folder, sanitized_base)
        marker_path = os.path.join(folder, ".local_source.json")
        info_path = os.path.join(folder, "download.info.json")

        # Disambiguate when another file from a different directory already owns `folder`.
        existing_owner = None
        if os.path.isfile(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    existing_owner = json.load(f).get("abspath")
            except Exception:
                existing_owner = None
        elif os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    raw_url = json.load(f).get("webpage_url")
                if raw_url:
                    existing_owner = os.path.abspath(raw_url)
            except Exception:
                existing_owner = None

        if existing_owner and os.path.normcase(existing_owner) != os.path.normcase(abs_src):
            short_hash = hashlib.sha1(os.path.normcase(abs_src).encode("utf-8")).hexdigest()[:8]
            folder = os.path.join(root_folder, f"{sanitized_base}_{short_hash}")
            marker_path = os.path.join(folder, ".local_source.json")
            info_path = os.path.join(folder, "download.info.json")

        os.makedirs(folder, exist_ok=True)
        target_video = os.path.join(folder, "download.mp4")

        cached_meta = None
        if os.path.isfile(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
            except Exception:
                cached_meta = None

        source_unchanged = (
            os.path.isfile(target_video)
            and os.path.getsize(target_video) > 0
            and cached_meta is not None
            and cached_meta.get("size") == src_meta["size"]
            and cached_meta.get("mtime_ns") == src_meta["mtime_ns"]
        )

        if not source_unchanged:
            if os.path.exists(target_video):
                try:
                    os.remove(target_video)
                except OSError:
                    pass
            _copy_or_remux_to_mp4(abs_src, target_video)
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(src_meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Prepared local video at {target_video}")
        else:
            logger.info(f"Using up-to-date local video: {target_video}")

        if not os.path.exists(info_path) or not source_unchanged:
            info = {
                "title": base_name or "Local Video",
                "uploader": "Local File",
                "description": "Generated from local video file.",
                "upload_date": time.strftime("%Y%m%d"),
                "categories": ["Local"],
                "tags": ["local", "test"],
                "webpage_url": abs_src,
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved local metadata: {info_path}")

    return folder


# ── step specs (params + upstream inputs + outputs) ───────────────────
def _cache_audio(folder: str) -> bool:
    audio_path = os.path.join(folder, "audio.wav")
    return os.path.isfile(audio_path) and os.path.getsize(audio_path) > 1024


def _has_speaker_wavs(folder: str) -> bool:
    spk_dir = os.path.join(folder, "SPEAKER")
    if not os.path.isdir(spk_dir):
        return False
    return any(
        fn.endswith(".wav") and os.path.getsize(os.path.join(spk_dir, fn)) > 44
        for fn in os.listdir(spk_dir)
    )


def _vocals_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str]]:
    return (
        'step010_demucs',
        {
            'demucs_model': cfg.demucs_model,
            'demucs_shifts': cfg.demucs_shifts,
            'upstream_video': _file_input_token(os.path.join(folder, 'download.mp4')),
        },
        [
            os.path.join(folder, 'audio_vocals.wav'),
            os.path.join(folder, 'audio_instruments.wav'),
        ],
    )


def _transcript_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str]]:
    return (
        'step020_whisperx',
        {
            'whisper_model': cfg.whisper_model,
            'source_language': cfg.source_language,
            'whisper_diarization': cfg.whisper_diarization,
            'upstream_vocals': _file_input_token(os.path.join(folder, 'audio_vocals.wav')),
        },
        [
            os.path.join(folder, 'transcript.json'),
        ],
    )


def _translation_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str]]:
    return (
        'step030_translation',
        {
            'translation_target_language': cfg.translation_target_language,
            'model_name': os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
            'tbatch': os.getenv('TRANSLATION_BATCH_SIZE', '10'),
            'upstream_transcript': _file_input_token(os.path.join(folder, 'transcript.json')),
        },
        [
            os.path.join(folder, 'translation.json'),
        ],
    )


def _tts_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str], list[str]]:
    return (
        'step040_tts',
        {
            'tts_engine': cfg.tts_engine,
            'text_normalization': normalization_identity(cfg.tts_engine),
            'target_language': cfg.translation_target_language,
            'formation_tts_style': os.getenv('FORMATION_TTS_STYLE', 'chinese_compact'),
            'upstream_translation': _file_input_token(os.path.join(folder, 'translation.json')),
            'upstream_vocals': _file_input_token(os.path.join(folder, 'audio_vocals.wav')),
            'upstream_instruments': _file_input_token(os.path.join(folder, 'audio_instruments.wav')),
        },
        [
            os.path.join(folder, 'audio_combined.wav'),
            os.path.join(folder, 'translation_tts.json'),
        ],
        [
            os.path.join(folder, 'wavs'),
            os.path.join(folder, 'audio_tts.wav'),
        ],
    )


def _video_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str]]:
    aligned_json = os.path.join(folder, 'translation_tts.json')
    sub_json = aligned_json if os.path.isfile(aligned_json) else os.path.join(folder, 'translation.json')
    return (
        'step050_synthesize_video',
        {
            'tts_engine': cfg.tts_engine,
            'target_language': cfg.translation_target_language,
            'subtitles': cfg.subtitles,
            'speed_up': cfg.speed_up,
            'fps': cfg.fps,
            'target_resolution': cfg.target_resolution,
            'upstream_video': _file_input_token(os.path.join(folder, 'download.mp4')),
            'upstream_audio': _file_input_token(os.path.join(folder, 'audio_combined.wav')),
            'upstream_subtitles_json': _file_input_token(sub_json),
        },
        [
            os.path.join(folder, 'video.mp4'),
        ],
    )


def _info_spec(folder: str, cfg: PipelineConfig) -> tuple[str, dict, list[str]]:
    return (
        'step060_generate_info',
        {
            'target_language': cfg.translation_target_language,
            'upstream_summary': _file_input_token(os.path.join(folder, 'summary.json')),
            'upstream_info': _file_input_token(os.path.join(folder, 'download.info.json')),
        },
        [
            os.path.join(folder, 'video.txt'),
        ],
    )


# ── model warm-up ─────────────────────────────────────────────────────
def _warm_up_models(cfg: PipelineConfig, devices: list[str], task_count: int = 1) -> None:
    """Pre-load only the active models across unique devices actually needed for the batch."""
    needed = max(1, int(task_count))
    unique_devices = list(dict.fromkeys((devices or ['auto'])[:needed]))

    def _warmup_tts_on(dev: str):
        if cfg.tts_engine == 'cosyvoice':
            from .step043_tts_cosyvoice import load_cosyvoice_model
            return load_cosyvoice_model(device=dev)
        from .step042_tts_xtts import init_TTS
        loaded = init_TTS(device=dev)
        if loaded is None:
            raise RuntimeError(
                'XTTS v2 weights are not configured (OMNIDUB_XTTS_LOCAL_PATH is unset). '
                'Configure local XTTS weights or switch tts_engine to "cosyvoice".'
            )
        return loaded

    for dev in unique_devices:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                f"demucs@{dev}": pool.submit(
                    init_demucs,
                    model_name=cfg.demucs_model,
                    device=dev,
                    shifts=cfg.demucs_shifts,
                ),
                f"tts@{dev}": pool.submit(_warmup_tts_on, dev),
                f"whisperx@{dev}": pool.submit(
                    init_whisperx,
                    model_name=cfg.whisper_model,
                    download_root=cfg.whisper_download_root,
                    device=dev,
                    diarization=cfg.whisper_diarization,
                ),
            }
            for name, fut in futures.items():
                try:
                    fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Model warmup failed [{name}]: {exc}") from exc


def _cfg_with_device(cfg: PipelineConfig, device: str) -> PipelineConfig:
    if not isinstance(device, str):
        return cfg
    return replace(
        cfg,
        demucs_device=device,
        whisper_device=device,
        tts_device=device,
    )


# ── core pipeline (single folder) ────────────────────────────────────
def _run_pipeline(folder: str, cfg: PipelineConfig) -> None:
    """Execute the full pipeline for one video folder, skipping cached steps."""

    logger.info(f"[Pipeline] START  folder={folder}")

    # ── Step 1: extract audio + vocal separation ──────────────────────
    s1_name, s1_params, s1_outputs = _vocals_spec(folder, cfg)
    if not should_skip_step(
        folder, s1_name, s1_params, s1_outputs,
        extra_cleanup_paths=[os.path.join(folder, 'audio.wav')],
    ):
        if not _cache_audio(folder):
            logger.info("[Step 1] Extracting audio from video ...")
            extract_audio_from_video(folder)
        logger.info("[Step 1] Separating vocals / instruments ...")
        separate_audio(
            folder,
            model_name=cfg.demucs_model,
            device=cfg.demucs_device,
            progress=True,
            shifts=cfg.demucs_shifts,
        )
        mark_step_done(folder, s1_name, s1_params, s1_outputs)
    else:
        logger.info("[Step 1] SKIP — vocals already separated")

    # ── Step 2: speech recognition ────────────────────────────────────
    s2_name, s2_params, s2_outputs = _transcript_spec(folder, cfg)
    s2_skip = should_skip_step(
        folder, s2_name, s2_params, s2_outputs,
        extra_cleanup_paths=[os.path.join(folder, 'SPEAKER')],
    )
    if s2_skip and not _has_speaker_wavs(folder):
        # Regenerate if SPEAKER reference wavs were removed independently.
        for stale_p in (s2_outputs[0], os.path.join(folder, f'.{s2_name}.fingerprint')):
            try:
                os.remove(stale_p)
            except OSError:
                pass
        s2_skip = False
    if not s2_skip:
        logger.info("[Step 2] Transcribing speech ...")
        transcribe_audio(
            folder,
            model_name=cfg.whisper_model,
            download_root=cfg.whisper_download_root,
            device=cfg.whisper_device,
            batch_size=cfg.whisper_batch_size,
            diarization=cfg.whisper_diarization,
            min_speakers=cfg.whisper_min_speakers,
            max_speakers=cfg.whisper_max_speakers,
            source_language=cfg.source_language,
        )
        mark_step_done(folder, s2_name, s2_params, s2_outputs)
    else:
        logger.info("[Step 2] SKIP — transcript already exists")

    # ── Step 3: translation ───────────────────────────────────────────
    s3_name, s3_params, s3_outputs = _translation_spec(folder, cfg)
    if not should_skip_step(
        folder, s3_name, s3_params, s3_outputs,
        extra_cleanup_paths=[os.path.join(folder, 'summary.json')],
    ):
        logger.info("[Step 3] Translating subtitles ...")
        translate(folder, target_language=cfg.translation_target_language)
        mark_step_done(folder, s3_name, s3_params, s3_outputs)
    else:
        logger.info("[Step 3] SKIP — translation already cached")

    # ── Step 4: TTS ───────────────────────────────────────────────────
    s4_name, s4_params, s4_outputs, s4_extra = _tts_spec(folder, cfg)
    if not should_skip_step(folder, s4_name, s4_params, s4_outputs, extra_cleanup_paths=s4_extra):
        logger.info("[Step 4] Generating TTS audio ...")
        generate_wavs(
            folder,
            tts_engine=cfg.tts_engine,
            tts_device=cfg.tts_device,
            target_language=cfg.translation_target_language,
        )
        mark_step_done(folder, s4_name, s4_params, s4_outputs)
    else:
        logger.info("[Step 4] SKIP — TTS audio already generated")

    # ── Step 5: video compositing ─────────────────────────────────────
    s5_name, s5_params, s5_outputs = _video_spec(folder, cfg)
    if not should_skip_step(
        folder, s5_name, s5_params, s5_outputs,
        extra_cleanup_paths=[os.path.join(folder, 'subtitles.srt')],
    ):
        logger.info("[Step 5] Compositing final video ...")
        synthesize_video(
            folder,
            subtitles=cfg.subtitles,
            speed_up=cfg.speed_up,
            fps=cfg.fps,
            resolution=cfg.target_resolution,
        )
        mark_step_done(folder, s5_name, s5_params, s5_outputs)
    else:
        logger.info("[Step 5] SKIP — final video already exists")

    # ── Step 6: info / thumbnail ──────────────────────────────────────
    s6_name, s6_params, s6_outputs = _info_spec(folder, cfg)
    if not should_skip_step(folder, s6_name, s6_params, s6_outputs):
        logger.info("[Step 6] Generating info / thumbnail ...")
        generate_info(folder)
        mark_step_done(folder, s6_name, s6_params, s6_outputs)
    else:
        logger.info("[Step 6] SKIP — info already generated")

    logger.info(f"[Pipeline] DONE   folder={folder}")


def _run_pipeline_with_retries(folder: str, cfg: PipelineConfig) -> bool:
    for attempt in range(1, cfg.max_retries + 1):
        try:
            _run_pipeline(folder, cfg)
            return True
        except Exception as exc:
            logger.error(
                f"Pipeline error (attempt {attempt}/{cfg.max_retries}) "
                f"folder={folder}: {exc}"
            )
    return False


def _process_online_info(info: dict, cfg: PipelineConfig) -> bool:
    folder = get_target_folder(info, cfg.root_folder)
    if folder is None:
        logger.warning(f"Cannot determine folder for: {info.get('title', '?')}")
        return False

    # Step 0 — download (has its own resolution-aware cache check inside)
    folder = download_single_video(info, cfg.root_folder, cfg.resolution)
    if folder is None:
        logger.warning(f"Download failed: {info.get('title', '?')}")
        return False

    return _run_pipeline_with_retries(folder, cfg)


def _process_online_info_on_device(info: dict, cfg: PipelineConfig, device: str) -> bool:
    logger.info(f"[Scheduler] Online task assigned to {device}: {info.get('title', '?')}")
    return _process_online_info(info, _cfg_with_device(cfg, device))


def _parse_multi_input(raw: str, is_local_paths: bool = False) -> list[str]:
    """Parse newline- or comma-separated inputs without splitting commas inside
    existing file paths (e.g. '你好，世界.mp4') or single URLs (e.g. 'https://a.com/x,y')."""
    if raw is None:
        return []
    raw_str = str(raw).strip()
    if not raw_str:
        return []

    items = []
    for raw_line in raw_str.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Never split an existing local path that literally contains ',' or '，'.
        if os.path.exists(line):
            items.append(line)
            continue

        if is_local_paths:
            parts = [p.strip() for p in re.split(r'[，,]', line) if p.strip()]
            # If splitting broke a single non-existent path vs multiple paths, keep parts
            items.extend(parts)
        else:
            # For URLs, only split on commas when followed by another URL scheme if schemes are present.
            if re.search(r'(?:https?|rtmp|rtsp|ftp)://', line, flags=re.IGNORECASE):
                parts = [
                    p.strip()
                    for p in re.split(r'[，,]\s*(?=(?:https?|rtmp|rtsp|ftp)://)', line, flags=re.IGNORECASE)
                    if p.strip()
                ]
            else:
                parts = [p.strip() for p in re.split(r'[，,]', line) if p.strip()]
            items.extend(parts)
    return items


def _validate_engine_and_language(
    tts_engine: str,
    target_language: str,
    xtts_cpml_ack: bool,
) -> None:
    if tts_engine == 'xtts' and not xtts_cpml_ack:
        logger.error('XTTS v2 requires CPML non-commercial acknowledgment. Aborting.')
        raise ValueError(
            'XTTS v2 requires explicit CPML acknowledgment. '
            'Set xtts_cpml_ack=True or switch tts_engine to "cosyvoice". '
            'See https://coqui.ai/cpml'
        )
    if tts_engine == 'xtts':
        logger.warning(
            'XTTS v2 is licensed under CPML — non-commercial use only. '
            'See https://coqui.ai/cpml'
        )
    target_norm = str(target_language or '').strip().lower()
    if tts_engine == 'cosyvoice' and target_norm in _COSYVOICE_UNSUPPORTED_TARGETS:
        raise ValueError(
            f'CosyVoice-300M does not support target language "{target_language}" '
            '(supported: 简体中文, 繁体中文, English, ja, ko). '
            'Switch tts_engine to "xtts" (with local XTTS v2 weights) or select a supported language.'
        )


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API — two entry points only
# ══════════════════════════════════════════════════════════════════════

def process_online_video(
    url: str,
    root_folder: str = "videos",
    download_resolution: str = "1080p",
    max_videos: int = 20,
    source_language: str = "auto",
    target_language: str = "简体中文",
    tts_engine: str = "cosyvoice",
    burn_subtitles: bool = True,
    speed_up: float = 1.05,
    max_workers: int = 1,
    xtts_cpml_ack: bool = False,
) -> str:
    """
    Full pipeline for one or more online video URLs.

    Parameters
    ----------
    url : str
        One URL or multiple URLs separated by commas / newlines.
    xtts_cpml_ack : bool, default False
        Defence-in-depth CPML acknowledgment. The Gradio UI also enforces this
        gate; passing True here is required to use ``tts_engine='xtts'`` from
        any caller (CLI, script, notebook).

    Returns
    -------
    str  – human-readable summary (e.g. "Success: 3  Fail: 0")
    """
    cfg = PipelineConfig(
        root_folder=root_folder,
        resolution=download_resolution,
        target_resolution=download_resolution,
        num_videos=max_videos,
        source_language=source_language,
        translation_target_language=target_language,
        tts_engine=tts_engine,
        subtitles=burn_subtitles,
        speed_up=speed_up,
        max_workers=max(1, int(max_workers)),
    )

    _validate_engine_and_language(cfg.tts_engine, cfg.translation_target_language, xtts_cpml_ack)

    urls = _parse_multi_input(url, is_local_paths=False)
    if not urls:
        return "Error: no valid URL provided."

    # backward-compat: single local path detected in URL field
    if len(urls) == 1 and os.path.isfile(urls[0]):
        logger.warning("Detected local file path in URL field, routing to local pipeline.")
        return process_local_video(
            urls[0],
            root_folder=root_folder,
            source_language=source_language,
            target_language=target_language,
            tts_engine=tts_engine,
            burn_subtitles=burn_subtitles,
            speed_up=speed_up,
            max_workers=max_workers,
            xtts_cpml_ack=xtts_cpml_ack,
            target_resolution=download_resolution,
        )

    info_list = list(get_info_list_from_url(urls, cfg.num_videos))
    if not info_list:
        return "Error: failed to extract any downloadable video from the provided URL(s)."

    ok, fail = 0, 0
    with reserved_cuda_devices(cfg.max_workers) as worker_devices:
        _warm_up_models(cfg, worker_devices, task_count=len(info_list))
        if cfg.max_workers == 1:
            logger.info(f"[Scheduler] Using device: {worker_devices[0]}")
            single_cfg = _cfg_with_device(cfg, worker_devices[0])
            for info in info_list:
                if _process_online_info(info, single_cfg):
                    ok += 1
                else:
                    fail += 1
        else:
            logger.info(f"[Scheduler] Parallel devices: {worker_devices}")
            logger.warning(f"Parallel mode enabled: max_workers={cfg.max_workers}")
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = [
                    pool.submit(
                        _process_online_info_on_device,
                        info,
                        cfg,
                        worker_devices[i % len(worker_devices)],
                    )
                    for i, info in enumerate(info_list)
                ]
                for fut in futures:
                    try:
                        if fut.result():
                            ok += 1
                        else:
                            fail += 1
                    except Exception as exc:
                        logger.error(f"Parallel worker crashed: {exc}")
                        fail += 1

    return f"Success: {ok}  Fail: {fail}"


def process_local_video(
    file_path: str,
    root_folder: str = "videos",
    source_language: str = "auto",
    target_language: str = "简体中文",
    tts_engine: str = "cosyvoice",
    burn_subtitles: bool = True,
    speed_up: float = 1.05,
    max_workers: int = 1,
    xtts_cpml_ack: bool = False,
    target_resolution: str = "1080p",
) -> str:
    """
    Full pipeline for one or more local video files.

    Parameters
    ----------
    file_path : str
        One or more local video paths separated by commas / newlines.
    xtts_cpml_ack : bool, default False
        Defence-in-depth CPML acknowledgment.
    target_resolution : str, default "1080p"
        Target output video resolution (e.g. "1080p", "2160p", "720p", "480p").

    Returns
    -------
    str  – human-readable summary
    """
    cfg = PipelineConfig(
        root_folder=root_folder,
        source_language=source_language,
        translation_target_language=target_language,
        tts_engine=tts_engine,
        subtitles=burn_subtitles,
        speed_up=speed_up,
        max_workers=max(1, int(max_workers)),
        target_resolution=target_resolution,
    )

    _validate_engine_and_language(cfg.tts_engine, cfg.translation_target_language, xtts_cpml_ack)

    file_paths = _parse_multi_input(file_path, is_local_paths=True)
    if not file_paths:
        return "Error: no valid local video path provided."

    def _process_single_local_on_device(
        path: str, base_cfg: PipelineConfig, device: str
    ) -> tuple[bool, str]:
        try:
            folder = _prepare_local_video_folder(path, base_cfg.root_folder)
            local_cfg = _cfg_with_device(base_cfg, device)
            logger.info(f"[Scheduler] Local task assigned to {device}: {path}")
            return _run_pipeline_with_retries(folder, local_cfg), folder
        except Exception as exc:
            logger.error(f"Pipeline error for local video {path}: {exc}")
            return False, path

    ok, fail = 0, 0
    folders = []
    with reserved_cuda_devices(cfg.max_workers) as worker_devices:
        _warm_up_models(cfg, worker_devices, task_count=len(file_paths))
        if cfg.max_workers == 1:
            logger.info(f"[Scheduler] Using device: {worker_devices[0]}")
            for path in file_paths:
                success, folder = _process_single_local_on_device(path, cfg, worker_devices[0])
                folders.append(folder)
                if success:
                    ok += 1
                else:
                    fail += 1
        else:
            logger.info(f"[Scheduler] Parallel devices: {worker_devices}")
            logger.warning(f"Parallel local mode enabled: max_workers={cfg.max_workers}")
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = [
                    pool.submit(
                        _process_single_local_on_device,
                        path,
                        cfg,
                        worker_devices[i % len(worker_devices)],
                    )
                    for i, path in enumerate(file_paths)
                ]
                for fut in futures:
                    try:
                        success, folder = fut.result()
                        folders.append(folder)
                        if success:
                            ok += 1
                        else:
                            fail += 1
                    except Exception as exc:
                        logger.error(f"Parallel local worker crashed: {exc}")
                        fail += 1

    return f"Success: {ok}  Fail: {fail}\nFolders: {folders}"
