from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import threading

import numpy as np
from scipy.io import wavfile
import torch

_AUTO_CUDA_DEVICE = None
_AUTO_CUDA_VISIBLE_ENV = None

_RESERVE_LOCK = threading.Lock()
_RESERVED = {}  # device_str -> int (reservation count)
_RESERVATION_MEMORY_MB = 200  # each reservation pretends to "occupy" 200MB


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    arr = np.asarray(wav, dtype=np.float32)
    if arr.size == 0:
        wavfile.write(output_path, sample_rate, np.zeros((0,), dtype=np.int16))
        return
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(arr, -1.0, 1.0)
    wav_int16 = np.round(clipped * 32767.0).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_int16)


def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate=24000):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    arr = np.asarray(wav, dtype=np.float32)
    if arr.size == 0:
        wavfile.write(output_path, sample_rate, np.zeros((0,), dtype=np.int16))
        return
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(arr)))
    wav_norm = arr * (32767.0 / max(0.01, peak))
    wav_int16 = np.clip(np.round(wav_norm), -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_int16)


def _query_nvidia_smi_rows() -> list[tuple[str, str, int]]:
    """Return [(index_str, uuid_str, used_mb), ...] across host GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,uuid,memory.used',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

    rows = []
    for idx_fallback, line in enumerate(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(',')]
        if len(cols) >= 3:
            idx_str, uuid_str, mem_str = cols[0], cols[1], cols[2]
        else:
            idx_str, uuid_str, mem_str = str(idx_fallback), '', cols[0]
        try:
            used_mb = int(mem_str)
        except ValueError:
            used_mb = 0
        rows.append((idx_str, uuid_str, used_mb))
    return rows


def _query_nvidia_smi_used_mb() -> list[int]:
    """Return per-GPU used memory (MB) via nvidia-smi, or empty list if unavailable."""
    return [row[2] for row in _query_nvidia_smi_rows()]


_DEFAULT_QUERY_USED_MB = _query_nvidia_smi_used_mb


def _map_visible_cuda_used_mb() -> list[int]:
    """Return used memory (MB) strictly aligned with PyTorch's logical cuda:0..cuda:K-1 ordinals,
    respecting CUDA_VISIBLE_DEVICES and clamping to torch.cuda.device_count()."""
    dev_count = int(torch.cuda.device_count()) if hasattr(torch.cuda, 'device_count') else 0
    visible_env = os.environ.get('CUDA_VISIBLE_DEVICES')

    if _query_nvidia_smi_used_mb is not _DEFAULT_QUERY_USED_MB:
        raw_used = _query_nvidia_smi_used_mb()
        rows = [(str(i), '', mb) for i, mb in enumerate(raw_used)]
    else:
        rows = _query_nvidia_smi_rows()

    mapped: list[int] = []
    if rows:
        if visible_env is not None:
            vis = visible_env.strip()
            if vis in {'', '-1', 'none', 'void', 'NoDevFiles'}:
                return []
            by_idx = {r[0]: r[2] for r in rows}
            for token in (t.strip() for t in vis.split(',')):
                if not token:
                    break
                if token in by_idx:
                    mapped.append(by_idx[token])
                else:
                    matched_uuid = next(
                        (r[2] for r in rows if r[1] and r[1].startswith(token)),
                        None,
                    )
                    if matched_uuid is not None:
                        mapped.append(matched_uuid)
                    else:
                        # Per CUDA runtime semantics, enumeration stops at the first invalid entry.
                        break
        else:
            mapped = [r[2] for r in rows]

    if dev_count > 0:
        if len(mapped) >= dev_count:
            return mapped[:dev_count]
        # Fallback to torch.cuda.mem_get_info when nvidia-smi is unavailable or masked by cgroups/MIG.
        if hasattr(torch.cuda, 'mem_get_info'):
            try:
                info_list = []
                for idx in range(dev_count):
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    info_list.append(max(0, int((total_b - free_b) // (1024 * 1024))))
                return info_list
            except Exception:
                pass
        if mapped:
            return mapped + [0] * (dev_count - len(mapped))
        return [0] * dev_count

    return mapped


def pick_least_used_cuda_device() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available.')
    used_list = _map_visible_cuda_used_mb()
    if not used_list:
        raise RuntimeError('No visible CUDA devices available.')
    best_idx = min(range(len(used_list)), key=lambda idx: used_list[idx])
    return best_idx


def reserve_cuda_devices(n: int) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    if not torch.cuda.is_available():
        return ['cpu']
    with _RESERVE_LOCK:
        used_list = _map_visible_cuda_used_mb()
        if not used_list:
            return ['cpu']

        scored = []
        for idx, used_mb in enumerate(used_list):
            device_str = f'cuda:{idx}'
            reserved_count = _RESERVED.get(device_str, 0)
            score = used_mb + _RESERVATION_MEMORY_MB * reserved_count
            scored.append((score, idx, device_str))
        scored.sort(key=lambda x: (x[0], x[1]))
        device_count = len(scored)
        if n <= device_count:
            chosen = [item[2] for item in scored[:n]]
        else:
            chosen = [scored[i % device_count][2] for i in range(n)]
        for device_str in chosen:
            _RESERVED[device_str] = _RESERVED.get(device_str, 0) + 1
        return chosen


def release_cuda_devices(devices) -> None:
    if not devices:
        return
    with _RESERVE_LOCK:
        for device in devices:
            if not str(device).startswith('cuda:'):
                continue
            current = _RESERVED.get(device, 0)
            if current <= 1:
                _RESERVED.pop(device, None)
            else:
                _RESERVED[device] = current - 1


def pick_cuda_devices_for_workers(max_workers: int) -> list[str]:
    return reserve_cuda_devices(max(1, int(max_workers)))


@contextmanager
def reserved_cuda_devices(max_workers: int):
    """Reserve GPU devices for workers and guarantee release on exit."""
    devices = pick_cuda_devices_for_workers(max_workers)
    try:
        yield devices
    finally:
        release_cuda_devices(devices)


def compute_fingerprint(d: dict) -> str:
    """
    计算参数字典的稳定 SHA256 摘要。先按 key 递归排序再 JSON 序列化，
    保证等价字典（满足 `==` 相等）总是产生相同的指纹，与字段插入顺序无关。
    """
    payload = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _file_exists_nonempty(path: str) -> bool:
    """判断文件是否存在且非空。"""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _remove_path_quietly(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.remove(path)
    except OSError:
        pass


def should_skip_step(
    folder: str,
    step_name: str,
    params: dict,
    output_paths: list,
    extra_cleanup_paths: list | None = None,
) -> bool:
    """
    参数感知的缓存有效性判断。

    仅当以下三个条件**同时**满足时返回 True：
      1. 旁路指纹文件 `<folder>/.<step_name>.fingerprint` 存在
      2. 其内容与当前 `params` 计算得到的 SHA256 摘要相等
      3. `output_paths` 中每条路径在磁盘上都存在且非空

    否则：
      - 删除过期的旁路指纹文件（防止半成品在重试时被误跳过）；
      - 删除 `output_paths`（及参数变更时的 `extra_cleanup_paths`）中已存在的过期产物；
      - 返回 False（步骤成功完成后应调用 `mark_step_done` 写入新指纹）。
    """
    sidecar = os.path.join(folder, f'.{step_name}.fingerprint')
    current = compute_fingerprint(params)

    cached = None
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                cached = f.read().strip()
        except OSError:
            cached = None

    outputs_ok = bool(output_paths) and all(_file_exists_nonempty(p) for p in output_paths)

    if cached == current and outputs_ok:
        return True

    _remove_path_quietly(sidecar)
    for path in output_paths:
        _remove_path_quietly(path)

    # When parameters changed (or fingerprint is missing), also purge auxiliary intermediates.
    if extra_cleanup_paths and cached != current:
        for path in extra_cleanup_paths:
            _remove_path_quietly(path)

    return False


def mark_step_done(
    folder: str,
    step_name: str,
    params: dict,
    output_paths: list,
) -> None:
    """校验本步输出均已非空落盘后，写入参数指纹旁路文件。"""
    missing = [p for p in output_paths if not _file_exists_nonempty(p)]
    if missing:
        raise RuntimeError(
            f'Step "{step_name}" finished without producing valid output(s): {missing}'
        )
    sidecar = os.path.join(folder, f'.{step_name}.fingerprint')
    current = compute_fingerprint(params)
    os.makedirs(folder, exist_ok=True)
    with open(sidecar, 'w', encoding='utf-8') as f:
        f.write(current)


def resolve_torch_device(device: str = 'auto'):
    global _AUTO_CUDA_DEVICE, _AUTO_CUDA_VISIBLE_ENV
    if device != 'auto':
        return device
    if not torch.cuda.is_available():
        return 'cpu'
    vis_now = os.environ.get('CUDA_VISIBLE_DEVICES')
    dev_count = int(torch.cuda.device_count()) if hasattr(torch.cuda, 'device_count') else 0
    cache_valid = False
    if _AUTO_CUDA_DEVICE is not None and _AUTO_CUDA_VISIBLE_ENV == vis_now:
        if _AUTO_CUDA_DEVICE.startswith('cuda:'):
            try:
                ordinal = int(_AUTO_CUDA_DEVICE.split(':', 1)[1])
                cache_valid = dev_count <= 0 or ordinal < dev_count
            except ValueError:
                cache_valid = False
        else:
            cache_valid = True
    if not cache_valid:
        try:
            _AUTO_CUDA_DEVICE = f'cuda:{pick_least_used_cuda_device()}'
        except Exception:
            _AUTO_CUDA_DEVICE = 'cuda:0' if dev_count > 0 else 'cpu'
        _AUTO_CUDA_VISIBLE_ENV = vis_now
    return _AUTO_CUDA_DEVICE
