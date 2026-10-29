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

_RESERVE_LOCK = threading.Lock()
_RESERVED = {}  # device_str -> int (reservation count)
_RESERVATION_MEMORY_MB = 200  # each reservation pretends to "occupy" 200MB


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * 32767
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def _query_nvidia_smi_used_mb() -> list[int]:
    """Return per-GPU used memory (MB) via nvidia-smi, or empty list if unavailable."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

    used_list = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first_col = line.split(',')[0].strip()
        try:
            used_list.append(int(first_col))
        except ValueError:
            used_list.append(0)
    return used_list


def pick_least_used_cuda_device() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available.')
    used_list = _query_nvidia_smi_used_mb()
    if not used_list:
        if torch.cuda.device_count() > 0:
            return 0
        raise RuntimeError('No GPU info from nvidia-smi.')
    best_idx = min(range(len(used_list)), key=lambda idx: used_list[idx])
    return best_idx


def reserve_cuda_devices(n: int) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    if not torch.cuda.is_available():
        return ['cpu']
    with _RESERVE_LOCK:
        used_list = _query_nvidia_smi_used_mb()
        if not used_list:
            dev_count = torch.cuda.device_count()
            if dev_count <= 0:
                return ['cpu']
            used_list = [0] * dev_count

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
    global _AUTO_CUDA_DEVICE
    if device != 'auto':
        return device
    if not torch.cuda.is_available():
        return 'cpu'
    if _AUTO_CUDA_DEVICE is None:
        try:
            _AUTO_CUDA_DEVICE = f'cuda:{pick_least_used_cuda_device()}'
        except Exception:
            _AUTO_CUDA_DEVICE = 'cuda'
    return _AUTO_CUDA_DEVICE
