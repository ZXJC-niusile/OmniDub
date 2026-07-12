import hashlib
import json

import numpy as np
from scipy.io import wavfile
import subprocess
import torch
import threading

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


def pick_least_used_cuda_device() -> int:
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ],
        capture_output=True,
        text=True,
        check=True
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError('No GPU info from nvidia-smi.')
    best_idx = 0
    best_used = None
    for idx, line in enumerate(lines):
        used_str, _ = [part.strip() for part in line.split(',')]
        used = int(used_str)
        if best_used is None or used < best_used:
            best_used = used
            best_idx = idx
    return best_idx


def list_cuda_devices_by_used_memory() -> list[str]:
    if not torch.cuda.is_available():
        return []
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ],
        capture_output=True,
        text=True,
        check=True
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    usage = []
    for idx, line in enumerate(lines):
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            continue
        used = int(parts[0])
        total = int(parts[1])
        usage.append((idx, used, total))
    usage.sort(key=lambda x: (x[1], x[1] / max(1, x[2]), x[0]))
    return [f'cuda:{idx}' for idx, _, _ in usage]


def reserve_cuda_devices(n: int) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    with _RESERVE_LOCK:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return ['cpu']
        scored = []
        for idx, line in enumerate(lines):
            used_mb = int(line.split(',')[0].strip())
            device_str = f'cuda:{idx}'
            reserved_count = _RESERVED.get(device_str, 0)
            score = used_mb + _RESERVATION_MEMORY_MB * reserved_count
            scored.append((score, device_str))
        scored.sort(key=lambda x: x[0])
        device_count = len(scored)
        if n <= device_count:
            chosen = [item[1] for item in scored[:n]]
        else:
            chosen = [scored[i % device_count][1] for i in range(n)]
        for device_str in chosen:
            _RESERVED[device_str] = _RESERVED.get(device_str, 0) + 1
        return chosen


def release_cuda_devices(devices) -> None:
    with _RESERVE_LOCK:
        for device in devices:
            current = _RESERVED.get(device, 0)
            if current > 0:
                _RESERVED[device] = current - 1


def pick_cuda_devices_for_workers(max_workers: int) -> list[str]:
    return reserve_cuda_devices(max(1, int(max_workers)))


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


def should_skip_step(
    folder: str,
    step_name: str,
    params: dict,
    output_paths: list,
) -> bool:
    """
    参数感知的缓存有效性判断。

    仅当以下三个条件**同时**满足时返回 True：
      1. 旁路指纹文件 `<folder>/.<step_name>.fingerprint` 存在
      2. 其内容与当前 `params` 计算得到的 SHA256 摘要相等
      3. `output_paths` 中每条路径在磁盘上都存在且非空

    否则：
      - 删除 `output_paths` 中已存在的过期产物；
      - 写入本次新的指纹旁路文件；
      - 返回 False（调用方应当强制重跑本步）。
    """
    import os

    sidecar = os.path.join(folder, f'.{step_name}.fingerprint')
    current = compute_fingerprint(params)

    cached = None
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                cached = f.read().strip()
        except Exception:
            cached = None

    outputs_ok = all(_file_exists_nonempty(p) for p in output_paths)

    if cached == current and outputs_ok:
        return True

    for path in output_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

    try:
        os.makedirs(folder, exist_ok=True)
        with open(sidecar, 'w', encoding='utf-8') as f:
            f.write(current)
    except Exception:
        pass

    return False


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
