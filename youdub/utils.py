import numpy as np
from scipy.io import wavfile
import subprocess
import torch

_AUTO_CUDA_DEVICE = None

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


def pick_cuda_devices_for_workers(max_workers: int) -> list[str]:
    ordered = list_cuda_devices_by_used_memory()
    if not ordered:
        return ['cpu']
    count = max(1, int(max_workers))
    if count <= len(ordered):
        return ordered[:count]
    # Workers > available GPUs: round-robin reuse.
    return [ordered[i % len(ordered)] for i in range(count)]


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
