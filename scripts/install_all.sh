#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-omnidub}"
PY_VER="${PY_VER:-3.10}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_XTTS="${INSTALL_XTTS:-0}"

echo "[precheck] verify required system binaries (repo: ${REPO_ROOT})"
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found in PATH"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: ffmpeg not found. Please install ffmpeg first.";
  echo "Ubuntu example: sudo apt-get update && sudo apt-get install -y ffmpeg";
  exit 1;
}

echo "[1/8] Ensure conda env: ${ENV_NAME} (python=${PY_VER})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env exists: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "[2/8] Install CUDA PyTorch stack"
conda run -n "${ENV_NAME}" pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" pip install --index-url "${CUDA_INDEX_URL}" torch torchvision torchaudio

echo "[3/8] Install Optional Coqui TTS first (only when INSTALL_XTTS=1) to prevent downgrading modern deps"
if [[ "${INSTALL_XTTS}" == "1" ]]; then
  conda run -n "${ENV_NAME}" pip install TTS
else
  echo "Skipping optional Coqui TTS (set INSTALL_XTTS=1 to install XTTS v2 runtime)."
fi

echo "[4/8] Install OmniDub base dependencies"
conda run -n "${ENV_NAME}" pip install -r "${REPO_ROOT}/requirements.txt"

echo "[5/8] Install demucs (vocal separation)"
conda run -n "${ENV_NAME}" pip install demucs

echo "[6/8] Install WhisperX (speech recognition)"
conda run -n "${ENV_NAME}" pip install git+https://github.com/m-bain/whisperx.git

echo "[7/8] Install CosyVoice runtime dependencies"
conda run -n "${ENV_NAME}" pip install -r "${REPO_ROOT}/CosyVoice/requirements.txt"
conda run -n "${ENV_NAME}" pip install matcha-tts || {
  echo "WARNING: pip install matcha-tts failed.";
  echo "  CosyVoice requires the Matcha-TTS package for flow matching modules.";
  echo "  Manual fallback: clone https://github.com/shivammehta25/Matcha-TTS.git";
  echo "  into CosyVoice/third_party/Matcha-TTS/ (replacing the stub).";
}

echo "[8/8] Sanity check runtime imports"
INSTALL_XTTS="${INSTALL_XTTS}" conda run -n "${ENV_NAME}" python - <<'PY'
import importlib
import os

mods = [
    "gradio", "loguru", "yt_dlp", "librosa", "dotenv", "openai", "audiostretchy",
    "PIL", "demucs", "whisperx",
]
if os.getenv("INSTALL_XTTS") == "1":
    mods.append("TTS")
for m in mods:
    importlib.import_module(m)
print("Import check passed.")
PY

echo ""
echo "Install finished."
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. cp .env.example .env  # then fill in your API keys"
echo "  3. python app.py"
