# 一键部署指南

本文档介绍如何在全新的服务器上部署 OmniDub。

## 1. 系统要求

- Linux 服务器（推荐）
- NVIDIA 驱动已安装（GPU 模式；建议预安装中文字体如 `fonts-noto-cjk` 以保证字幕烧录效果）
- `ffmpeg` 已加入 PATH
- Conda 已安装

## 2. 运行安装脚本

```bash
git clone <your-repo-url> OmniDub
cd OmniDub
bash scripts/install_all.sh
# 如需同时安装可选的 Coqui XTTS v2 运行库：
# INSTALL_XTTS=1 bash scripts/install_all.sh
```

脚本会自动锚定仓库根目录并完成以下步骤：

| 步骤 | 内容 |
|---|---|
| 1 | 创建 conda 环境 `omnidub` (Python 3.10) |
| 2 | 安装 CUDA 12.8 版 PyTorch (torch / torchvision / torchaudio) |
| 3 | （可选，仅 `INSTALL_XTTS=1`）先行安装 Coqui TTS，避免降级后续现代依赖 |
| 4 | 安装项目基础依赖 (`requirements.txt`) |
| 5 | 安装 Demucs (人声分离) |
| 6 | 安装 WhisperX (语音识别) |
| 7 | 安装 CosyVoice 运行时依赖 + Matcha-TTS |
| 8 | 运行导入完整性检查 |

## 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下必需项：

- `OPENAI_API_KEY` - OpenAI API 密钥
- `MODEL_NAME` - GPT 模型名称
- `HF_TOKEN` - HuggingFace Token（需在 HuggingFace 上接受 pyannote 模型使用条款）
- `OMNIDUB_XTTS_LOCAL_PATH` - **（仅当使用 `xtts` 引擎时必填）** 自行下载并同意 CPML 协议后的本地 XTTS v2 权重目录路径；默认使用 `cosyvoice`（Apache 2.0）无需配置此项。

完整环境变量说明见 [README.md](../README.md) 中的"配置环境变量"部分。

## 4. Matcha-TTS 安装失败的处理

`install_all.sh` 会尝试 `pip install matcha-tts`。预编译 wheel 仅支持 Linux / Python 3.10，其他环境可能安装失败。如果失败，手动操作：

```bash
cd CosyVoice/third_party/Matcha-TTS
rm -rf matcha
git clone https://github.com/shivammehta25/Matcha-TTS.git .
```

项目代码在检测到 `CosyVoice/third_party/Matcha-TTS/matcha/models` 存在时会自动将其加入 `sys.path`。

## 5. 启动服务

```bash
conda activate omnidub
python app.py
```

## 6. 首次运行说明

首次运行时，系统会按需自动下载以下模型权重（可能需要数分钟）：

- WhisperX `large-v3` 模型与对齐模型
- Demucs `htdemucs_ft` 模型
- CosyVoice-300M 模型（默认引擎，从 ModelScope 下载）
- pyannote 说话人分离模型（需要 `HF_TOKEN`）

> **注意**：受 Coqui CPML 协议限制，系统**不会**自动下载 XTTS v2 模型权重。如选择 `xtts` 引擎（例如翻译至德语/法语/俄语），需提前安装运行库（`INSTALL_XTTS=1`）、自行下载权重、配置 `OMNIDUB_XTTS_LOCAL_PATH` 并在界面勾选 CPML 非商用确认框。

建议先用一个短视频测试全流程是否正常，或运行回归单测验证本地代码完整性：

```bash
python -m unittest discover -s tests -v
```

## 7. 生产环境注意事项

- 确保 NVIDIA 驱动和 CUDA 运行时可用（无 GPU 环境会自动回退至 CPU，但推理较慢）
- `max_workers` 参数应根据可用 GPU 显存调整（多卡环境下支持 `CUDA_VISIBLE_DEVICES` 隔离与自动均衡调度）
- 翻译步骤会调用 OpenAI API，可通过 `TRANSLATION_BATCH_SIZE` 调节单次批处理行数
- 生成的视频和中间产物保存在 `videos/` 目录下
