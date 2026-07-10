# 一键部署指南

本文档介绍如何在全新的服务器上部署 OmniDub。

## 1. 系统要求

- Linux 服务器（推荐）
- NVIDIA 驱动已安装（GPU 模式）
- `ffmpeg` 已加入 PATH
- Conda 已安装

## 2. 运行安装脚本

```bash
git clone <your-repo-url> OmniDub
cd OmniDub
bash scripts/install_all.sh
```

脚本会自动完成以下步骤：

| 步骤 | 内容 |
|---|---|
| 1 | 创建 conda 环境 `omnidub` (Python 3.10) |
| 2 | 安装 CUDA 12.8 版 PyTorch (torch / torchvision / torchaudio) |
| 3 | 安装项目基础依赖 (requirements.txt) |
| 4 | 安装 Demucs (人声分离) |
| 5 | 安装 WhisperX (语音识别) |
| 6 | 安装 Coqui TTS (XTTS v2 引擎) |
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

完整环境变量说明见 [README.md](../README.md) 中的"配置环境变量"部分。

## 4. Matcha-TTS 安装失败的处理

`install_all.sh` 会尝试 `pip install matcha-tts`。预编译 wheel 仅支持 Linux / Python 3.10，其他环境可能安装失败。如果失败，手动操作：

```bash
cd CosyVoice/third_party/Matcha-TTS
rm -rf matcha
git clone https://github.com/shivammehta25/Matcha-TTS.git .
```

项目代码中的 `sys.path.append` 会自动找到本地克隆的 Matcha-TTS。

## 5. 启动服务

```bash
conda activate omnidub
python app.py
```

## 6. 首次运行说明

首次运行时，系统会自动下载以下模型权重（可能需要数分钟）：

- WhisperX large-v3 模型
- Demucs htdemucs_ft 模型
- Coqui XTTS v2 模型
- CosyVoice-300M 模型（从 ModelScope 下载）
- pyannote 说话人分离模型（需要 HF_TOKEN）

建议先用一个短视频测试全流程是否正常。

## 7. 生产环境注意事项

- 确保 NVIDIA 驱动和 CUDA 运行时可用
- `max_workers` 参数应根据可用 GPU 显存调整
- 翻译步骤会调用 OpenAI API，注意速率限制
- 生成的视频和中间产物保存在 `videos/` 目录下
