# OmniDub

自动化视频配音流水线：输入 YouTube 链接或本地视频，自动完成语音识别、翻译、AI 声音克隆配音、双语字幕烧录，输出中文配音视频。

## 功能

- **视频下载**：支持 YouTube 单视频 / 播放列表 / 频道（yt-dlp），也支持本地视频文件
- **人声分离**：Demucs (htdemucs_ft) 分离人声与背景音乐
- **语音识别**：WhisperX (large-v3) + pyannote 说话人分离
- **翻译**：OpenAI GPT，支持中 / 英 / 德 / 法 / 俄等目标语言
- **AI 配音**：两种 TTS 引擎可选
  - **XTTS v2**（Coqui TTS）— 多语言声音克隆，默认用于在线视频
  - **CosyVoice 300M** — 零样本声音克隆，默认用于本地视频
- **时间拉伸对齐**：TTS 输出按 0.82x-1.0x 拉伸匹配原始语速
- **双语字幕**：生成 SRT 字幕并烧录至视频
- **多 GPU 调度**：支持 1-8 并行 worker，自动选择最空闲 GPU
- **断点续传**：每一步都有缓存检查，崩溃后可恢复

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.10 |
| Web UI | Gradio |
| 视频下载 | yt-dlp |
| 音频分离 | ffmpeg + Demucs |
| 语音识别 | WhisperX + pyannote |
| 翻译 | OpenAI GPT API |
| TTS | Coqui XTTS v2 / CosyVoice 300M |
| GPU | PyTorch + CUDA 12.8 |

## 快速开始

### 1. 系统要求

- Linux 服务器（推荐）/ Windows / macOS
- NVIDIA GPU + 驱动程序（GPU 模式）
- [ffmpeg](https://ffmpeg.org/) 在 PATH 中
- [Conda](https://docs.conda.io/) 已安装

### 2. 一键安装

```bash
git clone <your-repo-url> OmniDub
cd OmniDub
bash scripts/install_all.sh
```

安装脚本会自动完成：
- 创建 conda 环境 `omnidub` (Python 3.10)
- 安装 CUDA 12.8 版 PyTorch
- 安装项目依赖 (demucs, whisperx, TTS, CosyVoice)
- 安装 Matcha-TTS (CosyVoice 的流匹配依赖)

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

必需的环境变量：

| 变量 | 说明 | 必填 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 密钥（用于翻译和摘要）| 是 |
| `MODEL_NAME` | 使用的 GPT 模型名称 | 是 |
| `HF_TOKEN` | HuggingFace Token（下载 WhisperX / pyannote 模型）| 是 |

可选的环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | 自定义 API 地址（兼容 OpenAI 格式的第三方服务）|
| `HUGGINGFACE_TOKEN` | — | `HF_TOKEN` 的别名，二选一即可 |
| `HF_ENDPOINT` | — | HuggingFace 镜像地址，如 `https://hf-mirror.com` |
| `WHISPERX_ALIGN_FALLBACK_LANGUAGE` | `en` | 当检测到的语言没有对齐模型时的回退语言 |
| `WHISPERX_DIARIZATION_MODEL` | — | 自定义 pyannote 说话人分离模型 |
| `FORMATION_TTS_STYLE` | `chinese_compact` | 足球阵型数字的 TTS 读法风格 |

### 4. 启动

```bash
conda activate omnidub
python app.py
```

浏览器打开 Gradio 提供的本地链接，选择"在线视频"或"本地视频"标签页开始使用。

## 项目结构

```
OmniDub/
├── app.py                          # Gradio Web 入口
├── requirements.txt                # 主依赖清单
├── .env.example                    # 环境变量模板
├── youdub/                         # 核心流水线
│   ├── pipeline.py                 #   v2.0 主编排器
│   ├── utils.py                    #   GPU 调度、音频工具
│   ├── cn_tx.py                    #   中文文本规范化
│   ├── step000_video_downloader.py #   1. 下载视频
│   ├── step010_demucs_vr.py        #   2. 人声分离
│   ├── step020_whisperx.py         #   3. 语音识别
│   ├── step030_translation.py      #   4. 翻译
│   ├── step040_tts.py              #   5. TTS 调度
│   ├── step042_tts_xtts.py         #   5a. XTTS v2 引擎
│   ├── step043_tts_cosyvoice.py    #   5b. CosyVoice 引擎
│   ├── step050_synthesize_video.py #   6. 视频合成
│   └── step060_genrate_info.py     #   7. 缩略图+摘要
├── CosyVoice/                      # CosyVoice TTS 引擎（内嵌源码）
│   ├── cosyvoice/                  #   模型代码（仅保留运行时必需部分）
│   ├── third_party/Matcha-TTS/     #   Matcha-TTS 依赖（见下方说明）
│   └── requirements.txt            #   CosyVoice 运行时依赖
├── docs/ONE_CLICK_INSTALL.md       # 部署指南
└── scripts/install_all.sh          # 一键安装脚本
```

## 关于 Matcha-TTS 依赖

CosyVoice 的流匹配模块 (`flow_matching.py`, `decoder.py`) 依赖 [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) 包。

`install_all.sh` 会尝试通过 `pip install matcha-tts` 安装。如果安装失败（预编译 wheel 仅支持 Linux / Python 3.10），可以手动操作：

```bash
cd CosyVoice/third_party/Matcha-TTS
rm -rf matcha  # 删除占位 stub
git clone https://github.com/shivammehta25/Matcha-TTS.git .
```

项目代码中的 `sys.path.append` 会自动找到本地克隆的 Matcha-TTS。

## TTS 引擎说明

| 引擎 | 适用场景 | 模型 | 特点 |
|---|---|---|---|
| XTTS v2 | 在线视频（默认）| Coqui TTS | 多语言声音克隆，GPU 加速 |
| CosyVoice | 本地视频（默认）| CosyVoice-300M | 零样本声音克隆，普通话优化 |

在 Gradio 界面中可以通过 `tts_engine` 参数切换。首次使用时会自动下载模型权重。

## 致谢

- [Demucs](https://github.com/facebookresearch/demucs) — 人声分离
- [WhisperX](https://github.com/m-bain/whisperX) — 语音识别
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — TTS 引擎
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — 流匹配 TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 引擎
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频下载

## License

本项目代码采用 [MIT License](LICENSE) 开源。

> **注意**：项目支持的 TTS 引擎中，Coqui XTTS v2 的**模型权重**采用 [Coqui Public Model License (CPML)](https://coqui.ai/cpml)，仅允许非商业用途。如需商业用途，请使用 CosyVoice 引擎（Apache 2.0）或自行获取 XTTS v2 的商业授权。
