# OmniDub

自动化视频配音流水线：输入 YouTube 链接或本地视频，自动完成语音识别、翻译、AI 声音克隆配音、双语字幕烧录，输出配音视频。

## 功能

- **视频下载**：支持 YouTube 单视频 / 播放列表 / 频道（yt-dlp），也支持本地视频文件（MP4 / MKV / WebM / MOV 等）
- **人声分离**：Demucs (htdemucs_ft) 分离人声与背景音乐
- **语音识别**：WhisperX (large-v3) + pyannote 说话人分离
- **翻译**：OpenAI GPT，支持中 / 英 / 德 / 法 / 俄等目标语言
- **AI 配音**：两种 TTS 引擎可选
  - **CosyVoice 300M** — 零样本 / 跨语种声音克隆（Apache 2.0），在线与本地视频默认引擎（支持中 / 英 / 日 / 韩等）
  - **XTTS v2**（Coqui TTS）— 多语言声音克隆（opt-in 集成，支持德 / 法 / 俄 / 西等语种，需本地权重与 CPML 确认）
- **时间拉伸对齐**：TTS 输出按 0.82x-1.0x 拉伸匹配原始语速
- **双语字幕**：生成 SRT 字幕并烧录至视频（自适应横屏 16:9 与竖屏 9:16 布局）
- **多 GPU 调度**：支持 1-8 并行 worker，尊重 `CUDA_VISIBLE_DEVICES` 并自动选择最空闲 GPU
- **断点续传**：参数与上游产物感知的 DAG 缓存指纹检查，崩溃或中途重跑后可精准恢复

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.10 |
| Web UI | Gradio |
| 视频下载 | yt-dlp |
| 音频分离 | ffmpeg + Demucs |
| 语音识别 | WhisperX + pyannote |
| 翻译 | OpenAI GPT API |
| TTS | CosyVoice 300M / Coqui XTTS v2 (opt-in) |
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
# 如需同时安装可选的 Coqui XTTS v2 运行库：
# INSTALL_XTTS=1 bash scripts/install_all.sh
```

安装脚本会自动完成：
- 创建 conda 环境 `omnidub` (Python 3.10)
- 安装 CUDA 12.8 版 PyTorch
- 安装项目依赖 (demucs, whisperx, CosyVoice；当设置 `INSTALL_XTTS=1` 时额外先行安装 Coqui `TTS`)
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
| `OMNIDUB_XTTS_LOCAL_PATH` | — | 仅启用 `xtts` 时必填，本地 XTTS v2 权重目录路径 |
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
│   ├── utils.py                    #   GPU 调度、缓存指纹、音频工具
│   ├── cn_tx.py                    #   中文文本规范化
│   ├── step000_video_downloader.py #   1. 下载视频
│   ├── step010_demucs_vr.py        #   2. 人声分离
│   ├── step020_whisperx.py         #   3. 语音识别
│   ├── step030_translation.py      #   4. 翻译
│   ├── step040_tts.py              #   5. TTS 调度
│   ├── step042_tts_xtts.py         #   5a. XTTS v2 引擎 (opt-in)
│   ├── step043_tts_cosyvoice.py    #   5b. CosyVoice 引擎 (默认)
│   ├── step050_synthesize_video.py #   6. 视频合成
│   └── step060_generate_info.py    #   7. 缩略图+摘要
├── CosyVoice/                      # CosyVoice TTS 引擎（内嵌源码）
│   ├── cosyvoice/                  #   模型代码（仅保留运行时必需部分）
│   ├── third_party/Matcha-TTS/     #   Matcha-TTS 依赖（见下方说明）
│   └── requirements.txt            #   CosyVoice 运行时依赖
├── tests/                          # 回归单元测试套件
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

项目代码中的 `sys.path` 检测会自动加载本地克隆的 Matcha-TTS。

## TTS 引擎说明

| 引擎 | 默认状态 | 模型 | 特点 |
|---|---|---|---|
| CosyVoice | 在线 & 本地默认 | CosyVoice-300M | 开箱即用（Apache 2.0），零样本/跨语种声音克隆（支持中/英/日/韩） |
| XTTS v2 | 可选启用 (Opt-in) | Coqui TTS | 多语言声音克隆（含德/法/俄/西等），需自备权重并确认 CPML |

在 Gradio 界面中可以通过 `TTS Engine` 参数切换。使用 `CosyVoice` 时首次运行会自动下载模型权重；若目标语言为德语 / 法语 / 俄语等欧陆语种，请切换至 `xtts` 引擎。

## 回归测试

```bash
python -m unittest discover -s tests -v
```

## 致谢

- [Demucs](https://github.com/facebookresearch/demucs) — 人声分离
- [WhisperX](https://github.com/m-bain/whisperX) — 语音识别
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — TTS 引擎
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — 流匹配 TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 引擎
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频下载

## License

本项目代码采用 [MIT License](LICENSE) 开源。

> **重要**：XTTS v2 在本项目中是 **opt-in 集成**——`youdub/step042_tts_xtts.py` **不会自动**从 HuggingFace / Coqui 下载 XTTS v2 模型权重（CPML 禁止再分发）。
>
> 如需启用 XTTS v2：
>
> 1. 安装运行库：`INSTALL_XTTS=1 bash scripts/install_all.sh`（或手动 `pip install TTS`）；
> 2. 自行从 [HuggingFace `coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) 或 Coqui TTS 模型库下载权重，并**由您本人同意 CPML**；
> 3. 设置环境变量 `OMNIDUB_XTTS_LOCAL_PATH=<权重本地目录>`，并在界面勾选 CPML 非商用确认框（或调用时显式传入 `xtts_cpml_ack=True`）。
>
> 如需商用，建议直接使用默认的 `tts_engine="cosyvoice"`（CosyVoice 300M，Apache 2.0），或自行向 Coqui 申请 XTTS v2 商业授权。详见 [https://coqui.ai/cpml](https://coqui.ai/cpml)。
