# CosyVoice 文本规范化边界

CosyVoice 配音由其原生前端独自负责读法转换、标点整理和分句。
OmniDub 仅保留首尾空白清理，以及空文本、纯标点和纯舞台说明的静音判断；
有正文的文本直接交给 CosyVoice，不执行应用层数字、日期、比分、阵型、型号或 AI 替换。
参考文本同样由原生推理接口处理。XTTS 保持原有预处理行为。

## 安装与运行

在运行 OmniDub 的环境中安装 `CosyVoice/requirements.txt`，其中固定 `wetext==0.2.0`。
已有的 `scripts/install_all.sh` 会安装该文件。原生前端优先使用可用的 ttsfrd，
否则使用 wetext；wetext 的中文和英文 normalizer 显式指定语言及 TN 模式。
如果两者均无法初始化，OmniDub 报错并提示检查安装和日志，不使用简化规则继续合成。

加载日志记录实际后端、规范化策略修订号及已安装的 ttsfrd/wetext 包版本。
CosyVoice 原生推理日志输出最终合成文本。策略和包版本进入步骤缓存与分段缓存指纹，
旧的应用层规范化音频不会在迁移后直接复用。原地修改第三方资源或代码但不改变版本
不在版本指纹的检测范围内；这种维护操作后应清理相应配音缓存并重启进程。

## 验证与限制

测试命令：`python -m unittest discover -s tests -v`。
真实文本集成测试需安装 `wetext==0.2.0` 和已有的 `regex` 依赖；缺少 wetext 时会明确跳过。
它执行仓库中未改写的 `text_normalize` 方法及真实文本工具、wetext 规则，
使用轻量 tokenizer 替身，不加载语音权重。其余测试覆盖原文传递、推理路由、
前端缺失时报错、加载失败不入缓存，以及包版本变化使两层缓存失效。

2026-10-31 在 Windows / Python 3.13 隔离环境验证真实 wetext：
`误差0.05` → `误差零点零五。`，`特工007` → `特工零零七。`，
中文日期和英文金额也经过原生分句链路验证。未运行 ttsfrd 或 GPU 语音合成，
文本测试不证明最终发音、音质或所有语言歧义均正确。

型号、版本号、比分等歧义内容沿用所选成熟前端的读法，不保证字符原样保留。
本改动不添加新的自研规则、词典、LLM 消歧或其他配音方案。

接口依据：
[CosyVoice 原生入口](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/cosyvoice.py#L91)、
[wetext 官方接口](https://github.com/pengzhendong/wetext/blob/c188fc775afecb69dc71ce79d0e4b910afacc961/README.md)。
