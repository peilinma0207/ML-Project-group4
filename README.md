# ML-Project-group4

一个结合 WhisperX、说话人分离、视觉理解、RAG 术语检索和大语言模型的多模态脚本修复 Agent。项目目标是从视频中提取音频、画面和背景知识证据，修复 ASR 在噪声、专有名词、多说话人场景下产生的错误，最终导出带时间戳、证据来源和人工复核标记的结构化稿件。

本项目不是一个常驻 Web 服务，而是一条命令行数据处理流水线。推荐使用根目录的 `Makefile` 启动和管理常用任务。

## 项目定位

在会议、访谈、课程或 TED/TED-Ed 演讲等视频场景中，自动语音识别常会遇到这些问题：

- 背景噪声、口音或压缩导致识别置信度下降。
- 技术术语、人物名、产品名容易被误听。
- 多说话人场景需要保留说话人和时间戳信息。
- 单纯依靠音频时，无法利用画面中的字幕、幻灯片或屏幕文字。

本项目通过“音频 + 视觉 + RAG + LLM”的证据融合方式，对原始 ASR 结果进行谨慎修复：只在证据支持时改写文本，并保留需要人工复核的片段。

## 核心流程

```text
video_ingest
  -> audio_extract
  -> audio_preprocess
  -> asr_transcribe
  -> frame_sample
  -> vlm_extract
  -> rag_retrieve
  -> evidence_merge
  -> script_repair
  -> export
```

各模块都位于 `src/agent/`，并暴露 `run()` 函数；`src/agent/cli.py` 负责串联完整流程。

| 阶段 | 作用 |
| --- | --- |
| 音频抽取与预处理 | 使用 `ffmpeg` 从视频抽取单声道 WAV，并进行高通滤波、响度归一化 |
| ASR 转写 | 使用 WhisperX 生成分段文本、词级时间戳和低置信度标记 |
| 抽帧与视觉理解 | 按间隔、场景变化、低置信度片段抽帧，并用 VLM/OCR 提取画面证据 |
| RAG 检索 | 从 `data/glossary.json` 中检索术语、别名和常见误听写法 |
| 证据合并 | 将 ASR、视觉事件和 RAG 命中结果按时间对齐 |
| 文本修复 | 调用 OpenAI-compatible 文本模型修复脚本，保留时间戳和说话人 |
| 结果导出 | 输出 `script.json` 和可读的 `script.md` |

## 目录结构

```text
.
├── src/agent/                 # Agent 流水线模块
├── docs/agent/                # 模块级设计说明
├── docs/project_presentation_zh.md
├── data/
│   ├── videos/                # 示例视频
│   ├── audio/                 # 已抽取音频
│   ├── transcripts_ground_truth/
│   ├── metadata/
│   └── rag_docs/
├── tests/                     # 单元测试
├── main.py
├── pyproject.toml
├── uv.lock
└── Makefile
```

## 环境要求

- Python `>=3.10, <3.14`
- `uv`：项目依赖和虚拟环境管理
- `ffmpeg` 与 `ffprobe`：音频抽取、预处理和抽帧
- WhisperX 运行所需的 PyTorch 环境；首次运行可能需要下载模型权重
- 可选：GPU/CUDA，用于加速 WhisperX 和本地模型推理
- 可选：HuggingFace token，用于 pyannote 说话人分离模型
- 可选：OpenAI-compatible VLM/Text API，用于完整视觉理解和文本修复流程

## 快速启动

先检查本机工具：

```bash
make doctor
```

安装项目和开发依赖：

```bash
make sync
```

运行轻量示例流程：

```bash
make run
```

`make run` 默认使用 `data/videos/video_01.mp4`，并开启 `--skip-vlm --skip-repair`，因此不会调用外部 VLM 或文本模型 API；它仍会执行音频抽取、预处理、WhisperX 转写、抽帧、RAG 检索和导出。

自定义视频、主题和任务 ID：

```bash
make run \
  VIDEO=data/videos/video_03.mp4 \
  TOPIC="AI agents and machine learning" \
  WHISPER_MODEL=tiny \
  JOB_ID=video03_demo
```

运行完整流程：

```bash
make run-full \
  VIDEO=data/videos/video_01.mp4 \
  TOPIC="AI and machine learning" \
  VLM_API_BASE=http://localhost:1234/v1 \
  TEXT_API_BASE=http://localhost:1234/v1
```

启用说话人分离：

```bash
make run DIARIZE=1
```

## Makefile 命令

| 命令 | 说明 |
| --- | --- |
| `make help` | 查看可用命令和常用变量 |
| `make doctor` | 检查 `uv`、`ffmpeg`、`ffprobe` 是否可用 |
| `make sync` | 使用 `uv sync --extra dev` 安装依赖 |
| `make run` | 启动轻量流水线，跳过 VLM 和文本修复 |
| `make run-full` | 启动完整流水线，可传入 VLM/Text API 地址 |
| `make test` | 运行单元测试 |
| `make clean` | 删除本地 `output/` 运行结果 |

常用变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VIDEO` | `data/videos/video_01.mp4` | 输入视频路径 |
| `TOPIC` | `AI and machine learning` | 视频主题提示 |
| `OUTPUT_DIR` | `output` | 输出目录 |
| `WHISPER_MODEL` | `base` | WhisperX 模型大小 |
| `GLOSSARY` | `data/glossary.json` | RAG 术语表 |
| `JOB_ID` | 空，自动生成 | 任务 ID |
| `DIARIZE` | 空 | 设为 `1` 时启用说话人分离 |
| `VLM_API_BASE` | 空 | OpenAI-compatible 视觉模型 API 地址 |
| `TEXT_API_BASE` | 空 | OpenAI-compatible 文本模型 API 地址 |

## 直接使用 CLI

也可以绕过 Makefile，直接运行 CLI：

```bash
uv run python -m src.agent.cli \
  --video data/videos/video_01.mp4 \
  --topic "AI and machine learning" \
  --output-dir ./output \
  --whisper-model base \
  --glossary data/glossary.json \
  --skip-vlm \
  --skip-repair
```

完整参数可查看：

```bash
uv run python -m src.agent.cli --help
```

## 输出结果

每次运行会在 `OUTPUT_DIR/JOB_ID/` 下生成结果。若没有手动指定 `JOB_ID`，系统会自动生成短 ID。

```text
output/<job_id>/
├── audio.wav
├── audio_processed.wav
├── frames/
├── script.json
└── script.md
```

- `script.json`：结构化结果，便于程序读取和评估。
- `script.md`：人类可读稿件，包含时间戳、说话人、文本、证据来源、置信度和复核标记。
- `frames/`：抽取出的关键帧，用于视觉证据分析。

## 数据集

`data/` 目录包含 10 个 TED/TED-Ed AI 相关视频样本、对应音频、人工整理的 ground truth transcript、元数据和 RAG 背景材料。数据说明见 `data/README.md`。

## 数据合成（损坏数据生成）

为测试 ASR 修复 Agent 的能力，我们实现了三种数据损坏方法，生成不同难度的测试数据：

| 方法 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| 音频加噪 | `.wav` 音频 | 加噪音频 | 白噪声 + 混响，3 个 SNR 等级（20/10/3 dB） |
| 同音词替换 | GT 文本 | 替换文本 | 基于 CMU 发音词典自动找同音词，3 个比例（10%/25%/50%） |
| 对抗扰动 | `.wav` 音频 | 扰动音频 | FGSM 攻击 Whisper tiny，ε=0.01，人耳几乎不可感知 |

损坏数据目录结构：

```text
data/corrupted/
├── noise/audio/           # 60 个加噪音频
├── homophone/text/        # 30 个同音词替换文本
├── adversarial/audio/     # 10 个对抗扰动音频
├── docs/                  # DATA_CORRUPTION.md 详细文档
└── metadata/              # 统计报告
```

使用方法：

```bash
# 运行全部损坏方法
python3 -m src.agent.corrupt.runner

# 只运行部分方法
python3 -m src.agent.corrupt.runner --skip-adversarial

# 查看统计
python3 -m src.agent.corrupt.stats
```

详细说明见 `data/corrupted/docs/DATA_CORRUPTION.md`。

## 开发与测试

运行测试：

```bash
make test
```

或直接运行：

```bash
uv run pytest tests/ -v
```

模块级设计文档位于 `docs/agent/`：

- `docs/agent/spec.md`：整体流水线规范
- `docs/agent/audio.md`：音频抽取与预处理
- `docs/agent/asr.md`：ASR 与说话人分离
- `docs/agent/visual.md`：抽帧与视觉理解
- `docs/agent/rag.md`：RAG 术语检索
- `docs/agent/repair.md`：证据合并与脚本修复
- `docs/agent/export.md`：导出与 CLI

## 当前限制

- 当前阶段尚未接入 LangGraph，流程由 CLI 顺序串联。
- RAG 使用本地 JSON 术语表模拟检索，尚未接入真实向量数据库。
- 暂无批处理、Web UI、SRT/VTT 字幕导出和模型对比功能。
- 完整视觉理解和文本修复依赖外部或本地 OpenAI-compatible API。
- 首次运行 WhisperX 可能需要较长下载和初始化时间。

## 中文演讲稿

项目介绍演讲稿见 `docs/project_presentation_zh.md`，适合课程展示、项目答辩或组会汇报时使用。
