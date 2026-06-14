# 视频受损音频文本脚本修复 Agent 计划

## 1. 项目目标

建设一个基于 LangChain 生态的 Agent 流程，用于修复视频中受损音频对应的带时间戳文本脚本。系统从音频、视觉和 RAG 三类证据中提取信息，再由纯文本模型完成脚本修复。

核心目标包括：

1. 从视频中抽取受损音频，并生成带时间戳的初始转写。
2. 从视频画面中提取场景、动作、画面文字、人物状态和主题线索。
3. 从外部整理的 RAG 资料中检索术语、别名、缩写、人物名、产品名和常见误听词。
4. 由纯文本模型综合音频证据、视觉证据和 RAG 证据，生成带时间戳的修复脚本。
5. 保留每段修复内容的证据来源，支持人工复核。

## 2. 已确认约束

| 约束项 | 当前设定 |
|---|---|
| 视觉模型运行环境 | 本地 GPU 或云端 GPU |
| 视频内容 | 内容类型可能多样，但每个任务主题明确 |
| 交付产物 | 带时间戳的脚本 |
| RAG 资料 | 由他人整理，当前计划中仅记录接口和数据要求 |
| 文本模型规模 | 可以比视觉模型稍大，最大 8B |
| 模型许可证 | 无特殊限制 |
| 成本边界 | 端到端大型多模态模型成本过高，主流程采用小型视觉模型加纯文本模型 |
| 对比组 | 加入原生多模态模型和云端大模型 API，用于质量上限评估 |

## 3. 总体技术方案

建议用 LangGraph 负责流程编排，用 LangChain 负责模型、检索器、工具调用和提示模板封装。该任务的输入源和处理步骤相对固定，图状态流程比完全自主的 Agent 更适合保留中间证据、重试失败节点、恢复历史任务和接入人工复核。

推荐流程如下：

```text
video_ingest
  -> audio_extract
  -> audio_preprocess
  -> asr_transcribe
  -> word_align_and_diarize
  -> frame_sample
  -> ocr_extract
  -> vlm_scene_extract
  -> rag_term_retrieve
  -> evidence_merge
  -> script_repair
  -> quality_check
  -> export_timestamped_script
```

LangGraph 状态建议包含以下字段：

```json
{
  "job_id": "string",
  "video_uri": "string",
  "audio_uri": "string",
  "topic_hint": "string",
  "asr_segments": [],
  "word_timestamps": [],
  "speaker_segments": [],
  "sampled_frames": [],
  "ocr_events": [],
  "visual_events": [],
  "rag_hits": [],
  "merged_evidence": [],
  "repair_candidates": [],
  "final_script": [],
  "review_flags": [],
  "metrics": {}
}
```

## 4. 音频处理设计

音频侧建议使用 `faster-whisper` 或 WhisperX。WhisperX 的价值在于词级时间戳、VAD、词级时间匹配和说话人分离，适合受损音频和长视频处理。

音频节点建议包含：

1. 音频抽取：使用 `ffmpeg` 从视频抽取单声道音频，保存采样率、声道、码率和时长。
2. 音频增强：执行响度标准化、降噪、静音裁切和必要的高通滤波。
3. VAD 分段：按语音活动切分，保留短重叠区间，减少词语被截断。
4. ASR 转写：生成片段级文本、词级时间戳和置信度。
5. 说话人分离：对会议、访谈、课程视频保留 speaker id。
6. 低置信片段标记：把低置信、重复词、异常长静音、疑似幻听片段交给后续视觉和 RAG 辅助修复。

ASR 输出建议结构：

```json
{
  "segment_id": "seg_0001",
  "start": 12.34,
  "end": 18.92,
  "speaker": "SPEAKER_01",
  "text": "原始转写文本",
  "words": [
    {
      "word": "term",
      "start": 13.01,
      "end": 13.42,
      "confidence": 0.61
    }
  ],
  "quality_flags": ["low_confidence", "possible_term_error"]
}
```

## 5. 视觉模型选型

视觉模型的任务定位是证据抽取，而非独立完成脚本修复。模型输出应采用 JSON，避免让小型视觉模型生成长段自然语言。

视觉模型通过 OpenAI 兼容 API 调用，不在本地加载模型权重。推理服务由外部提供（如 vLLM、SGLang、LM Studio 或云端 API）。视觉请求以 base64 编码图片通过 chat/completions 接口的 `image_url` 字段发送。

### 5.1 主候选模型

| 模型 | 规模 | 适合任务 | 计划定位 |
|---|---:|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | 视频理解、OCR、空间关系、长上下文、多图输入 | 通用 VLM 主候选 |
| `NemoStation/Marlin-2B` | 2B | 视频 dense caption、事件时间定位、结构化 Scene 与 Event 输出 | 视频时间线主候选 |
| `tencent/Penguin-VL-2B` | 2B | 长视频理解、文档 OCR、画面知识、低成本推理 | 小模型主候选 |
| `tencent/Youtu-VL-4B-Instruct` | 4B | 通用视觉理解、OCR、视觉定位、检测和分割类视觉任务 | 通用视觉补充候选 |
| `OpenGVLab/InternVL3_5-2B-HF` | 2.3B | 通用视觉理解、OCR、图像推理、HF 标准接口 | 成熟基线候选 |
| `LiquidAI/LFM2.5-VL-1.6B` | 1.6B | 高速视觉理解、多语言 OCR、实时视频流描述 | 低延迟候选 |
| `PaddlePaddle/PaddleOCR-VL-1.6` | 约 1.0B | 画面文字、幻灯片、表格、公式、图表和文档结构 | 专用 OCR 节点 |

### 5.2 推荐优先级

第一批测试建议采用 `Qwen3-VL-4B-Instruct`、`Marlin-2B`、`Penguin-VL-2B` 和 `PaddleOCR-VL-1.6`。这四者分别覆盖通用视觉质量、视频时间定位、小模型视频效率和专用 OCR。

第二批测试加入 `Youtu-VL-4B-Instruct`、`InternVL3_5-2B-HF` 和 `LFM2.5-VL-1.6B`。这三者用于补充通用视觉能力、成熟生态基线和低延迟帧描述能力。

### 5.3 视觉抽取策略

视觉侧建议按三类帧采样：

1. 固定间隔采样：每 2 到 5 秒取一帧，覆盖整体场景变化。
2. 镜头边界采样：检测画面变化，在镜头切换处取关键帧。
3. ASR 低置信片段采样：围绕低置信音频片段，在前后 2 到 5 秒内加密取帧。

视觉模型输出结构：

```json
{
  "time_range": [12.0, 18.0],
  "scene": "会议室投影演示",
  "people": ["讲者", "听众"],
  "actions": ["讲者指向屏幕", "幻灯片切换"],
  "objects": ["投影屏幕", "笔记本电脑"],
  "visible_text": [
    {
      "text": "Vector Database",
      "bbox": [120, 80, 640, 150],
      "confidence": 0.88
    }
  ],
  "term_candidates": ["Vector Database", "Hybrid Search"],
  "confidence": 0.79,
  "evidence_frame": "frames/000123.jpg"
}
```

## 6. RAG 设计记录

RAG 资料由他人整理，当前计划只约定数据接口和使用方式。RAG 在本项目中承担术语纠正和上下文注入，而非大段内容补写。

RAG 数据建议包含：

1. 术语表：标准写法、别名、缩写、同音误听词。
2. 主题资料：课程标题、会议议程、产品文档、项目背景。
3. 人名和组织名：姓名、职务、组织、常见误拼。
4. 历史字幕和人工修订稿：用于相似表达检索。
5. 来源元数据：资料来源、更新时间、适用主题、可信等级。

检索结果建议结构：

```json
{
  "query": "whisper 低置信片段或候选术语",
  "hits": [
    {
      "term": "Hybrid Search",
      "aliases": ["hybrid retrieval", "混合检索"],
      "common_mishearings": ["high bread search", "hybrid source"],
      "source": "glossary_v1",
      "updated_at": "2026-05-20",
      "score": 0.91
    }
  ]
}
```

## 7. 文本综合模型

最终修复模型可以使用 8B 以内纯文本模型。首选 `Qwen3-8B`，备选 `Llama-3.1-8B-Instruct` 或同规模指令模型。文本模型输入为结构化证据，而非原始视频帧。

文本模型通过 OpenAI 兼容 API 调用，不在本地加载模型权重。推理服务由外部提供（如 vLLM、SGLang、LM Studio 或云端 API）。文本请求通过 chat/completions 接口发送。

文本模型职责：

1. 修正 ASR 中的误听词、断句和标点。
2. 根据视觉证据补足画面中明确出现的术语、标题、产品名和环境信息。
3. 根据 RAG 证据修正专有名词和缩写。
4. 为每段脚本保留证据类型和置信等级。
5. 输出带时间戳的脚本，仅在证据充足时写成确定文本。

输出结构：

```json
{
  "start": 12.34,
  "end": 18.92,
  "speaker": "SPEAKER_01",
  "text": "修复后的脚本文本",
  "evidence": {
    "audio": ["seg_0001"],
    "visual": ["frame_000123"],
    "rag": ["glossary_v1:Hybrid Search"]
  },
  "confidence": 0.84,
  "review_required": false
}
```

## 8. 对比组设计

对比组用于评估主流程相对端到端多模态模型的质量差距和成本差距。主流程仍以 8B 以内本地或云端 GPU 模型为准。

| 组别 | 方案 | 目的 |
|---|---|---|
| A 组 | WhisperX 加 RAG 加文本 8B | 音频与术语基础能力基线 |
| B 组 | WhisperX 加小型 VLM 加 RAG 加文本 8B | 主流程候选 |
| C 组 | WhisperX 加 PaddleOCR-VL-1.6 加 RAG 加文本 8B | 验证专用 OCR 对术语修复的贡献 |
| D 组 | Gemma 4 12B 原生多模态 | 本地原生多模态质量参考，超过主模型规模上限 |
| E 组 | 云端大模型 API | 质量上限与成本参考 |

Gemma 4 12B 可作为原生多模态对比组，因为它是 12B 级统一多模态模型，适合衡量“小型视觉抽取加文本综合”与“原生多模态理解”之间的差距。它超过项目设定的 8B 主流程上限，因此仅作为评测参考。

云端大模型 API 建议只在抽样集上评估，用于建立质量上限和人工复核参考。由于成本约束明确，默认批处理流程采用低成本主方案。

## 9. 评测指标

建议建立 30 到 100 条人工标注片段作为评测集，每条片段包含原视频区间、受损音频、人工参考脚本、术语表引用和画面关键帧。

评测指标包括：

1. 字词错误率：WER 或 CER，用于衡量脚本文本相对人工参考稿的差异。
2. 术语准确率：专有名词、缩写、人名、产品名是否修正为标准写法。
3. 时间戳偏差：片段开始和结束时间相对人工标注的差值。
4. 视觉证据命中率：画面文字和场景线索是否被正确使用。
5. 无依据补写率：模型是否加入证据中缺失的内容。
6. 人工修订时间：人工完成复核所需时间。
7. 推理成本：每小时视频的 GPU 时间、API 成本和处理耗时。

## 10. 实施阶段

### 阶段一：最小原型

目标是跑通单视频处理流程。

工作内容：

1. 使用 `ffmpeg` 抽取音频和关键帧。
2. 使用 WhisperX 生成片段级转写、词级时间戳和说话人标签。
3. 使用一个小型 VLM 输出视觉 JSON。
4. 使用临时术语表模拟 RAG 检索结果。
5. 使用 8B 文本模型生成带时间戳修复脚本。
6. 输出 JSON 和 Markdown 两种格式。

### 阶段二：模型对比

目标是评估小型视觉模型与对比组。

测试模型：

1. `Qwen/Qwen3-VL-4B-Instruct`
2. `tencent/Penguin-VL-2B`
3. `NemoStation/Marlin-2B`
4. `tencent/Youtu-VL-4B-Instruct`
5. `OpenGVLab/InternVL3_5-2B-HF`
6. `LiquidAI/LFM2.5-VL-1.6B`
7. `PaddlePaddle/PaddleOCR-VL-1.6`
8. `Gemma 4 12B`
9. 选定的云端大模型 API

评估重点：

1. 术语修复准确率。
2. 画面文字提取准确率。
3. 低置信音频片段修复贡献。
4. 每小时视频处理成本。
5. 人工复核效率。

### 阶段三：工程化处理

目标是让流程支持批处理、重试和人工复核。

工作内容：

1. 用 LangGraph 管理节点状态和检查点。
2. 为每个节点建立输入输出 schema。
3. 增加缓存，避免重复转写和重复视觉推理。
4. 增加人工复核界面所需的证据字段。
5. 增加失败重试和节点级日志。
6. 增加导出格式：JSON、Markdown、SRT、VTT。

## 11. 关键工程注意事项

1. 小型视觉模型只输出结构化证据，最终脚本由纯文本模型生成。
2. 视觉证据应按时间戳归入相邻 ASR 片段，避免跨片段错误引用。
3. OCR 建议独立成节点，画面文字对术语修复的价值通常高于普通场景描述。
4. RAG 检索结果需要保留来源和更新时间，最终输出中应区分音频证据、视觉证据和术语证据。
5. 低置信 ASR 片段应进入人工复核队列。
6. 文本模型提示词应要求模型保留时间戳，禁止扩大时间范围。
7. 对比组只用于抽样评测，避免把高成本模型带入默认处理流程。

## 12. 建议的初始技术栈

| 模块 | 建议组件 |
|---|---|
| 流程编排 | LangGraph |
| 模型与检索封装 | LangChain |
| 音频抽取 | ffmpeg |
| ASR | WhisperX 或 faster-whisper |
| 说话人分离 | pyannote.audio，经 WhisperX 调用 |
| 视觉推理 | OpenAI 兼容 API（vLLM、SGLang、LM Studio 或云端 API） |
| OCR | PaddleOCR-VL-1.6 或常规 PaddleOCR |
| 向量库 | Milvus、Qdrant、pgvector 或 Chroma |
| 文本模型服务 | OpenAI 兼容 API（vLLM、SGLang、LM Studio、Ollama 或云端 API） |
| 追踪与调试 | LangSmith 或自建日志表 |

## 13. 资料来源记录

截至 2026-06-08，本计划参考的公开资料包括：

1. Qwen3-VL-4B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
2. Tencent Penguin-VL-2B 模型卡：https://huggingface.co/tencent/Penguin-VL-2B
3. LiquidAI LFM2.5-VL-1.6B 模型卡：https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
4. InternVL3.5-2B-HF 模型卡：https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF
5. NemoStation Marlin-2B 模型卡：https://huggingface.co/NemoStation/Marlin-2B
6. Tencent Youtu-VL-4B-Instruct 模型卡：https://huggingface.co/tencent/Youtu-VL-4B-Instruct
7. PaddleOCR-VL-1.6 模型卡：https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
8. PaddleOCR-VL-1.6 论文：https://arxiv.org/abs/2606.03264
9. Gemma 4 12B Developer Guide：https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
10. Gemma 4 12B Hugging Face 模型卡：https://huggingface.co/google/gemma-4-12B-it
11. WhisperX 仓库：https://github.com/m-bain/whisperX
12. LangGraph Persistence 文档：https://docs.langchain.com/oss/python/langgraph/persistence
13. Gemini API 视频理解文档：https://ai.google.dev/gemini-api/docs/video-understanding
