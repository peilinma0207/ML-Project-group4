# Schema Reference

All inter-node data types are defined in `src/agent/schema.py` as Pydantic models.

## Configuration

### JobConfig

Pipeline-level configuration. Created from CLI arguments.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `job_id` | `str` | — | Unique job identifier |
| `video_uri` | `str` | — | Path to input video |
| `topic_hint` | `str` | `""` | Topic context for VLM and repair prompts |
| `output_dir` | `str` | `"./output"` | Root output directory |
| `whisper_model` | `str` | `"base"` | WhisperX model size |
| `vlm_model` | `str` | `"Qwen/Qwen3-VL-4B-Instruct"` | VLM model name |
| `text_model` | `str` | `"Qwen/Qwen3-8B"` | Text repair model name |
| `glossary_path` | `str` | `"data/glossary.json"` | Path to RAG glossary |
| `enable_diarization` | `bool` | `False` | Enable speaker diarization |
| `frame_interval` | `float` | `3.0` | Seconds between interval frames |
| `low_confidence_threshold` | `float` | `0.6` | Word confidence threshold |

## Audio Types

### AudioMeta

Metadata for an audio file produced by extraction or preprocessing.

| Field | Type | Default |
|-------|------|---------|
| `audio_uri` | `str` | — |
| `sample_rate` | `int` | `16000` |
| `channels` | `int` | `1` |
| `duration` | `float` | `0.0` |
| `codec` | `str` | `"pcm_s16le"` |

### WordTimestamp

Single word with timing and confidence from ASR.

| Field | Type |
|-------|------|
| `word` | `str` |
| `start` | `float` |
| `end` | `float` |
| `confidence` | `float` |

### ASRSegment

A transcribed segment with word-level timestamps.

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | `str` | e.g. `"seg_0001"` |
| `start` / `end` | `float` | Segment time range |
| `speaker` | `str` | Speaker label (if diarization enabled) |
| `text` | `str` | Transcribed text |
| `words` | `list[WordTimestamp]` | Word-level detail |
| `quality_flags` | `list[str]` | e.g. `["low_confidence"]` |

## Visual Types

### SampledFrame

A keyframe extracted from the video.

| Field | Type | Description |
|-------|------|-------------|
| `frame_path` | `str` | Path to saved JPEG |
| `timestamp` | `float` | Frame time in video |
| `sample_reason` | `str` | `"interval"`, `"scene_change"`, or `"low_confidence"` |

### OCRHit

Text detected in a frame via OCR.

| Field | Type |
|-------|------|
| `text` | `str` |
| `bbox` | `list[int]` |
| `confidence` | `float` |

### VisualEvent

Aggregated visual evidence over a time range.

| Field | Type |
|-------|------|
| `time_range` | `tuple[float, float]` |
| `scene` | `str` |
| `people` | `list[str]` |
| `actions` | `list[str]` |
| `objects` | `list[str]` |
| `visible_text` | `list[OCRHit]` |
| `term_candidates` | `list[str]` |
| `confidence` | `float` |
| `evidence_frame` | `str` |

## RAG Types

### RAGHit

A glossary match result.

| Field | Type |
|-------|------|
| `term` | `str` |
| `aliases` | `list[str]` |
| `common_mishearings` | `list[str]` |
| `source` | `str` |
| `score` | `float` |

## Merge & Repair Types

### MergedEvidence

All evidence aligned to one ASR segment.

| Field | Type |
|-------|------|
| `segment_id` | `str` |
| `asr` | `ASRSegment` |
| `visual_events` | `list[VisualEvent]` |
| `rag_hits` | `list[RAGHit]` |

### EvidenceSource

Tracks which evidence sources contributed to a repair.

| Field | Type |
|-------|------|
| `audio` | `list[str]` |
| `visual` | `list[str]` |
| `rag` | `list[str]` |

### RepairedSegment

Final repaired script segment.

| Field | Type | Description |
|-------|------|-------------|
| `start` / `end` | `float` | Original timestamps (preserved) |
| `speaker` | `str` | Original speaker label |
| `text` | `str` | Corrected text |
| `evidence` | `EvidenceSource` | Contributing sources |
| `confidence` | `float` | Repair confidence |
| `review_required` | `bool` | Flagged for human review |

## Export & Pipeline Types

### ExportResult

Paths to exported files.

| Field | Type |
|-------|------|
| `json_path` | `str` |
| `markdown_path` | `str` |

### PipelineState

Full pipeline state carrying all intermediate results. Used for checkpointing.
