# Phase 1 MVP Specification

## Goal

Run a single video through the full pipeline and produce a timestamped repaired script.

## Architecture

Phase 1 builds each processing step as a standalone Python module under `src/`. No LangGraph yet — each module exposes a single `run()` function that takes typed input and returns typed output. A top-level CLI script chains them together.

## Directory Layout

```
agent/
  plan.md          # overall plan (existing)
  spec.md          # this file
  TODO.1 … TODO.N  # per-step task files
src/
  __init__.py
  schema.py        # shared Pydantic models for all inter-node data
  audio_extract.py # ffmpeg audio extraction
  audio_preprocess.py  # loudness norm, denoise, VAD trim
  asr_transcribe.py    # WhisperX transcription + word timestamps + diarization
  frame_sample.py      # keyframe extraction (interval + shot-boundary + low-conf)
  vlm_extract.py       # VLM scene + OCR extraction
  rag_retrieve.py      # mock RAG term lookup (JSON glossary)
  evidence_merge.py    # merge audio + visual + RAG evidence per segment
  script_repair.py     # text model repair pass
  export.py            # JSON + Markdown export
  cli.py               # CLI entry point chaining all steps
tests/
  conftest.py
  test_schema.py
  test_audio_extract.py
  test_audio_preprocess.py
  test_asr_transcribe.py
  test_frame_sample.py
  test_vlm_extract.py
  test_rag_retrieve.py
  test_evidence_merge.py
  test_script_repair.py
  test_export.py
data/
  glossary.json    # mock RAG glossary for Phase 1
```

## Shared State Schema (schema.py)

All inter-node data is defined as Pydantic models. Key types:

- `JobConfig` — job_id, video_uri, topic_hint
- `AudioMeta` — audio_uri, sample_rate, channels, duration
- `ASRSegment` — segment_id, start, end, speaker, text, words (list of `WordTimestamp`), quality_flags
- `WordTimestamp` — word, start, end, confidence
- `SampledFrame` — frame_path, timestamp, sample_reason
- `VisualEvent` — time_range, scene, people, actions, objects, visible_text, term_candidates, confidence, evidence_frame
- `OCRHit` — text, bbox, confidence
- `RAGHit` — term, aliases, common_mishearings, source, score
- `MergedEvidence` — segment_id, asr, visual_events, rag_hits
- `RepairedSegment` — start, end, speaker, text, evidence, confidence, review_required
- `PipelineState` — the full state object carrying all intermediate results

## Module Contracts

Each module follows this pattern:

```python
def run(input: InputType, config: JobConfig) -> OutputType:
    ...
```

### 1. audio_extract

- Input: `JobConfig` (contains `video_uri`)
- Output: `AudioMeta`
- Uses: ffmpeg subprocess to extract mono WAV, 16kHz
- Tests: mock subprocess, verify output metadata

### 2. audio_preprocess

- Input: `AudioMeta`
- Output: `AudioMeta` (updated path to processed audio)
- Uses: ffmpeg filters (loudness normalization, high-pass filter)
- Tests: verify filter commands, handle missing audio

### 3. asr_transcribe

- Input: `AudioMeta`
- Output: `list[ASRSegment]`
- Uses: WhisperX for transcription, word alignment, speaker diarization
- Tests: mock WhisperX, verify segment structure, low-confidence flagging

### 4. frame_sample

- Input: `JobConfig`, `list[ASRSegment]`
- Output: `list[SampledFrame]`
- Uses: ffmpeg for frame extraction, scene-change detection
- Tests: verify sampling strategies, frame file naming

### 5. vlm_extract

- Input: `list[SampledFrame]`, `JobConfig`
- Output: `list[VisualEvent]`
- Uses: Transformers / vLLM for VLM inference, structured JSON output
- Tests: mock model, verify JSON schema compliance

### 6. rag_retrieve

- Input: `list[ASRSegment]`, `list[VisualEvent]`
- Output: `list[RAGHit]`
- Uses: JSON glossary file lookup (mock for Phase 1)
- Tests: term matching, alias resolution, empty glossary

### 7. evidence_merge

- Input: `list[ASRSegment]`, `list[VisualEvent]`, `list[RAGHit]`
- Output: `list[MergedEvidence]`
- Uses: time-range alignment to match visual/RAG evidence to ASR segments
- Tests: overlapping ranges, no-match segments, boundary conditions

### 8. script_repair

- Input: `list[MergedEvidence]`, `JobConfig`
- Output: `list[RepairedSegment]`
- Uses: 8B text model (Qwen3-8B via vLLM/Ollama), structured prompt
- Tests: mock model, verify output schema, timestamp preservation

### 9. export

- Input: `list[RepairedSegment]`, `JobConfig`
- Output: file paths (JSON file, Markdown file)
- Tests: verify file content format, timestamp formatting

## CLI Interface

```bash
uv run python -m src.cli --video path/to/video.mp4 --topic "meeting about vector databases" --output-dir ./output
```

## Testing Strategy

- Unit tests per module with mocked external deps (ffmpeg, WhisperX, models)
- Integration test with a short sample video (manual, not in CI)
- pytest as test runner
- All tests runnable without GPU

## Phase 1 Non-Goals

- No LangGraph integration (deferred to Phase 3)
- No real RAG vector store (mock glossary only)
- No batch processing
- No web UI or human review interface
- No SRT/VTT export (deferred to Phase 3)
- No model comparison (deferred to Phase 2)
