# Agent Pipeline Specification

Video damaged-audio script repair pipeline. Extracts audio, visual, and RAG evidence from a video, then uses a text model to produce a corrected timestamped script.

## Architecture

```
video_ingest
  → audio_extract → audio_preprocess → asr_transcribe
  → frame_sample → vlm_extract
  → rag_retrieve
  → evidence_merge → script_repair → export
```

Each step is a standalone module under `src/agent/` exposing a `run()` function. A CLI (`src/agent/cli.py`) chains them together. No LangGraph yet (deferred to Phase 3).

## Modules

| Module | Description | Detail |
|--------|-------------|--------|
| `schema.py` | Pydantic models for all inter-node data | [schema.md](schema.md) |
| `audio_extract.py` | ffmpeg mono WAV extraction | [audio.md](audio.md) |
| `audio_preprocess.py` | Loudness normalization, high-pass filter | [audio.md](audio.md) |
| `asr_transcribe.py` | WhisperX transcription with word timestamps | [asr.md](asr.md) |
| `frame_sample.py` | Keyframe extraction (interval/scene/low-conf) | [visual.md](visual.md) |
| `vlm_extract.py` | VLM scene description and OCR | [visual.md](visual.md) |
| `rag_retrieve.py` | Mock RAG term lookup from JSON glossary | [rag.md](rag.md) |
| `evidence_merge.py` | Time-aligned evidence merging per ASR segment | [repair.md](repair.md) |
| `script_repair.py` | 8B text model repair with structured prompting | [repair.md](repair.md) |
| `export.py` | JSON and Markdown script export | [export.md](export.md) |
| `cli.py` | CLI entry point | [export.md](export.md) |

## CLI Usage

```bash
uv run python -m src.agent.cli \
  --video path/to/video.mp4 \
  --topic "meeting about vector databases" \
  --output-dir ./output
```

## Testing

- `uv run pytest tests/ -v` — 129 unit tests, all mocked (no GPU/ffmpeg required)
- Tests live in `tests/test_*.py`, fixtures in `tests/conftest.py`

## Phase 1 Non-Goals

- No LangGraph (Phase 3)
- No real vector store (mock glossary only)
- No batch processing, web UI, or SRT/VTT export
- No model comparison (Phase 2)
