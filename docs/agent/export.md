# Export & CLI

Covers `export.py` and `cli.py`.

## export

Exports repaired segments to JSON and Markdown files.

**Function:** `run(segments: list[RepairedSegment], config: JobConfig) -> ExportResult`

### JSON Export

`{output_dir}/{job_id}/script.json` — array of `RepairedSegment` objects.

### Markdown Export

`{output_dir}/{job_id}/script.md` — human-readable script with:
- Header: video path, topic, job ID
- Per-segment blocks: timestamp range, speaker, text, evidence sources, confidence
- Review flags: segments needing human review are marked with `:warning: REVIEW REQUIRED`

### Timestamp Format

`HH:MM:SS.mmm` (e.g. `00:01:05.123`)

## cli

CLI entry point chaining all pipeline steps.

**Usage:**

```bash
uv run python -m src.agent.cli \
  --video path/to/video.mp4 \
  --topic "meeting about vector databases" \
  --output-dir ./output \
  --whisper-model base \
  --glossary data/glossary.json \
  --diarize
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Input video path |
| `--topic` | `""` | Topic hint |
| `--output-dir` | `./output` | Output directory |
| `--whisper-model` | `base` | WhisperX model size |
| `--vlm-model` | `Qwen/Qwen3-VL-4B-Instruct` | VLM model |
| `--text-model` | `Qwen/Qwen3-8B` | Text repair model |
| `--glossary` | `data/glossary.json` | Glossary path |
| `--diarize` | `false` | Enable speaker diarization |
| `--job-id` | auto-generated | Job identifier |

### Pipeline Steps

1. Audio extraction (ffmpeg)
2. Audio preprocessing (loudnorm, highpass)
3. ASR transcription (WhisperX)
4. Frame sampling (interval + scene + low-conf)
5. VLM extraction (scene + OCR)
6. RAG retrieval (glossary lookup)
7. Evidence merge + script repair (text model)
8. Export (JSON + Markdown)

Prints a summary at the end: segment count, review-flagged count, output paths.
