# Visual Processing

Covers `frame_sample.py` and `vlm_extract.py`.

## frame_sample

Extracts keyframes from video using three strategies.

**Function:** `run(config: JobConfig, segments: list[ASRSegment]) -> list[SampledFrame]`

### Sampling Strategies

1. **Interval sampling** — one frame every `config.frame_interval` seconds (default 3s)
2. **Scene-change detection** — ffmpeg `select='gt(scene,0.3)'` filter identifies shot boundaries
3. **Low-confidence region sampling** — dense 1s-interval frames around ASR segments flagged with `low_confidence`, extending 2s before/after the segment

### Post-processing

- **Deduplication:** frames within 0.5s of each other are deduplicated (first wins)
- **Renumbering:** final frames are renumbered sequentially as `frame_000001.jpg`, etc.

Frames are saved to `{output_dir}/{job_id}/frames/`.

## vlm_extract

Extracts structured visual evidence from sampled frames using a vision-language model.

**Function:** `run(frames: list[SampledFrame], config: JobConfig) -> list[VisualEvent]`

### Process

1. Loads VLM (default: `Qwen/Qwen3-VL-4B-Instruct` via Transformers)
2. For each frame, prompts the model to output structured JSON:
   - Scene description, people, actions, objects
   - Visible text (OCR) with bounding boxes and confidence
   - Term candidates relevant to the topic
3. Parses JSON output, validates against schema
4. Merges nearby events (within 2s) into a single `VisualEvent`

### Merging

When events are within `max_gap` seconds:
- Time ranges are extended to cover both
- People, actions, objects, term candidates are unioned
- Visible text lists are concatenated
- Higher confidence is kept

### Error Handling

- Missing frames are skipped with a warning
- Malformed model output is logged and skipped (not crash)
- Falls back gracefully on model inference errors
