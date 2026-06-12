# ASR Transcription

Covers `asr_transcribe.py`.

## Overview

Wraps WhisperX to produce word-level timestamped transcription with optional speaker diarization.

**Function:** `run(audio: AudioMeta, config: JobConfig) -> list[ASRSegment]`

## Process

1. Detects device (CUDA if available, else CPU)
2. Loads WhisperX model (size from `config.whisper_model`)
3. Transcribes audio → raw segments
4. Aligns words for word-level timestamps via `whisperx.align()`
5. Optionally runs speaker diarization via `pyannote.audio`
6. Flags low-confidence words (below `config.low_confidence_threshold`)

## Quality Flags

Segments receive quality flags based on word confidence:

- `low_confidence` — at least one word below the confidence threshold (default 0.6)

## Config

| Field | Effect |
|-------|--------|
| `whisper_model` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `enable_diarization` | Enable pyannote speaker diarization |
| `low_confidence_threshold` | Word confidence cutoff for flagging |

## Notes

- WhisperX requires a HuggingFace token for pyannote speaker diarization models
- Word timestamps come from the alignment step, not directly from Whisper
- Segment IDs are sequential: `seg_0000`, `seg_0001`, etc.
