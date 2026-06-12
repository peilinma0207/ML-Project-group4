from __future__ import annotations

import logging
from pathlib import Path

import whisperx

from .schema import ASRSegment, AudioMeta, JobConfig, WordTimestamp

logger = logging.getLogger(__name__)


def run(audio: AudioMeta, config: JobConfig) -> list[ASRSegment]:
    audio_path = Path(audio.audio_uri)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(
        config.whisper_model,
        device=device,
        compute_type=compute_type,
    )

    audio_data = whisperx.load_audio(str(audio_path))
    raw_result = model.transcribe(audio_data)

    align_model, align_meta = whisperx.load_align_model(
        language_code=raw_result.get("language", "en"),
        device=device,
    )
    aligned = whisperx.align(
        raw_result["segments"],
        align_model,
        align_meta,
        audio_data,
        device=device,
    )

    segments = aligned.get("segments", raw_result.get("segments", []))

    if config.enable_diarization:
        segments = _apply_diarization(audio_data, segments, device)

    return _build_asr_segments(segments, config.low_confidence_threshold)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _apply_diarization(
    audio_data,
    segments: list[dict],
    device: str,
) -> list[dict]:
    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_result = diarize_model(audio_data)
    return whisperx.assign_word_speakers(diarize_result, {"segments": segments})["segments"]


def _build_asr_segments(
    raw_segments: list[dict],
    confidence_threshold: float,
) -> list[ASRSegment]:
    results = []
    for i, seg in enumerate(raw_segments):
        words = []
        has_low_confidence = False

        for w in seg.get("words", []):
            confidence = w.get("score", w.get("confidence", 1.0))
            word = WordTimestamp(
                word=w.get("word", ""),
                start=w.get("start", seg.get("start", 0.0)),
                end=w.get("end", seg.get("end", 0.0)),
                confidence=confidence,
            )
            words.append(word)
            if confidence < confidence_threshold:
                has_low_confidence = True

        quality_flags = []
        if has_low_confidence:
            quality_flags.append("low_confidence")

        text = seg.get("text", "").strip()
        if not text and words:
            text = " ".join(w.word for w in words)

        results.append(ASRSegment(
            segment_id=f"seg_{i:04d}",
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            speaker=seg.get("speaker", ""),
            text=text,
            words=words,
            quality_flags=quality_flags,
        ))

    return results
