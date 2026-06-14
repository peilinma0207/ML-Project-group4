from __future__ import annotations

import argparse
import json
import logging
import subprocess
import uuid
from pathlib import Path

from .schema import AudioMeta, JobConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video damaged-audio script repair pipeline",
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--audio", default="", help="Path to pre-extracted audio (skips extraction + preprocessing)")
    parser.add_argument("--topic", default="", help="Topic hint for the video content")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--whisper-model", default="base", help="WhisperX model size")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--vlm-api-base", default="", help="OpenAI-compatible API base URL for VLM (e.g. http://localhost:1234/v1)")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--text-api-base", default="", help="OpenAI-compatible API base URL for text model (e.g. http://localhost:1234/v1)")
    parser.add_argument("--glossary", default="data/glossary.json", help="Path to glossary JSON")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--job-id", default=None, help="Job ID (auto-generated if not set)")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM visual extraction (still samples frames)")
    parser.add_argument("--skip-repair", action="store_true", help="Skip text model repair (use raw ASR output)")
    return parser.parse_args(argv)


def _probe_audio(audio_path: Path) -> AudioMeta:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    fmt = info.get("format", {})
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    return AudioMeta(
        audio_uri=str(audio_path),
        sample_rate=int(stream.get("sample_rate", 16000)),
        channels=int(stream.get("channels", 1)),
        duration=float(fmt.get("duration", 0.0)),
        codec=stream.get("codec_name", "unknown"),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    config = JobConfig(
        job_id=args.job_id or uuid.uuid4().hex[:8],
        video_uri=args.video,
        audio_uri=args.audio,
        topic_hint=args.topic,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        vlm_model=args.vlm_model,
        vlm_api_base=args.vlm_api_base,
        text_model=args.text_model,
        text_api_base=args.text_api_base,
        glossary_path=args.glossary,
        enable_diarization=args.diarize,
    )

    from . import audio_extract, audio_preprocess, asr_transcribe
    from . import frame_sample, vlm_extract, rag_retrieve
    from . import evidence_merge, script_repair, export

    logger.info("Starting pipeline for video: %s", config.video_uri)

    if config.audio_uri:
        logger.info("Step 1-2/8: Using pre-extracted audio: %s", config.audio_uri)
        audio = _probe_audio(Path(config.audio_uri))
    else:
        logger.info("Step 1/8: Extracting audio...")
        audio = audio_extract.run(config)
        logger.info("Step 2/8: Preprocessing audio...")
        audio = audio_preprocess.run(audio, config)

    logger.info("Step 3/8: Transcribing with WhisperX (%s)...", config.whisper_model)
    segments = asr_transcribe.run(audio, config)
    logger.info("  -> %d segments transcribed", len(segments))
    low_conf = sum(1 for s in segments if "low_confidence" in s.quality_flags)
    logger.info("  -> %d low-confidence segments", low_conf)

    logger.info("Step 4/8: Sampling frames...")
    frames = frame_sample.run(config, segments)
    logger.info("  -> %d frames sampled", len(frames))

    if args.skip_vlm:
        logger.info("Step 5/8: Skipping VLM extraction (--skip-vlm)")
        visual_events = []
    else:
        logger.info("Step 5/8: Extracting visual evidence...")
        visual_events = vlm_extract.run(frames, config)
        logger.info("  -> %d visual events", len(visual_events))

    logger.info("Step 6/8: Retrieving RAG terms...")
    rag_hits = rag_retrieve.run(segments, visual_events, config.glossary_path)
    logger.info("  -> %d RAG hits", len(rag_hits))
    for hit in rag_hits:
        logger.info("     %s (score=%.2f, source=%s)", hit.term, hit.score, hit.source)

    logger.info("Step 7/8: Merging evidence and repairing script...")
    merged = evidence_merge.run(segments, visual_events, rag_hits)

    if args.skip_repair:
        logger.info("  Skipping text model repair (--skip-repair), using raw ASR")
        from .schema import EvidenceSource, RepairedSegment
        repaired = []
        for m in merged:
            repaired.append(RepairedSegment(
                start=m.asr.start,
                end=m.asr.end,
                speaker=m.asr.speaker,
                text=m.asr.text,
                evidence=EvidenceSource(
                    audio=[m.segment_id],
                    rag=[f"{h.source}:{h.term}" for h in m.rag_hits],
                ),
                confidence=min((w.confidence for w in m.asr.words), default=1.0),
                review_required="low_confidence" in m.asr.quality_flags,
            ))
    else:
        repaired = script_repair.run(merged, config)
    logger.info("  -> %d segments repaired", len(repaired))

    logger.info("Step 8/8: Exporting...")
    result = export.run(repaired, config)

    review_count = sum(1 for s in repaired if s.review_required)
    logger.info("Pipeline complete!")
    logger.info("  Segments: %d", len(repaired))
    logger.info("  Flagged for review: %d", review_count)
    logger.info("  JSON: %s", result.json_path)
    logger.info("  Markdown: %s", result.markdown_path)


if __name__ == "__main__":
    main()
