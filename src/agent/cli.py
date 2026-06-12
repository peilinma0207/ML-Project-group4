from __future__ import annotations

import argparse
import logging
import sys
import uuid

from .schema import JobConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video damaged-audio script repair pipeline",
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--topic", default="", help="Topic hint for the video content")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--whisper-model", default="base", help="WhisperX model size")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--glossary", default="data/glossary.json", help="Path to glossary JSON")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--job-id", default=None, help="Job ID (auto-generated if not set)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    config = JobConfig(
        job_id=args.job_id or uuid.uuid4().hex[:8],
        video_uri=args.video,
        topic_hint=args.topic,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        vlm_model=args.vlm_model,
        text_model=args.text_model,
        glossary_path=args.glossary,
        enable_diarization=args.diarize,
    )

    from . import audio_extract, audio_preprocess, asr_transcribe
    from . import frame_sample, vlm_extract, rag_retrieve
    from . import evidence_merge, script_repair, export

    logger.info("Starting pipeline for video: %s", config.video_uri)

    logger.info("Step 1/8: Extracting audio...")
    audio = audio_extract.run(config)

    logger.info("Step 2/8: Preprocessing audio...")
    audio = audio_preprocess.run(audio, config)

    logger.info("Step 3/8: Transcribing with WhisperX...")
    segments = asr_transcribe.run(audio, config)
    logger.info("  -> %d segments transcribed", len(segments))

    logger.info("Step 4/8: Sampling frames...")
    frames = frame_sample.run(config, segments)
    logger.info("  -> %d frames sampled", len(frames))

    logger.info("Step 5/8: Extracting visual evidence...")
    visual_events = vlm_extract.run(frames, config)
    logger.info("  -> %d visual events", len(visual_events))

    logger.info("Step 6/8: Retrieving RAG terms...")
    rag_hits = rag_retrieve.run(segments, visual_events, config.glossary_path)
    logger.info("  -> %d RAG hits", len(rag_hits))

    logger.info("Step 7/8: Merging evidence and repairing script...")
    merged = evidence_merge.run(segments, visual_events, rag_hits)
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
