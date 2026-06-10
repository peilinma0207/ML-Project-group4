from __future__ import annotations

import argparse
import sys


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Pipeline not yet implemented. Video: {args.video}, Topic: {args.topic}")


if __name__ == "__main__":
    main()
