from __future__ import annotations

import json
from pathlib import Path

from src.schema import ExportResult, JobConfig, RepairedSegment


def run(
    segments: list[RepairedSegment],
    config: JobConfig,
) -> ExportResult:
    job_dir = Path(config.output_dir) / config.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    json_path = job_dir / "script.json"
    md_path = job_dir / "script.md"

    _export_json(segments, json_path)
    _export_markdown(segments, md_path, config)

    return ExportResult(
        json_path=str(json_path),
        markdown_path=str(md_path),
    )


def _export_json(segments: list[RepairedSegment], path: Path) -> None:
    data = [seg.model_dump() for seg in segments]
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _export_markdown(
    segments: list[RepairedSegment],
    path: Path,
    config: JobConfig,
) -> None:
    lines = [
        f"# Repaired Script",
        f"",
        f"**Video:** {config.video_uri}  ",
        f"**Topic:** {config.topic_hint}  ",
        f"**Job ID:** {config.job_id}  ",
        f"",
        f"---",
        f"",
    ]

    for seg in segments:
        ts_start = _format_timestamp(seg.start)
        ts_end = _format_timestamp(seg.end)
        speaker = f"**{seg.speaker}**" if seg.speaker else "**Unknown**"
        review = " :warning: REVIEW REQUIRED" if seg.review_required else ""

        lines.append(f"### [{ts_start} - {ts_end}] {speaker}{review}")
        lines.append(f"")
        lines.append(seg.text)
        lines.append(f"")

        if seg.evidence.audio or seg.evidence.visual or seg.evidence.rag:
            lines.append(f"_Evidence: audio={seg.evidence.audio}, visual={seg.evidence.visual}, rag={seg.evidence.rag}_")
            lines.append(f"")

        lines.append(f"Confidence: {seg.confidence:.2f}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
