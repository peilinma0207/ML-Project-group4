from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.schema import ASRSegment, JobConfig, SampledFrame


def run(config: JobConfig, segments: list[ASRSegment]) -> list[SampledFrame]:
    video = Path(config.video_uri)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    frames_dir = Path(config.output_dir) / config.job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video)

    frames: list[SampledFrame] = []
    frames.extend(_interval_sample(video, frames_dir, duration, config.frame_interval))
    frames.extend(_scene_change_sample(video, frames_dir))
    frames.extend(_low_confidence_sample(video, frames_dir, segments))

    frames = _deduplicate(frames, min_gap=0.5)
    frames = _renumber(frames, frames_dir)

    return frames


def _probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _interval_sample(
    video: Path,
    frames_dir: Path,
    duration: float,
    interval: float,
) -> list[SampledFrame]:
    frames = []
    t = 0.0
    while t < duration:
        frame_path = frames_dir / f"interval_{t:.3f}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", str(video),
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
            ],
            check=True,
            capture_output=True,
        )
        frames.append(SampledFrame(
            frame_path=str(frame_path),
            timestamp=t,
            sample_reason="interval",
        ))
        t += interval
    return frames


def _scene_change_sample(video: Path, frames_dir: Path) -> list[SampledFrame]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_frames",
            "-f", "lavfi",
            f"movie={str(video)},select='gt(scene\\,0.3)'",
        ],
        capture_output=True,
        text=True,
    )

    frames = []
    if result.returncode != 0:
        return frames

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return frames

    for frame_info in data.get("frames", []):
        t = float(frame_info.get("pts_time", frame_info.get("best_effort_timestamp_time", 0)))
        frame_path = frames_dir / f"scene_{t:.3f}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", str(video),
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
            ],
            check=True,
            capture_output=True,
        )
        frames.append(SampledFrame(
            frame_path=str(frame_path),
            timestamp=t,
            sample_reason="scene_change",
        ))

    return frames


def _low_confidence_sample(
    video: Path,
    frames_dir: Path,
    segments: list[ASRSegment],
) -> list[SampledFrame]:
    frames = []
    for seg in segments:
        if "low_confidence" not in seg.quality_flags:
            continue

        start = max(0.0, seg.start - 2.0)
        end = seg.end + 2.0
        step = 1.0
        t = start
        while t <= end:
            frame_path = frames_dir / f"lowconf_{t:.3f}.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(t),
                    "-i", str(video),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(frame_path),
                ],
                check=True,
                capture_output=True,
            )
            frames.append(SampledFrame(
                frame_path=str(frame_path),
                timestamp=t,
                sample_reason="low_confidence",
            ))
            t += step

    return frames


def _deduplicate(frames: list[SampledFrame], min_gap: float) -> list[SampledFrame]:
    if not frames:
        return frames

    sorted_frames = sorted(frames, key=lambda f: f.timestamp)
    result = [sorted_frames[0]]
    for frame in sorted_frames[1:]:
        if frame.timestamp - result[-1].timestamp >= min_gap:
            result.append(frame)
    return result


def _renumber(frames: list[SampledFrame], frames_dir: Path) -> list[SampledFrame]:
    result = []
    for i, frame in enumerate(frames):
        new_name = f"frame_{i + 1:06d}.jpg"
        new_path = frames_dir / new_name
        result.append(SampledFrame(
            frame_path=str(new_path),
            timestamp=frame.timestamp,
            sample_reason=frame.sample_reason,
        ))
    return result
