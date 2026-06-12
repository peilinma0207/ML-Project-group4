from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .schema import AudioMeta, JobConfig


def run(config: JobConfig) -> AudioMeta:
    video = Path(config.video_uri)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    job_dir = Path(config.output_dir) / config.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "audio.wav"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    duration = _probe_duration(output_path)

    return AudioMeta(
        audio_uri=str(output_path),
        sample_rate=16000,
        channels=1,
        duration=duration,
        codec="pcm_s16le",
    )


def _probe_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])
