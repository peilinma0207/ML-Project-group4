from __future__ import annotations

import subprocess
from pathlib import Path

from .schema import AudioMeta, JobConfig


def run(audio: AudioMeta, config: JobConfig) -> AudioMeta:
    input_path = Path(audio.audio_uri)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    job_dir = Path(config.output_dir) / config.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "audio_processed.wav"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", _build_filter_chain(),
            "-ar", str(audio.sample_rate),
            "-ac", str(audio.channels),
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return AudioMeta(
        audio_uri=str(output_path),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        duration=audio.duration,
        codec=audio.codec,
    )


def _build_filter_chain() -> str:
    filters = [
        "highpass=f=80",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ]
    return ",".join(filters)
