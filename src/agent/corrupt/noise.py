"""Audio noise corruption using ffmpeg.

Generates noisy audio at multiple SNR levels to simulate real-world conditions
that degrade ASR transcription quality.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class NoiseResult:
    video_id: str
    original_path: str
    outputs: list[dict]  # [{snr, noise_type, output_path}]


def add_noise(
    audio_path: str | Path,
    video_id: str,
    output_dir: str | Path,
    snr_levels: list[int] | None = None,
    noise_types: list[str] | None = None,
) -> NoiseResult:
    """Add noise to audio at multiple SNR levels.

    Args:
        audio_path: Path to input WAV file.
        video_id: Identifier like "video_01".
        output_dir: Directory to write noisy audio files.
        snr_levels: List of SNR values in dB (default: [20, 10, 3]).
        noise_types: List of noise types (default: ["white", "reverb"]).

    Returns:
        NoiseResult with paths to all generated files.
    """
    if snr_levels is None:
        snr_levels = [20, 10, 3]
    if noise_types is None:
        noise_types = ["white", "reverb"]

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = _get_duration(audio_path)
    outputs = []

    for noise_type in noise_types:
        for snr in snr_levels:
            out_name = f"{video_id}_{noise_type}_snr{snr}.wav"
            out_path = output_dir / out_name

            if noise_type == "white":
                _add_white_noise(audio_path, out_path, snr, duration)
            elif noise_type == "reverb":
                _add_reverb(audio_path, out_path, snr)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            outputs.append({
                "snr": snr,
                "noise_type": noise_type,
                "output_path": str(out_path),
            })

    return NoiseResult(
        video_id=video_id,
        original_path=str(audio_path),
        outputs=outputs,
    )


def _get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _add_white_noise(
    input_path: Path,
    output_path: Path,
    snr_db: int,
    duration: float,
) -> None:
    """Add pink noise at specified SNR using ffmpeg.

    SNR = 10 * log10(signal_power / noise_power)
    For a normalized signal (loudnorm'd), we scale noise relative to target SNR.
    """
    # noise volume: lower SNR = louder noise
    # For SNR in dB: noise_rms = signal_rms / 10^(snr/20)
    # We use anoisesrc with amplitude derived from SNR
    # amplitude ~ 10^(-snr/20) relative to full scale
    amplitude = 10 ** (-snr_db / 20.0)
    amplitude = min(amplitude, 1.0)

    filter_str = (
        f"anoisesrc=d={duration}:c=pink:r=16000:a={amplitude:.4f}[noise];"
        f"[in][noise]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        capture_output=True, check=True,
    )


def _add_reverb(
    input_path: Path,
    output_path: Path,
    snr_db: int,
) -> None:
    """Add echo/reverb effect simulating a room environment.

    Uses aecho filter with multiple delays to simulate conference room.
    SNR controls how much dry vs wet signal (lower SNR = more reverb).
    """
    # Wet/dry mix: higher SNR = less reverb
    # in_gain (dry) and out_gain (wet) control the mix
    wet_ratio = 10 ** (-snr_db / 20.0)
    dry_gain = 1.0
    wet_gain = wet_ratio

    # Delays in ms and decays for room simulation
    delays = "40|60|80"
    decays = "0.3|0.25|0.2"

    filter_str = f"aecho={dry_gain}:{wet_gain}:{delays}:{decays}"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_str,
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        capture_output=True, check=True,
    )


def batch_add_noise(
    data_dir: str | Path,
    output_dir: str | Path,
    snr_levels: list[int] | None = None,
    noise_types: list[str] | None = None,
) -> list[NoiseResult]:
    """Process all 10 video audio files.

    Args:
        data_dir: Root data directory containing audio/.
        output_dir: Directory for corrupted audio output.
        snr_levels: SNR levels to generate.
        noise_types: Noise types to apply.

    Returns:
        List of NoiseResult for each video.
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    results = []

    for i in range(1, 11):
        video_id = f"video_{i:02d}"
        audio_path = audio_dir / f"{video_id}.wav"
        if not audio_path.exists():
            print(f"  [skip] {audio_path} not found")
            continue

        print(f"  [noise] {video_id} ...")
        result = add_noise(
            audio_path=audio_path,
            video_id=video_id,
            output_dir=output_dir,
            snr_levels=snr_levels,
            noise_types=noise_types,
        )
        results.append(result)

    return results
