"""FGSM adversarial attack on Whisper ASR.

Implements Fast Gradient Sign Method on audio waveform to generate
adversarial examples that cause ASR transcription errors.

Uses OpenAI Whisper tiny model on Apple MPS (or CPU fallback).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    video_id: str
    original_path: str
    output_path: str
    epsilon: float
    original_transcription: str
    adversarial_transcription: str
    word_error_rate: float
    max_perturbation: float
    snr_db: float


def fgsm_attack(
    audio_path: str | Path,
    video_id: str,
    output_dir: str | Path,
    epsilon: float = 0.01,
    model_name: str = "tiny",
    device: str | None = None,
    ground_truth_text: str = "",
) -> AdversarialResult:
    """Apply FGSM adversarial attack to audio.

    Args:
        audio_path: Path to input WAV file.
        video_id: Identifier like "video_01".
        output_dir: Directory to write adversarial audio.
        epsilon: Perturbation magnitude (default: 0.01).
        model_name: Whisper model to use (default: "tiny").
        device: Device for inference ("mps", "cpu", "cuda"). Auto-detect if None.
        ground_truth_text: Optional GT text for computing WER.

    Returns:
        AdversarialResult with attack statistics.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = _detect_device()
    logger.info(f"Using device: {device}")

    # Load model
    model = whisper.load_model(model_name, device=device)
    model.eval()

    # Load audio
    audio = whisper.load_audio(str(audio_path))
    audio_tensor = torch.from_numpy(audio).float().to(device)

    # Get original transcription
    original_text = _transcribe(model, audio_tensor)

    # FGSM attack (untargeted: maximize prediction entropy)
    audio_adv, max_pert, snr = _fgsm_step(
        model=model,
        audio=audio_tensor,
        epsilon=epsilon,
        device=device,
    )

    # Get adversarial transcription
    adv_text = _transcribe(model, audio_adv)

    # Compute WER
    ref_text = ground_truth_text if ground_truth_text else original_text
    wer = _compute_wer(ref_text, adv_text)

    # Save adversarial audio
    out_name = f"{video_id}_adv_eps{epsilon:.4f}.wav"
    out_path = output_dir / out_name
    _save_wav(audio_adv.cpu(), out_path, sample_rate=16000)

    return AdversarialResult(
        video_id=video_id,
        original_path=str(audio_path),
        output_path=str(out_path),
        epsilon=epsilon,
        original_transcription=original_text,
        adversarial_transcription=adv_text,
        word_error_rate=round(wer, 4),
        max_perturbation=round(float(max_pert), 6),
        snr_db=round(float(snr), 2),
    )


def _detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _transcribe(model, audio_tensor: torch.Tensor) -> str:
    """Transcribe audio tensor using Whisper model.transcribe()."""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()

    # Use Whisper's built-in transcribe method (handles padding, mel, decode)
    result = model.transcribe(
        audio_tensor.cpu().numpy(),
        language="en",
        fp16=False,
    )
    return result.get("text", "").strip()


def _fgsm_step(
    model,
    audio: torch.Tensor,
    epsilon: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one FGSM step on the audio.

    Strategy: compute mel spectrogram with gradient, run through encoder,
    then maximize the encoder output norm (untargeted perturbation).

    Args:
        model: Whisper model.
        audio: Input audio tensor [T].
        epsilon: Perturbation magnitude.
        device: Computation device.

    Returns:
        Tuple of (adversarial_audio, max_perturbation, snr_db).
    """
    # Pad/trim to 30 seconds (Whisper's expected input length)
    audio_padded = whisper.pad_or_trim(audio)

    # We need gradients through the mel spectrogram
    # Whisper's log_mel_spectrogram uses STFT which is differentiable
    audio_grad = audio_padded.clone().detach().requires_grad_(True)

    # Compute mel spectrogram (differentiable)
    mel = whisper.log_mel_spectrogram(audio_grad)

    # Run through encoder
    # The encoder output is what the decoder uses; perturbing it
    # will change the transcription
    encoder_out = model.encoder(mel.unsqueeze(0).to(device))

    # Untargeted attack: maximize the L2 norm of encoder output
    # This destabilizes the decoder's ability to produce correct tokens
    loss = encoder_out.pow(2).mean()

    # Backward pass to get gradients on audio
    loss.backward()

    # FGSM perturbation
    grad = audio_grad.grad
    if grad is None:
        # Fallback: use zero perturbation
        perturbation = torch.zeros_like(audio_padded)
    else:
        perturbation = epsilon * grad.sign()

    # Apply perturbation
    audio_adv = audio_padded + perturbation

    # Clip to valid audio range [-1, 1]
    audio_adv = torch.clamp(audio_adv, -1.0, 1.0)

    # Compute perturbation stats
    actual_pert = audio_adv - audio_padded
    max_pert = actual_pert.abs().max()

    # Compute SNR
    signal_power = audio_padded.pow(2).mean()
    noise_power = actual_pert.pow(2).mean()
    if noise_power > 0:
        snr = 10 * torch.log10(signal_power / noise_power + 1e-10)
    else:
        snr = torch.tensor(float("inf"))

    # Trim back to original length
    audio_adv = audio_adv[:len(audio)]

    return audio_adv.detach(), max_pert, snr


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    n = len(ref_words)
    if n == 0:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j-1] + 1,  # substitution
                )

    return d[n][len(hyp_words)] / n


def _save_wav(audio: torch.Tensor, path: Path, sample_rate: int = 16000) -> None:
    """Save audio tensor as WAV file."""
    import scipy.io.wavfile as wavfile

    audio_np = audio.cpu().numpy()
    # Convert to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, audio_int16)


def batch_fgsm_attack(
    data_dir: str | Path,
    output_dir: str | Path,
    epsilon: float = 0.01,
    model_name: str = "tiny",
    device: str | None = None,
) -> list[AdversarialResult]:
    """Run FGSM attack on all 10 video audio files.

    Args:
        data_dir: Root data directory.
        output_dir: Directory for adversarial audio output.
        epsilon: Perturbation magnitude.
        model_name: Whisper model name.
        device: Computation device.

    Returns:
        List of AdversarialResult for each video.
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    gt_dir = data_dir / "transcripts_ground_truth"
    results = []

    # Load model once for all files
    if device is None:
        device = _detect_device()

    for i in range(1, 11):
        video_id = f"video_{i:02d}"
        audio_path = audio_dir / f"{video_id}.wav"
        gt_path = gt_dir / f"{video_id}_ground_truth.txt"

        if not audio_path.exists():
            print(f"  [skip] {audio_path} not found")
            continue

        # Load ground truth if available
        gt_text = ""
        if gt_path.exists():
            raw = gt_path.read_text(encoding="utf-8").strip()
            # Strip timestamps
            lines = raw.split("\n")
            gt_text = " ".join(
                l.strip() for l in lines
                if not l.strip().startswith(("00:", "01:", "02:"))
            ).strip()

        print(f"  [adversarial] {video_id} (eps={epsilon}) ...")
        try:
            result = fgsm_attack(
                audio_path=audio_path,
                video_id=video_id,
                output_dir=output_dir,
                epsilon=epsilon,
                model_name=model_name,
                device=device,
                ground_truth_text=gt_text,
            )
            results.append(result)
        except Exception as e:
            print(f"  [error] {video_id}: {e}")

    return results


def generate_adversarial_report(
    results: list[AdversarialResult],
    output_path: str | Path,
) -> None:
    """Generate a JSON report of adversarial attack results."""
    report = [asdict(r) for r in results]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
