"""Unified runner for all data corruption methods.

Processes all 10 video samples through noise, homophone, and adversarial
corruption pipelines, and generates summary reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .noise import batch_add_noise, NoiseResult
from .homophone import batch_corrupt_text, generate_corruption_report, HomophoneResult
from .adversarial import batch_fgsm_attack, generate_adversarial_report, AdversarialResult

logger = logging.getLogger(__name__)


def run_all(
    data_dir: str | Path = "data",
    output_dir: str | Path = "data/corrupted",
    noise_snr_levels: list[int] | None = None,
    noise_types: list[str] | None = None,
    homophone_ratios: list[float] | None = None,
    adv_epsilon: float = 0.01,
    adv_model: str = "tiny",
    device: str | None = None,
    skip_noise: bool = False,
    skip_homophone: bool = False,
    skip_adversarial: bool = False,
) -> dict:
    """Run all corruption methods on the dataset.

    Args:
        data_dir: Root data directory (default: "data").
        output_dir: Output directory for corrupted data (default: "data/corrupted").
        noise_snr_levels: SNR levels for noise corruption.
        noise_types: Noise types to apply.
        homophone_ratios: Corruption ratios for homophone substitution.
        adv_epsilon: Epsilon for FGSM adversarial attack.
        adv_model: Whisper model for adversarial attack.
        device: Computation device (auto-detect if None).
        skip_noise: Skip noise corruption.
        skip_homophone: Skip homophone corruption.
        skip_adversarial: Skip adversarial attack.

    Returns:
        Dictionary with results from all methods.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    results = {
        "noise": [],
        "homophone": [],
        "adversarial": [],
    }

    # 1. Audio noise corruption
    if not skip_noise:
        print("\n=== Audio Noise Corruption ===")
        noise_results = batch_add_noise(
            data_dir=data_dir,
            output_dir=output_dir / "noise" / "audio",
            snr_levels=noise_snr_levels,
            noise_types=noise_types,
        )
        results["noise"] = noise_results
        print(f"  Generated {sum(len(r.outputs) for r in noise_results)} noisy audio files")
    else:
        print("\n=== Skipping Audio Noise ===")

    # 2. Homophone substitution
    if not skip_homophone:
        print("\n=== Homophone Substitution ===")
        homophone_results = batch_corrupt_text(
            data_dir=data_dir,
            output_dir=output_dir / "homophone" / "text",
            ratios=homophone_ratios,
        )
        results["homophone"] = homophone_results
        print(f"  Generated {len(homophone_results)} corrupted text files")

        # Generate report
        generate_corruption_report(
            homophone_results,
            output_dir / "metadata" / "homophone_report.json",
        )
    else:
        print("\n=== Skipping Homophone ===")

    # 3. Adversarial attack
    if not skip_adversarial:
        print("\n=== Adversarial Attack (FGSM) ===")
        adv_results = batch_fgsm_attack(
            data_dir=data_dir,
            output_dir=output_dir / "adversarial" / "audio",
            epsilon=adv_epsilon,
            model_name=adv_model,
            device=device,
        )
        results["adversarial"] = adv_results
        print(f"  Generated {len(adv_results)} adversarial audio files")

        # Generate report
        generate_adversarial_report(
            adv_results,
            output_dir / "metadata" / "adversarial_report.json",
        )
    else:
        print("\n=== Skipping Adversarial ===")

    # Generate summary
    summary = _build_summary(results, output_dir)
    summary_path = output_dir / "metadata" / "corruption_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n=== Summary saved to {summary_path} ===")

    return results


def _build_summary(results: dict, output_dir: Path) -> dict:
    """Build summary statistics from all corruption results."""
    summary = {
        "output_dir": str(output_dir),
        "methods": {},
    }

    # Noise summary
    noise_results: list[NoiseResult] = results.get("noise", [])
    if noise_results:
        noise_files = []
        for r in noise_results:
            for o in r.outputs:
                noise_files.append(o["output_path"])
        summary["methods"]["noise"] = {
            "n_videos": len(noise_results),
            "n_files": len(noise_files),
            "snr_levels": list(set(o["snr"] for r in noise_results for o in r.outputs)),
            "noise_types": list(set(o["noise_type"] for r in noise_results for o in r.outputs)),
        }

    # Homophone summary
    homo_results: list[HomophoneResult] = results.get("homophone", [])
    if homo_results:
        total_subs = sum(len(r.substitutions) for r in homo_results)
        summary["methods"]["homophone"] = {
            "n_files": len(homo_results),
            "total_substitutions": total_subs,
            "ratios": list(set(r.ratio for r in homo_results)),
            "avg_substitutions_per_file": round(total_subs / len(homo_results), 1) if homo_results else 0,
        }

    # Adversarial summary
    adv_results: list[AdversarialResult] = results.get("adversarial", [])
    if adv_results:
        avg_wer = sum(r.word_error_rate for r in adv_results) / len(adv_results)
        avg_snr = sum(r.snr_db for r in adv_results) / len(adv_results)
        summary["methods"]["adversarial"] = {
            "n_files": len(adv_results),
            "epsilon": adv_results[0].epsilon if adv_results else None,
            "avg_word_error_rate": round(avg_wer, 4),
            "avg_snr_db": round(avg_snr, 2),
            "model": "tiny",
        }

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all data corruption methods")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--output-dir", default="data/corrupted", help="Output directory")
    parser.add_argument("--noise-snrs", nargs="+", type=int, default=[20, 10, 3],
                        help="SNR levels for noise (default: 20 10 3)")
    parser.add_argument("--noise-types", nargs="+", default=["white", "reverb"],
                        help="Noise types (default: white reverb)")
    parser.add_argument("--homophone-ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5],
                        help="Homophone corruption ratios (default: 0.1 0.25 0.5)")
    parser.add_argument("--adv-epsilon", type=float, default=0.01,
                        help="FGSM epsilon (default: 0.01)")
    parser.add_argument("--adv-model", default="tiny", help="Whisper model for FGSM")
    parser.add_argument("--device", default=None, help="Device (mps/cpu/cuda)")
    parser.add_argument("--skip-noise", action="store_true", help="Skip noise corruption")
    parser.add_argument("--skip-homophone", action="store_true", help="Skip homophone")
    parser.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        noise_snr_levels=args.noise_snrs,
        noise_types=args.noise_types,
        homophone_ratios=args.homophone_ratios,
        adv_epsilon=args.adv_epsilon,
        adv_model=args.adv_model,
        device=args.device,
        skip_noise=args.skip_noise,
        skip_homophone=args.skip_homophone,
        skip_adversarial=args.skip_adversarial,
    )
