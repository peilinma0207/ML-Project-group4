"""Corruption data statistics.

Reports synthesis stats for each method:
- Homophone: how many words/characters changed
- Noise: SNR levels and noise types applied
- Adversarial: perturbation epsilon and SNR
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class HomophoneStats:
    video_id: str
    ratio: float
    n_words_original: int
    n_words_corrupted: int
    n_words_changed: int
    word_change_pct: float
    n_chars_original: int
    n_chars_corrupted: int
    n_chars_changed: int
    char_change_pct: float


@dataclass
class NoiseStats:
    video_id: str
    noise_type: str
    snr_db: int
    output_path: str


@dataclass
class AdversarialStats:
    video_id: str
    epsilon: float
    max_perturbation: float
    snr_db: float
    output_path: str


def _strip_timestamps(text: str) -> str:
    lines = text.strip().split("\n")
    return " ".join(
        l.strip() for l in lines
        if not re.match(r"^\d{2}:\d{2}(:\d{2})?$", l.strip())
    ).strip()


def _word_diff(a: str, b: str) -> tuple[int, int, float]:
    words_a = a.lower().split()
    words_b = b.lower().split()
    n = max(len(words_a), len(words_b))
    if n == 0:
        return 0, 0, 0.0
    changed = sum(1 for i in range(min(len(words_a), len(words_b))) if words_a[i] != words_b[i])
    changed += abs(len(words_a) - len(words_b))
    return n, changed, changed / n


def _char_diff(a: str, b: str) -> tuple[int, int, float]:
    a_clean = a.lower().replace(" ", "")
    b_clean = b.lower().replace(" ", "")
    n = max(len(a_clean), len(b_clean))
    if n == 0:
        return 0, 0, 0.0
    # Levenshtein at char level
    la, lb = len(a_clean), len(b_clean)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a_clean[i-1] == b_clean[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return n, d[la][lb], d[la][lb] / n


def collect_stats(
    data_dir: str | Path = "data",
    corrupted_dir: str | Path = "data/corrupted",
) -> dict:
    data_dir = Path(data_dir)
    corrupted_dir = Path(corrupted_dir)

    # Load ground truth
    gt_dir = data_dir / "transcripts_ground_truth"
    gts = {}
    for i in range(1, 11):
        vid = f"video_{i:02d}"
        p = gt_dir / f"{vid}_ground_truth.txt"
        if p.exists():
            gts[vid] = _strip_timestamps(p.read_text(encoding="utf-8"))

    result = {"homophone": [], "noise": [], "adversarial": []}

    # --- Homophone ---
    homo_dir = corrupted_dir / "homophone" / "text"
    for f in sorted(homo_dir.glob("*.txt")):
        parts = f.stem.split("_")
        vid = f"{parts[0]}_{parts[1]}"
        ratio = int(parts[-1][1:]) / 100.0
        gt = gts.get(vid, "")
        if not gt:
            continue
        corrupted = _strip_timestamps(f.read_text(encoding="utf-8"))
        nw, ncw, wpct = _word_diff(gt, corrupted)
        nc, ncc, cpct = _char_diff(gt, corrupted)
        result["homophone"].append(asdict(HomophoneStats(
            video_id=vid, ratio=ratio,
            n_words_original=nw, n_words_corrupted=len(corrupted.split()),
            n_words_changed=ncw, word_change_pct=round(wpct * 100, 1),
            n_chars_original=nc, n_chars_corrupted=len(corrupted.replace(" ", "")),
            n_chars_changed=ncc, char_change_pct=round(cpct * 100, 1),
        )))

    # --- Noise ---
    noise_dir = corrupted_dir / "noise" / "audio"
    for f in sorted(noise_dir.glob("*.wav")):
        parts = f.stem.split("_")
        vid = f"{parts[0]}_{parts[1]}"
        noise_type = parts[2]
        snr = int(parts[3].replace("snr", ""))
        result["noise"].append(asdict(NoiseStats(
            video_id=vid, noise_type=noise_type, snr_db=snr, output_path=str(f),
        )))

    # --- Adversarial ---
    adv_report = corrupted_dir / "metadata" / "adversarial_report.json"
    if adv_report.exists():
        for entry in json.loads(adv_report.read_text(encoding="utf-8")):
            result["adversarial"].append(asdict(AdversarialStats(
                video_id=entry["video_id"],
                epsilon=entry["epsilon"],
                max_perturbation=entry["max_perturbation"],
                snr_db=entry["snr_db"],
                output_path=entry["output_path"],
            )))

    return result


def print_summary(stats: dict) -> None:
    print("=" * 70)
    print("Data Corruption Statistics")
    print("=" * 70)

    # Homophone
    homo = stats["homophone"]
    if homo:
        print("\n--- Homophone Substitution ---")
        print(f"{'Video':<10} {'Ratio':>6} {'Words Changed':>14} {'Chars Changed':>14}")
        print("-" * 50)
        for h in homo:
            print(f"{h['video_id']:<10} {h['ratio']:>5.0%} {h['n_words_changed']:>5}/{h['n_words_original']:<5} ({h['word_change_pct']:>5.1f}%) {h['n_chars_changed']:>5}/{h['n_chars_original']:<5} ({h['char_change_pct']:>5.1f}%)")
        # Averages by ratio
        from collections import defaultdict
        by_ratio = defaultdict(list)
        for h in homo:
            by_ratio[h['ratio']].append(h)
        print("-" * 50)
        for r in sorted(by_ratio):
            items = by_ratio[r]
            avg_w = sum(h['word_change_pct'] for h in items) / len(items)
            avg_c = sum(h['char_change_pct'] for h in items) / len(items)
            print(f"{'Average':<10} {r:>5.0%} {'':>14} {'':>5}      {'':>5}      (word {avg_w:.1f}%, char {avg_c:.1f}%)")

    # Noise
    noise = stats["noise"]
    if noise:
        print("\n--- Audio Noise ---")
        print(f"{'Noise Type':<12} {'SNR (dB)':>10} {'Files':>6}")
        print("-" * 30)
        from collections import Counter
        counts = Counter((n['noise_type'], n['snr_db']) for n in noise)
        for (nt, snr), cnt in sorted(counts.items()):
            print(f"{nt:<12} {snr:>10} {cnt:>6}")

    # Adversarial
    adv = stats["adversarial"]
    if adv:
        print("\n--- Adversarial Attack (FGSM) ---")
        print(f"{'Video':<10} {'Epsilon':>8} {'Max Pert':>10} {'SNR (dB)':>10}")
        print("-" * 42)
        for a in adv:
            print(f"{a['video_id']:<10} {a['epsilon']:>8.4f} {a['max_perturbation']:>10.6f} {a['snr_db']:>10.1f}")
        avg_snr = sum(a['snr_db'] for a in adv) / len(adv)
        avg_pert = sum(a['max_perturbation'] for a in adv) / len(adv)
        print("-" * 42)
        print(f"{'Average':<10} {'':>8} {avg_pert:>10.6f} {avg_snr:>10.1f}")

    print("\n" + "=" * 70)


def save_stats(stats: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--corrupted-dir", default="data/corrupted")
    parser.add_argument("--output", default="data/corrupted/metadata/corruption_stats.json")
    args = parser.parse_args()

    stats = collect_stats(args.data_dir, args.corrupted_dir)
    print_summary(stats)
    save_stats(stats, args.output)
    print(f"\nSaved to {args.output}")
