"""Automated homophone substitution using CMU Pronouncing Dictionary.

Finds phonetically similar words and replaces original words at controlled ratios
to simulate ASR transcription errors.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pronouncing
from rapidfuzz import fuzz


@dataclass
class Substitution:
    original: str
    replacement: str
    position: int  # word index in sentence
    phonetic_similarity: float
    edit_similarity: float


@dataclass
class HomophoneResult:
    video_id: str
    original_text: str
    corrupted_text: str
    ratio: float
    substitutions: list[Substitution]
    output_path: str


# Common English homophones not in CMU (fallback)
BUILTIN_HOMOPHONES: dict[str, list[str]] = {
    "their": ["there", "they're"],
    "there": ["their", "they're"],
    "your": ["you're"],
    "to": ["too", "two"],
    "too": ["to", "two"],
    "two": ["to", "too"],
    "write": ["right", "rite"],
    "right": ["write", "rite"],
    "know": ["no"],
    "no": ["know"],
    "knows": ["nose"],
    "nose": ["knows"],
    "see": ["sea"],
    "sea": ["see"],
    "hear": ["here"],
    "here": ["hear"],
    "wear": ["where"],
    "where": ["wear"],
    "won": ["one"],
    "one": ["won"],
    "by": ["buy", "bye"],
    "buy": ["by", "bye"],
    "bye": ["by", "buy"],
    "break": ["brake"],
    "brake": ["break"],
    "flower": ["flour"],
    "flour": ["flower"],
    "pair": ["pear", "pare"],
    "pear": ["pair", "pare"],
    "plain": ["plane"],
    "plane": ["plain"],
    "weather": ["whether"],
    "whether": ["weather"],
    "week": ["weak"],
    "weak": ["week"],
    "whole": ["hole"],
    "hole": ["whole"],
    "would": ["wood"],
    "wood": ["would"],
    "peace": ["piece"],
    "piece": ["peace"],
    "manor": ["manner"],
    "manner": ["manor"],
    "morning": ["mourning"],
    "mourning": ["morning"],
    "principal": ["principle"],
    "principle": ["principal"],
    "stationary": ["stationery"],
    "stationery": ["stationary"],
    "than": ["then"],
    "then": ["than"],
    "affect": ["effect"],
    "effect": ["affect"],
    "accept": ["except"],
    "except": ["accept"],
    "allowed": ["aloud"],
    "aloud": ["allowed"],
    "assent": ["ascent"],
    "ascent": ["assent"],
    "bald": ["bawled"],
    "bawled": ["bald"],
    "band": ["banned"],
    "banned": ["band"],
    "bare": ["bear"],
    "bear": ["bare"],
    "berry": ["bury"],
    "bury": ["berry"],
    "board": ["bored"],
    "bored": ["board"],
    "bridal": ["bridle"],
    "bridle": ["bridal"],
    "cereal": ["serial"],
    "serial": ["cereal"],
    "complement": ["compliment"],
    "compliment": ["complement"],
    "council": ["counsel"],
    "counsel": ["council"],
    "currant": ["current"],
    "current": ["currant"],
    "descent": ["dissent"],
    "dissent": ["descent"],
    "discreet": ["discrete"],
    "discrete": ["discreet"],
    "elicit": ["illicit"],
    "illicit": ["elicit"],
    "fair": ["fare"],
    "fare": ["fair"],
    "fir": ["fur"],
    "fur": ["fir"],
    "guessed": ["guest"],
    "guest": ["guessed"],
    "idle": ["idol"],
    "idol": ["idle"],
    "incite": ["insight"],
    "insight": ["incite"],
    "knead": ["need"],
    "need": ["knead"],
    "knight": ["night"],
    "night": ["knight"],
    "not": ["knot"],
    "knot": ["not"],
    "led": ["lead"],
    "maize": ["maze"],
    "maze": ["maize"],
    "medal": ["metal", "meddle"],
    "metal": ["medal", "meddle"],
    "patience": ["patients"],
    "patients": ["patience"],
    "pedal": ["peddle"],
    "peddle": ["pedal"],
    "precede": ["proceed"],
    "proceed": ["precede"],
    "presence": ["presents"],
    "presents": ["presence"],
    "profit": ["prophet"],
    "prophet": ["profit"],
    "sauce": ["source"],
    "source": ["sauce"],
    "sight": ["site", "cite"],
    "site": ["sight", "cite"],
    "cite": ["sight", "site"],
    "sole": ["soul"],
    "soul": ["sole"],
    "some": ["sum"],
    "sum": ["some"],
    "tail": ["tale"],
    "tale": ["tail"],
    "vain": ["vein", "vane"],
    "vein": ["vain", "vane"],
    "vane": ["vain", "vein"],
    "wade": ["weighed"],
    "weighed": ["wade"],
    "wait": ["weight"],
    "weight": ["wait"],
    "wood": ["would"],
    "would": ["wood"],
}


def corrupt_text(
    text_path: str | Path,
    video_id: str,
    output_dir: str | Path,
    ratios: list[float] | None = None,
    seed: int = 42,
) -> list[HomophoneResult]:
    """Replace words with homophones at multiple corruption ratios.

    Args:
        text_path: Path to ground truth text file.
        video_id: Identifier like "video_01".
        output_dir: Directory to write corrupted text files.
        ratios: List of corruption ratios (default: [0.1, 0.25, 0.5]).
        seed: Random seed for reproducibility.

    Returns:
        List of HomophoneResult for each ratio.
    """
    if ratios is None:
        ratios = [0.1, 0.25, 0.5]

    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_text = text_path.read_text(encoding="utf-8").strip()
    # Strip timestamp prefix if present (e.g., "00:00\nActual text...")
    clean_text = _strip_timestamps(raw_text)
    words = clean_text.split()

    results = []
    for ratio in ratios:
        rng = random.Random(seed)
        corrupted_words, substitutions = _apply_substitutions(words, ratio, rng)

        corrupted_text = " ".join(corrupted_words)
        # Preserve original timestamp prefix if present
        timestamp_prefix = _extract_timestamp_prefix(raw_text)
        if timestamp_prefix:
            corrupted_text = f"{timestamp_prefix}\n{corrupted_text}"

        out_name = f"{video_id}_homophone_r{int(ratio*100):02d}.txt"
        out_path = output_dir / out_name
        out_path.write_text(corrupted_text, encoding="utf-8")

        results.append(HomophoneResult(
            video_id=video_id,
            original_text=clean_text,
            corrupted_text=corrupted_text,
            ratio=ratio,
            substitutions=substitutions,
            output_path=str(out_path),
        ))

    return results


def _strip_timestamps(text: str) -> str:
    """Remove timestamp lines like '00:00' or '00:42' from transcript."""
    lines = text.strip().split("\n")
    content_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just timestamps
        if re.match(r"^\d{2}:\d{2}(:\d{2})?$", stripped):
            continue
        content_lines.append(stripped)
    return " ".join(content_lines).strip()


def _extract_timestamp_prefix(text: str) -> str:
    """Extract timestamp prefix if present."""
    lines = text.strip().split("\n")
    timestamps = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d{2}:\d{2}(:\d{2})?$", stripped):
            timestamps.append(stripped)
        else:
            break
    return "\n".join(timestamps) if timestamps else ""


def _apply_substitutions(
    words: list[str],
    ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[Substitution]]:
    """Apply homophone substitutions to a list of words.

    Args:
        words: List of words to potentially corrupt.
        ratio: Fraction of replaceable words to substitute.
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of (corrupted_words, list_of_substitutions).
    """
    # Find replaceable words (have homophone candidates)
    replaceable_indices = []
    for i, word in enumerate(words):
        candidates = _get_candidates(word)
        if candidates:
            replaceable_indices.append(i)

    if not replaceable_indices:
        return list(words), []

    # Select subset based on ratio
    n_replace = max(1, int(len(replaceable_indices) * ratio))
    n_replace = min(n_replace, len(replaceable_indices))
    selected = rng.sample(replaceable_indices, n_replace)

    corrupted = list(words)
    substitutions = []

    for idx in selected:
        original = corrupted[idx]
        candidates = _get_candidates(original)
        if not candidates:
            continue

        replacement = rng.choice(candidates)

        # Compute similarities
        phones_orig = pronouncing.phones_for_word(original.lower())
        phones_repl = pronouncing.phones_for_word(replacement.lower())
        phonetic_sim = 0.0
        if phones_orig and phones_repl:
            phonetic_sim = _phone_similarity(phones_orig[0], phones_repl[0])

        edit_sim = fuzz.ratio(original.lower(), replacement.lower()) / 100.0

        # Preserve case pattern
        replacement = _match_case(original, replacement)

        corrupted[idx] = replacement
        substitutions.append(Substitution(
            original=original,
            replacement=replacement,
            position=idx,
            phonetic_similarity=round(phonetic_sim, 3),
            edit_similarity=round(edit_sim, 3),
        ))

    return corrupted, substitutions


# Pre-built reverse index: phones -> words (lazy loaded)
_PHONE_INDEX: dict[str, list[str]] | None = None


def _build_phone_index() -> dict[str, list[str]]:
    """Build reverse index from pronunciation to words (cached)."""
    global _PHONE_INDEX
    if _PHONE_INDEX is not None:
        return _PHONE_INDEX

    index: dict[str, list[str]] = {}
    for word in pronouncing.cmudict.entries():
        word_str = word[0].lower()
        phones_str = " ".join(word[1])
        if phones_str not in index:
            index[phones_str] = []
        index[phones_str].append(word_str)

    _PHONE_INDEX = index
    return index


def _get_candidates(word: str) -> list[str]:
    """Get homophone candidates for a word from CMU dict + built-in table.

    Uses pre-built reverse index for fast lookup instead of scanning all entries.

    Args:
        word: Input word.

    Returns:
        List of candidate replacement words (lowercase).
    """
    word_lower = word.lower().strip(".,!?;:\"'")

    candidates = []

    # 1. Check built-in homophones
    if word_lower in BUILTIN_HOMOPHONES:
        candidates.extend(BUILTIN_HOMOPHONES[word_lower])

    # 2. CMU pronouncing dictionary via reverse index
    phones = pronouncing.phones_for_word(word_lower)
    if phones:
        phones_str = phones[0]  # Take first pronunciation
        index = _build_phone_index()
        if phones_str in index:
            for w in index[phones_str]:
                w_clean = w.lower().strip(".,!?;:\"'")
                if w_clean != word_lower and w_clean not in candidates:
                    candidates.append(w_clean)

    # Filter out very short words (1-2 chars) to avoid noise
    candidates = [c for c in candidates if len(c) >= 2]

    return candidates


def _phone_similarity(phones_a: str, phones_b: str) -> float:
    """Compute phonetic similarity between two phone sequences.

    Uses simple Jaccard similarity on phone tokens.
    """
    set_a = set(phones_a.split())
    set_b = set(phones_b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _match_case(original: str, replacement: str) -> str:
    """Match the case pattern of the original word."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def batch_corrupt_text(
    data_dir: str | Path,
    output_dir: str | Path,
    ratios: list[float] | None = None,
    seed: int = 42,
) -> list[HomophoneResult]:
    """Process all 10 ground truth transcript files.

    Args:
        data_dir: Root data directory containing transcripts_ground_truth/.
        output_dir: Directory for corrupted text output.
        ratios: Corruption ratios to apply.
        seed: Random seed.

    Returns:
        List of HomophoneResult for each video and ratio.
    """
    data_dir = Path(data_dir)
    gt_dir = data_dir / "transcripts_ground_truth"
    results = []

    for i in range(1, 11):
        video_id = f"video_{i:02d}"
        gt_path = gt_dir / f"{video_id}_ground_truth.txt"
        if not gt_path.exists():
            print(f"  [skip] {gt_path} not found")
            continue

        print(f"  [homophone] {video_id} ...")
        video_results = corrupt_text(
            text_path=gt_path,
            video_id=video_id,
            output_dir=output_dir,
            ratios=ratios,
            seed=seed,
        )
        results.extend(video_results)

    return results


def generate_corruption_report(results: list[HomophoneResult], output_path: str | Path) -> None:
    """Generate a JSON report of all substitutions made."""
    report = []
    for r in results:
        report.append({
            "video_id": r.video_id,
            "ratio": r.ratio,
            "n_substitutions": len(r.substitutions),
            "substitutions": [asdict(s) for s in r.substitutions],
            "output_path": r.output_path,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
