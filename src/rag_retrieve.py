from __future__ import annotations

import json
import logging
from pathlib import Path

from rapidfuzz import fuzz

from src.schema import ASRSegment, RAGHit, VisualEvent

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 75


def run(
    segments: list[ASRSegment],
    visual_events: list[VisualEvent],
    glossary_path: str,
) -> list[RAGHit]:
    glossary = _load_glossary(glossary_path)
    if not glossary:
        return []

    queries: set[str] = set()

    for seg in segments:
        for word in seg.words:
            if word.confidence < 0.6:
                queries.add(word.word.lower())
        if "low_confidence" in seg.quality_flags:
            queries.add(seg.text.lower())
            for token in seg.text.split():
                queries.add(token.lower())
            queries.update(_ngrams(seg.text.lower(), 2))
            queries.update(_ngrams(seg.text.lower(), 3))

    for event in visual_events:
        for candidate in event.term_candidates:
            queries.add(candidate.lower())
        for ocr in event.visible_text:
            queries.add(ocr.text.lower())

    hits: dict[str, RAGHit] = {}

    for query in queries:
        for entry in glossary:
            score = _match_entry(query, entry)
            if score > 0 and entry["term"] not in hits:
                hits[entry["term"]] = RAGHit(
                    term=entry["term"],
                    aliases=entry.get("aliases", []),
                    common_mishearings=entry.get("common_mishearings", []),
                    source=entry.get("source", ""),
                    score=score / 100.0,
                )

    return list(hits.values())


def _load_glossary(path: str) -> list[dict]:
    glossary_path = Path(path)
    if not glossary_path.exists():
        logger.warning("Glossary not found: %s", path)
        return []
    try:
        return json.loads(glossary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load glossary: %s", path)
        return []


def _ngrams(text: str, n: int) -> list[str]:
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _match_entry(query: str, entry: dict) -> float:
    term = entry.get("term", "")
    if query == term.lower():
        return 100.0

    for alias in entry.get("aliases", []):
        if query == alias.lower():
            return 95.0

    for mishearing in entry.get("common_mishearings", []):
        if query == mishearing.lower():
            return 90.0

    best = fuzz.ratio(query, term.lower())

    for alias in entry.get("aliases", []):
        best = max(best, fuzz.ratio(query, alias.lower()))

    if best >= FUZZY_THRESHOLD:
        return best

    return 0.0
