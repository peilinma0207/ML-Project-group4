from __future__ import annotations

import json
import logging
from pathlib import Path

from rapidfuzz import fuzz

from .schema import ASRSegment, RAGHit, VisualEvent

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 75
MIN_QUERY_LENGTH = 2


def run(
    segments: list[ASRSegment],
    visual_events: list[VisualEvent],
    glossary_path: str,
) -> list[RAGHit]:
    glossary = _load_glossary(glossary_path)
    if not glossary:
        return []

    queries = _collect_queries(segments, visual_events)
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


def _collect_queries(segments: list[ASRSegment], visual_events: list[VisualEvent]) -> set[str]:
    queries: set[str] = set()

    for seg in segments:
        for word in seg.words:
            if word.confidence < 0.6:
                _add_query(queries, word.word)
        if "low_confidence" in seg.quality_flags:
            _add_query(queries, seg.text)
            for token in seg.text.split():
                _add_query(queries, token)
            for size in (2, 3):
                for ngram in _ngrams(seg.text.lower(), size):
                    _add_query(queries, ngram)

    for event in visual_events:
        for candidate in event.term_candidates:
            _add_query(queries, candidate)
        for ocr in event.visible_text:
            _add_query(queries, ocr.text)

    return queries


def _add_query(queries: set[str], value: str) -> None:
    normalized = _normalize(value)
    if len(normalized) >= MIN_QUERY_LENGTH:
        queries.add(normalized)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


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
    term = _normalize(entry.get("term", ""))
    if query == term:
        return 100.0

    for alias in entry.get("aliases", []):
        if query == _normalize(alias):
            return 95.0

    for mishearing in entry.get("common_mishearings", []):
        if query == _normalize(mishearing):
            return 90.0

    best = fuzz.ratio(query, term)

    for alias in entry.get("aliases", []):
        best = max(best, fuzz.ratio(query, _normalize(alias)))

    if best >= FUZZY_THRESHOLD:
        return best

    return 0.0
