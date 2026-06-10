import json

import pytest

from src.rag_retrieve import run, _load_glossary, _match_entry
from src.schema import ASRSegment, OCRHit, RAGHit, VisualEvent, WordTimestamp


@pytest.fixture
def glossary_file(tmp_path):
    glossary = [
        {
            "term": "Hybrid Search",
            "aliases": ["hybrid retrieval"],
            "common_mishearings": ["high bread search", "hybrid source"],
            "source": "glossary_v1",
        },
        {
            "term": "Vector Database",
            "aliases": ["vector db", "vector store"],
            "common_mishearings": ["vector data base"],
            "source": "glossary_v1",
        },
    ]
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps(glossary, ensure_ascii=False))
    return str(path)


@pytest.fixture
def low_conf_segments():
    return [
        ASRSegment(
            segment_id="seg_0001",
            start=0.0,
            end=5.0,
            text="high bread search is important",
            words=[
                WordTimestamp(word="high", start=0.0, end=0.3, confidence=0.4),
                WordTimestamp(word="bread", start=0.4, end=0.7, confidence=0.3),
                WordTimestamp(word="search", start=0.8, end=1.1, confidence=0.5),
                WordTimestamp(word="is", start=1.2, end=1.3, confidence=0.99),
                WordTimestamp(word="important", start=1.4, end=1.8, confidence=0.95),
            ],
            quality_flags=["low_confidence"],
        ),
    ]


class TestRAGRetrieve:
    def test_finds_mishearing(self, glossary_file, low_conf_segments):
        result = run(low_conf_segments, [], glossary_file)
        terms = [h.term for h in result]
        assert "Hybrid Search" in terms

    def test_finds_term_from_visual(self, glossary_file):
        events = [
            VisualEvent(
                time_range=(0.0, 5.0),
                term_candidates=["Vector Database"],
            ),
        ]
        result = run([], events, glossary_file)
        terms = [h.term for h in result]
        assert "Vector Database" in terms

    def test_finds_term_from_ocr(self, glossary_file):
        events = [
            VisualEvent(
                time_range=(0.0, 5.0),
                visible_text=[OCRHit(text="vector db", confidence=0.9)],
            ),
        ]
        result = run([], events, glossary_file)
        terms = [h.term for h in result]
        assert "Vector Database" in terms

    def test_no_matches(self, glossary_file):
        segments = [
            ASRSegment(
                segment_id="seg_0001",
                start=0.0,
                end=5.0,
                text="completely unrelated content",
                words=[
                    WordTimestamp(word="completely", start=0.0, end=0.5, confidence=0.3),
                ],
                quality_flags=["low_confidence"],
            ),
        ]
        result = run(segments, [], glossary_file)
        assert result == []

    def test_empty_glossary(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        result = run([], [], str(path))
        assert result == []

    def test_missing_glossary(self, tmp_path):
        result = run([], [], str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_deduplication(self, glossary_file):
        segments = [
            ASRSegment(
                segment_id="s1",
                start=0.0,
                end=5.0,
                text="hybrid search hybrid search",
                words=[
                    WordTimestamp(word="hybrid", start=0.0, end=0.3, confidence=0.4),
                    WordTimestamp(word="search", start=0.4, end=0.7, confidence=0.4),
                    WordTimestamp(word="hybrid", start=0.8, end=1.1, confidence=0.4),
                    WordTimestamp(word="search", start=1.2, end=1.5, confidence=0.4),
                ],
                quality_flags=["low_confidence"],
            ),
        ]
        events = [
            VisualEvent(
                time_range=(0.0, 5.0),
                term_candidates=["Hybrid Search"],
            ),
        ]
        result = run(segments, events, glossary_file)
        hybrid_hits = [h for h in result if h.term == "Hybrid Search"]
        assert len(hybrid_hits) == 1


class TestLoadGlossary:
    def test_valid_file(self, glossary_file):
        data = _load_glossary(glossary_file)
        assert len(data) == 2

    def test_missing_file(self):
        data = _load_glossary("/nonexistent.json")
        assert data == []

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        data = _load_glossary(str(path))
        assert data == []


class TestMatchEntry:
    def test_exact_match(self):
        entry = {"term": "Hybrid Search", "aliases": [], "common_mishearings": []}
        assert _match_entry("hybrid search", entry) == 100.0

    def test_alias_match(self):
        entry = {"term": "Hybrid Search", "aliases": ["hybrid retrieval"], "common_mishearings": []}
        assert _match_entry("hybrid retrieval", entry) == 95.0

    def test_mishearing_match(self):
        entry = {"term": "Hybrid Search", "aliases": [], "common_mishearings": ["high bread search"]}
        assert _match_entry("high bread search", entry) == 90.0

    def test_fuzzy_match(self):
        entry = {"term": "Hybrid Search", "aliases": [], "common_mishearings": []}
        score = _match_entry("hybrd search", entry)
        assert score >= 75.0

    def test_no_match(self):
        entry = {"term": "Hybrid Search", "aliases": [], "common_mishearings": []}
        assert _match_entry("banana", entry) == 0.0
