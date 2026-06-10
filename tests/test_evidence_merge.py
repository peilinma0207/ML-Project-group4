import pytest

from src.evidence_merge import run, _find_overlapping_events, _find_matching_rag
from src.schema import (
    ASRSegment,
    MergedEvidence,
    OCRHit,
    RAGHit,
    VisualEvent,
    WordTimestamp,
)


@pytest.fixture
def segments():
    return [
        ASRSegment(
            segment_id="seg_0000",
            start=0.0,
            end=5.0,
            text="hello world",
            words=[
                WordTimestamp(word="hello", start=0.0, end=0.5, confidence=0.95),
                WordTimestamp(word="world", start=0.6, end=1.0, confidence=0.88),
            ],
        ),
        ASRSegment(
            segment_id="seg_0001",
            start=5.5,
            end=10.0,
            text="hybrid search is important",
            words=[
                WordTimestamp(word="hybrid", start=5.5, end=6.0, confidence=0.45),
                WordTimestamp(word="search", start=6.1, end=6.5, confidence=0.50),
            ],
            quality_flags=["low_confidence"],
        ),
    ]


@pytest.fixture
def visual_events():
    return [
        VisualEvent(
            time_range=(0.0, 4.0),
            scene="office",
            term_candidates=["Vector Database"],
        ),
        VisualEvent(
            time_range=(6.0, 9.0),
            scene="presentation",
            visible_text=[OCRHit(text="Hybrid Search", confidence=0.9)],
            term_candidates=["Hybrid Search"],
        ),
    ]


@pytest.fixture
def rag_hits():
    return [
        RAGHit(
            term="Hybrid Search",
            aliases=["hybrid retrieval"],
            common_mishearings=["high bread search"],
            source="glossary_v1",
            score=0.9,
        ),
        RAGHit(
            term="Vector Database",
            aliases=["vector db"],
            common_mishearings=[],
            source="glossary_v1",
            score=0.85,
        ),
    ]


class TestEvidenceMerge:
    def test_all_segments_get_evidence(self, segments, visual_events, rag_hits):
        result = run(segments, visual_events, rag_hits)
        assert len(result) == 2
        assert all(isinstance(m, MergedEvidence) for m in result)

    def test_visual_matching(self, segments, visual_events, rag_hits):
        result = run(segments, visual_events, rag_hits)
        assert len(result[0].visual_events) == 1
        assert result[0].visual_events[0].scene == "office"
        assert len(result[1].visual_events) == 1
        assert result[1].visual_events[0].scene == "presentation"

    def test_rag_matching(self, segments, visual_events, rag_hits):
        result = run(segments, visual_events, rag_hits)
        seg1_rag_terms = [h.term for h in result[1].rag_hits]
        assert "Hybrid Search" in seg1_rag_terms

    def test_no_evidence(self):
        segments = [
            ASRSegment(segment_id="s1", start=100.0, end=105.0, text="nothing"),
        ]
        result = run(segments, [], [])
        assert len(result) == 1
        assert result[0].visual_events == []
        assert result[0].rag_hits == []

    def test_empty_segments(self):
        result = run([], [], [])
        assert result == []


class TestFindOverlappingEvents:
    def test_full_overlap(self):
        seg = ASRSegment(segment_id="s1", start=2.0, end=8.0, text="t")
        events = [VisualEvent(time_range=(3.0, 7.0), scene="a")]
        assert len(_find_overlapping_events(seg, events)) == 1

    def test_partial_overlap(self):
        seg = ASRSegment(segment_id="s1", start=5.0, end=10.0, text="t")
        events = [VisualEvent(time_range=(3.0, 6.0), scene="a")]
        assert len(_find_overlapping_events(seg, events)) == 1

    def test_no_overlap(self):
        seg = ASRSegment(segment_id="s1", start=0.0, end=3.0, text="t")
        events = [VisualEvent(time_range=(5.0, 8.0), scene="a")]
        assert len(_find_overlapping_events(seg, events)) == 0

    def test_boundary_touch(self):
        seg = ASRSegment(segment_id="s1", start=0.0, end=5.0, text="t")
        events = [VisualEvent(time_range=(5.0, 8.0), scene="a")]
        assert len(_find_overlapping_events(seg, events)) == 1

    def test_multiple_overlaps(self):
        seg = ASRSegment(segment_id="s1", start=0.0, end=10.0, text="t")
        events = [
            VisualEvent(time_range=(1.0, 3.0), scene="a"),
            VisualEvent(time_range=(5.0, 7.0), scene="b"),
            VisualEvent(time_range=(20.0, 25.0), scene="c"),
        ]
        assert len(_find_overlapping_events(seg, events)) == 2
