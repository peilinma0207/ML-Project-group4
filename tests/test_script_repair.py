import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.script_repair import (
    run,
    _build_prompt,
    _parse_repair_output,
    _build_evidence_source,
    _repair_segment,
)
from src.agent.schema import (
    ASRSegment,
    EvidenceSource,
    MergedEvidence,
    OCRHit,
    RAGHit,
    RepairedSegment,
    VisualEvent,
    WordTimestamp,
)


@pytest.fixture
def config():
    from src.agent.schema import JobConfig
    return JobConfig(
        job_id="test_001",
        video_uri="dummy.mp4",
        topic_hint="vector databases",
    )


@pytest.fixture
def merged_evidence():
    asr = ASRSegment(
        segment_id="seg_0001",
        start=5.5,
        end=10.0,
        speaker="SPEAKER_01",
        text="high bread search is important",
        words=[
            WordTimestamp(word="high", start=5.5, end=5.8, confidence=0.4),
            WordTimestamp(word="bread", start=5.9, end=6.2, confidence=0.3),
            WordTimestamp(word="search", start=6.3, end=6.6, confidence=0.5),
            WordTimestamp(word="is", start=6.7, end=6.8, confidence=0.99),
            WordTimestamp(word="important", start=6.9, end=7.5, confidence=0.95),
        ],
        quality_flags=["low_confidence"],
    )
    visual = [
        VisualEvent(
            time_range=(5.0, 9.0),
            scene="presentation",
            visible_text=[OCRHit(text="Hybrid Search", confidence=0.9)],
            term_candidates=["Hybrid Search"],
            evidence_frame="frames/frame_000010.jpg",
        ),
    ]
    rag = [
        RAGHit(
            term="Hybrid Search",
            aliases=["hybrid retrieval"],
            common_mishearings=["high bread search"],
            source="glossary_v1",
            score=0.9,
        ),
    ]
    return MergedEvidence(
        segment_id="seg_0001",
        asr=asr,
        visual_events=visual,
        rag_hits=rag,
    )


MOCK_REPAIR_JSON = {
    "text": "Hybrid Search is important",
    "confidence": 0.85,
    "review_required": False,
    "corrections": ["high bread search -> Hybrid Search"],
}


class TestScriptRepair:
    @patch("src.agent.script_repair._load_model")
    def test_success(self, mock_load, merged_evidence, config):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]
        mock_tokenizer.decode.return_value = json.dumps(MOCK_REPAIR_JSON)

        result = run([merged_evidence], config)

        assert len(result) == 1
        assert isinstance(result[0], RepairedSegment)
        assert result[0].text == "Hybrid Search is important"
        assert result[0].start == 5.5
        assert result[0].end == 10.0
        assert result[0].speaker == "SPEAKER_01"

    @patch("src.agent.script_repair._load_model")
    def test_preserves_timestamps(self, mock_load, merged_evidence, config):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]
        mock_tokenizer.decode.return_value = json.dumps(MOCK_REPAIR_JSON)

        result = run([merged_evidence], config)
        assert result[0].start == merged_evidence.asr.start
        assert result[0].end == merged_evidence.asr.end

    @patch("src.agent.script_repair._load_model")
    def test_model_failure_falls_back(self, mock_load, merged_evidence, config):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_tokenizer.side_effect = Exception("model error")

        result = run([merged_evidence], config)
        assert len(result) == 1
        assert result[0].text == merged_evidence.asr.text
        assert result[0].review_required is True
        assert result[0].confidence == 0.0


class TestBuildPrompt:
    def test_includes_asr_text(self, merged_evidence):
        prompt = _build_prompt(merged_evidence, "test topic")
        assert "high bread search is important" in prompt

    def test_includes_visual_evidence(self, merged_evidence):
        prompt = _build_prompt(merged_evidence, "test topic")
        assert "Hybrid Search" in prompt
        assert "presentation" in prompt

    def test_includes_rag_evidence(self, merged_evidence):
        prompt = _build_prompt(merged_evidence, "test topic")
        assert "glossary_v1" not in prompt  # source is in evidence, not prompt
        assert "high bread search" in prompt

    def test_includes_topic(self, merged_evidence):
        prompt = _build_prompt(merged_evidence, "vector databases")
        assert "vector databases" in prompt

    def test_no_evidence(self):
        asr = ASRSegment(segment_id="s1", start=0.0, end=5.0, text="hello")
        merged = MergedEvidence(segment_id="s1", asr=asr)
        prompt = _build_prompt(merged, "topic")
        assert "None" in prompt


class TestParseRepairOutput:
    def test_valid_json(self):
        text = json.dumps(MOCK_REPAIR_JSON)
        result = _parse_repair_output(text)
        assert result["text"] == "Hybrid Search is important"

    def test_json_with_preamble(self):
        text = "Here is the repair:\n" + json.dumps(MOCK_REPAIR_JSON)
        result = _parse_repair_output(text)
        assert result is not None

    def test_no_json(self):
        assert _parse_repair_output("no json") is None

    def test_invalid_json(self):
        assert _parse_repair_output("{bad json}") is None


class TestBuildEvidenceSource:
    def test_all_sources(self, merged_evidence):
        source = _build_evidence_source(merged_evidence)
        assert isinstance(source, EvidenceSource)
        assert "seg_0001" in source.audio
        assert "frames/frame_000010.jpg" in source.visual
        assert "glossary_v1:Hybrid Search" in source.rag

    def test_no_evidence(self):
        asr = ASRSegment(segment_id="s1", start=0.0, end=5.0, text="t")
        merged = MergedEvidence(segment_id="s1", asr=asr)
        source = _build_evidence_source(merged)
        assert source.audio == ["s1"]
        assert source.visual == []
        assert source.rag == []
