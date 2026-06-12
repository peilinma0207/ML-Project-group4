import json

from src.agent.schema import (
    ASRSegment,
    AudioMeta,
    EvidenceSource,
    ExportResult,
    JobConfig,
    MergedEvidence,
    OCRHit,
    PipelineState,
    RAGHit,
    RepairedSegment,
    SampledFrame,
    VisualEvent,
    WordTimestamp,
)


class TestJobConfig:
    def test_create_minimal(self):
        config = JobConfig(job_id="j1", video_uri="v.mp4")
        assert config.job_id == "j1"
        assert config.topic_hint == ""

    def test_create_full(self):
        config = JobConfig(
            job_id="j1",
            video_uri="v.mp4",
            topic_hint="topic",
            output_dir="/out",
            whisper_model="large",
            enable_diarization=True,
        )
        assert config.enable_diarization is True
        assert config.whisper_model == "large"

    def test_json_roundtrip(self):
        config = JobConfig(job_id="j1", video_uri="v.mp4", topic_hint="t")
        data = config.model_dump_json()
        restored = JobConfig.model_validate_json(data)
        assert restored == config


class TestAudioMeta:
    def test_defaults(self):
        meta = AudioMeta(audio_uri="a.wav")
        assert meta.sample_rate == 16000
        assert meta.channels == 1

    def test_json_roundtrip(self):
        meta = AudioMeta(audio_uri="a.wav", duration=120.5)
        restored = AudioMeta.model_validate_json(meta.model_dump_json())
        assert restored == meta


class TestWordTimestamp:
    def test_create(self):
        w = WordTimestamp(word="hello", start=1.0, end=1.5, confidence=0.9)
        assert w.word == "hello"


class TestASRSegment:
    def test_with_words(self):
        seg = ASRSegment(
            segment_id="s1",
            start=0.0,
            end=5.0,
            text="hello world",
            words=[
                WordTimestamp(word="hello", start=0.0, end=0.5, confidence=0.9),
                WordTimestamp(word="world", start=0.6, end=1.0, confidence=0.8),
            ],
        )
        assert len(seg.words) == 2

    def test_quality_flags(self):
        seg = ASRSegment(
            segment_id="s1",
            start=0.0,
            end=5.0,
            text="test",
            quality_flags=["low_confidence"],
        )
        assert "low_confidence" in seg.quality_flags

    def test_json_roundtrip(self):
        seg = ASRSegment(segment_id="s1", start=0.0, end=5.0, text="t")
        restored = ASRSegment.model_validate_json(seg.model_dump_json())
        assert restored == seg


class TestSampledFrame:
    def test_create(self):
        f = SampledFrame(frame_path="f.jpg", timestamp=3.0, sample_reason="interval")
        assert f.sample_reason == "interval"


class TestOCRHit:
    def test_create(self):
        hit = OCRHit(text="Hello", bbox=[0, 0, 100, 50], confidence=0.95)
        assert hit.text == "Hello"


class TestVisualEvent:
    def test_create(self):
        event = VisualEvent(
            time_range=(0.0, 5.0),
            scene="office",
            visible_text=[OCRHit(text="Title", confidence=0.9)],
            term_candidates=["Vector Database"],
        )
        assert event.scene == "office"
        assert len(event.visible_text) == 1

    def test_json_roundtrip(self):
        event = VisualEvent(time_range=(1.0, 2.0), scene="s")
        restored = VisualEvent.model_validate_json(event.model_dump_json())
        assert restored == event


class TestRAGHit:
    def test_create(self):
        hit = RAGHit(
            term="Hybrid Search",
            aliases=["hybrid retrieval"],
            common_mishearings=["high bread search"],
            source="glossary_v1",
            score=0.9,
        )
        assert hit.term == "Hybrid Search"


class TestMergedEvidence:
    def test_create(self):
        asr = ASRSegment(segment_id="s1", start=0.0, end=5.0, text="t")
        merged = MergedEvidence(segment_id="s1", asr=asr)
        assert merged.visual_events == []
        assert merged.rag_hits == []


class TestRepairedSegment:
    def test_create(self):
        seg = RepairedSegment(
            start=0.0,
            end=5.0,
            text="repaired text",
            evidence=EvidenceSource(audio=["s1"], visual=["f1"]),
            confidence=0.85,
        )
        assert seg.review_required is False

    def test_json_roundtrip(self):
        seg = RepairedSegment(start=0.0, end=5.0, text="t")
        restored = RepairedSegment.model_validate_json(seg.model_dump_json())
        assert restored == seg


class TestExportResult:
    def test_create(self):
        result = ExportResult(json_path="a.json", markdown_path="a.md")
        assert result.json_path == "a.json"


class TestPipelineState:
    def test_minimal(self):
        config = JobConfig(job_id="j1", video_uri="v.mp4")
        state = PipelineState(config=config)
        assert state.audio is None
        assert state.asr_segments == []

    def test_json_roundtrip(self):
        config = JobConfig(job_id="j1", video_uri="v.mp4")
        state = PipelineState(
            config=config,
            audio=AudioMeta(audio_uri="a.wav", duration=10.0),
            asr_segments=[
                ASRSegment(segment_id="s1", start=0.0, end=5.0, text="t")
            ],
        )
        data = state.model_dump_json()
        restored = PipelineState.model_validate_json(data)
        assert restored.config.job_id == "j1"
        assert restored.audio.duration == 10.0
        assert len(restored.asr_segments) == 1
