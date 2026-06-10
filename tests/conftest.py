import pytest
from pathlib import Path


@pytest.fixture
def sample_job_config():
    from src.schema import JobConfig
    return JobConfig(
        job_id="test_001",
        video_uri="test_video.mp4",
        topic_hint="test topic",
        output_dir="./test_output",
    )


@pytest.fixture
def sample_audio_meta():
    from src.schema import AudioMeta
    return AudioMeta(
        audio_uri="test_output/test_001/audio.wav",
        sample_rate=16000,
        channels=1,
        duration=60.0,
    )


@pytest.fixture
def sample_asr_segments():
    from src.schema import ASRSegment, WordTimestamp
    return [
        ASRSegment(
            segment_id="seg_0001",
            start=0.0,
            end=5.0,
            speaker="SPEAKER_01",
            text="hello world",
            words=[
                WordTimestamp(word="hello", start=0.0, end=0.5, confidence=0.95),
                WordTimestamp(word="world", start=0.6, end=1.0, confidence=0.88),
            ],
            quality_flags=[],
        ),
        ASRSegment(
            segment_id="seg_0002",
            start=5.5,
            end=10.0,
            speaker="SPEAKER_01",
            text="hybrid search is important",
            words=[
                WordTimestamp(word="hybrid", start=5.5, end=6.0, confidence=0.45),
                WordTimestamp(word="search", start=6.1, end=6.5, confidence=0.50),
                WordTimestamp(word="is", start=6.6, end=6.8, confidence=0.99),
                WordTimestamp(word="important", start=6.9, end=7.5, confidence=0.92),
            ],
            quality_flags=["low_confidence"],
        ),
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def glossary_path(tmp_path):
    import json
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
