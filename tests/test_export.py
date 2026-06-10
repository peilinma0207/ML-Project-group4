import json
from pathlib import Path

import pytest

from src.export import run, _format_timestamp
from src.schema import (
    EvidenceSource,
    ExportResult,
    JobConfig,
    RepairedSegment,
)


@pytest.fixture
def config(tmp_path):
    return JobConfig(
        job_id="test_001",
        video_uri="test_video.mp4",
        topic_hint="vector databases",
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def repaired_segments():
    return [
        RepairedSegment(
            start=0.0,
            end=5.0,
            speaker="SPEAKER_01",
            text="Hello world",
            evidence=EvidenceSource(audio=["seg_0000"]),
            confidence=0.95,
            review_required=False,
        ),
        RepairedSegment(
            start=5.5,
            end=10.0,
            speaker="SPEAKER_01",
            text="Hybrid Search is important",
            evidence=EvidenceSource(
                audio=["seg_0001"],
                visual=["frame_000010.jpg"],
                rag=["glossary_v1:Hybrid Search"],
            ),
            confidence=0.85,
            review_required=False,
        ),
        RepairedSegment(
            start=10.5,
            end=15.0,
            speaker="SPEAKER_02",
            text="unclear segment here",
            confidence=0.35,
            review_required=True,
        ),
    ]


class TestExport:
    def test_run_creates_files(self, config, repaired_segments):
        result = run(repaired_segments, config)
        assert isinstance(result, ExportResult)
        assert Path(result.json_path).exists()
        assert Path(result.markdown_path).exists()

    def test_json_format(self, config, repaired_segments):
        result = run(repaired_segments, config)
        data = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["text"] == "Hello world"
        assert data[1]["text"] == "Hybrid Search is important"
        assert data[2]["review_required"] is True

    def test_json_roundtrip(self, config, repaired_segments):
        result = run(repaired_segments, config)
        data = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        for item in data:
            seg = RepairedSegment.model_validate(item)
            assert isinstance(seg, RepairedSegment)

    def test_markdown_contains_text(self, config, repaired_segments):
        result = run(repaired_segments, config)
        md = Path(result.markdown_path).read_text(encoding="utf-8")
        assert "Hello world" in md
        assert "Hybrid Search is important" in md
        assert "REVIEW REQUIRED" in md

    def test_markdown_contains_timestamps(self, config, repaired_segments):
        result = run(repaired_segments, config)
        md = Path(result.markdown_path).read_text(encoding="utf-8")
        assert "00:00:00.000" in md
        assert "00:00:05.000" in md

    def test_markdown_contains_speakers(self, config, repaired_segments):
        result = run(repaired_segments, config)
        md = Path(result.markdown_path).read_text(encoding="utf-8")
        assert "SPEAKER_01" in md
        assert "SPEAKER_02" in md

    def test_empty_segments(self, config):
        result = run([], config)
        data = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
        assert data == []

    def test_output_dir_created(self, config, repaired_segments):
        output_dir = Path(config.output_dir) / config.job_id
        assert not output_dir.exists()
        run(repaired_segments, config)
        assert output_dir.exists()


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00.000"

    def test_seconds(self):
        assert _format_timestamp(5.5) == "00:00:05.500"

    def test_minutes(self):
        assert _format_timestamp(65.123) == "00:01:05.123"

    def test_hours(self):
        assert _format_timestamp(3661.5) == "01:01:01.500"

    def test_subsecond_precision(self):
        assert _format_timestamp(1.001) == "00:00:01.001"
