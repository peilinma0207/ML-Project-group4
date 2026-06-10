import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.frame_sample import (
    run,
    _deduplicate,
    _interval_sample,
    _low_confidence_sample,
    _renumber,
    _scene_change_sample,
)
from src.schema import ASRSegment, JobConfig, SampledFrame, WordTimestamp


@pytest.fixture
def config(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video")
    return JobConfig(
        job_id="test_001",
        video_uri=str(video),
        output_dir=str(tmp_path / "output"),
        frame_interval=3.0,
    )


@pytest.fixture
def segments():
    return [
        ASRSegment(
            segment_id="seg_0000",
            start=0.0,
            end=5.0,
            text="hello world",
            quality_flags=[],
        ),
        ASRSegment(
            segment_id="seg_0001",
            start=5.5,
            end=10.0,
            text="hybrid search",
            quality_flags=["low_confidence"],
        ),
    ]


class TestFrameSampleRun:
    @patch("src.frame_sample._low_confidence_sample", return_value=[])
    @patch("src.frame_sample._scene_change_sample", return_value=[])
    @patch("src.frame_sample._interval_sample")
    @patch("src.frame_sample._probe_duration", return_value=10.0)
    def test_success(self, mock_dur, mock_interval, mock_scene, mock_low, config, segments):
        mock_interval.return_value = [
            SampledFrame(frame_path="f1.jpg", timestamp=0.0, sample_reason="interval"),
            SampledFrame(frame_path="f2.jpg", timestamp=3.0, sample_reason="interval"),
        ]
        result = run(config, segments)
        assert len(result) == 2
        assert all(isinstance(f, SampledFrame) for f in result)

    def test_missing_video(self, tmp_path):
        config = JobConfig(
            job_id="test_002",
            video_uri=str(tmp_path / "nonexistent.mp4"),
            output_dir=str(tmp_path / "output"),
        )
        with pytest.raises(FileNotFoundError):
            run(config, [])


class TestIntervalSample:
    @patch("src.frame_sample.subprocess.run")
    def test_generates_frames_at_interval(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        video = Path("test.mp4")
        frames_dir = Path("frames")
        result = _interval_sample(video, frames_dir, 7.0, 3.0)
        assert len(result) == 3
        assert result[0].timestamp == 0.0
        assert result[1].timestamp == 3.0
        assert result[2].timestamp == 6.0
        assert all(f.sample_reason == "interval" for f in result)

    @patch("src.frame_sample.subprocess.run")
    def test_zero_duration(self, mock_run):
        result = _interval_sample(Path("v.mp4"), Path("f"), 0.0, 3.0)
        assert result == []


class TestSceneChangeSample:
    @patch("src.frame_sample.subprocess.run")
    def test_with_scene_changes(self, mock_run):
        probe_result = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "frames": [
                    {"pts_time": "2.5"},
                    {"pts_time": "8.1"},
                ]
            }),
        )
        extract_result = MagicMock(returncode=0)
        mock_run.side_effect = [probe_result, extract_result, extract_result]

        result = _scene_change_sample(Path("v.mp4"), Path("frames"))
        assert len(result) == 2
        assert result[0].timestamp == 2.5
        assert result[1].timestamp == 8.1
        assert all(f.sample_reason == "scene_change" for f in result)

    @patch("src.frame_sample.subprocess.run")
    def test_no_scene_changes(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"frames": []}),
        )
        result = _scene_change_sample(Path("v.mp4"), Path("frames"))
        assert result == []

    @patch("src.frame_sample.subprocess.run")
    def test_probe_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _scene_change_sample(Path("v.mp4"), Path("frames"))
        assert result == []


class TestLowConfidenceSample:
    @patch("src.frame_sample.subprocess.run")
    def test_samples_around_flagged_segments(self, mock_run, segments):
        mock_run.return_value = MagicMock(returncode=0)
        result = _low_confidence_sample(Path("v.mp4"), Path("frames"), segments)
        assert len(result) > 0
        assert all(f.sample_reason == "low_confidence" for f in result)
        timestamps = [f.timestamp for f in result]
        assert min(timestamps) >= 3.5
        assert max(timestamps) <= 12.0

    @patch("src.frame_sample.subprocess.run")
    def test_no_flagged_segments(self, mock_run):
        segments = [
            ASRSegment(segment_id="s1", start=0.0, end=5.0, text="ok", quality_flags=[])
        ]
        result = _low_confidence_sample(Path("v.mp4"), Path("frames"), segments)
        assert result == []


class TestDeduplicate:
    def test_removes_close_frames(self):
        frames = [
            SampledFrame(frame_path="a.jpg", timestamp=0.0, sample_reason="interval"),
            SampledFrame(frame_path="b.jpg", timestamp=0.3, sample_reason="scene_change"),
            SampledFrame(frame_path="c.jpg", timestamp=1.0, sample_reason="interval"),
        ]
        result = _deduplicate(frames, min_gap=0.5)
        assert len(result) == 2
        assert result[0].timestamp == 0.0
        assert result[1].timestamp == 1.0

    def test_keeps_all_when_spread(self):
        frames = [
            SampledFrame(frame_path="a.jpg", timestamp=0.0, sample_reason="interval"),
            SampledFrame(frame_path="b.jpg", timestamp=3.0, sample_reason="interval"),
            SampledFrame(frame_path="c.jpg", timestamp=6.0, sample_reason="interval"),
        ]
        result = _deduplicate(frames, min_gap=0.5)
        assert len(result) == 3

    def test_empty_input(self):
        assert _deduplicate([], min_gap=0.5) == []

    def test_single_frame(self):
        frames = [SampledFrame(frame_path="a.jpg", timestamp=1.0, sample_reason="interval")]
        result = _deduplicate(frames, min_gap=0.5)
        assert len(result) == 1


class TestRenumber:
    def test_sequential_naming(self):
        frames = [
            SampledFrame(frame_path="old_a.jpg", timestamp=0.0, sample_reason="interval"),
            SampledFrame(frame_path="old_b.jpg", timestamp=3.0, sample_reason="scene_change"),
        ]
        result = _renumber(frames, Path("frames"))
        assert result[0].frame_path.endswith("frame_000001.jpg")
        assert result[1].frame_path.endswith("frame_000002.jpg")
        assert result[0].timestamp == 0.0
        assert result[1].sample_reason == "scene_change"

    def test_empty_input(self):
        result = _renumber([], Path("frames"))
        assert result == []
