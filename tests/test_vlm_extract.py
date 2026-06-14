import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.agent.vlm_extract import (
    run,
    _extract_single_frame,
    _merge_nearby_events,
    _parse_json_output,
)
from src.agent.schema import JobConfig, OCRHit, SampledFrame, VisualEvent


@pytest.fixture
def config():
    return JobConfig(
        job_id="test_001",
        video_uri="dummy.mp4",
        topic_hint="vector databases",
        vlm_api_base="http://localhost:1234/v1",
    )


@pytest.fixture
def frames(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        img.save(str(p))
        paths.append(p)
    return [
        SampledFrame(frame_path=str(paths[0]), timestamp=0.0, sample_reason="interval"),
        SampledFrame(frame_path=str(paths[1]), timestamp=1.5, sample_reason="interval"),
        SampledFrame(frame_path=str(paths[2]), timestamp=5.0, sample_reason="scene_change"),
    ]


MOCK_VLM_JSON = {
    "scene": "conference room",
    "people": ["speaker"],
    "actions": ["presenting"],
    "objects": ["screen", "laptop"],
    "visible_text": [
        {"text": "Vector Database", "bbox": [10, 20, 200, 50], "confidence": 0.9}
    ],
    "term_candidates": ["Vector Database", "HNSW"],
}


def _make_mock_response(data):
    response_body = json.dumps({
        "choices": [{"message": {"content": json.dumps(data)}}]
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestVLMExtract:
    @patch("src.agent.vlm_extract.urllib.request.urlopen")
    def test_run_success(self, mock_urlopen, frames, config):
        mock_urlopen.return_value = _make_mock_response(MOCK_VLM_JSON)
        result = run(frames, config)
        assert len(result) >= 1
        assert all(isinstance(e, VisualEvent) for e in result)
        assert mock_urlopen.call_count == 3

    @patch("src.agent.vlm_extract.urllib.request.urlopen")
    def test_run_api_failure(self, mock_urlopen, frames, config):
        mock_urlopen.side_effect = Exception("connection refused")
        result = run(frames, config)
        assert result == []

    def test_missing_frame_skipped(self, config):
        frames = [SampledFrame(frame_path="/nonexistent.jpg", timestamp=0.0, sample_reason="interval")]
        result = run(frames, config)
        assert result == []


class TestParseJsonOutput:
    def test_valid_json(self):
        text = json.dumps(MOCK_VLM_JSON)
        result = _parse_json_output(text)
        assert result["scene"] == "conference room"

    def test_json_with_surrounding_text(self):
        text = 'Here is the analysis:\n' + json.dumps(MOCK_VLM_JSON) + '\nDone.'
        result = _parse_json_output(text)
        assert result is not None
        assert result["scene"] == "conference room"

    def test_no_json(self):
        assert _parse_json_output("no json here") is None

    def test_invalid_json(self):
        assert _parse_json_output("{invalid json}") is None

    def test_empty_string(self):
        assert _parse_json_output("") is None


class TestMergeNearbyEvents:
    def test_merge_close_events(self):
        events = [
            VisualEvent(time_range=(0.0, 0.0), scene="office", people=["a"]),
            VisualEvent(time_range=(1.0, 1.0), scene="", people=["b"]),
        ]
        result = _merge_nearby_events(events, max_gap=2.0)
        assert len(result) == 1
        assert result[0].time_range == (0.0, 1.0)
        assert result[0].scene == "office"
        assert set(result[0].people) == {"a", "b"}

    def test_no_merge_far_events(self):
        events = [
            VisualEvent(time_range=(0.0, 0.0), scene="a"),
            VisualEvent(time_range=(5.0, 5.0), scene="b"),
        ]
        result = _merge_nearby_events(events, max_gap=2.0)
        assert len(result) == 2

    def test_empty_events(self):
        assert _merge_nearby_events([], max_gap=2.0) == []

    def test_single_event(self):
        events = [VisualEvent(time_range=(1.0, 1.0), scene="s")]
        result = _merge_nearby_events(events, max_gap=2.0)
        assert len(result) == 1

    def test_merges_term_candidates(self):
        events = [
            VisualEvent(time_range=(0.0, 0.0), term_candidates=["A"]),
            VisualEvent(time_range=(1.0, 1.0), term_candidates=["B"]),
        ]
        result = _merge_nearby_events(events, max_gap=2.0)
        assert set(result[0].term_candidates) == {"A", "B"}

    def test_merges_visible_text(self):
        events = [
            VisualEvent(
                time_range=(0.0, 0.0),
                visible_text=[OCRHit(text="Hello", confidence=0.9)],
            ),
            VisualEvent(
                time_range=(1.0, 1.0),
                visible_text=[OCRHit(text="World", confidence=0.8)],
            ),
        ]
        result = _merge_nearby_events(events, max_gap=2.0)
        assert len(result[0].visible_text) == 2
