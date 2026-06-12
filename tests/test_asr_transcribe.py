from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.asr_transcribe import run, _build_asr_segments, _detect_device
from src.agent.schema import ASRSegment, AudioMeta, JobConfig


@pytest.fixture
def config():
    return JobConfig(
        job_id="test_001",
        video_uri="dummy.mp4",
        output_dir="./test_output",
        whisper_model="tiny",
        low_confidence_threshold=0.6,
    )


@pytest.fixture
def audio_meta(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake audio")
    return AudioMeta(audio_uri=str(audio_file), duration=10.0)


MOCK_RAW_RESULT = {
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "hello world",
        },
        {
            "start": 3.5,
            "end": 7.0,
            "text": "hybrid search",
        },
    ],
}

MOCK_ALIGNED_RESULT = {
    "segments": [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.95},
                {"word": "world", "start": 0.6, "end": 1.0, "score": 0.88},
            ],
        },
        {
            "start": 3.5,
            "end": 7.0,
            "text": "hybrid search",
            "words": [
                {"word": "hybrid", "start": 3.5, "end": 4.0, "score": 0.45},
                {"word": "search", "start": 4.1, "end": 4.5, "score": 0.50},
            ],
        },
    ],
}


class TestASRTranscribe:
    @patch("src.agent.asr_transcribe._detect_device", return_value="cpu")
    @patch("src.agent.asr_transcribe.whisperx")
    def test_success(self, mock_wx, mock_device, audio_meta, config):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = MOCK_RAW_RESULT
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = b"audio_data"
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = MOCK_ALIGNED_RESULT

        result = run(audio_meta, config)

        assert len(result) == 2
        assert all(isinstance(s, ASRSegment) for s in result)
        assert result[0].text == "hello world"
        assert result[1].text == "hybrid search"

    @patch("src.agent.asr_transcribe._detect_device", return_value="cpu")
    @patch("src.agent.asr_transcribe.whisperx")
    def test_word_timestamps(self, mock_wx, mock_device, audio_meta, config):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = MOCK_RAW_RESULT
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = b"audio_data"
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = MOCK_ALIGNED_RESULT

        result = run(audio_meta, config)

        assert len(result[0].words) == 2
        assert result[0].words[0].word == "hello"
        assert result[0].words[0].confidence == 0.95

    @patch("src.agent.asr_transcribe._detect_device", return_value="cpu")
    @patch("src.agent.asr_transcribe.whisperx")
    def test_low_confidence_flagging(self, mock_wx, mock_device, audio_meta, config):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = MOCK_RAW_RESULT
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = b"audio_data"
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = MOCK_ALIGNED_RESULT

        result = run(audio_meta, config)

        assert "low_confidence" not in result[0].quality_flags
        assert "low_confidence" in result[1].quality_flags

    @patch("src.agent.asr_transcribe._detect_device", return_value="cpu")
    @patch("src.agent.asr_transcribe.whisperx")
    def test_segment_ids(self, mock_wx, mock_device, audio_meta, config):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = MOCK_RAW_RESULT
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = b"audio_data"
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = MOCK_ALIGNED_RESULT

        result = run(audio_meta, config)

        assert result[0].segment_id == "seg_0000"
        assert result[1].segment_id == "seg_0001"

    def test_missing_audio(self, config):
        audio = AudioMeta(audio_uri="/nonexistent/audio.wav", duration=10.0)
        with pytest.raises(FileNotFoundError):
            run(audio, config)


class TestBuildASRSegments:
    def test_basic_segments(self):
        raw = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "hello",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9}
                ],
            }
        ]
        result = _build_asr_segments(raw, 0.6)
        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].quality_flags == []

    def test_low_confidence_word(self):
        raw = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "test",
                "words": [
                    {"word": "test", "start": 0.0, "end": 0.5, "score": 0.3}
                ],
            }
        ]
        result = _build_asr_segments(raw, 0.6)
        assert "low_confidence" in result[0].quality_flags

    def test_empty_segments(self):
        result = _build_asr_segments([], 0.6)
        assert result == []

    def test_segment_without_words(self):
        raw = [{"start": 0.0, "end": 3.0, "text": "hello"}]
        result = _build_asr_segments(raw, 0.6)
        assert result[0].text == "hello"
        assert result[0].words == []

    def test_speaker_assignment(self):
        raw = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "hello",
                "speaker": "SPEAKER_01",
                "words": [],
            }
        ]
        result = _build_asr_segments(raw, 0.6)
        assert result[0].speaker == "SPEAKER_01"


class TestDetectDevice:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available(self, mock_cuda):
        assert _detect_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, mock_cuda):
        assert _detect_device() == "cpu"
