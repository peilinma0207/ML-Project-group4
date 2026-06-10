import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.audio_extract import run, _probe_duration
from src.schema import AudioMeta, JobConfig


@pytest.fixture
def config(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video")
    return JobConfig(
        job_id="test_001",
        video_uri=str(video),
        output_dir=str(tmp_path / "output"),
    )


class TestAudioExtract:
    @patch("src.audio_extract._probe_duration", return_value=60.5)
    @patch("src.audio_extract.subprocess.run")
    def test_success(self, mock_run, mock_probe, config):
        mock_run.return_value = MagicMock(returncode=0)

        result = run(config)

        assert isinstance(result, AudioMeta)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.duration == 60.5
        assert result.audio_uri.endswith("audio.wav")

    @patch("src.audio_extract.subprocess.run")
    def test_ffmpeg_args(self, mock_run, config):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("src.audio_extract._probe_duration", return_value=10.0):
            run(config)

        call_args = mock_run.call_args_list[0]
        cmd = call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd
        assert "-ar" in cmd
        idx = cmd.index("-ar")
        assert cmd[idx + 1] == "16000"
        assert "-ac" in cmd
        idx = cmd.index("-ac")
        assert cmd[idx + 1] == "1"

    def test_missing_video(self, tmp_path):
        config = JobConfig(
            job_id="test_002",
            video_uri=str(tmp_path / "nonexistent.mp4"),
            output_dir=str(tmp_path / "output"),
        )
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            run(config)

    @patch("src.audio_extract.subprocess.run")
    def test_ffmpeg_failure(self, mock_run, config):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        with pytest.raises(subprocess.CalledProcessError):
            run(config)

    def test_output_dir_created(self, config):
        output_dir = Path(config.output_dir) / config.job_id
        assert not output_dir.exists()
        with patch("src.audio_extract.subprocess.run") as mock_run, \
             patch("src.audio_extract._probe_duration", return_value=5.0):
            mock_run.return_value = MagicMock(returncode=0)
            run(config)
        assert output_dir.exists()


class TestProbeDuration:
    @patch("src.audio_extract.subprocess.run")
    def test_parse_duration(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"format": {"duration": "123.456"}}),
            returncode=0,
        )
        result = _probe_duration(Path("test.wav"))
        assert result == 123.456

    @patch("src.audio_extract.subprocess.run")
    def test_ffprobe_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        with pytest.raises(subprocess.CalledProcessError):
            _probe_duration(Path("test.wav"))
