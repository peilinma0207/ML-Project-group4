import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.audio_preprocess import run, _build_filter_chain
from src.schema import AudioMeta, JobConfig


@pytest.fixture
def config(tmp_path):
    return JobConfig(
        job_id="test_001",
        video_uri="dummy.mp4",
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def audio_meta(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake audio")
    return AudioMeta(
        audio_uri=str(audio_file),
        sample_rate=16000,
        channels=1,
        duration=60.0,
    )


class TestAudioPreprocess:
    @patch("src.audio_preprocess.subprocess.run")
    def test_success(self, mock_run, audio_meta, config):
        mock_run.return_value = MagicMock(returncode=0)
        result = run(audio_meta, config)

        assert isinstance(result, AudioMeta)
        assert result.audio_uri.endswith("audio_processed.wav")
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.duration == 60.0

    @patch("src.audio_preprocess.subprocess.run")
    def test_ffmpeg_filter_chain(self, mock_run, audio_meta, config):
        mock_run.return_value = MagicMock(returncode=0)
        run(audio_meta, config)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        af_idx = cmd.index("-af")
        filter_str = cmd[af_idx + 1]
        assert "highpass=f=80" in filter_str
        assert "loudnorm" in filter_str

    @patch("src.audio_preprocess.subprocess.run")
    def test_preserves_sample_rate(self, mock_run, audio_meta, config):
        mock_run.return_value = MagicMock(returncode=0)
        audio_meta.sample_rate = 44100
        result = run(audio_meta, config)
        assert result.sample_rate == 44100

        cmd = mock_run.call_args[0][0]
        ar_idx = cmd.index("-ar")
        assert cmd[ar_idx + 1] == "44100"

    def test_missing_audio_file(self, config):
        audio = AudioMeta(audio_uri="/nonexistent/audio.wav", duration=10.0)
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            run(audio, config)

    @patch("src.audio_preprocess.subprocess.run")
    def test_ffmpeg_failure(self, mock_run, audio_meta, config):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        with pytest.raises(subprocess.CalledProcessError):
            run(audio_meta, config)

    @patch("src.audio_preprocess.subprocess.run")
    def test_output_dir_created(self, mock_run, audio_meta, config):
        mock_run.return_value = MagicMock(returncode=0)
        output_dir = Path(config.output_dir) / config.job_id
        assert not output_dir.exists()
        run(audio_meta, config)
        assert output_dir.exists()


class TestBuildFilterChain:
    def test_contains_highpass(self):
        chain = _build_filter_chain()
        assert "highpass=f=80" in chain

    def test_contains_loudnorm(self):
        chain = _build_filter_chain()
        assert "loudnorm" in chain

    def test_comma_separated(self):
        chain = _build_filter_chain()
        parts = chain.split(",")
        assert len(parts) == 2
