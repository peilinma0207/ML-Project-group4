# Audio Processing

Covers `audio_extract.py` and `audio_preprocess.py`.

## audio_extract

Extracts mono audio from video using ffmpeg.

**Function:** `run(config: JobConfig) -> AudioMeta`

**Process:**
1. Validates video file exists
2. Creates output directory `{output_dir}/{job_id}/`
3. Runs ffmpeg: `-vn -acodec pcm_s16le -ar 16000 -ac 1`
4. Probes output duration via ffprobe
5. Returns `AudioMeta` with path, sample rate, duration

**Config used:** `video_uri`, `output_dir`, `job_id`

**System dependency:** `ffmpeg` and `ffprobe` must be on PATH.

**Error handling:** Raises `FileNotFoundError` if video missing, `subprocess.CalledProcessError` on ffmpeg failure.

## audio_preprocess

Applies loudness normalization and filtering to extracted audio.

**Function:** `run(audio: AudioMeta, config: JobConfig) -> AudioMeta`

**Filter chain:**
1. `highpass=f=80` — removes low-frequency rumble
2. `loudnorm=I=-16:TP=-1.5:LRA=11` — EBU R128 loudness normalization

**Output:** `{output_dir}/{job_id}/audio_processed.wav` with updated `AudioMeta`.

Preserves the original sample rate and channel count from input.
