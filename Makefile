.DEFAULT_GOAL := help

.PHONY: help doctor sync test cli run run-light run-full clean

VIDEO ?= data/videos/video_01.mp4
TOPIC ?= AI and machine learning
OUTPUT_DIR ?= output
WHISPER_MODEL ?= base
GLOSSARY ?= data/glossary.json
JOB_ID ?=
DIARIZE ?=
AUDIO ?=
VLM_MODEL ?= Qwen/Qwen3-VL-4B-Instruct
TEXT_MODEL ?= Qwen/Qwen3-8B
VLM_API_BASE ?=
TEXT_API_BASE ?=
CLI_ARGS ?= --help

PYTHON := uv run python
CLI := $(PYTHON) -m src.agent.cli

AUDIO_ARG := $(if $(strip $(AUDIO)),--audio "$(AUDIO)",)
JOB_ARG := $(if $(strip $(JOB_ID)),--job-id "$(JOB_ID)",)
DIARIZE_ARG := $(if $(strip $(DIARIZE)),--diarize,)
VLM_API_ARG := $(if $(strip $(VLM_API_BASE)),--vlm-api-base "$(VLM_API_BASE)",)
TEXT_API_ARG := $(if $(strip $(TEXT_API_BASE)),--text-api-base "$(TEXT_API_BASE)",)
COMMON_ARGS := --video "$(VIDEO)" $(AUDIO_ARG) --topic "$(TOPIC)" --output-dir "$(OUTPUT_DIR)" --whisper-model "$(WHISPER_MODEL)" --glossary "$(GLOSSARY)" $(JOB_ARG) $(DIARIZE_ARG)

help:
	@printf "%s\n" "ML-Project-group4 commands"
	@printf "%s\n" ""
	@printf "%s\n" "Targets:"
	@printf "%s\n" "  make doctor     Check uv, ffmpeg and ffprobe"
	@printf "%s\n" "  make sync       Install project and dev dependencies with uv"
	@printf "%s\n" "  make cli        Run raw CLI command (default: --help)"
	@printf "%s\n" "  make run        Run lightweight pipeline (--skip-vlm --skip-repair)"
	@printf "%s\n" "  make run-full   Run full VLM + text repair pipeline"
	@printf "%s\n" "  make test       Run pytest suite"
	@printf "%s\n" "  make clean      Remove OUTPUT_DIR"
	@printf "%s\n" ""
	@printf "%s\n" "Common variables:"
	@printf "%s\n" "  VIDEO=data/videos/video_01.mp4"
	@printf "%s\n" "  TOPIC=\"AI and machine learning\""
	@printf "%s\n" "  OUTPUT_DIR=output"
	@printf "%s\n" "  WHISPER_MODEL=base"
	@printf "%s\n" "  JOB_ID=demo"
	@printf "%s\n" "  DIARIZE=1"
	@printf "%s\n" "  CLI_ARGS=\"--help\""
	@printf "%s\n" "  VLM_API_BASE=http://localhost:1234/v1"
	@printf "%s\n" "  TEXT_API_BASE=http://localhost:1234/v1"

doctor:
	@command -v uv >/dev/null || { echo "Missing uv. Install uv before running this project."; exit 1; }
	@command -v ffmpeg >/dev/null || { echo "Missing ffmpeg. Install ffmpeg before processing videos."; exit 1; }
	@command -v ffprobe >/dev/null || { echo "Missing ffprobe. It is usually installed with ffmpeg."; exit 1; }
	@echo "OK: uv, ffmpeg and ffprobe are available."

sync:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

cli:
	$(CLI) $(CLI_ARGS)

run: run-light

run-light:
	$(CLI) $(COMMON_ARGS) --skip-vlm --skip-repair

run-full:
	$(CLI) $(COMMON_ARGS) --vlm-model "$(VLM_MODEL)" --text-model "$(TEXT_MODEL)" $(VLM_API_ARG) $(TEXT_API_ARG)

clean:
	@test -n "$(strip $(OUTPUT_DIR))" || { echo "OUTPUT_DIR is empty; refusing to remove it."; exit 1; }
	@test "$(OUTPUT_DIR)" != "/" || { echo "Refusing to remove /."; exit 1; }
	rm -rf "$(OUTPUT_DIR)"
