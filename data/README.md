# TED AI Dataset

This folder contains the collected TED and TED-Ed video dataset for the damaged transcript repair project.

## Dataset Purpose

The goal of this dataset is to support a damaged transcript repair Agent. Each sample provides a video clip, an extracted audio file, and a ground truth transcript.

These files can be used by later modules such as Whisper, vision-only models, RAG, and the final transcript repair Agent.

## Data Source

The original project plan mentioned YouTube videos. During data collection, YouTube downloading was blocked by bot verification and cookie authentication issues. Therefore, this dataset uses TED and TED-Ed open video sources instead.

The selected videos are related to artificial intelligence, machine learning, deep learning, AI agents, misinformation, or AI understanding.

## Dataset Scale

This dataset contains:

- 10 selected TED / TED-Ed video clips
- 10 extracted WAV audio files
- 10 ground truth transcript files
- 1 metadata CSV file
- RAG background documents

## Folder Structure

data/
- videos/
- audio/
- transcripts_ground_truth/
- metadata/
- rag_docs/

## Folder Description

### videos/

This folder contains the selected video clips.

Example:

video_01.mp4  
video_02.mp4  
...  
video_10.mp4  

Each video clip corresponds to one source sample. The selected time range is recorded in metadata/videos_metadata.csv.

### audio/

This folder contains audio files extracted from the selected video clips.

Example:

video_01.wav  
video_02.wav  
...  
video_10.wav  

The audio files are saved in WAV format and can be used by Whisper or other speech recognition models.

### transcripts_ground_truth/

This folder contains the ground truth transcript files.

Example:

video_01_ground_truth.txt  
video_02_ground_truth.txt  
...  
video_10_ground_truth.txt  

These transcripts are copied and organized from the official TED / TED-Ed transcript pages. They should be used as reference answers for later evaluation.

### metadata/

This folder contains the metadata file:

videos_metadata.csv

The metadata file records the basic information of each sample, including video ID, title, source platform, URL, speaker, theme, start time, end time, language, local file paths, and notes.

### rag_docs/

This folder contains background documents for the RAG module.

The RAG documents provide useful information about:

- AI and machine learning terms
- speaker background
- video topic summaries
- dataset source notes

These documents are intended to help the Agent repair damaged transcripts by recognizing technical terms, speaker names, topic-specific vocabulary, and background context.

## Relationship with Later Project Steps

The data synthesis member can use the ground truth transcripts to create damaged transcripts with different noise levels.

The Agent framework member can use:

- video clips as visual input
- audio files for Whisper transcription
- ground truth transcripts for evaluation
- RAG documents as external background knowledge

## Notes

Only the selected video clips are necessary for experiments. Full original videos are not required in the repository because the source URLs and selected time ranges are already recorded in the metadata file.

