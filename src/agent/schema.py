from __future__ import annotations

from pydantic import BaseModel, Field


class JobConfig(BaseModel):
    job_id: str
    video_uri: str
    audio_uri: str = ""
    topic_hint: str = ""
    output_dir: str = "./output"
    whisper_model: str = "base"
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_api_base: str = ""
    text_model: str = "Qwen/Qwen3-8B"
    text_api_base: str = ""
    glossary_path: str = "data/glossary.json"
    enable_diarization: bool = False
    frame_interval: float = 3.0
    low_confidence_threshold: float = 0.6


class AudioMeta(BaseModel):
    audio_uri: str
    sample_rate: int = 16000
    channels: int = 1
    duration: float = 0.0
    codec: str = "pcm_s16le"


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 1.0


class ASRSegment(BaseModel):
    segment_id: str
    start: float
    end: float
    speaker: str = ""
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class SampledFrame(BaseModel):
    frame_path: str
    timestamp: float
    sample_reason: str


class OCRHit(BaseModel):
    text: str
    bbox: list[int] = Field(default_factory=list)
    confidence: float = 1.0


class VisualEvent(BaseModel):
    time_range: tuple[float, float]
    scene: str = ""
    people: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    visible_text: list[OCRHit] = Field(default_factory=list)
    term_candidates: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_frame: str = ""


class RAGHit(BaseModel):
    term: str
    aliases: list[str] = Field(default_factory=list)
    common_mishearings: list[str] = Field(default_factory=list)
    source: str = ""
    score: float = 0.0


class EvidenceSource(BaseModel):
    audio: list[str] = Field(default_factory=list)
    visual: list[str] = Field(default_factory=list)
    rag: list[str] = Field(default_factory=list)


class MergedEvidence(BaseModel):
    segment_id: str
    asr: ASRSegment
    visual_events: list[VisualEvent] = Field(default_factory=list)
    rag_hits: list[RAGHit] = Field(default_factory=list)


class RepairedSegment(BaseModel):
    start: float
    end: float
    speaker: str = ""
    text: str
    evidence: EvidenceSource = Field(default_factory=EvidenceSource)
    confidence: float = 0.0
    review_required: bool = False


class ExportResult(BaseModel):
    json_path: str
    markdown_path: str


class PipelineState(BaseModel):
    config: JobConfig
    audio: AudioMeta | None = None
    asr_segments: list[ASRSegment] = Field(default_factory=list)
    sampled_frames: list[SampledFrame] = Field(default_factory=list)
    visual_events: list[VisualEvent] = Field(default_factory=list)
    rag_hits: list[RAGHit] = Field(default_factory=list)
    merged_evidence: list[MergedEvidence] = Field(default_factory=list)
    repaired_segments: list[RepairedSegment] = Field(default_factory=list)
    export: ExportResult | None = None
