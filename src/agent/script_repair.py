from __future__ import annotations

import json
import logging

from .schema import (
    EvidenceSource,
    JobConfig,
    MergedEvidence,
    RepairedSegment,
)

logger = logging.getLogger(__name__)

REPAIR_PROMPT_TEMPLATE = """You are a script repair assistant. Fix the ASR transcription using the provided evidence.

## ASR Transcript
Text: {asr_text}
Time: {start:.2f}s - {end:.2f}s
Speaker: {speaker}
Word confidences: {word_confidences}

## Visual Evidence
{visual_evidence}

## RAG Term Matches
{rag_evidence}

## Topic
{topic}

## Instructions
1. Fix misheard words using visual and RAG evidence.
2. Preserve the original timestamps — do NOT change start/end times.
3. Keep the speaker label unchanged.
4. Only fix words where evidence supports the correction.
5. Set review_required to true if confidence is below 0.7.

Output ONLY a JSON object:
{{
  "text": "corrected transcript text",
  "confidence": 0.0-1.0,
  "review_required": true/false,
  "corrections": ["list of corrections made"]
}}"""


def run(
    evidence: list[MergedEvidence],
    config: JobConfig,
) -> list[RepairedSegment]:
    model, tokenizer = _load_model(config.text_model)

    results = []
    for merged in evidence:
        repaired = _repair_segment(model, tokenizer, merged, config.topic_hint)
        results.append(repaired)

    return results


def _load_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    return model, tokenizer


def _repair_segment(
    model,
    tokenizer,
    merged: MergedEvidence,
    topic_hint: str,
) -> RepairedSegment:
    prompt = _build_prompt(merged, topic_hint)

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data = _parse_repair_output(text)
    except Exception:
        logger.warning("Repair failed for segment %s, using original", merged.segment_id)
        data = None

    if data is None:
        return RepairedSegment(
            start=merged.asr.start,
            end=merged.asr.end,
            speaker=merged.asr.speaker,
            text=merged.asr.text,
            evidence=_build_evidence_source(merged),
            confidence=0.0,
            review_required=True,
        )

    return RepairedSegment(
        start=merged.asr.start,
        end=merged.asr.end,
        speaker=merged.asr.speaker,
        text=data.get("text", merged.asr.text),
        evidence=_build_evidence_source(merged),
        confidence=data.get("confidence", 0.0),
        review_required=data.get("review_required", True),
    )


def _build_prompt(merged: MergedEvidence, topic_hint: str) -> str:
    word_confs = ", ".join(
        f"{w.word}({w.confidence:.2f})" for w in merged.asr.words
    )

    visual_parts = []
    for ve in merged.visual_events:
        parts = [f"Scene: {ve.scene}"]
        if ve.visible_text:
            ocr_texts = [f"'{o.text}'({o.confidence:.2f})" for o in ve.visible_text]
            parts.append(f"Visible text: {', '.join(ocr_texts)}")
        if ve.term_candidates:
            parts.append(f"Terms: {', '.join(ve.term_candidates)}")
        visual_parts.append("; ".join(parts))
    visual_evidence = "\n".join(visual_parts) if visual_parts else "None"

    rag_parts = []
    for hit in merged.rag_hits:
        parts = [f"Term: {hit.term}"]
        if hit.aliases:
            parts.append(f"Aliases: {', '.join(hit.aliases)}")
        if hit.common_mishearings:
            parts.append(f"Common mishearings: {', '.join(hit.common_mishearings)}")
        rag_parts.append("; ".join(parts))
    rag_evidence = "\n".join(rag_parts) if rag_parts else "None"

    return REPAIR_PROMPT_TEMPLATE.format(
        asr_text=merged.asr.text,
        start=merged.asr.start,
        end=merged.asr.end,
        speaker=merged.asr.speaker,
        word_confidences=word_confs,
        visual_evidence=visual_evidence,
        rag_evidence=rag_evidence,
        topic=topic_hint,
    )


def _parse_repair_output(text: str) -> dict | None:
    text = text.strip()
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _build_evidence_source(merged: MergedEvidence) -> EvidenceSource:
    audio = [merged.segment_id]
    visual = [ve.evidence_frame for ve in merged.visual_events if ve.evidence_frame]
    rag = [f"{hit.source}:{hit.term}" for hit in merged.rag_hits]
    return EvidenceSource(audio=audio, visual=visual, rag=rag)
