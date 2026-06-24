from __future__ import annotations

import json
import logging
import urllib.request

from .schema import (
    EvidenceSource,
    JobConfig,
    MergedEvidence,
    RepairedSegment,
)

logger = logging.getLogger(__name__)

BATCH_REPAIR_PROMPT_TEMPLATE = """You are a script repair assistant. Fix the ASR transcription segments using the provided evidence.

## Segments to repair
{segments_block}

## Topic
{topic}

## Instructions
1. Fix misheard words using visual and RAG evidence.
2. Preserve the original timestamps — do NOT change start/end times.
3. Keep the speaker label unchanged.
4. Only fix words where evidence supports the correction.
5. Set review_required to true if average word confidence is below 0.7.

Output ONLY a JSON array with one object per segment, in the same order:
[
  {{"segment_id": "seg_0000", "text": "corrected text", "confidence": 0.0-1.0, "review_required": true/false, "corrections": ["list"]}},
  ...
]"""

SEGMENT_TEMPLATE = """### Segment {segment_id}
- Time: {start:.2f}s - {end:.2f}s
- Speaker: {speaker}
- Text: {text}
- Word confidences: {word_confidences}
- Visual evidence: {visual_evidence}
- RAG terms: {rag_evidence}"""


def run(
    evidence: list[MergedEvidence],
    config: JobConfig,
) -> list[RepairedSegment]:
    if not evidence:
        return []

    prompt = _build_batch_prompt(evidence, config.topic_hint)

    try:
        text = _call_api(config.text_api_base, config.text_model, prompt, config.text_api_key)
        logger.info("Batch repair API returned %d chars", len(text) if text else 0)
        parsed = _parse_batch_output(text)
    except Exception as exc:
        logger.warning("Batch repair failed: %r — falling back to originals", exc)
        parsed = []

    results = []
    for i, merged in enumerate(evidence):
        data = parsed[i] if i < len(parsed) else None
        if data and data.get("text"):
            results.append(RepairedSegment(
                start=merged.asr.start,
                end=merged.asr.end,
                speaker=merged.asr.speaker,
                text=data.get("text", merged.asr.text),
                evidence=_build_evidence_source(merged),
                confidence=data.get("confidence", 0.0),
                review_required=data.get("review_required", True),
            ))
        else:
            results.append(RepairedSegment(
                start=merged.asr.start,
                end=merged.asr.end,
                speaker=merged.asr.speaker,
                text=merged.asr.text,
                evidence=_build_evidence_source(merged),
                confidence=0.0,
                review_required=True,
            ))

    return results


def _call_api(api_base: str, model_name: str, prompt: str, api_key: str = "") -> str:
    import time as _time
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 65536,
        "temperature": 0.1,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(5):
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                wait = 2 ** attempt * 10
                logger.warning("Repair HTTP %d, retrying in %ds...", e.code, wait)
                _time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError) as e:
            if attempt < 4:
                wait = 2 ** attempt * 10
                logger.warning("Repair timeout/network error, retrying in %ds...", wait)
                _time.sleep(wait)
            else:
                raise


def _build_batch_prompt(evidence: list[MergedEvidence], topic: str) -> str:
    segments_block = "\n\n".join(
        _format_segment(merged) for merged in evidence
    )
    return BATCH_REPAIR_PROMPT_TEMPLATE.format(
        segments_block=segments_block,
        topic=topic,
    )


def _format_segment(merged: MergedEvidence) -> str:
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
    visual_evidence = " | ".join(visual_parts) if visual_parts else "None"

    rag_parts = []
    for hit in merged.rag_hits:
        parts = [hit.term]
        if hit.aliases:
            parts.append(f"(aliases: {', '.join(hit.aliases[:3])})")
        rag_parts.append(" ".join(parts))
    rag_evidence = ", ".join(rag_parts) if rag_parts else "None"

    return SEGMENT_TEMPLATE.format(
        segment_id=merged.segment_id,
        start=merged.asr.start,
        end=merged.asr.end,
        speaker=merged.asr.speaker,
        text=merged.asr.text,
        word_confidences=word_confs,
        visual_evidence=visual_evidence,
        rag_evidence=rag_evidence,
    )


def _parse_batch_output(text: str) -> list[dict]:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # fallback: single object
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1:
        try:
            obj = json.loads(text[obj_start:obj_end + 1])
            return [obj]
        except json.JSONDecodeError:
            pass
    return []


def _build_evidence_source(merged: MergedEvidence) -> EvidenceSource:
    audio = [merged.segment_id]
    visual = [ve.evidence_frame for ve in merged.visual_events if ve.evidence_frame]
    rag = [f"{hit.source}:{hit.term}" for hit in merged.rag_hits]
    return EvidenceSource(audio=audio, visual=visual, rag=rag)
