from __future__ import annotations

import json
import logging
from pathlib import Path

from src.schema import JobConfig, OCRHit, SampledFrame, VisualEvent

logger = logging.getLogger(__name__)

VLM_PROMPT_TEMPLATE = """Analyze this video frame and output a JSON object with the following fields:
- "scene": brief scene description
- "people": list of people visible (e.g. ["speaker", "audience"])
- "actions": list of actions happening
- "objects": list of notable objects
- "visible_text": list of objects with "text", "bbox" (4 ints), "confidence" (0-1)
- "term_candidates": list of technical terms or proper nouns visible or implied

Topic context: {topic}

Output ONLY valid JSON, no other text."""


def run(frames: list[SampledFrame], config: JobConfig) -> list[VisualEvent]:
    model, processor = _load_model(config.vlm_model)

    raw_events: list[VisualEvent] = []
    for frame in frames:
        frame_path = Path(frame.frame_path)
        if not frame_path.exists():
            logger.warning("Frame not found: %s", frame_path)
            continue

        event = _extract_single_frame(model, processor, frame, config.topic_hint)
        if event is not None:
            raw_events.append(event)

    return _merge_nearby_events(raw_events, max_gap=2.0)


def _load_model(model_name: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    return model, processor


def _extract_single_frame(
    model,
    processor,
    frame: SampledFrame,
    topic_hint: str,
) -> VisualEvent | None:
    from PIL import Image

    prompt = VLM_PROMPT_TEMPLATE.format(topic=topic_hint)

    try:
        image = Image.open(frame.frame_path)
    except Exception:
        logger.warning("Failed to open frame: %s", frame.frame_path)
        return None

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(outputs[0], skip_special_tokens=True)
        data = _parse_json_output(text)
    except Exception:
        logger.warning("VLM inference failed for frame: %s", frame.frame_path)
        return None

    if data is None:
        return None

    visible_text = []
    for vt in data.get("visible_text", []):
        if isinstance(vt, dict) and "text" in vt:
            visible_text.append(OCRHit(
                text=vt["text"],
                bbox=vt.get("bbox", []),
                confidence=vt.get("confidence", 0.0),
            ))

    return VisualEvent(
        time_range=(frame.timestamp, frame.timestamp),
        scene=data.get("scene", ""),
        people=data.get("people", []),
        actions=data.get("actions", []),
        objects=data.get("objects", []),
        visible_text=visible_text,
        term_candidates=data.get("term_candidates", []),
        confidence=0.5,
        evidence_frame=frame.frame_path,
    )


def _parse_json_output(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _merge_nearby_events(
    events: list[VisualEvent],
    max_gap: float,
) -> list[VisualEvent]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.time_range[0])
    merged: list[VisualEvent] = [sorted_events[0]]

    for event in sorted_events[1:]:
        prev = merged[-1]
        if event.time_range[0] - prev.time_range[1] <= max_gap:
            merged[-1] = VisualEvent(
                time_range=(prev.time_range[0], event.time_range[1]),
                scene=prev.scene if prev.scene else event.scene,
                people=list(set(prev.people + event.people)),
                actions=list(set(prev.actions + event.actions)),
                objects=list(set(prev.objects + event.objects)),
                visible_text=prev.visible_text + event.visible_text,
                term_candidates=list(set(prev.term_candidates + event.term_candidates)),
                confidence=max(prev.confidence, event.confidence),
                evidence_frame=prev.evidence_frame,
            )
        else:
            merged.append(event)

    return merged
