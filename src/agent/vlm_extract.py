from __future__ import annotations

import base64
import json
import logging
import urllib.request
from pathlib import Path

from .schema import JobConfig, OCRHit, SampledFrame, VisualEvent

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
    raw_events: list[VisualEvent] = []
    for frame in frames:
        frame_path = Path(frame.frame_path)
        if not frame_path.exists():
            logger.warning("Frame not found: %s", frame_path)
            continue

        event = _extract_single_frame(frame, config)
        if event is not None:
            raw_events.append(event)

    return _merge_nearby_events(raw_events, max_gap=2.0)


def _extract_single_frame(
    frame: SampledFrame,
    config: JobConfig,
) -> VisualEvent | None:
    prompt = VLM_PROMPT_TEMPLATE.format(topic=config.topic_hint)

    try:
        image_data = Path(frame.frame_path).read_bytes()
        image_b64 = base64.b64encode(image_data).decode()
    except Exception:
        logger.warning("Failed to read frame: %s", frame.frame_path)
        return None

    try:
        text = _call_vlm_api(config.vlm_api_base, config.vlm_model, prompt, image_b64)
        data = _parse_json_output(text)
    except Exception:
        logger.warning("VLM API call failed for frame: %s", frame.frame_path)
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


def _call_vlm_api(api_base: str, model_name: str, prompt: str, image_b64: str) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


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
