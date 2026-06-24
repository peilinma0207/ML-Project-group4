from __future__ import annotations

import base64
import json
import logging
import urllib.request
from pathlib import Path

from .schema import JobConfig, OCRHit, SampledFrame, VisualEvent

logger = logging.getLogger(__name__)

VLM_PROMPT_TEMPLATE = """Analyze these video frames and output a JSON array. For each frame, produce one object with:
- "frame_index": the frame number (0-based, matching the order of images provided)
- "scene": brief scene description
- "people": list of people visible
- "actions": list of actions happening
- "objects": list of notable objects
- "visible_text": list of objects with "text", "bbox" (4 ints), "confidence" (0-1)
- "term_candidates": list of technical terms or proper nouns visible or implied

Topic context: {topic}

Output ONLY a valid JSON array, one object per frame."""

BATCH_SIZE = 10


def run(frames: list[SampledFrame], config: JobConfig) -> list[VisualEvent]:
    raw_events: list[VisualEvent] = []

    for batch_start in range(0, len(frames), BATCH_SIZE):
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        valid_frames = [(f, Path(f.frame_path)) for f in batch if Path(f.frame_path).exists()]
        if not valid_frames:
            continue
        events = _extract_batch(valid_frames, config)
        raw_events.extend(events)

    return _merge_nearby_events(raw_events, max_gap=2.0)


def _extract_batch(
    frames_with_paths: list[tuple[SampledFrame, Path]],
    config: JobConfig,
) -> list[VisualEvent]:
    prompt = VLM_PROMPT_TEMPLATE.format(topic=config.topic_hint)

    content_parts: list[dict] = [{"type": "text", "text": prompt}]
    for frame, frame_path in frames_with_paths:
        try:
            image_data = frame_path.read_bytes()
            image_b64 = base64.b64encode(image_data).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        except Exception:
            logger.warning("Failed to read frame: %s", frame_path)

    if len(content_parts) <= 1:
        return []

    try:
        text = _call_vlm_api(config.vlm_api_base, config.vlm_model, content_parts, config.vlm_api_key)
        results = _parse_batch_output(text)
    except Exception as exc:
        logger.warning("VLM batch call failed: %s", exc)
        return []

    events = []
    for i, (frame, _) in enumerate(frames_with_paths):
        data = results[i] if i < len(results) else None
        if data is None:
            continue

        visible_text = []
        for vt in data.get("visible_text", []):
            if isinstance(vt, dict) and "text" in vt:
                visible_text.append(OCRHit(
                    text=vt["text"],
                    bbox=vt.get("bbox", []),
                    confidence=vt.get("confidence", 0.0),
                ))

        events.append(VisualEvent(
            time_range=(frame.timestamp, frame.timestamp),
            scene=data.get("scene", ""),
            people=data.get("people", []),
            actions=data.get("actions", []),
            objects=data.get("objects", []),
            visible_text=visible_text,
            term_candidates=data.get("term_candidates", []),
            confidence=0.5,
            evidence_frame=frame.frame_path,
        ))

    return events


def _call_vlm_api(api_base: str, model_name: str, content_parts: list[dict], api_key: str = "") -> str:
    import time as _time
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": content_parts}],
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
                logger.warning("VLM HTTP %d, retrying in %ds...", e.code, wait)
                _time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError) as e:
            if attempt < 4:
                wait = 2 ** attempt * 10
                logger.warning("VLM timeout/network error, retrying in %ds...", wait)
                _time.sleep(wait)
            else:
                raise


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
    # fallback: try single object
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1:
        try:
            obj = json.loads(text[obj_start:obj_end + 1])
            return [obj]
        except json.JSONDecodeError:
            pass
    return []


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
