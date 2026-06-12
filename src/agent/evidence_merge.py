from __future__ import annotations

from .schema import ASRSegment, MergedEvidence, RAGHit, VisualEvent


def run(
    segments: list[ASRSegment],
    visual_events: list[VisualEvent],
    rag_hits: list[RAGHit],
) -> list[MergedEvidence]:
    result = []
    for seg in segments:
        matched_visual = _find_overlapping_events(seg, visual_events)
        matched_rag = _find_matching_rag(seg, matched_visual, rag_hits)
        result.append(MergedEvidence(
            segment_id=seg.segment_id,
            asr=seg,
            visual_events=matched_visual,
            rag_hits=matched_rag,
        ))
    return result


def _find_overlapping_events(
    segment: ASRSegment,
    events: list[VisualEvent],
) -> list[VisualEvent]:
    result = []
    for event in events:
        ev_start, ev_end = event.time_range
        if ev_start <= segment.end and ev_end >= segment.start:
            result.append(event)
    return result


def _find_matching_rag(
    segment: ASRSegment,
    visual_events: list[VisualEvent],
    rag_hits: list[RAGHit],
) -> list[RAGHit]:
    segment_terms: set[str] = set()

    for word in segment.words:
        segment_terms.add(word.word.lower())

    for token in segment.text.lower().split():
        segment_terms.add(token)

    for event in visual_events:
        for candidate in event.term_candidates:
            segment_terms.add(candidate.lower())
        for ocr in event.visible_text:
            segment_terms.add(ocr.text.lower())

    matched = []
    for hit in rag_hits:
        all_terms = [hit.term.lower()] + [a.lower() for a in hit.aliases] + [m.lower() for m in hit.common_mishearings]
        if any(t in segment_terms for t in all_terms):
            matched.append(hit)
            continue
        for term in all_terms:
            for seg_term in segment_terms:
                if term in seg_term or seg_term in term:
                    matched.append(hit)
                    break
            else:
                continue
            break

    return matched
