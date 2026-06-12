# Evidence Merge & Script Repair

Covers `evidence_merge.py` and `script_repair.py`.

## evidence_merge

Aligns visual and RAG evidence to ASR segments by time range.

**Function:** `run(segments, visual_events, rag_hits) -> list[MergedEvidence]`

### Matching Logic

- **Visual events** match a segment if their time ranges overlap (inclusive boundary)
- **RAG hits** match a segment if any of the hit's terms, aliases, or mishearings appear in the segment's words, text tokens, or associated visual event term candidates

Every ASR segment gets a `MergedEvidence` entry, even if no visual or RAG evidence matched.

## script_repair

Uses an 8B text model to correct ASR transcription based on merged evidence.

**Function:** `run(evidence: list[MergedEvidence], config: JobConfig) -> list[RepairedSegment]`

### Prompt Design

For each segment, the prompt includes:
- Original ASR text with per-word confidence scores
- Visual evidence: scene descriptions, visible text (OCR), term candidates
- RAG evidence: correct terms, aliases, common mishearings
- Topic context from `config.topic_hint`

The model is instructed to output JSON with: corrected text, confidence, review flag.

### Constraints

- **Timestamps are preserved** — the model cannot change start/end times
- **Speaker labels are preserved** — carried through from ASR
- **Evidence-only corrections** — model should only fix words where evidence supports it
- **Review flagging** — segments with confidence < 0.7 get `review_required=true`

### Fallback

If model inference fails for a segment, the original ASR text is kept with `confidence=0.0` and `review_required=true`.

### Evidence Tracking

Each `RepairedSegment` records which sources contributed:
- `audio`: segment IDs
- `visual`: frame file paths
- `rag`: `{source}:{term}` identifiers
