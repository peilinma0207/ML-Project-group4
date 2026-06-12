# RAG Retrieval

Covers `rag_retrieve.py`.

## Overview

Mock RAG term lookup using a local JSON glossary file. Phase 1 substitute for a real vector store.

**Function:** `run(segments: list[ASRSegment], visual_events: list[VisualEvent], glossary_path: str) -> list[RAGHit]`

## Query Generation

Queries are collected from:
- Low-confidence ASR words (confidence < 0.6)
- All tokens from segments flagged with `low_confidence`
- N-grams (2-word, 3-word) from low-confidence segment text
- VLM term candidates
- OCR visible text

## Matching

Each query is checked against glossary entries in priority order:

1. **Exact match** against term (score: 100)
2. **Alias match** (score: 95)
3. **Mishearing match** (score: 90)
4. **Fuzzy match** via `rapidfuzz.fuzz.ratio` (threshold: 75)

Results are deduplicated by term — each glossary entry appears at most once.

## Glossary Format

`data/glossary.json`:

```json
[
  {
    "term": "Hybrid Search",
    "aliases": ["hybrid retrieval"],
    "common_mishearings": ["high bread search"],
    "source": "glossary_v1"
  }
]
```

## Notes

- Phase 2+ will replace this with a real vector store (Milvus/Qdrant/Chroma)
- The `source` field traces which glossary contributed each hit
