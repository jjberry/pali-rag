# CLAUDE.md — pali-rag

RAG over the Pāli Canon. Full design + review notes: `DESIGN.md`.

## Data (local, never cloned by this repo)
- `~/sc-data/sc_bilara_data/` — Pāli root (`root/pli/ms/sutta/`) + Sujato English
  (`translation/en/sujato/sutta/`), joined on shared segment IDs (`mn1:1.3`).
- `~/dpd.db` — full Digital Pāḷi Dictionary SQLite (read-only).
- All paths come from `config.py`; override via `SC_DATA` / `DPD_DB` env vars.

## Measured scale (don't re-estimate)
~4,167 paired suttas → 148,496 English segments → ~30,124 paragraph chunks.
Small enough that embedding is minutes on CPU; ChromaDB is for persistence, not
scale.

## Design rules baked into the code (keep them)
1. Embed **English only**; Pāli + citations are metadata. No bilingual concat.
2. Term archaeology expands the query via `dpd.db` inflection/sandhi tables
   (`inflection_templates`, `lookup.deconstructor`) — never naive substring.
3. Extract is an **inner join on the English side**: Pāli-only suttas are
   dropped on purpose (Vinaya/Abhidhamma, parts of KN).
4. Cap chunk size at `config.MAX_CHUNK_CHARS`, sub-splitting oversized
   paragraphs.
5. Generation: `claude-sonnet-4-6` default, `claude-opus-4-8` for `--hq`.

## Status
Runnable now (stdlib): `extract_segments.py`, `chunk.py`, `cli.py check`,
`term_lookup.py` headword lookup.
Stubbed (need deps / implementation): `embed_and_index.py`, `rag/retriever.py`,
`rag/pipeline.py`, and the inflection expansion in `term_lookup.py`.

## Build order
extract_segments → chunk → embed_and_index → retriever → pipeline → cli ask.
