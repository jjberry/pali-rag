# pali-rag

Retrieval-augmented exploration of the Pāli Canon, grounded in primary texts.
English semantic queries surface actual sutta passages (Sujato's CC0 English
with the Pāli alongside) and Claude answers with sutta-UID citations.

See [`DESIGN.md`](DESIGN.md) for the full design and review notes.

## Data sources (local, not vendored)

This project reads data already on disk — it does **not** clone anything:

- `~/sc-data/sc_bilara_data/` — segmented Pāli root text + Sujato English,
  joined on shared segment IDs (e.g. `mn1:1.3`).
- `~/dpd.db` — full Digital Pāḷi Dictionary (grammar, roots, inflections,
  sandhi), used for term archaeology.

Override locations with the `SC_DATA` / `DPD_DB` env vars (see `config.py`).

## Layout

```
config.py                 paths + pipeline parameters (points at ~/sc-data, ~/dpd.db)
cli.py                    entry point: check | ask | term
scripts/
  extract_segments.py     [runnable]  join Pāli+English -> data/segments.jsonl
  chunk.py                [runnable]  paragraph chunks   -> data/chunks.jsonl
  embed_and_index.py      [stub]      embed (English only) -> ChromaDB
  term_lookup.py          [partial]   DPD term expansion (inflection TODO)
rag/
  retriever.py            [stub]      ChromaDB query
  prompts.py              [done]      grounding system prompt
  pipeline.py             [stub]      retrieve -> Claude
data/                     generated artifacts (gitignored)
```

## Quick start

```bash
# 1. Confirm the local data is where config.py expects:
python3 cli.py check

# 2. Build the aligned dataset and chunks (stdlib only, ~30K chunks, fast):
python3 scripts/extract_segments.py
python3 scripts/chunk.py

# 3. Install ML/LLM deps for the rest:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. (to implement) embed + index, then query:
python3 scripts/embed_and_index.py
python3 cli.py ask "What does the Buddha say about the cause of suffering?"
python3 cli.py term vedanā
```

## Design decisions carried into the code

- **Embed English only**, store Pāli as metadata (avoids the bilingual-concat
  flaw); an English-optimized embedder is the default in `config.EMBED_MODEL`.
- **Term archaeology runs through `dpd.db`** (inflection + sandhi expansion),
  not naive substring search.
- **Inner-join on the English side** intentionally drops Pāli-only suttas.
- **Chunk size is capped** (`config.MAX_CHUNK_CHARS`) to avoid silent
  truncation at embedding time.
```
