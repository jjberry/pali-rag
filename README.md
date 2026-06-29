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
  extract_segments.py     join Pāli+English      -> data/segments.jsonl
  chunk.py                size-merged chunks      -> data/chunks.jsonl
  embed_and_index.py      embed (English only)    -> ChromaDB (data/chroma/)
  term_lookup.py          DPD inflection expansion + Pāli-field search
rag/
  retriever.py            ChromaDB query
  prompts.py              grounding system prompt
  pipeline.py             retrieve -> Claude
data/                     generated artifacts (gitignored)
```

## Setup

```bash
# 1. Confirm the local data is where config.py expects:
python3 cli.py check

# 2. Build the aligned dataset and chunks (stdlib only, fast):
python3 scripts/extract_segments.py    # -> ~148K segments
python3 scripts/chunk.py               # -> ~14.7K chunks

# 3. Install the ML/LLM deps:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Embed + build the vector index (downloads the embed model on first run;
#    ~7 min on CPU). Use --reset to rebuild an existing index.
python3 scripts/embed_and_index.py --reset
```

## Running queries

Both query commands assume the venv is active (`source .venv/bin/activate`).

**Grounded RAG answer** (`ask`) — retrieves passages and has Claude answer with
sutta-UID citations. Requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 cli.py ask "What does the Buddha say about the cause of suffering?"
python3 cli.py ask "..." --hq          # use the higher-quality model (opus)
```

A Pāli technical term in the query (e.g. `dukkha`) is automatically expanded
with its DPD English glosses before retrieval — so it matches the English-only
index even though the indexed text says "suffering", not "dukkha". The model
still answers your original wording; the expansion is logged to stderr.

**Term archaeology** (`term`) — expands a Pāli headword to its inflected forms
via `dpd.db`, then whole-word searches the Pāli text for every occurrence, with
exact segment-ID citations and a per-Nikāya breakdown. No API key needed:

```bash
python3 cli.py term vedanā
python3 cli.py term satipaṭṭhāna

# --compounds also matches sandhi-fused compounds containing the term (via
# lookup.deconstructor), recovering hits like 'satipaṭṭhānāti' (+iti) that
# whole-word matching misses:
python3 cli.py term satipaṭṭhāna --compounds
```

## Design decisions carried into the code

- **Embed English only**, store Pāli as metadata (avoids the bilingual-concat
  flaw); an English-optimized embedder is the default in `config.EMBED_MODEL`.
- **Term archaeology runs through `dpd.db`** (inflection + sandhi expansion),
  not naive substring search.
- **Inner-join on the English side** intentionally drops Pāli-only suttas.
- **Chunks are size-bounded**: capped at `config.MAX_CHUNK_CHARS` to avoid
  silent truncation, and merged up to `config.MIN_CHUNK_CHARS` so tiny
  one-sentence / title-only paragraphs don't out-rank real content.
