# pali-rag

Retrieval-augmented exploration of the Pāli Canon, grounded in primary texts.
English semantic queries surface actual sutta passages (Sujato's CC0 English
with the Pāli alongside) and Claude answers with sutta-UID citations.

See [`DESIGN.md`](DESIGN.md) for the full design and review notes.

## Data sources (local, not vendored)

This project reads data already on disk — it does **not** vendor or clone
anything itself. You supply two inputs:

- `~/sc-data/sc_bilara_data/` — segmented Pāli root text + Sujato English,
  joined on shared segment IDs (e.g. `mn1:1.3`). Get it from SuttaCentral's
  bilara data: <https://github.com/suttacentral/bilara-data>
  (`git clone` it and point `SC_DATA` at the parent, or symlink so that
  `$SC_DATA/sc_bilara_data` resolves).
- `~/dpd.db` — the Digital Pāḷi Dictionary SQLite database (grammar, roots,
  inflections, sandhi), used for term archaeology and query expansion. Download
  from the DPD project: <https://github.com/digitalpalidictionary/dpd-db>
  (releases include the prebuilt `dpd.db`).

Override either location with the `SC_DATA` / `DPD_DB` env vars (see
`config.py`). Run `python3 cli.py check` to confirm both resolve.

### Credits & licensing of the data

This tool is just plumbing; the substance is other people's scholarship:

- **Bhikkhu Sujato's English translations** and the SuttaCentral segmented
  texts are dedicated to the public domain (CC0).
- **The Digital Pāḷi Dictionary** (Bodhirasa) is released under CC BY-NC-SA
  4.0 — please review its terms before any redistribution or non-personal use.

The data is not included in this repository; these are pointers to the
upstream projects. The code in this repo is MIT-licensed (see `LICENSE`).

## Layout

```
config.py                 paths + pipeline parameters (points at ~/sc-data, ~/dpd.db)
cli.py                    entry point: check | ask | chat | web | term
scripts/
  extract_segments.py     join Pāli+English      -> data/segments.jsonl
  chunk.py                size-merged chunks      -> data/chunks.jsonl
  embed_and_index.py      embed (English only)    -> ChromaDB (data/chroma/)
  term_lookup.py          DPD inflection expansion + Pāli-field search
rag/
  retriever.py            ChromaDB query
  prompts.py              grounding system prompt
  pipeline.py             retrieve -> Claude
  chat.py                 multi-turn session (condense + retrieve + history)
web/
  app.py                  stdlib server: read / ask / chat
  render.py               Markdown -> HTML (tables on)
  templates/, static/     Jinja2 pages + CSS
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
python3 cli.py ask "..." --secondary   # add a second, secondary-sources answer
```

A Pāli technical term in the query (e.g. `dukkha`) is automatically expanded
with its DPD English glosses before retrieval — so it matches the English-only
index even though the indexed text says "suffering", not "dukkha". The model
still answers your original wording; the expansion is logged to stderr.

**Secondary sources** (`--secondary`) — adds a second section that answers the
same question from Mahāyāna / secondary literature via
[Bibliotheca Polyglotta](https://www2.hf.uio.no/polyglotta/) (the `bp/` pilot:
live keyword search → fetch → semantic re-rank → cited answer). The two answers
appear under `## From the Pāli Canon` and `## From the secondary literature`
headings and are saved together. It runs concurrently with the Pāli answer, but
is slower (live, rate-limited network requests) and needs a network connection;
without `--secondary` nothing changes. Works on `ask` and `chat`, and as a
checkbox in the web UI's Ask and Chat pages.

**Multi-turn conversation** (`chat`) — `ask` is one-shot; `chat` is an
interactive REPL that keeps the conversation going. Each follow-up is condensed
into a standalone search query (using the history) before retrieval, so
references like "expand on that" still retrieve the right passages. Sessions
can be saved and resumed:

```bash
python3 cli.py chat                       # ephemeral session
python3 cli.py chat --session anatta      # save under a name
python3 cli.py chat --resume anatta       # continue it later
python3 cli.py chat --secondary           # add secondary sources each turn
```

**Saving answers to re-read later** — answers are written as Markdown under
`data/answers/` so you can revisit them without re-running the query:

```bash
python3 cli.py ask "..." --save           # auto-named data/answers/<ts>-<slug>.md
python3 cli.py ask "..." --save notes.md  # or a path you choose
```

In `chat`, type `:save [path]` to export a transcript on demand; a
`--session`-named chat also auto-exports `data/answers/<name>.md` on exit.

**Web UI** (`web`) — a local browser front-end that renders answers as proper
Markdown (tables, citations, and all), so long responses are easier to read
than in the terminal:

```bash
python3 cli.py web                        # http://127.0.0.1:8000
python3 cli.py web --port 8080 --hq
```

Three sections:

- **Read** — browse and re-read everything saved under `data/answers/`,
  rendered to HTML. Works with no API key.
- **Ask** — one-shot questions; the answer is rendered inline and auto-saved
  to `data/answers/`. An *include secondary sources* checkbox adds the
  Bibliotheca Polyglotta section (see `--secondary` above).
- **Chat** — multi-turn, history-aware conversation in the browser (same
  condense + retrieve pipeline as the REPL); each session auto-saves a
  resumable transcript. The same *secondary sources* checkbox is available and
  can be toggled per message.

It's a dependency-light stdlib server (`markdown-it-py` + Jinja2, both already
pinned), single-user, and binds to `127.0.0.1` only — Read needs no API key,
Ask/Chat require `ANTHROPIC_API_KEY`.

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
- **Title-fusion retrieval**: a parallel one-entry-per-sutta title index is
  fused (reciprocal rank fusion) with body-chunk hits, so the stock formulaic
  suttas — terse, elided bodies that embed weakly — still surface when their
  topical title matches (e.g. SN 22.59 "The Characteristic of Not-Self").

## License

Code: MIT (see [`LICENSE`](LICENSE)). The Pāli/English texts and the dictionary
are separate works under their own licenses — see *Credits & licensing of the
data* above.
