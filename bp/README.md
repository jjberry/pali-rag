# bp/ — Bibliotheca Polyglotta pilot (secondary literature)

A **prototype**, separate from the working v1 Pāli-canon pipeline. It explores
extending the RAG idea to secondary/Mahāyāna literature via the University of
Oslo's **Thesaurus Literaturae Buddhicae (TLB)** — starting with Śāntideva's
**Bodhicaryāvatāra** (Sanskrit / Tibetan×2 / Chinese / Mongolian / French / two
English translations, aligned segment-by-segment).

## Why not scrape-and-index like v1?

TLB is a huge (~100+ texts), copyright-encumbered corpus with **no open license,
no API, and no bulk/TEI export**. So instead of building a local index of the
whole corpus, this pilot uses BP's own search to pick relevant *texts*, harvests
each one whole (once), and reranks it semantically:

```
question
  → expand      Claude → lexical search terms (English + karuṇā/bodhicitta…)
  → search      BP's server-side search = the relevance step, used to pick which
                TEXTS are worth harvesting (no local index needed up front)
  → harvest     pull each relevant text's whole "Complete text" continuous view
                in ONE request (cache-first), parsing every aligned segment
  → rerank      embed all the harvested English with v1's model, score vs. the
                question — semantic recall over each text's full content; drop
                runt segments and down-weight headings (ported from v1) so a
                chapter title can't out-rank the verses beneath it
  → answer      Claude answers, citing source text + segment / permalink
```

The chicken-and-egg ("we don't know which texts to fetch") dissolves because
**BP's search engine is the relevance step** — we delegate it rather than
pre-embedding everything.

**Why text-level (continuous view), not per-segment (record view):** the
per-segment `page=record` view is **AJAX-only for many texts** (Lotus,
Laṅkāvatāra… render an empty shell server-side), so scraping it silently loses
those texts. The `page=fulltext&view=fulltext&vid=N` "Complete text" view
renders server-side for **every** text, in a clean `class='BolkContainer'` →
`class='English'`/`'Sanskrit'`/… structure, and returns the whole text in one
request. Harvesting at the text level therefore (a) fixes the AJAX-only texts
and (b) lifts recall from BP's *lexical* search to *semantic* rerank over each
text's full content — so a passage no longer has to contain the exact keyword to
be found.

## The cache and the licensing bright line

Every harvested text is written to a **never-expiring** SQLite cache
(`data/bp_cache.sqlite`, **gitignored**): a `texts` row marks the text as
harvested and every segment lands in `records` (uid `vid:seg:i`). Scholarly
edition text is immutable, so permanent caching is correct, not stale; the cache
accumulates the texts a user actually queries and steadily cuts load on BP —
after the first harvest a text re-reads from disk in ~10 ms.

- We cache **texts/records** (immutable). We do **not** persistently cache
  **search results** — recall must stay fresh.
- The cache is **per-user and local**, built from the user's own BP access —
  same posture as v1's gitignored `data/chroma/` (itself a derived copy of
  Sujato text). Copyright's teeth are in **redistribution, not local storage**.
- **Bright line:** never commit or ship the cache DB or an index derived from
  it. Rate-limit and identify the User-Agent while fetching (see `config.py`).
  Keep attribution to BP/TLB. Two English translations here (Barnett 1947,
  Matics 1970) are plausibly still under copyright — fine for local research,
  not for republication.

## Semantic retrieval is now the default (not a future upgrade)

Because we harvest whole texts, the rerank already runs semantically over each
relevant text's full content — the "usage-driven upgrade" the earlier sketch
deferred is just how retrieval works now. `RecordCache.coverage(vid)` /
`has_text(vid)` report what's cached. Segment **embeddings** are cached too
(content-addressed by sha256 of the text + model name), so after a text is first
embedded every later query over it only embeds the *question* — the reranked
corpus itself is never re-embedded.

## Files

| file | role |
|------|------|
| `config.py` | BP endpoints, cache path, User-Agent, text→vid map; reuses v1's embedder |
| `cache.py`  | immutable cache: `records` (segments) + `texts` (harvest marker) |
| `client.py` | polite BP client: `search()`, `fetch_text()` (continuous view) + `fetch_record()` + parsers |
| `pilot.py`  | the expand → pick-texts → harvest → rerank → answer loop + CLI |

## Run

```bash
# needs ANTHROPIC_API_KEY, and the v1 .venv (sentence-transformers, anthropic)
python -m bp.pilot "What is the nature of bodhicitta?"
python -m bp.pilot --all-libraries --hq "How is patience (kṣānti) cultivated?"
python -m bp.pilot --max-texts 20 "..."   # raise the per-question text cap
```

`--max-texts` (default **12**) caps how many texts are harvested per question;
`-k` is how many segments are fed to the model. The first harvest of a text is
one HTTP request (~2–4 s); after that it re-reads from cache in ~10 ms.

## Status / not-yet

- **Verified live end-to-end** (2026-07-02): the continuous-view harvest works
  for every curated text incl. the previously-AJAX-only Lotus & Laṅkāvatāra;
  a bodhicitta query returned an answer citing all three.
- **Rerank hygiene ported from v1**: `MIN_SEGMENT_CHARS` floor + `HEADING_PENALTY`
  on `maintitle`/`subtitle`/`chaptertitle` segments (parser captures each
  segment's content-type `kind`). Fixed a case where a Bca subtitle out-ranked
  its own verses; tunable in `config.py`.
- Reranks **all** harvested English per query (e.g. ~3958 segments across 6
  texts), but **segment embeddings are now cached** (content-addressed:
  `embeddings(text_hash, model) → float32 blob`). First run over a text embeds
  it once (cold); every later query — even a different question — re-embeds
  nothing, only the query itself. Verified: cold `3958 new / 0 cached` →
  warm `0 new / 3958 cached`. Switching embedders re-embeds (model is in the key).
- `fetch_record()` (per-segment, UUID-cited) is kept for reference/debug but is
  no longer the retrieval path.
- No `robots.txt` / ToS check — add before any heavier use.
