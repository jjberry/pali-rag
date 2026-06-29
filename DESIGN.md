# Pāli Canon RAG System — Design Document

> **Revision note (2026-06-29):** This document was originally drafted without access to the local data. It has since been checked against the actual files on disk (`~/sc-data`, `~/dpd.db`). Corrections are applied inline below and summarized in [Review Notes & Revisions](#review-notes--revisions) at the end. Read that section before building.

## Project Overview

A retrieval-augmented generation (RAG) system for exploring the Pāli Canon in a way that is grounded in primary texts rather than secondary literature. The system allows semantic queries in English that surface actual sutta passages, with Pāli text alongside, enabling term-level and thematic research anchored directly to canonical sources.

**Goals:**
- Answer questions about the Buddha's teachings with citations to specific suttas
- Enable term archaeology: tracing how Pāli terms are used across the Nikāyas
- Support comparison across schools by identifying which canonical passages each position draws from
- Serve practical, philosophical, and scholarly inquiry simultaneously

---

## Data Sources

### Primary: bilara-data (already present locally inside sc-data)
- **Repo:** `https://github.com/suttacentral/bilara-data`
- **No separate clone needed.** This segmented corpus already lives inside the local sc-data clone at `~/sc-data/sc_bilara_data/`. Verified paths:
  - Pāli root text: `~/sc-data/sc_bilara_data/root/pli/ms/sutta/{dn,mn,sn,an,kn}/…_root-pli-ms.json`
  - Sujato English: `~/sc-data/sc_bilara_data/translation/en/sujato/sutta/{dn,mn,sn,an,kn}/…_translation-en-sujato.json`
- Contains the full Pāli Canon in segment-level JSON, with aligned translations
- Bhikkhu Sujato's English translation is the most complete and is licensed CC0 — free for any use
- Each segment has a unique ID (e.g., `mn1:1.3`) encoding Nikāya, sutta number, and position. Verified: the Pāli and English files share identical keys, so the join is exact.
- Structure: separate JSON files per sutta per language, keyed by segment ID
- **Coverage gap (verified):** 5,764 Pāli suttas exist but only 4,167 have a Sujato English translation → 1,597 Pāli-only suttas (much of Vinaya/Abhidhamma + parts of KN such as Jātaka). The four main Nikāyas (DN/MN/SN/AN) are fully covered. See revision #3.

### Secondary: sc-data (cloned locally at `~/sc-data`)
- **Repo:** `https://github.com/suttacentral/sc-data`
- Contains metadata, sutta tree structure, parallel mappings across traditions (Pāli, Chinese Āgamas, Sanskrit, Tibetan), and blurbs
- The parallels file (`relationship/parallels.json`) maps cross-tradition relationships — useful for later cross-tradition features

### Dictionary: Pāli Digital Dictionary (DPD)
- **Present locally at `~/dpd.db`** (full DPD SQLite database; verified 89,050 headwords, 753 roots)
- To be integrated for term-level lookup
- Provides definitions, grammatical analysis, and root information for Pāli terms
- Complements the corpus embeddings for precise term archaeology
- **Critical for term archaeology:** the `lookup` table's `deconstructor` column (sandhi/compound splits) plus `inflection_templates` give the lemmatization needed to find inflected forms of a term. This must be used from day one, not deferred — see revision #4.

---

## Architecture

### Phase 1: Data Pipeline

**Step 1: Clone repositories**
```bash
git clone https://github.com/suttacentral/bilara-data
git clone https://github.com/suttacentral/sc-data
```

**Step 2: Extract aligned segment pairs**

Walk bilara-data to extract Pāli (`root/pli/ms/`) and English (`translation/en/sujato/`) JSON files, pairing them by segment ID. Output a structured dataset:

```
segment_id  | pali_text | english_text | nikaya | sutta_uid | sutta_title
mn1:1.3     | ...       | ...          | MN     | mn1       | Mūlapariyāya Sutta
```

Target collections for initial build (Early Buddhism focus):
- DN — Dīgha Nikāya (Long Discourses)
- MN — Majjhima Nikāya (Middle Length Discourses)
- SN — Saṁyutta Nikāya (Linked Discourses)
- AN — Aṅguttara Nikāya (Numerical Discourses)
- Dhp, Ud, Iti, Snp — selected Khuddaka Nikāya texts

**Step 3: Chunking**

Chunk at paragraph level: group consecutive segments into chunks that approximate natural paragraph breaks. Each chunk retains:
- All constituent segment IDs (for citation)
- Combined Pāli text
- Combined English text
- Nikāya, sutta UID, sutta title metadata

Paragraph-level chunking is preferred over single-segment (too narrow) or full-sutta (too broad). The formulaic repetition in the suttas means similar teachings appear many times — paragraph chunking lets retrieval surface multiple instances naturally.

**Step 4: Construct embedding input**

> **REVISED (#2):** The original plan was to embed a bilingual concatenation `[PALI] {pali_text} [EN] {english_text}`. This is now considered a flaw: for an *English* query against a general multilingual model, the English content dominates the vector while the Pāli mostly adds noise — it does not improve retrieval and can degrade it.
>
> **Revised approach:** embed the **English text alone** for semantic retrieval, and store the Pāli as metadata on the chunk. Handle Pāli term lookup through a separate path (the term-archaeology mode backed by `dpd.db`, revision #4). This also frees the embedding-model choice from the multilingual constraint — see Phase 2.

Embedding input per chunk:

```
{english_text}
```

with `pali_text`, segment IDs, and sutta metadata stored alongside as retrievable metadata (not embedded).

**Step 4b: Enforce a max chunk size (#5)**

Paragraph grouping by the major segment number is usually small (median ~158 chars, p99 ~1,400) but a few chunks balloon (observed max ~32,661 chars where grouping is too coarse, e.g. long repetition blocks or verse texts). Cap chunk size (e.g. ~2,000 chars / ~512 tokens) and sub-split anything larger so it is not silently truncated at embedding time.

---

### Phase 2: Embedding and Indexing

**Embedding model**

> **REVISED (#2):** Since we now embed English alone (not a Pāli+English concat), an **English-optimized** embedder is the better default — e.g. `BAAI/bge-large-en-v1.5`, `thenlper/gte-large`, or an `e5` model — which will outperform a multilingual model on English-query→English-passage retrieval. Reserve the multilingual models (LaBSE, `paraphrase-multilingual-mpnet-base-v2`) for if/when we add genuine cross-lingual (Pāli-side) retrieval.

Runs locally on CPU (Mac Studio). No GPU required for inference.

> **REVISED (#1):** The original "a few hours on CPU" assumed ~500K–1M chunks. The real corpus is far smaller (see vector database below), so embedding the full English Canon is a matter of **minutes, not hours**.

**Vector database**

Use ChromaDB for local development:
- Runs entirely in-process or as a persistent local store
- No server required
- Python-native, simple API
- Handles the Canon's scale comfortably

> **REVISED (#1):** Measured scale is **~30,124 paragraph-chunks** (from 148,496 Sujato English segments) — roughly 20× smaller than the original "~500K–1M" estimate. At this size the data fits in RAM and even brute-force numpy cosine similarity is effectively instant. ChromaDB is still a fine, convenient choice, but the only real reason to use it here is persistence — not scale. Don't over-engineer the indexing.

Store each chunk with:
- Embedding vector
- Full metadata (segment IDs, nikaya, sutta_uid, sutta_title, pali_text, english_text)

**Fine-tuning (deferred)**

The bilara-data Pāli-English segment pairs are ideal training data for fine-tuning a multilingual sentence transformer to better align Pāli with English in the embedding space. This is deferred to Phase 3 — validate the system with a pretrained model first.

---

### Phase 3: Query and Generation

**Query flow:**

1. User submits a natural language query in English
2. Query is embedded using the same sentence transformer
3. ChromaDB returns top-k most similar chunks (k=5–10 to start)
4. Retrieved chunks (with Pāli, English, and citations) are passed to Claude as context
5. Claude generates a grounded response, citing specific suttas by UID

**System prompt design**

The system prompt should instruct Claude to:
- Ground all claims in the retrieved passages
- Always cite sutta UIDs (e.g., MN 1, SN 22.59)
- Note when retrieved passages are formulaic repetitions vs. distinct teachings
- Flag when the query touches areas where retrieved passages are ambiguous or sparse
- Distinguish between the text's own framing and later interpretive glosses

**Term archaeology mode**

Secondary query mode for Pāli term lookup:
1. User specifies a Pāli term (e.g., *vedanā*, *paṭicca-samuppāda*)
2. Search is run against the Pāli field specifically
3. All chunks containing the term are retrieved and grouped by Nikāya
4. Claude synthesizes how the term is used across contexts, noting consistency and variation
5. DPD definition is included as additional context

> **REVISED (#4):** Step 2 cannot be naive substring search. Pāli is heavily inflected and uses **sandhi** (words fuse: *vedanā* → *vedanānaṁ*, fused into compounds, etc.), so a raw match misses most occurrences. Use `dpd.db` to expand the query term to its inflected/sandhi forms before searching:
> - `inflection_templates` (joined to `dpd_headwords.pattern`) to generate declensional forms
> - `lookup.deconstructor` (JSON sandhi/compound splits) to catch fused occurrences
>
> This was filed under "Future Directions" in the original draft, but the term-archaeology mode genuinely depends on it — it must be in scope from the first version of this mode, not deferred.

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for NLP/ML tooling |
| Embedding | sentence-transformers | Local inference, no API cost, good multilingual support |
| Vector DB | ChromaDB | Simple, local, no server, Python-native |
| LLM | Claude via Anthropic API | `claude-sonnet-4-6` for generation (cost/latency); use `claude-opus-4-8`, the current top model, where max generation quality matters |
| Data format | JSON (bilara-data native) | No conversion needed |
| Interface | CLI first, then optional web UI | Fastest to build and iterate |

---

## Project Structure

```
pali-rag/
├── CLAUDE.md               # This document (for Claude Code sessions)
├── data/
│   ├── bilara-data/        # git submodule or clone
│   └── sc-data/            # git submodule or clone
├── scripts/
│   ├── extract_segments.py # Step 2: build aligned segment dataset
│   ├── chunk.py            # Step 3: paragraph-level chunking
│   ├── embed_and_index.py  # Step 4+Phase 2: embed chunks, build ChromaDB index
│   └── term_lookup.py      # Term archaeology utility
├── rag/
│   ├── retriever.py        # ChromaDB query interface
│   ├── prompts.py          # System prompt templates
│   └── pipeline.py         # End-to-end query → response
├── cli.py                  # Main entry point
├── requirements.txt
└── README.md
```

---

## Build Order

1. `extract_segments.py` — parse bilara-data, output `segments.jsonl`
2. `chunk.py` — group segments into paragraph chunks, output `chunks.jsonl`
3. `embed_and_index.py` — embed all chunks, persist ChromaDB collection
4. `retriever.py` — query interface over ChromaDB
5. `pipeline.py` — wire retriever to Claude API
6. `cli.py` — simple query interface for testing

---

## Key Design Decisions

**Why bilara-data over sc-data for text?**
bilara-data has the actual sutta text in clean segment-level JSON. sc-data has structure and metadata. Both are needed but bilara-data is the text source.

**Why Sujato's translation?**
Most complete coverage of the four main Nikāyas, CC0 licensed, modern readable English. Bhikkhu Bodhi's translations are more scholarly but not freely licensed for this use.

**Why paragraph chunks rather than full suttas?**
Full suttas are too long and dilute retrieval precision. Single segments are too short and lose context. Paragraph chunks balance semantic coherence with retrieval specificity. Many suttas are also highly repetitive — chunking at paragraph level lets the index capture distinct teaching moments rather than duplicating large repetition blocks.

**Why local embeddings rather than OpenAI embeddings?**
Cost at scale, no data leaving local machine (the Pāli texts are CC0 but keeping the pipeline local is clean), and sentence-transformers are well-suited to this domain.

**Why ChromaDB rather than Pinecone or Weaviate?**
Entirely local, no account or API key, zero infrastructure. For a research tool used by one person on a Mac Studio, operational simplicity wins.

---

## Future Directions

- **Fine-tuned embeddings:** Use bilara-data Pāli-English pairs to fine-tune a multilingual model for better cross-lingual alignment
- **Cross-tradition queries:** Use sc-data parallels to surface Chinese Āgama parallels alongside Pāli suttas
- **Commentary layer:** Optionally index Buddhaghosa's commentaries (Visuddhimagga, etc.) as a separate collection, queryable independently — allowing explicit comparison of canonical vs. commentarial framing
- **Web UI:** Simple React interface with sutta citation links back to suttacentral.net
- **Pāli morphological search:** Integrate a Pāli stemmer/lemmatizer so term searches handle inflected forms

---

## Review Notes & Revisions

The design was reviewed on 2026-06-29 against the actual local data (`~/sc-data`, `~/dpd.db`). The original draft was written on a phone without data access. Verified facts and corrections:

**Verified sound:**
- The bilara segment corpus is already present at `~/sc-data/sc_bilara_data/` — Step 1 (clone) is effectively done; no separate `bilara-data` clone needed.
- The segment-ID join (`mn1:1.3`) is exact: Pāli and English files share identical keys.
- DPD DB live at `~/dpd.db`: 89,050 headwords, 753 roots.
- Paragraph chunking fits the embedding context window: median chunk ~158 chars, p99 ~1,400; only 130 of ~30K chunks exceed ~2,000 chars.

**Revisions (referenced inline above):**
1. **Scale was overestimated ~20×.** Real corpus ≈ 30,124 paragraph-chunks / 148,496 English segments, not "500K–1M." Embedding is minutes not hours; ChromaDB is convenience, not necessity.
2. **Drop the bilingual `[PALI]…[EN]…` single-embedding.** Embed English alone for retrieval; store Pāli as metadata. This also unlocks a stronger English-only embedder over the multilingual default.
3. **Make the Pāli↔English join explicit.** Inner-join silently drops 1,597 Pāli-only suttas. Harmless for the DN/MN/SN/AN focus (fully covered), but log it as a decision and verify per-text KN coverage (Dhp/Ud/Iti/Snp).
4. **Term archaeology must use `dpd.db` from day one.** Naive Pāli substring search fails on inflection/sandhi; use `inflection_templates` + `lookup.deconstructor` to expand terms. Pulled forward from "Future Directions."
5. **Add a max-chunk-size guard.** One observed chunk hit ~32,661 chars; cap and sub-split to avoid silent truncation.

**Minor:** `claude-sonnet-4-6` is fine for generation; `claude-opus-4-8` is the current top model for max quality. Fine-tuning correctly deferred and even less urgent at this scale.

---

## References

- bilara-data: https://github.com/suttacentral/bilara-data
- sc-data: https://github.com/suttacentral/sc-data
- SuttaCentral: https://suttacentral.net
- ChromaDB docs: https://docs.trychroma.com
- sentence-transformers: https://www.sbert.net
- Pāli Digital Dictionary: https://digitalpalidictionary.github.io
