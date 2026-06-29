"""Central configuration for the Pāli Canon RAG project.

Paths point at the data already present locally (see the design doc's
Review Notes): the bilara segment corpus lives *inside* the sc-data clone,
so there is no separate bilara-data clone. Override any path via env var.
"""
from __future__ import annotations

import os
from pathlib import Path

# The embedding model is downloaded once and cached; there is no need to ping
# the HF Hub on every load. Run offline by default (silences the unauthenticated
# rate-limit warning and skips the startup network round-trip). Must be set
# before sentence-transformers / huggingface_hub import, so it lives here since
# config is imported first by every entry point. An explicit env var still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --- External data (already on disk; not part of this repo) ---------------
HOME = Path.home()

# sc-data clone. The segmented bilara corpus is under sc_bilara_data/.
SC_DATA = Path(os.environ.get("SC_DATA", HOME / "sc-data"))
SC_BILARA = SC_DATA / "sc_bilara_data"

# Verified source roots (segment-ID keyed JSON, identical keys across langs).
PALI_ROOT = SC_BILARA / "root" / "pli" / "ms" / "sutta"
EN_SUJATO = SC_BILARA / "translation" / "en" / "sujato" / "sutta"

# Full Digital Pāḷi Dictionary SQLite DB (grammar, roots, inflections, sandhi).
DPD_DB = Path(os.environ.get("DPD_DB", HOME / "dpd.db"))

# --- Generated artifacts (live in this repo, gitignored) ------------------
DATA_DIR = Path(__file__).parent / "data"
SEGMENTS_JSONL = DATA_DIR / "segments.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"

# --- Pipeline parameters --------------------------------------------------
# Nikāyas to ingest. DN/MN/SN/AN are fully covered by Sujato's English;
# 'kn' is partial (see coverage gap in the design doc).
COLLECTIONS = ("dn", "mn", "sn", "an", "kn")

# Revised design (#2): embed English alone, store Pāli as metadata.
# English-optimized embedder beats a multilingual one for English queries.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# Revised design (#5): cap chunk size and sub-split anything larger so it is
# not silently truncated at embedding time (~512 tokens ≈ 2000 chars).
MAX_CHUNK_CHARS = 2000

# Retrieval-quality follow-up: the suttas segment into many one-sentence
# paragraphs (and a title-only ':0' paragraph per sutta). Emitting one chunk
# each produced tiny, keyword-only chunks (e.g. a bare sutta title) that
# out-ranked real content. Merge consecutive paragraphs until a chunk reaches
# this soft floor, breaking only at a paragraph boundary past it.
MIN_CHUNK_CHARS = 350

CHROMA_COLLECTION = "pali_canon"

# --- Generation -----------------------------------------------------------
# sonnet for cost/latency; opus is the current top model for max quality.
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-sonnet-4-6")
GEN_MODEL_HQ = "claude-opus-4-8"
TOP_K = 8


def check_data() -> list[str]:
    """Return a list of human-readable problems with the local data layout."""
    problems = []
    for label, p in [
        ("sc-data", SC_DATA),
        ("sc_bilara_data", SC_BILARA),
        ("Pāli root text", PALI_ROOT),
        ("Sujato English", EN_SUJATO),
        ("dpd.db", DPD_DB),
    ]:
        if not p.exists():
            problems.append(f"missing {label}: {p}")
    return problems
