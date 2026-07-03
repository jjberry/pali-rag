"""BP-pilot configuration, kept separate from the v1 project config so this
exploration can evolve without disturbing the working pipeline. Reuses the v1
embedding model (so rerank scores are comparable) and the v1 data dir (so the
cache lands under the already-gitignored data/)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as v1  # noqa: E402  (the v1 project config)

BASE = "https://www2.hf.uio.no/polyglotta/"

# Immutable record cache. Under data/, which is gitignored in the repo already,
# so the cache is never committed or shipped (the licensing bright line).
CACHE_DB = v1.DATA_DIR / "bp_cache.sqlite"

# Be a good citizen: identify the tool and never hammer the server.
USER_AGENT = "pali-rag-bp-pilot/0.1 (+research; contact jeffreyjberry@gmail.com)"
MIN_REQUEST_INTERVAL = 1.0  # seconds between live requests to BP

# Reuse v1's English embedder so rerank distances mean the same thing (#2).
EMBED_MODEL = v1.EMBED_MODEL

# Generation model for expansion + answering (same defaults as v1).
GEN_MODEL = v1.GEN_MODEL
GEN_MODEL_HQ = v1.GEN_MODEL_HQ

# Known text -> volume-id (vid) map, from the TLB library listing
# (index.php?page=library&bid=2). Extend as the pilot covers more texts.
# Restricting search to a curated subset keeps recall focused and cuts noise
# from the ~100+ other TLB texts. None => search all libraries (context=a).
TEXTS: dict[str, int] = {
    "bodhicaryavatara": 1120,   # Śāntideva — the pilot text (Skt/Tib/Chi/Eng×2)
    "vimalakirti": 37,
    "lotus": 483,               # Saddharmapuṇḍarīka
    "lankavatara": 1265,
    "buddhacarita": 77,
    "dhammapada": 80,
}

# Language-family prefixes used in BP edition tags (e.g. "Eng: Barnett (1947)").
LANG_PREFIX = {
    "San": "Sanskrit", "Pāli": "Pali", "Pali": "Pali", "Chi": "Chinese",
    "Tib": "Tibetan", "Mon": "Mongolian", "Fre": "French", "Eng": "English",
    "Ger": "German", "Skt": "Sanskrit",
}

# We embed/rerank the English rendering only, mirroring v1 design rule #2.
RERANK_LANG = "English"

# Rerank hygiene, ported from v1's title/min-length handling (v1 used a
# MIN_CHUNK_CHARS floor so tiny keyword chunks couldn't out-rank real content,
# and gated a separate title index). The continuous view labels each segment's
# content type in its span class, so we can act on it directly:
#   - drop runt segments below the floor (the MIN_CHUNK_CHARS analog);
#   - down-weight heading segments so a topical chapter title can't out-rank the
#     verses under it (the gated-title analog) — penalized, not dropped, since a
#     strongly on-topic heading is still a useful signal.
MIN_SEGMENT_CHARS = 60
HEADING_KINDS = frozenset({"maintitle", "subtitle", "chaptertitle"})
HEADING_PENALTY = 0.15  # subtracted from the cosine score of heading segments
