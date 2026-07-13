"""Immutable, never-expiring local cache of BP records (SQLite).

Rationale: a fetched TLB record is a set of *scholarly edition* segments
(`San: Minayeff (1889) 157,26` ...) — the content does not change, so caching
it permanently is correct, not stale. The cache slowly accumulates the texts a
user actually queries, so repeat questions never re-hit BP.

Licensing bright line: this DB lives under data/ (gitignored) and is per-user,
built from the user's own BP access — same posture as v1's gitignored
data/chroma/ (itself a derived copy of Sujato text). Copyright's teeth are in
*redistribution*, not local storage: never commit or ship this file.

We cache *records* (immutable). We deliberately do NOT cache search results
here — recall must stay fresh so BP additions and better query terms are seen.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import config as C

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    uid         TEXT PRIMARY KEY,   -- BP permalink UUID, or 'vid:seg:i' for
                                    -- segments harvested from the continuous view
    vid         INTEGER NOT NULL,
    mid         INTEGER NOT NULL,   -- BP segment mid, or the segment index for
                                    -- continuous-view harvests
    source_text TEXT,               -- e.g. 'bodhicaryavatara'
    fetched_at  REAL NOT NULL,
    raw_html    TEXT,               -- kept so we can re-parse if the parser improves
    parsed_json TEXT NOT NULL       -- {language: [{edition, text, span_id?}, ...]}
);
CREATE INDEX IF NOT EXISTS idx_records_vid_mid ON records(vid, mid);

-- Marks a whole text as harvested from its continuous view (page=fulltext).
-- This is the unit the retriever works in: search picks relevant texts, we
-- fetch each once, and every segment lands in `records` under the same vid.
CREATE TABLE IF NOT EXISTS texts (
    vid         INTEGER PRIMARY KEY,
    source_text TEXT,
    n_segments  INTEGER NOT NULL,
    fetched_at  REAL NOT NULL
);

-- Content-addressed embedding cache. The embedding of a fixed string under a
-- fixed model never changes, so — like the records above — it is cacheable
-- forever. Keying on the text's hash (not a segment uid) is deliberate: a
-- segment may have several English renderings, and content-addressing also
-- survives parser changes and re-harvests. `model` is part of the key so
-- switching embedders re-embeds rather than mixing incompatible vectors.
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT NOT NULL,   -- sha256 hex of the embedded (normalized) text
    model     TEXT NOT NULL,   -- embedding model name
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,   -- normalized float32 vector, .tobytes()
    PRIMARY KEY (text_hash, model)
);
"""

# SQLite's default limit on host parameters in one statement is 999; stay well
# under it when batching an IN (...) lookup of many text hashes.
_SQLITE_VAR_CHUNK = 500


class RecordCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or C.CACHE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the web server hands each request to a fresh
        # thread, so a cached BPClient/RecordCache singleton is reached from
        # different threads over time. Access is still serialized upstream (the
        # web app's _lock), so dropping the affinity check is safe here.
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    # --- reads ---------------------------------------------------------------
    def get_by_mid(self, vid: int, mid: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM records WHERE vid=? AND mid=?", (vid, mid)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get(self, uid: str) -> dict | None:
        row = self.db.execute("SELECT * FROM records WHERE uid=?", (uid,)).fetchone()
        return self._row_to_record(row) if row else None

    def coverage(self, vid: int) -> int:
        """How many segments of a text are cached — the signal for the
        usage-driven upgrade from lexical-live to local-semantic retrieval."""
        return self.db.execute(
            "SELECT COUNT(*) FROM records WHERE vid=?", (vid,)
        ).fetchone()[0]

    def has_text(self, vid: int) -> bool:
        """True once a text's continuous view has been fully harvested."""
        return self.db.execute(
            "SELECT 1 FROM texts WHERE vid=?", (vid,)
        ).fetchone() is not None

    def get_text_segments(self, vid: int) -> list[dict]:
        """All cached segments of a harvested text, in segment order."""
        rows = self.db.execute(
            "SELECT * FROM records WHERE vid=? ORDER BY mid", (vid,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # --- embedding cache (content-addressed) ---------------------------------
    def get_vectors(self, hashes: list[str], model: str) -> dict[str, bytes]:
        """Return {text_hash: raw float32 bytes} for whichever of `hashes` are
        already embedded under `model`. Chunked to respect SQLite's variable
        limit; the caller reconstructs vectors (keeps this module numpy-free)."""
        out: dict[str, bytes] = {}
        for i in range(0, len(hashes), _SQLITE_VAR_CHUNK):
            chunk = hashes[i:i + _SQLITE_VAR_CHUNK]
            q = ("SELECT text_hash, vec FROM embeddings WHERE model=? AND "
                 "text_hash IN (%s)" % ",".join("?" * len(chunk)))
            for row in self.db.execute(q, (model, *chunk)):
                out[row["text_hash"]] = row["vec"]
        return out

    def put_vectors(self, items: list[tuple[str, int, bytes]], model: str) -> None:
        """Store embeddings. `items` is (text_hash, dim, vec_bytes). Idempotent:
        an already-cached (text_hash, model) is left untouched (immutable)."""
        if not items:
            return
        self.db.executemany(
            "INSERT OR IGNORE INTO embeddings (text_hash, model, dim, vec) "
            "VALUES (?,?,?,?)",
            [(h, model, dim, vec) for (h, dim, vec) in items],
        )
        self.db.commit()

    # --- writes --------------------------------------------------------------
    def put(self, record: dict, *, refresh: bool = False) -> None:
        """Insert a record. Idempotent: skips if the uid is already cached
        (immutable content) unless refresh=True (the rare edition-correction
        escape hatch)."""
        uid = record["uid"]
        if not refresh and self.get(uid) is not None:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO records "
            "(uid, vid, mid, source_text, fetched_at, raw_html, parsed_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                uid,
                record["vid"],
                record["mid"],
                record.get("source_text"),
                time.time(),
                record.get("raw_html"),
                json.dumps(record["languages"], ensure_ascii=False),
            ),
        )
        self.db.commit()

    def put_text(self, vid: int, source_text: str | None, records: list[dict],
                 *, refresh: bool = False) -> None:
        """Store every segment of a continuous-view harvest in one transaction
        and mark the text as fetched. Skips if already harvested unless refresh."""
        if not refresh and self.has_text(vid):
            return
        now = time.time()
        self.db.executemany(
            "INSERT OR REPLACE INTO records "
            "(uid, vid, mid, source_text, fetched_at, raw_html, parsed_json) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (r["uid"], r["vid"], r["mid"], r.get("source_text"), now,
                 r.get("raw_html"), json.dumps(r["languages"], ensure_ascii=False))
                for r in records
            ],
        )
        self.db.execute(
            "INSERT OR REPLACE INTO texts (vid, source_text, n_segments, fetched_at) "
            "VALUES (?,?,?,?)",
            (vid, source_text, len(records), now),
        )
        self.db.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        return {
            "uid": row["uid"],
            "vid": row["vid"],
            "mid": row["mid"],
            "source_text": row["source_text"],
            "fetched_at": row["fetched_at"],
            "raw_html": row["raw_html"],
            "languages": json.loads(row["parsed_json"]),
        }
