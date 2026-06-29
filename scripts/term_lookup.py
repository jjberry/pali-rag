#!/usr/bin/env python3
"""Term-archaeology helper — expand a Pāli term to its inflected/sandhi forms.

Revision #4: naive substring search over the Pāli field fails because Pāli is
heavily inflected and uses sandhi (vedanā -> vedanānaṁ, fused into compounds).
This module uses the local DPD database (~/dpd.db) to expand a headword into
the surface forms to actually search for in the corpus.

Read-only against dpd.db via stdlib sqlite3.

DPD pre-computes the surface forms for every headword in the `inflections`
column (a comma-separated list, e.g. vedanā -> vedanā,vedanāyo,vedanaṃ,…), so
we read those directly rather than re-deriving them from the encoded
`inflection_templates` grid (stem + ending per cell). Same result, no parsing.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def connect() -> sqlite3.Connection:
    if not config.DPD_DB.exists():
        sys.exit(f"missing DPD database: {config.DPD_DB}")
    con = sqlite3.connect(f"file:{config.DPD_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def headwords(con: sqlite3.Connection, lemma: str) -> list[sqlite3.Row]:
    """Dictionary entries whose lemma matches (DPD strips trailing digits on
    homonyms, e.g. 'dhamma 1'); match on the lemma stem."""
    return con.execute(
        "SELECT id, lemma_1, pos, grammar, pattern, meaning_1, inflections "
        "FROM dpd_headwords WHERE lemma_1 = ? OR lemma_1 LIKE ? "
        "ORDER BY id",
        (lemma, f"{lemma} %"),
    ).fetchall()


def inflected_forms(hw: sqlite3.Row) -> list[str]:
    """Surface forms for a headword, read from DPD's pre-computed comma-
    separated `inflections` column. Indeclinables may have an empty column."""
    raw = hw["inflections"]
    if not raw:
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def deconstructions(con: sqlite3.Connection, surface: str) -> list[str]:
    """Sandhi/compound splits for a surface form, from lookup.deconstructor
    (a JSON array of ' + '-joined splits)."""
    row = con.execute(
        "SELECT deconstructor FROM lookup WHERE lookup_key = ? AND deconstructor != ''",
        (surface,),
    ).fetchone()
    if not row:
        return []
    try:
        return json.loads(row["deconstructor"])
    except (ValueError, TypeError):
        return []


def expand(term: str) -> set[str]:
    """All surface forms worth searching the corpus for, given a headword."""
    con = connect()
    try:
        forms = {term}
        for hw in headwords(con, term):
            forms.update(inflected_forms(hw))
        return forms
    finally:
        con.close()


# Pāli words only: unicode letters (incl. diacritics ā ī ṅ ñ ṭ ḍ ṇ ḷ ṃ …),
# no digits/punctuation. Used to whole-word match rather than substring match,
# which is the whole point of expansion (design rule #4: 'vedana' must not
# match inside 'vedanākkhandha' by accident).
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize_pali(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text)}


def search_corpus(forms: set[str], limit: int | None = None) -> list[dict]:
    """Scan the segment corpus for segments whose Pāli contains one of the
    surface forms as a whole word. Returns occurrence records with the exact
    segment ID for citation.

    Note: this catches inflected forms but not sandhi-fused compounds (e.g.
    'sati' inside 'satova'); splitting those needs lookup.deconstructor per
    corpus token (see deconstructions) — left as a future enhancement.
    """
    if not config.SEGMENTS_JSONL.exists():
        sys.exit(f"missing {config.SEGMENTS_JSONL}; run extract_segments.py first")
    forms = {f.lower() for f in forms}
    hits: list[dict] = []
    with config.SEGMENTS_JSONL.open() as f:
        for line in f:
            seg = json.loads(line)
            matched = tokenize_pali(seg["pali"]) & forms
            if matched:
                hits.append({
                    "segment_id": seg["segment_id"],
                    "sutta_uid": seg["sutta_uid"],
                    "nikaya": seg["nikaya"],
                    "matched": sorted(matched),
                    "pali": seg["pali"],
                    "english": seg["english"],
                })
                if limit and len(hits) >= limit:
                    break
    return hits


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: term_lookup.py <pali-term>")
    term = sys.argv[1]
    con = connect()
    try:
        rows = headwords(con, term)
        if not rows:
            print(f"No DPD headword found for {term!r}.")
        for hw in rows:
            print(f"{hw['lemma_1']}  [{hw['pos']}]  pattern={hw['pattern']}")
            print(f"    {hw['meaning_1']}")
            forms = inflected_forms(hw)
            if forms:
                print(f"    {len(forms)} inflected forms: {', '.join(forms)}")
    finally:
        con.close()

    all_forms = expand(term)
    print(f"\nExpanded to {len(all_forms)} surface form(s); searching the Pāli field…")

    hits = search_corpus(all_forms)
    if not hits:
        print(f"No occurrences of {term!r} (or its inflections) in the corpus.")
        return

    by_nikaya = Counter(h["nikaya"] for h in hits)
    spread = ", ".join(f"{nik} {n}" for nik, n in sorted(by_nikaya.items()))
    print(f"{len(hits)} occurrence(s) across {len(by_nikaya)} Nikāya(s): {spread}\n")

    shown = 25
    for h in hits[:shown]:
        forms = "/".join(h["matched"])
        print(f"  {h['segment_id']:<16} ({forms})")
        print(f"      {h['pali']}")
        print(f"      {h['english']}")
    if len(hits) > shown:
        print(f"\n  … and {len(hits) - shown} more occurrence(s).")


if __name__ == "__main__":
    main()
