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

import argparse
import json
import os
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

# Diacritics that mark a token as unmistakably Pāli (a query word carrying one
# is expanded even if short). ASCII Pāli loanwords (dukkha, sati, dhamma) are
# caught by the headword lookup + length guard instead.
PALI_DIACRITICS = set("āīūṁṃṅñṭḍṇḷṛṝḹ")

# Pāli terms so ubiquitous in the canon that their glosses are noise for
# retrieval (proper nouns / framing vocatives), not the conceptual terms a
# user is asking about. The Pāli analogue of QUERY_STOPWORDS.
PALI_SKIP = {
    "buddha", "bhagava", "bhagavant", "bhagavā", "tathāgata", "bhikkhu",
    "bhikkhave", "bhikkhū", "bhante", "āvuso", "āyasma", "sutta", "nikāya",
}

# Common English words long enough to clear the length guard; never expanded
# even on the off chance one is also a DPD headword.
QUERY_STOPWORDS = {
    "about", "after", "again", "against", "another", "because", "been", "being",
    "between", "both", "could", "does", "doing", "down", "during", "each",
    "ever", "from", "have", "having", "here", "into", "just", "like", "made",
    "make", "many", "mean", "meant", "more", "most", "much", "must", "only",
    "other", "over", "said", "same", "should", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "very", "were", "what", "when", "where",
    "which", "while", "with", "would", "your",
}


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


def compound_forms(base_forms: set[str]) -> set[str]:
    """Sandhi-fused compound surface forms in which one of `base_forms` is a
    component, via lookup.deconstructor (e.g. searching 'satipaṭṭhāna' also
    finds 'satipaṭṭhānasuttaṃ', which whole-word matching alone misses).

    The deconstructor column is a JSON array of ' + '-joined candidate splits;
    we keep a compound if any split has a component exactly equal to one of our
    forms. A single LIKE on the forms' common prefix narrows 1.3M rows cheaply;
    the exact-component test then removes coincidental substring matches.
    """
    forms = {f.lower() for f in base_forms}
    prefix = os.path.commonprefix(sorted(forms))
    patterns = [prefix] if len(prefix) >= 3 else list(forms)

    con = connect()
    try:
        rows: dict[str, str] = {}
        for pat in patterns:
            for r in con.execute(
                "SELECT lookup_key, deconstructor FROM lookup "
                "WHERE deconstructor LIKE ? AND deconstructor != ''",
                (f"%{pat}%",),
            ):
                rows[r["lookup_key"]] = r["deconstructor"]

        out: set[str] = set()
        for key, decon in rows.items():
            try:
                splits = json.loads(decon)
            except (ValueError, TypeError):
                continue
            for split in splits:
                parts = {p.strip().lower() for p in split.split("+")}
                if forms & parts:
                    out.add(key.lower())
                    break
        # Plain inflected forms are already handled by the whole-word pass.
        return out - forms
    finally:
        con.close()


def _is_pali_term(token: str) -> bool:
    """Heuristic: worth looking up as a Pāli term in an English query."""
    if token in QUERY_STOPWORDS or token in PALI_SKIP:
        return False
    return any(c in PALI_DIACRITICS for c in token) or len(token) >= 4


def expand_query(query: str) -> tuple[str, dict[str, list[str]]]:
    """Augment an English query with the DPD English glosses of any Pāli term
    it contains, so the English-only index can retrieve a Pāli technical term
    (e.g. 'dukkha' -> '… suffering; unease; unsatisfactoriness …').

    Returns (text_to_embed, {term: [glosses]}). The returned text is for
    retrieval only — the caller still answers the user's original question.
    On any DPD problem (e.g. no dpd.db) the query is returned unchanged.
    """
    if not config.DPD_DB.exists():
        return query, {}
    con = connect()
    try:
        found: dict[str, list[str]] = {}
        for token in sorted({t.lower() for t in WORD_RE.findall(query)}):
            if not _is_pali_term(token):
                continue
            glosses: list[str] = []
            for hw in headwords(con, token):
                meaning = (hw["meaning_1"] or "").strip()
                if meaning and meaning not in glosses:
                    glosses.append(meaning)
            if glosses:
                found[token] = glosses
    finally:
        con.close()

    if not found:
        return query, {}
    extra = "; ".join(g for glosses in found.values() for g in glosses)
    return f"{query} {extra}", found


def main() -> None:
    ap = argparse.ArgumentParser(description="DPD term-archaeology lookup")
    ap.add_argument("term", help="Pāli headword to trace")
    ap.add_argument(
        "--compounds",
        action="store_true",
        help="also match sandhi-fused compounds containing the term "
        "(via lookup.deconstructor), not just whole-word inflections",
    )
    args = ap.parse_args()
    term = args.term

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

    base = expand(term)
    comp: set[str] = set()
    if args.compounds:
        comp = compound_forms(base)
        print(
            f"\nExpanded to {len(base)} inflected + {len(comp)} compound "
            "surface form(s); searching the Pāli field…"
        )
    else:
        print(f"\nExpanded to {len(base)} surface form(s); searching the Pāli field…")

    hits = search_corpus(base | comp)
    if not hits:
        print(f"No occurrences of {term!r} (or its inflections) in the corpus.")
        return

    by_nikaya = Counter(h["nikaya"] for h in hits)
    spread = ", ".join(f"{nik} {n}" for nik, n in sorted(by_nikaya.items()))
    n_compound = sum(1 for h in hits if set(h["matched"]) & comp)
    extra = f" ({n_compound} via compounds)" if comp else ""
    print(
        f"{len(hits)} occurrence(s){extra} across {len(by_nikaya)} "
        f"Nikāya(s): {spread}\n"
    )

    shown = 25
    for h in hits[:shown]:
        tag = " [compound]" if set(h["matched"]) & comp else ""
        print(f"  {h['segment_id']:<16} ({'/'.join(h['matched'])}){tag}")
        print(f"      {h['pali']}")
        print(f"      {h['english']}")
    if len(hits) > shown:
        print(f"\n  … and {len(hits) - shown} more occurrence(s).")


if __name__ == "__main__":
    main()
