#!/usr/bin/env python3
"""Term-archaeology helper — expand a Pāli term to its inflected/sandhi forms.

Revision #4: naive substring search over the Pāli field fails because Pāli is
heavily inflected and uses sandhi (vedanā -> vedanānaṁ, fused into compounds).
This module uses the local DPD database (~/dpd.db) to expand a headword into
the surface forms to actually search for in the corpus.

Read-only against dpd.db via stdlib sqlite3. The inflection-table expansion is
stubbed; the headword + deconstructor lookups are wired up as a starting point.
"""
from __future__ import annotations

import sqlite3
import sys
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
        "SELECT id, lemma_1, pos, grammar, pattern, meaning_1 "
        "FROM dpd_headwords WHERE lemma_1 = ? OR lemma_1 LIKE ? "
        "ORDER BY id",
        (lemma, f"{lemma} %"),
    ).fetchall()


def inflected_forms(con: sqlite3.Connection, pattern: str) -> list[str]:
    """Expand a declension/conjugation pattern into surface forms via
    inflection_templates. TODO: parse the template's encoded inflection grid
    (column 'data') and concatenate stem + endings for each cell."""
    # row = con.execute(
    #     "SELECT data FROM inflection_templates WHERE pattern = ?", (pattern,)
    # ).fetchone()
    # ... parse row["data"] (JSON inflection grid) -> list of endings ...
    return []


def deconstructions(con: sqlite3.Connection, surface: str) -> list[str]:
    """Sandhi/compound splits for a surface form, from lookup.deconstructor
    (a JSON array of ' + '-joined splits)."""
    row = con.execute(
        "SELECT deconstructor FROM lookup WHERE lookup_key = ? AND deconstructor != ''",
        (surface,),
    ).fetchone()
    if not row:
        return []
    import json
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
            if hw["pattern"]:
                forms.update(inflected_forms(con, hw["pattern"]))
        return forms
    finally:
        con.close()


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
    finally:
        con.close()
    print("\nNote: inflection expansion is stubbed (see inflected_forms TODO).")


if __name__ == "__main__":
    main()
