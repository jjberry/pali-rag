"""Bibliotheca Polyglotta / TLB pilot — live-search-as-tool retrieval.

This package is an *exploration* separate from the working v1 Pāli-canon
pipeline. It does not scrape-and-index the TLB corpus; instead it treats
Bibliotheca Polyglotta's own server-side search as the relevance step
(dissolving the "which texts do we fetch?" chicken-and-egg), fetches only the
matching records, and re-ranks them locally with the same embedder v1 uses.

Every fetched record is written to an immutable, never-expiring, *gitignored*
SQLite cache (scholarly edition text does not change), so repeat queries hit
disk instead of BP and load on their servers falls over time. See
bp/README.md for the full design and the licensing bright line.

Modules:
  cache.py   immutable record cache (SQLite, keyed by BP permalink UUID)
  client.py  polite BP HTTP client: search() + fetch_record(), cache-backed
  pilot.py   expand -> search -> fetch -> JIT semantic rerank -> answer
"""
