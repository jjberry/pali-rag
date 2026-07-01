"""Markdown -> HTML for the web UI.

Uses markdown-it-py (already present as a transitive dep, pinned in
requirements.txt) with GFM tables + strikethrough enabled so the answer tables
render. Raw HTML in the source is escaped by default, so rendering
Claude-generated content is safe.
"""
from __future__ import annotations

from functools import lru_cache

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark").enable(["table", "strikethrough"])


@lru_cache(maxsize=256)
def markdown_to_html(text: str) -> str:
    return _md.render(text)
