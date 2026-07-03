"""Polite, cache-backed HTTP client for Bibliotheca Polyglotta.

BP is an old PHP site whose reading pages are an AJAX shell, but two endpoints
render server-side and parse cleanly with the stdlib (no headless browser):

  search:  index.php?page=search&sokefelt=<term>&searchmode=word|phrase|phraseregex|free
                     &context=a|v<vid>&Sok=Search
  record:  index.php?page=record&view=record&vid=<vid>&mid=<mid>

A record view returns one segment aligned across every available language, each
line tagged with an edition+page citation and a stable permalink UUID, e.g.
  San: Minayeff (1889) 157,26   bodhicaryāvatāre bodhicittānuśaṃsaḥ ...
  Eng: Barnett (1947) 37,1-2    Chapter I: The Praise of the Thought of Enlightenment
  uid=a61b0e1a-6bec-11df-...

Parsing is heuristic (regex over the tag-stripped token stream), which is why
we keep the raw HTML in the cache — if the parser improves we can re-parse
without re-fetching.
"""
from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request

from . import config as C
from .cache import RecordCache

# An edition tag: a language-family prefix (optionally qualified, e.g. "Tib Dūn")
# then ": ..." — captured up to the segment text that follows it in the stream.
_LANG_ALT = "|".join(re.escape(p) for p in C.LANG_PREFIX)
_EDITION_RE = re.compile(rf"^((?:{_LANG_ALT})\b[^:]*):\s*(.*)$")
_UID_RE = re.compile(r"uid=([0-9a-f-]{8,})")
_RECORD_HREF_RE = re.compile(r"page=record[^'\"]*?vid=(\d+)[^'\"]*?mid=(\d+)")

# Continuous-view ("Complete text") structure: one aligned segment per
# BolkContainer; within it, each language column is a div carrying the language
# name as its class, wrapping a <span id='<n>bi' class='...'>text</span>. The
# span's class marks the *content type* and varies by text — 'paragraph' for
# prose but 'verse' for verse texts (Bca, Buddhacarita, Dhammapada), plus
# 'maintitle'/'subtitle'/'chaptertitle' for headings — so we key on the stable
# id='<n>bi' marker, not the class. This view renders server-side for *every*
# TLB text (including ones whose per-segment record view is AJAX-only).
_BOLK_RE = re.compile(r"<div class='BolkContainer'>(.*?)<div class='clear'></div></div>", re.S)
_LANG_SPAN_RE = re.compile(
    r"<div class='(Sanskrit|Chinese|Tibetan|English|Pali|Pāli|Mongolian|French|German)'>"
    r".*?<span id='(\d+)bi'[^>]*?class='([^']*)'[^>]*>(.*?)</span>", re.S)


class BPClient:
    def __init__(self, cache: RecordCache | None = None) -> None:
        self.cache = cache or RecordCache()
        self._last_request = 0.0

    # --- low-level fetch (rate-limited, identified) --------------------------
    def _get(self, params: dict) -> str:
        # Space out live requests so we never hammer BP.
        wait = C.MIN_REQUEST_INTERVAL - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        url = C.BASE + "index.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": C.USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        self._last_request = time.time()
        return body

    # --- search (never cached: recall must stay fresh) -----------------------
    def search(self, term: str, *, vid: int | None = None, mode: str = "word",
               limit: int = 25) -> list[tuple[int, int]]:
        """Return up to `limit` (vid, mid) record hits for `term`. `vid=None`
        searches all libraries (context=a); a vid restricts to one text."""
        params = {
            "page": "search", "sokefelt": term, "searchmode": mode,
            "context": f"v{vid}" if vid else "a", "Sok": "Search",
        }
        page_html = self._get(params)
        seen: list[tuple[int, int]] = []
        for m in _RECORD_HREF_RE.finditer(page_html):
            hit = (int(m.group(1)), int(m.group(2)))
            if hit not in seen:
                seen.append(hit)
            if len(seen) >= limit:
                break
        return seen

    # --- record fetch (cache-first) ------------------------------------------
    def fetch_record(self, vid: int, mid: int, *, source_text: str | None = None,
                     refresh: bool = False) -> dict:
        """Return the aligned multilingual record for (vid, mid), from cache if
        present (immutable content), otherwise fetch + parse + cache."""
        if not refresh:
            cached = self.cache.get_by_mid(vid, mid)
            if cached is not None:
                return cached
        raw = self._get({"page": "record", "view": "record", "vid": vid, "mid": mid})
        record = self._parse_record(raw, vid, mid, source_text)
        self.cache.put(record, refresh=refresh)
        return record

    # --- whole-text harvest via the continuous view --------------------------
    def fetch_text(self, vid: int, *, source_text: str | None = None,
                   refresh: bool = False) -> list[dict]:
        """Harvest an entire text from its continuous ("Complete text") view in
        one request, parsing every aligned segment. Cache-first: returns cached
        segments if the text was already harvested. This is the uniform path
        that also works for texts whose per-segment record view is AJAX-only."""
        if not refresh and self.cache.has_text(vid):
            return self.cache.get_text_segments(vid)
        raw = self._get({"page": "fulltext", "view": "fulltext", "vid": vid})
        records = self._parse_fulltext(raw, vid, source_text)
        self.cache.put_text(vid, source_text, records, refresh=refresh)
        return records

    @classmethod
    def _parse_fulltext(cls, page_html: str, vid: int,
                        source_text: str | None) -> list[dict]:
        """Parse a continuous-view page into per-segment records. Each segment
        gets a synthetic stable uid ('vid:seg:i') and mid (its index), since the
        continuous view exposes paragraph span-ids rather than permalink UUIDs."""
        records = []
        for i, block in enumerate(_BOLK_RE.findall(page_html)):
            languages: dict[str, list[dict]] = {}
            for lang, span_id, kind, raw_text in _LANG_SPAN_RE.findall(block):
                text = html.unescape(re.sub(r"<[^>]+>", " ", raw_text))
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                lang = "Pali" if lang == "Pāli" else lang
                languages.setdefault(lang, []).append(
                    {"edition": source_text or str(vid), "text": text,
                     "span_id": span_id, "kind": kind}  # kind: verse|paragraph|*title
                )
            if not languages:
                continue
            records.append({
                "uid": f"{vid}:seg:{i}", "vid": vid, "mid": i,
                "source_text": source_text, "raw_html": None,
                "languages": languages,
            })
        return records

    # --- parsing (per-segment record view) -----------------------------------
    @staticmethod
    def _tokenize(page_html: str) -> list[str]:
        """Tag-stripped, unescaped, whitespace-collapsed token stream — the same
        view used to eyeball the record structure by hand."""
        body = re.sub(r"<script.*?</script>", "", page_html, flags=re.S)
        parts = re.split(r"<[^>]+>", body)
        toks = []
        for p in parts:
            t = html.unescape(p).strip()
            t = re.sub(r"\s+", " ", t)
            if t:
                toks.append(t)
        return toks

    @classmethod
    def _parse_record(cls, page_html: str, vid: int, mid: int,
                      source_text: str | None) -> dict:
        toks = cls._tokenize(page_html)
        uid_m = _UID_RE.search(page_html)
        uid = uid_m.group(1) if uid_m else f"{vid}:{mid}"  # fallback pseudo-key

        # Walk the stream: an edition-tag token is followed by its segment text.
        languages: dict[str, list[dict]] = {}
        i = 0
        while i < len(toks):
            em = _EDITION_RE.match(toks[i])
            if em and i + 1 < len(toks):
                citation = toks[i].strip()              # 'Eng: Barnett (1947) 37,1-2'
                family = em.group(1).split()[0]         # 'Eng'
                lang = C.LANG_PREFIX.get(family, family)
                text = toks[i + 1].strip()
                # Skip if the "text" is actually the next edition tag (empty line).
                if not _EDITION_RE.match(text):
                    languages.setdefault(lang, []).append(
                        {"edition": citation, "text": text}
                    )
                    i += 2
                    continue
            i += 1

        return {
            "uid": uid, "vid": vid, "mid": mid, "source_text": source_text,
            "raw_html": page_html, "languages": languages,
        }
