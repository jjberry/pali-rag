"""Courtesy robots.txt / terms check for the live Bibliotheca Polyglotta endpoint.

BP offers no API and no explicit licence — it only describes itself as "open
access" — so before any *live* request we do the polite thing: consult the
host's robots.txt and refuse paths it disallows. Policy (RFC 9309-aligned, which
is deliberately *less* strict than Python's stdlib default here):

  * 2xx robots.txt  -> parse it and honour `can_fetch`; a real Disallow blocks us.
  * 4xx (absent)    -> no usable robots.txt, so no restrictions -> allowed.
    (The BP host actually returns 403 for /robots.txt — a legacy-Apache
    "missing file" quirk, not an intentional `Disallow: /`, which would be a
    200 with a body. The stdlib RobotFileParser treats 403 as disallow-all,
    which would wrongly refuse the openly-served site; we don't.)
  * 5xx / network   -> could not verify; proceed with a one-time warning rather
    than block on a transient failure.

The decision is computed once per host (memoised). Escape hatch: set
`BP_IGNORE_ROBOTS=1` to skip the check entirely. On the first live request we
also print a one-time notice reminding that fetched text is cached locally only
(never redistributed) and that the source's own terms should be respected.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
import urllib.robotparser
from functools import lru_cache
from urllib.parse import urlsplit

from . import config as C

ROBOTS_TIMEOUT = 15  # seconds; the check must not stall a session


class RobotsDisallowed(RuntimeError):
    """Raised when robots.txt explicitly disallows a path we were about to fetch."""


_notice_shown = False
_noted: set[str] = set()   # host_roots whose absent/error note we've printed


def _print_notice() -> None:
    global _notice_shown
    if _notice_shown:
        return
    _notice_shown = True
    print(
        "[bp] Using Bibliotheca Polyglotta live "
        f"({C.BASE}). Respecting robots.txt and a "
        f"{C.MIN_REQUEST_INTERVAL:g}s rate limit; fetched text is cached locally "
        "only and never redistributed. Review the source's own terms before "
        "heavy use.",
        file=sys.stderr,
    )


@lru_cache(maxsize=None)
def _parser_for(host_root: str, user_agent: str):
    """Fetch + parse robots.txt once per host. Returns (parser_or_None, note):
    a parser when there are real rules to honour, or None when robots.txt is
    absent/unverifiable (=> no restrictions). `note` is a one-line explanation
    for absent/error cases (printed once), else None."""
    robots_url = host_root + "/robots.txt"
    req = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=ROBOTS_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            return None, f"robots.txt returned {e.code}; treating as no restrictions"
        return None, f"robots.txt returned {e.code}; could not verify — proceeding"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, f"robots.txt fetch failed ({e}); could not verify — proceeding"

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser, None


def ensure_allowed(url: str, user_agent: str = C.USER_AGENT) -> None:
    """Gate a live fetch of `url`. Raises RobotsDisallowed if robots.txt forbids
    it; otherwise returns (printing the courtesy notice on first use). Cheap to
    call on every request — the robots.txt fetch/parse is memoised per host."""
    _print_notice()
    if os.environ.get("BP_IGNORE_ROBOTS"):
        return
    parts = urlsplit(url)
    host_root = f"{parts.scheme}://{parts.netloc}"
    parser, note = _parser_for(host_root, user_agent)
    if note:  # absent/unverifiable robots.txt -> allowed; explain once per host
        if host_root not in _noted:
            _noted.add(host_root)
            print(f"[bp] {note}", file=sys.stderr)
        return
    if not parser.can_fetch(user_agent, url):
        raise RobotsDisallowed(
            f"robots.txt at {host_root}/robots.txt disallows {url} for "
            f"'{user_agent}'. Set BP_IGNORE_ROBOTS=1 to override (at your own "
            "discretion)."
        )
