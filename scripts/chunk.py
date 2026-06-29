#!/usr/bin/env python3
"""Step 3 — group segments into chunks of a useful size.

Segments carry paragraph numbers in the ID (``mn1:1.3`` and ``mn1:1.4`` ->
paragraph ``mn1:1``). Suttas segment very finely, though: a title-only ``:0``
paragraph plus many one-sentence paragraphs. Emitting one chunk per paragraph
produced tiny, keyword-only chunks (a bare sutta title would out-rank real
content). So within each sutta we pack consecutive segments greedily and break
only at a paragraph boundary once past config.MIN_CHUNK_CHARS, while never
exceeding config.MAX_CHUNK_CHARS (revision #5; an oversized single paragraph is
still sub-split, never silently truncated). Every constituent segment ID is
kept for citation.

Output: data/chunks.jsonl. Pure stdlib — runnable immediately.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

PARA_RE = re.compile(r"^(.*?):(\d+)\.")


def paragraph_key(segment_id: str) -> str:
    """'mn1:1.3' -> 'mn1:1'. Falls back to the full id if it doesn't match."""
    m = PARA_RE.match(segment_id)
    return f"{m.group(1)}:{m.group(2)}" if m else segment_id


def pack(segs: list[dict]) -> list[list[dict]]:
    """Greedily pack one sutta's segments into groups: never exceed
    MAX_CHUNK_CHARS, and only break at a paragraph boundary once the group is
    past MIN_CHUNK_CHARS — so single-sentence paragraphs merge with neighbours
    instead of becoming runts."""
    groups, cur, cur_len = [], [], 0
    for i, seg in enumerate(segs):
        seg_len = len(seg["english"])
        if cur and cur_len + seg_len > config.MAX_CHUNK_CHARS:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += seg_len

        last = i + 1 == len(segs)
        at_boundary = last or paragraph_key(segs[i + 1]["segment_id"]) != paragraph_key(
            seg["segment_id"]
        )
        if at_boundary and cur_len >= config.MIN_CHUNK_CHARS:
            groups.append(cur)
            cur, cur_len = [], 0
    if cur:
        groups.append(cur)
    return groups


def make_chunks(segs: list[dict]) -> list[dict]:
    """Build chunk records for one sutta, with unique paragraph-based ids."""
    groups = [g for g in pack(segs) if g]
    bases = [paragraph_key(g[0]["segment_id"]) for g in groups]
    repeated = {b for b, n in Counter(bases).items() if n > 1}

    out, dup_seen = [], {}
    for group, base in zip(groups, bases):
        if base in repeated:  # only oversized paragraphs that got sub-split
            dup_seen[base] = dup_seen.get(base, 0)
            chunk_id = f"{base}#{dup_seen[base]}"
            dup_seen[base] += 1
        else:
            chunk_id = base
        first = group[0]
        out.append({
            "chunk_id": chunk_id,
            "segment_ids": [s["segment_id"] for s in group],
            "pali": " ".join(s["pali"] for s in group if s["pali"]).strip(),
            "english": " ".join(s["english"] for s in group if s["english"]).strip(),
            "nikaya": first["nikaya"],
            "sutta_uid": first["sutta_uid"],
            "sutta_title": first["sutta_title"],
        })
    return out


def main() -> None:
    if not config.SEGMENTS_JSONL.exists():
        sys.exit(f"missing {config.SEGMENTS_JSONL}; run extract_segments.py first")

    n_chunks = 0
    cur_uid = None
    buf: list[dict] = []
    with config.SEGMENTS_JSONL.open() as f, config.CHUNKS_JSONL.open("w") as out:
        def emit():
            nonlocal n_chunks
            for chunk in make_chunks(buf):
                # skip chunks with no usable English (e.g. blank segments)
                if not chunk["english"]:
                    continue
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1

        for line in f:
            seg = json.loads(line)
            if seg["sutta_uid"] != cur_uid and buf:
                emit()
                buf = []
            cur_uid = seg["sutta_uid"]
            buf.append(seg)
        if buf:
            emit()

    print(f"Wrote {n_chunks} chunks -> {config.CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
