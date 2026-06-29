#!/usr/bin/env python3
"""Step 3 — group segments into paragraph-level chunks.

Segments are grouped by the major paragraph number in the segment ID
(``mn1:1.3`` and ``mn1:1.4`` -> paragraph ``mn1:1``). Each chunk keeps every
constituent segment ID for citation, the combined Pāli and English text, and
sutta metadata.

Revision #5: a hard size cap (config.MAX_CHUNK_CHARS) sub-splits any oversized
paragraph so it is not silently truncated when embedded.

Output: data/chunks.jsonl. Pure stdlib — runnable immediately.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

PARA_RE = re.compile(r"^(.*?):(\d+)\.")


def paragraph_key(segment_id: str) -> str:
    """'mn1:1.3' -> 'mn1:1'. Falls back to the full id if it doesn't match."""
    m = PARA_RE.match(segment_id)
    return f"{m.group(1)}:{m.group(2)}" if m else segment_id


def flush(buf: list[dict], para: str) -> list[dict]:
    """Turn a buffer of segments (one paragraph) into one or more chunks,
    sub-splitting on the char cap while never breaking a segment."""
    chunks, cur, cur_len = [], [], 0
    for seg in buf:
        seg_len = len(seg["english"])
        if cur and cur_len + seg_len > config.MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += seg_len
    if cur:
        chunks.append(cur)

    out = []
    for i, group in enumerate(chunks):
        first = group[0]
        suffix = f"#{i}" if len(chunks) > 1 else ""
        out.append({
            "chunk_id": para + suffix,
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
    cur_para = None
    buf: list[dict] = []
    with config.SEGMENTS_JSONL.open() as f, config.CHUNKS_JSONL.open("w") as out:
        def emit():
            nonlocal n_chunks
            for chunk in flush(buf, cur_para):
                # skip chunks with no usable English (e.g. blank segments)
                if not chunk["english"]:
                    continue
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1

        for line in f:
            seg = json.loads(line)
            para = paragraph_key(seg["segment_id"])
            if para != cur_para and buf:
                emit()
                buf = []
            cur_para = para
            buf.append(seg)
        if buf:
            emit()

    print(f"Wrote {n_chunks} chunks -> {config.CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
