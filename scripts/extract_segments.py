#!/usr/bin/env python3
"""Step 2 — extract aligned Pāli/English segment pairs from local sc-data.

Walks the Sujato English translation files and joins each to its Pāli root
file by shared segment ID (e.g. ``mn1:1.3``). This is an INNER join on the
English side: Pāli-only suttas (Vinaya/Abhidhamma, parts of KN) are dropped
by design — see coverage-gap note in the design doc (revision #3).

Output: data/segments.jsonl, one JSON object per segment:
    {segment_id, pali, english, nikaya, sutta_uid, sutta_title}

Pure stdlib — runnable immediately, no dependencies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def sutta_uid_from_path(p: Path) -> str:
    # ".../mn1_translation-en-sujato.json" -> "mn1"
    return p.name.split("_")[0]


def nikaya_from_uid(uid: str) -> str:
    # leading alpha run: "mn1" -> "mn", "sn22.59" -> "sn"
    i = 0
    while i < len(uid) and uid[i].isalpha():
        i += 1
    return uid[:i].upper()


def iter_segments():
    problems = config.check_data()
    if problems:
        sys.exit("Data layout problems:\n  " + "\n  ".join(problems))

    en_files = sorted(config.EN_SUJATO.rglob("*_translation-en-sujato.json"))
    paired = dropped = 0
    for en_path in en_files:
        uid = sutta_uid_from_path(en_path)
        # Mirror the full relative path (nesting can be deep, e.g. sn/sn1/),
        # swapping only the translation filename suffix for the root one.
        rel_dir = en_path.relative_to(config.EN_SUJATO).parent
        # The collection is the top-level dir (dn/mn/sn/an/kn). KN texts carry
        # their own UID prefixes (dhp, ud, iti…), so filter by directory.
        collection = rel_dir.parts[0] if rel_dir.parts else ""
        if config.COLLECTIONS and collection not in config.COLLECTIONS:
            continue
        pli_path = config.PALI_ROOT / rel_dir / f"{uid}_root-pli-ms.json"
        if not pli_path.exists():
            dropped += 1
            continue
        paired += 1
        en = json.loads(en_path.read_text())
        pli = json.loads(pli_path.read_text())
        nikaya = nikaya_from_uid(uid)
        # title: the ":0.2" segment is conventionally the sutta name.
        title = en.get(f"{uid}:0.2", "").strip()
        for seg_id, en_text in en.items():
            yield {
                "segment_id": seg_id,
                "pali": pli.get(seg_id, "").strip(),
                "english": en_text.strip(),
                "nikaya": nikaya,
                "sutta_uid": uid,
                "sutta_title": title,
            }
    print(f"  suttas paired: {paired}  (english files with no Pāli root: {dropped})",
          file=sys.stderr)


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    n = 0
    with config.SEGMENTS_JSONL.open("w") as out:
        for seg in iter_segments():
            out.write(json.dumps(seg, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} segments -> {config.SEGMENTS_JSONL}")


if __name__ == "__main__":
    main()
