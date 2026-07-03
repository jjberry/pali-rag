"""BP pilot pipeline: expand -> live search -> fetch -> JIT rerank -> answer.

The retrieval flow, and why it differs from v1's index-everything approach:

  1. expand   Claude turns the question into lexical search terms (English +
              technical terms like 'karuṇā') — BP search is keyword, not
              semantic, so recall depends on good terms (cf. v1 DPD expansion).
  2. search   Query BP live per term; BP's own search is the relevance step,
              so we never need a local index to know which texts to fetch.
  3. fetch    Pull the full multilingual record for each candidate (cache-first).
  4. rerank   Embed the fetched *English* renderings with v1's model and score
              against the original question — restores semantic precision on the
              small candidate set (design rule #2: English-only for retrieval).
  5. answer   Claude answers over the top segments, citing mid / permalink /
              edition refs. The `tradition` facet is free: the source text.

Run:  python -m bp.pilot "What is the nature of bodhicitta?"
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from . import config as C
from .client import BPClient

_EXPAND_SYSTEM = (
    "You turn a user's question about Buddhist thought into search keywords for "
    "a LEXICAL (exact-word) search engine over English translations of Sanskrit "
    "Buddhist texts. Return 4-8 terms: common English words AND relevant "
    "technical terms (transliterated Sanskrit, e.g. karuṇā, bodhicitta). "
    "Respond ONLY as a JSON array of strings, no prose."
)


def expand_query(question: str, client) -> list[str]:
    """Ask Claude for lexical search terms. Falls back to the bare question."""
    resp = client.messages.create(
        model=C.GEN_MODEL,
        max_tokens=256,
        system=_EXPAND_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = resp.content[0].text.strip()
    try:
        terms = json.loads(text)
        return [t for t in terms if isinstance(t, str) and t.strip()]
    except (json.JSONDecodeError, TypeError):
        return [question]


def _english_candidates(record: dict) -> list[dict]:
    """One rerank candidate per English rendering of a segment (a segment may
    have several, e.g. Barnett + Matics), carrying enough to cite + show."""
    out = []
    for eng in record["languages"].get(C.RERANK_LANG, []):
        if eng["text"]:
            out.append({
                "uid": record["uid"], "vid": record["vid"], "mid": record["mid"],
                "source_text": record.get("source_text"),
                "edition": eng["edition"], "english": eng["text"],
                "span_id": eng.get("span_id"), "kind": eng.get("kind"),
                "languages": record["languages"],
            })
    return out


def _relevant_texts(terms: list[str], *, bp: BPClient,
                    texts: dict[str, int] | None, per_term: int,
                    max_texts: int) -> dict[int, str | None]:
    """Use BP's lexical search to decide *which texts* are worth harvesting —
    the relevance step that dissolves the chicken-and-egg. In curated mode a
    single hit marks a text relevant; in all-libraries mode we collect the vids
    of the segment hits. Capped at `max_texts`."""
    vids: dict[int, str | None] = {}
    if texts:
        for term in terms:
            for name, vid in texts.items():
                if vid not in vids and bp.search(term, vid=vid, limit=1):
                    vids[vid] = name
    else:
        for term in terms:
            for v, _mid in bp.search(term, limit=per_term):
                vids.setdefault(v, None)
    return dict(list(vids.items())[:max_texts])


def retrieve(question: str, *, anthropic_client, embed_model, bp: BPClient,
             texts: dict[str, int] | None = None, k: int = 8,
             per_term: int = 15, max_texts: int = 12) -> list[dict]:
    """Search-to-pick-texts, then harvest each relevant text whole (continuous
    view, cache-first) and rerank its English segments semantically against the
    question. Working at the text level fixes the AJAX-only record pages *and*
    lifts recall from lexical to semantic over each text's full content."""
    terms = expand_query(question, anthropic_client)
    print(f"[expand] {terms}", file=sys.stderr)

    vids = _relevant_texts(terms, bp=bp, texts=texts, per_term=per_term,
                           max_texts=max_texts)
    print(f"[texts] {len(vids)} relevant: {sorted(vids)}", file=sys.stderr)

    # Harvest each text once (one request, then cached) → English candidates.
    # Drop runt segments below the length floor (v1's MIN_CHUNK_CHARS analog):
    # a bare heading or one-word line can't out-rank real content if it's gone.
    candidates: list[dict] = []
    for vid, name in vids.items():
        for rec in bp.fetch_text(vid, source_text=name):
            for c in _english_candidates(rec):
                if len(c["english"]) >= C.MIN_SEGMENT_CHARS:
                    candidates.append(c)
    if not candidates:
        return []
    print(f"[rerank] {len(candidates)} English segments", file=sys.stderr)

    # Semantic rerank vs. the ORIGINAL question (not the expanded terms).
    import hashlib
    import numpy as np  # lazy; only needed when we actually have candidates
    qv = embed_model.encode([question], normalize_embeddings=True)[0].astype(np.float32)

    # Reuse cached segment embeddings: the vector of a fixed string under a fixed
    # model is immutable, so content-addressing (sha256 of the text) lets repeat
    # queries skip re-embedding everything and only embed the query + any segments
    # seen for the first time. This is the pipeline's main recurring cost.
    texts = [c["english"] for c in candidates]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    cached = bp.cache.get_vectors(hashes, C.EMBED_MODEL)
    dv = np.empty((len(texts), qv.shape[0]), dtype=np.float32)
    for i, h in enumerate(hashes):
        if h in cached:
            dv[i] = np.frombuffer(cached[h], dtype=np.float32)
    missing = [i for i, h in enumerate(hashes) if h not in cached]
    if missing:
        fresh = embed_model.encode(
            [texts[i] for i in missing], normalize_embeddings=True, batch_size=64,
        ).astype(np.float32)
        to_store, stored = [], set()
        for j, i in enumerate(missing):
            dv[i] = fresh[j]
            if hashes[i] not in stored:          # dedupe repeats within this batch
                to_store.append((hashes[i], fresh[j].shape[0], fresh[j].tobytes()))
                stored.add(hashes[i])
        bp.cache.put_vectors(to_store, C.EMBED_MODEL)
    print(f"[embed] {len(missing)} new, {len(texts) - len(missing)} cached",
          file=sys.stderr)

    scores = dv @ qv  # cosine; vectors are normalized
    # Down-weight heading segments so a topical chapter title can't out-rank the
    # verses beneath it (v1's gated-title analog) — penalized, not removed.
    penalty = np.array([
        C.HEADING_PENALTY if c.get("kind") in C.HEADING_KINDS else 0.0
        for c in candidates
    ])
    scores = scores - penalty
    order = np.argsort(scores)[::-1]

    ranked, seen = [], set()
    for idx in order:
        c = candidates[int(idx)]
        if c["uid"] in seen:                # keep best rendering per segment
            continue
        seen.add(c["uid"])
        c["score"] = float(scores[int(idx)])
        ranked.append(c)
        if len(ranked) >= k:
            break
    return ranked


def _cite(seg: dict) -> str:
    """A human-readable source line. Record-view hits carry a permalink UUID;
    continuous-view harvests cite text + segment index + paragraph span-id."""
    if re.fullmatch(r"[0-9a-f-]{8,}", seg["uid"]):
        return ("http://www2.hf.uio.no/common/apps/permlink/permlink.php?"
                f"app=polyglotta&context=record&uid={seg['uid']}")
    span = seg.get("span_id")
    where = f"seg {seg['mid']}" + (f", ¶{span}" if span else "")
    return f"{seg['source_text'] or seg['vid']} ({where}) — {C.BASE}"


def answer(question: str, *, hq: bool = False, all_libraries: bool = False,
           k: int = 8, max_texts: int = 12) -> None:
    """End-to-end pilot: retrieve then have Claude answer with citations."""
    import anthropic
    from sentence_transformers import SentenceTransformer

    client = anthropic.Anthropic()
    embed_model = SentenceTransformer(C.EMBED_MODEL)
    bp = BPClient()

    texts = None if all_libraries else C.TEXTS
    segs = retrieve(question, anthropic_client=client, embed_model=embed_model,
                    bp=bp, texts=texts, k=k, max_texts=max_texts)
    if not segs:
        sys.exit("no segments retrieved (try --all-libraries or rephrasing)")

    context = "\n\n".join(
        f"[{i+1}] ({s['source_text'] or s['vid']}; seg {s['mid']})\n{s['english']}"
        for i, s in enumerate(segs)
    )
    system = (
        "Answer using ONLY the provided TLB passages. Cite each claim with its "
        "[n] and the source text. These are Mahāyāna/secondary-literature "
        "sources; note where they differ from early-canon usage if relevant."
    )
    resp = client.messages.create(
        model=C.GEN_MODEL_HQ if hq else C.GEN_MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user",
                   "content": f"Question: {question}\n\nPassages:\n{context}"}],
    )
    print(resp.content[0].text)
    print("\n--- sources ---")
    for i, s in enumerate(segs):
        print(f"[{i+1}] {_cite(s)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bibliotheca Polyglotta RAG pilot")
    ap.add_argument("question")
    ap.add_argument("--hq", action="store_true", help="use the opus model")
    ap.add_argument("--all-libraries", action="store_true",
                    help="search all TLB texts, not just the curated subset")
    ap.add_argument("-k", type=int, default=8, help="segments to retrieve")
    ap.add_argument("--max-texts", type=int, default=12,
                    help="cap how many texts to harvest per question (default 12)")
    args = ap.parse_args()
    answer(args.question, hq=args.hq, all_libraries=args.all_libraries,
           k=args.k, max_texts=args.max_texts)


if __name__ == "__main__":
    main()
