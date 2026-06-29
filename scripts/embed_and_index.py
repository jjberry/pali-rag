#!/usr/bin/env python3
"""Step 4 + Phase 2 — embed chunks and build the ChromaDB index.

Revision #1: the corpus is ~30K chunks, so this runs in minutes on CPU and
ChromaDB is used for persistence/convenience, not scale.
Revision #2: embed ENGLISH ONLY; store Pāli + citations as metadata.

Requires: sentence-transformers, chromadb (see requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Encode + add in batches so we never hold all embeddings in memory at once
# and never exceed ChromaDB's per-call max batch size.
BATCH = 512


def load_chunks() -> list[dict]:
    if not config.CHUNKS_JSONL.exists():
        sys.exit(f"missing {config.CHUNKS_JSONL}; run chunk.py first")
    return [json.loads(line) for line in config.CHUNKS_JSONL.open()]


def to_metadata(chunk: dict) -> dict:
    """Everything except the English document becomes metadata.

    ChromaDB metadata values must be scalars, so lists are JSON-encoded
    (the retriever reverses this) and None values are dropped — the
    retriever treats a missing key the same as null.
    """
    meta: dict = {}
    for key, value in chunk.items():
        if key == "english":
            continue
        if value is None:
            continue
        meta[key] = json.dumps(value) if isinstance(value, list) else value
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reset",
        action="store_true",
        help="drop and rebuild the collection (default: refuse if it already "
        "has data, to avoid silent duplicate/stale rows)",
    )
    args = ap.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks; embedding English with {config.EMBED_MODEL}")

    from sentence_transformers import SentenceTransformer
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if args.reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass  # didn't exist yet

    col = client.get_or_create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    existing = col.count()
    if existing and not args.reset:
        sys.exit(
            f"collection '{config.CHROMA_COLLECTION}' already has {existing:,} "
            f"items at {config.CHROMA_DIR}. Re-run with --reset to rebuild."
        )

    model = SentenceTransformer(config.EMBED_MODEL)
    # Respect ChromaDB's add() ceiling if the build exposes it.
    batch = BATCH
    try:
        batch = min(batch, client.get_max_batch_size())
    except Exception:
        pass

    total = len(chunks)
    for start in range(0, total, batch):
        part = chunks[start : start + batch]
        texts = [c["english"] for c in part]  # English only (#2)
        embs = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=embs.tolist(),
            documents=texts,
            metadatas=[to_metadata(c) for c in part],
        )
        done = min(start + batch, total)
        print(f"  indexed {done:,}/{total:,}", end="\r", flush=True)

    print()
    print(f"Done. Collection '{config.CHROMA_COLLECTION}' has {col.count():,} items.")

    build_titles(client, model, chunks)


def build_titles(client, model, chunks: list[dict]) -> None:
    """One entry per sutta (its title), for the title-fusion retrieval signal."""
    titles: dict[str, str] = {}
    for c in chunks:
        uid = c["sutta_uid"]
        if uid not in titles and c.get("sutta_title"):
            titles[uid] = c["sutta_title"]
    if not titles:
        print("No sutta titles found; skipping title index.")
        return

    try:
        client.delete_collection(config.CHROMA_TITLES_COLLECTION)
    except Exception:
        pass
    tcol = client.get_or_create_collection(
        config.CHROMA_TITLES_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    uids = list(titles)
    embs = model.encode(
        [titles[u] for u in uids],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    tcol.add(
        ids=uids,
        embeddings=embs.tolist(),
        documents=[titles[u] for u in uids],
        metadatas=[{"sutta_uid": u} for u in uids],
    )
    print(f"Done. Collection '{config.CHROMA_TITLES_COLLECTION}' has {tcol.count():,} titles.")


if __name__ == "__main__":
    main()
