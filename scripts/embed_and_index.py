#!/usr/bin/env python3
"""Step 4 + Phase 2 — embed chunks and build the ChromaDB index.

Revision #1: the corpus is ~30K chunks, so this runs in minutes on CPU and
ChromaDB is used for persistence/convenience, not scale.
Revision #2: embed ENGLISH ONLY; store Pāli + citations as metadata.

Requires: sentence-transformers, chromadb (see requirements.txt). Stub — the
embedding/indexing body is left to implement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_chunks() -> list[dict]:
    if not config.CHUNKS_JSONL.exists():
        sys.exit(f"missing {config.CHUNKS_JSONL}; run chunk.py first")
    return [json.loads(line) for line in config.CHUNKS_JSONL.open()]


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks for embedding with {config.EMBED_MODEL}")

    # TODO:
    #   from sentence_transformers import SentenceTransformer
    #   import chromadb
    #   model = SentenceTransformer(config.EMBED_MODEL)
    #   texts = [c["english"] for c in chunks]            # English only (#2)
    #   embs  = model.encode(texts, batch_size=64, show_progress_bar=True,
    #                        normalize_embeddings=True)
    #   client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    #   col = client.get_or_create_collection(config.CHROMA_COLLECTION,
    #                                         metadata={"hnsw:space": "cosine"})
    #   col.add(ids=[c["chunk_id"] for c in chunks],
    #           embeddings=embs.tolist(),
    #           documents=texts,
    #           metadatas=[{k: (json.dumps(v) if isinstance(v, list) else v)
    #                       for k, v in c.items() if k != "english"}
    #                      for c in chunks])
    raise SystemExit("embed_and_index.py is a stub — implement the body (see TODO).")


if __name__ == "__main__":
    main()
