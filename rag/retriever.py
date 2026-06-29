"""Query interface over the ChromaDB index.

Embeds the English query with the same model used for indexing (#2) and
returns the top-k chunks with Pāli + citation metadata. Body-chunk hits are
fused (reciprocal rank fusion) with a parallel sutta-title index so that a
sutta whose body is terse/elided still surfaces when its topical title matches
the query (e.g. SN 22.59 'The Characteristic of Not-Self').
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

RRF_K = 60  # standard reciprocal-rank-fusion damping constant


class Retriever:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        self.model = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            self.col = client.get_collection(config.CHROMA_COLLECTION)
        except Exception:
            sys.exit(
                f"no '{config.CHROMA_COLLECTION}' collection at {config.CHROMA_DIR}; "
                "run scripts/embed_and_index.py first"
            )
        # Optional: absent on an index built before the title signal existed.
        try:
            self.titles = client.get_collection(config.CHROMA_TITLES_COLLECTION)
        except Exception:
            self.titles = None

    def query(self, text: str, k: int = config.TOP_K) -> list[dict]:
        # Same model + normalization as indexing (#2) so cosine is comparable.
        qv = self.model.encode([text], normalize_embeddings=True).tolist()

        # Body candidates: a pool wider than k so title-promoted suttas compete.
        pool = max(k * 2, 20)
        body = self.col.query(
            query_embeddings=qv,
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        scores: dict[str, float] = defaultdict(float)
        record: dict[str, dict] = {}
        for rank, (m, d, dist) in enumerate(
            zip(body["metadatas"][0], body["documents"][0], body["distances"][0])
        ):
            hit = self._rehydrate(m, d, dist)
            cid = hit["chunk_id"]
            scores[cid] += 1.0 / (RRF_K + rank)
            record[cid] = hit

        self._fuse_titles(qv, scores, record)

        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)[:k]
        return [record[c] for c in ranked]

    def _fuse_titles(self, qv, scores: dict, record: dict) -> None:
        """Add the best body chunk of each strongly title-matched sutta into the
        RRF pool, so a topical title can surface a sutta whose body ranks low."""
        if self.titles is None:
            return
        tres = self.titles.query(
            query_embeddings=qv,
            n_results=config.TITLE_FUSION_K,
            include=["metadatas", "distances"],
        )
        for rank, (meta, dist) in enumerate(
            zip(tres["metadatas"][0], tres["distances"][0])
        ):
            if dist > config.TITLE_MAX_DIST:
                continue  # only a clearly on-topic title may promote its sutta
            uid = meta["sutta_uid"]
            sub = self.col.query(
                query_embeddings=qv,
                n_results=1,
                where={"sutta_uid": uid},
                include=["documents", "metadatas", "distances"],
            )
            if not sub["ids"] or not sub["ids"][0]:
                continue
            hit = self._rehydrate(
                sub["metadatas"][0][0], sub["documents"][0][0], sub["distances"][0][0]
            )
            cid = hit["chunk_id"]
            scores[cid] += 1.0 / (RRF_K + rank)
            record.setdefault(cid, hit)

    @staticmethod
    def _rehydrate(meta: dict, document: str, distance: float | None = None) -> dict:
        """Undo the list->json encoding done at index time and re-attach the
        English document text."""
        out = dict(meta)
        if isinstance(out.get("segment_ids"), str):
            out["segment_ids"] = json.loads(out["segment_ids"])
        out["english"] = document
        if distance is not None:
            out["distance"] = distance
        return out
