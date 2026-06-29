"""Query interface over the ChromaDB index.

Embeds the English query with the same model used for indexing (#2) and
returns the top-k chunks with Pāli + citation metadata. Stub — fill in the
ChromaDB query body.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class Retriever:
    def __init__(self) -> None:
        # TODO:
        #   from sentence_transformers import SentenceTransformer
        #   import chromadb
        #   self.model = SentenceTransformer(config.EMBED_MODEL)
        #   client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        #   self.col = client.get_collection(config.CHROMA_COLLECTION)
        raise NotImplementedError("Retriever is a stub — wire up ChromaDB.")

    def query(self, text: str, k: int = config.TOP_K) -> list[dict]:
        # q = self.model.encode([text], normalize_embeddings=True)
        # res = self.col.query(query_embeddings=q.tolist(), n_results=k,
        #                      include=["documents", "metadatas"])
        # return [self._rehydrate(m, d) for m, d
        #         in zip(res["metadatas"][0], res["documents"][0])]
        raise NotImplementedError

    @staticmethod
    def _rehydrate(meta: dict, document: str) -> dict:
        """Undo the list->json encoding done at index time and re-attach the
        English document text."""
        out = dict(meta)
        if isinstance(out.get("segment_ids"), str):
            out["segment_ids"] = json.loads(out["segment_ids"])
        out["english"] = document
        return out
