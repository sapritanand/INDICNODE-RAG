"""
Hybrid Retrieval  –  Dense (FAISS) + Sparse (BM25) → RRF fusion

Why hybrid?
-----------
Dense retrieval (embedding similarity) captures semantic meaning but misses
exact keyword matches ("FAISS", "BM25", proper nouns).  Sparse BM25 excels at
keyword overlap but fails for paraphrases.  Fusing both with Reciprocal Rank
Fusion (RRF) reliably outperforms either alone on most retrieval benchmarks
(BEIR, MS-MARCO) without requiring any fine-tuning.

Reciprocal Rank Fusion (Cormack 2009):
  rrf_score(d) = Σ  1 / (k + rank(d, list_i))
  where k = 60 (empirically optimal), summed over both ranking lists.
  The constant k dampens the influence of very high-ranked documents,
  making the fusion robust to outlier scores.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk
from app.rag.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)

_BM25_FILE = "bm25.pkl"
_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenisation for BM25."""
    return text.lower().split()


class HybridRetriever:
    def __init__(self, embedding_index: EmbeddingIndex, index_dir: str):
        self.emb = embedding_index
        self.index_dir = Path(index_dir)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        """Build BM25 index (FAISS is handled by EmbeddingIndex)."""
        corpus = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_chunks = chunks
        self._save_bm25()
        logger.info("BM25 index built with %d documents", len(chunks))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 5,
        top_k_sparse: int = 5,
        top_k_final: int = 3,
    ) -> List[Tuple[Chunk, float]]:
        """
        Return top_k_final (chunk, normalised_score) pairs via RRF fusion.
        Score is normalised to [0, 1] for readability.
        """
        dense = self.emb.search(query, top_k=top_k_dense)
        sparse = self._bm25_search(query, top_k=top_k_sparse)

        fused = self._rrf_fuse(dense, sparse)

        # Normalise scores to [0, 1]
        if fused:
            max_score = fused[0][1]
            if max_score > 0:
                fused = [(c, s / max_score) for c, s in fused]

        return fused[:top_k_final]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_bm25(self) -> bool:
        path = self.index_dir / _BM25_FILE
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._bm25_chunks = data["chunks"]
            logger.info("BM25 index loaded (%d docs)", len(self._bm25_chunks))
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25: %s", exc)
            return False

    def _save_bm25(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / _BM25_FILE, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._bm25_chunks}, f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._bm25_chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _rrf_fuse(
        self,
        dense: List[Tuple[Chunk, float]],
        sparse: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        scores: dict[int, float] = {}
        chunk_map: dict[int, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense):
            cid = id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunk_map[cid] = chunk

        for rank, (chunk, _) in enumerate(sparse):
            # Match sparse chunks to their dense counterparts by content hash
            cid = next(
                (k for k, c in chunk_map.items() if c.content == chunk.content),
                id(chunk),
            )
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in ranked]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and self.emb.is_ready
