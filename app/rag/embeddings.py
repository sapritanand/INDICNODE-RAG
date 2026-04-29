"""
Embedding layer – wraps sentence-transformers and FAISS.

Design decisions
----------------
Model: all-MiniLM-L6-v2 (384 dims)
  ~14 ms/query on CPU, good quality for English text, small footprint (~80 MB).
  Outperforms ada-002 on several MTEB benchmarks at 1/10 the cost.

Index: FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine sim)
  Exact search (no approximation) is acceptable at knowledge-base sizes
  < 100 k chunks.  Switching to IndexIVFFlat or HNSW is a one-line change if
  the corpus grows.

Persistence: index + metadata pickle to `index_dir` so the server starts
  instantly without re-embedding on every restart.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    INDEX_FILE = "vectors.faiss"
    META_FILE = "metadata.pkl"

    def __init__(self, model_name: str, index_dir: str, embedding_dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.index_dir = Path(index_dir)
        self.dim = embedding_dim
        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build a FAISS index from scratch."""
        logger.info("Embedding %d chunks …", len(chunks))
        texts = [c.content for c in chunks]
        vecs = self._embed_batch(texts)

        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs)

        self._index = index
        self._chunks = chunks
        self._save()
        logger.info("FAISS index built with %d vectors", index.ntotal)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return top-k (chunk, cosine_score) pairs."""
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = self._embed_single(query)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Return normalised 1-D embedding for a query string."""
        return self._embed_single(query)[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._save()

    def load(self) -> bool:
        """Return True if a persisted index was successfully loaded."""
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("Loaded FAISS index (%d vectors)", self._index.ntotal)
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return False

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
        with open(self.index_dir / self.META_FILE, "wb") as f:
            pickle.dump(self._chunks, f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def _embed_single(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
