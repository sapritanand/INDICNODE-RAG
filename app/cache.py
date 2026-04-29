"""
Semantic Query Cache
---------------------
Design
------
Standard LRU caches key on the exact query string, which misses near-duplicate
questions ("What is RAG?" vs "Explain RAG to me").  A semantic cache uses
embedding cosine similarity to return a cached response for semantically
equivalent queries.

Algorithm:
  1. Embed the incoming query → q_vec
  2. Compute cosine similarity against all cached query embeddings
  3. If max(similarity) ≥ threshold (default 0.93) → cache hit, return stored response
  4. Otherwise → cache miss, process and store (q_vec, response) pair

Eviction: when the cache exceeds `max_size`, the oldest (FIFO) entry is evicted.
This is simpler than LRU but acceptable because semantic caches are dense — any
near-duplicate hits the entry regardless of recency.

Threshold choice:
  0.93 is conservative.  Two phrasings must be nearly semantically identical to
  hit.  Lowering to 0.85 increases hit rate but risks returning stale answers for
  subtly different questions.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.93):
        self._threshold = similarity_threshold
        self._max_size = max_size
        # OrderedDict preserves insertion order for FIFO eviction
        self._store: OrderedDict[int, Tuple[np.ndarray, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_vec: np.ndarray) -> Optional[Any]:
        """Return a cached response if a near-duplicate query exists, else None."""
        if not self._store:
            self._misses += 1
            return None

        best_sim, best_key = self._find_best_match(query_vec)
        if best_sim >= self._threshold:
            self._hits += 1
            logger.debug("Semantic cache HIT (similarity=%.3f)", best_sim)
            return self._store[best_key][1]

        self._misses += 1
        return None

    def put(self, query_vec: np.ndarray, response: Any) -> None:
        """Store a response keyed by query embedding."""
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # evict oldest

        key = id(query_vec)
        self._store[key] = (query_vec.copy(), response)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_best_match(self, query_vec: np.ndarray) -> Tuple[float, int]:
        best_sim = -1.0
        best_key = -1
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        for key, (cached_vec, _) in self._store.items():
            c = cached_vec / (np.linalg.norm(cached_vec) + 1e-10)
            sim = float(np.dot(q, c))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_sim, best_key
