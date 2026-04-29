"""Tests for the semantic cache."""

import numpy as np
import pytest
from app.cache import SemanticCache


def make_vec(values):
    v = np.array(values, dtype="float32")
    return v / np.linalg.norm(v)


class TestSemanticCache:
    def test_miss_on_empty_cache(self):
        cache = SemanticCache(similarity_threshold=0.90)
        v = make_vec([1.0, 0.0, 0.0])
        assert cache.get(v) is None

    def test_exact_hit(self):
        cache = SemanticCache(similarity_threshold=0.90)
        v = make_vec([1.0, 0.5, 0.3])
        cache.put(v, "cached response")
        result = cache.get(v.copy())
        assert result == "cached response"

    def test_near_duplicate_hit(self):
        cache = SemanticCache(similarity_threshold=0.92)
        v1 = make_vec([1.0, 0.01, 0.0])
        v2 = make_vec([1.0, 0.02, 0.0])  # very similar
        cache.put(v1, "response A")
        assert cache.get(v2) == "response A"

    def test_dissimilar_miss(self):
        cache = SemanticCache(similarity_threshold=0.92)
        v1 = make_vec([1.0, 0.0, 0.0])
        v2 = make_vec([0.0, 1.0, 0.0])  # orthogonal
        cache.put(v1, "response A")
        assert cache.get(v2) is None

    def test_eviction_when_full(self):
        cache = SemanticCache(max_size=2, similarity_threshold=0.90)
        v1 = make_vec([1.0, 0.0, 0.0])
        v2 = make_vec([0.0, 1.0, 0.0])
        v3 = make_vec([0.0, 0.0, 1.0])
        cache.put(v1, "r1")
        cache.put(v2, "r2")
        cache.put(v3, "r3")  # should evict v1
        assert cache.size == 2
        # v1 should be evicted, v3 should be there
        assert cache.get(v3.copy()) == "r3"

    def test_hit_rate_accounting(self):
        cache = SemanticCache(similarity_threshold=0.90)
        v = make_vec([1.0, 0.2, 0.3])
        cache.put(v, "response")
        cache.get(v.copy())  # hit
        cache.get(make_vec([0.0, 1.0, 0.0]))  # miss
        assert cache.hits == 1
        assert cache.misses == 1  # 1 miss on dissimilar vector
        assert abs(cache.hit_rate - 1 / 2) < 0.01
