"""
RAG Pipeline  –  Orchestrates the full query lifecycle

Flow
----
  User query
    │
    ├─ [1] Injection guard  →  reject if attack detected
    │
    ├─ [2] Semantic cache lookup  →  return cached response if hit
    │
    ├─ [3] Hybrid retrieval (FAISS + BM25 + RRF)
    │
    ├─ [4] Domain relevance check  →  reject if top-chunk score < threshold
    │       (uses raw FAISS cosine similarity, not RRF-normalised score)
    │
    ├─ [5] Multi-model routing  →  fast tier (Haiku) or smart tier (GPT-4o-mini)
    │       with automatic fallback if primary model fails
    │
    ├─ [6] Update semantic cache
    │
    └─ [7] Record metrics  →  return QueryResponse
"""

from __future__ import annotations

import logging
import time
from typing import Union

from app.cache import SemanticCache
from app.config import Settings
from app.models import (
    HealthResponse,
    MetricsSummary,
    QueryRequest,
    QueryResponse,
    RejectedResponse,
    RejectionReason,
    SourceChunk,
)
from app.rag.embeddings import EmbeddingIndex
from app.rag.retrieval import HybridRetriever
from app.rag.router import ModelRouter
from app.security.injection import InjectionGuard

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self._settings = settings

        self._emb_index = EmbeddingIndex(
            model_name=settings.embedding_model,
            index_dir=settings.index_dir,
            embedding_dim=settings.embedding_dim,
        )
        self._retriever = HybridRetriever(
            embedding_index=self._emb_index,
            index_dir=settings.index_dir,
        )
        self._router = ModelRouter(
            api_key=settings.openrouter_api_key,
            fast_model=settings.fast_model,
            smart_model=settings.smart_model,
            fallback_model=settings.fallback_model,
        )
        self._cache = SemanticCache(
            max_size=settings.cache_max_size,
            similarity_threshold=settings.cache_semantic_threshold,
        )
        self._guard = InjectionGuard()

        # Metrics
        self._total_queries = 0
        self._rejected_queries = 0
        self._fallback_used = 0
        self._total_response_time_ms = 0.0
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        faiss_ok = self._emb_index.load()
        bm25_ok = self._retriever.load_bm25()
        if faiss_ok and bm25_ok:
            logger.info("Indexes loaded.  %d chunks ready.", self._emb_index.total_chunks)
        else:
            logger.warning("Index not found on disk.  Run scripts/ingest.py first.")

    @property
    def is_ready(self) -> bool:
        return self._retriever.is_ready

    # ------------------------------------------------------------------
    # Main query handler
    # ------------------------------------------------------------------

    async def query(
        self, request: QueryRequest
    ) -> Union[QueryResponse, RejectedResponse]:
        start = time.perf_counter()
        self._total_queries += 1

        # ── 1. Injection guard ──────────────────────────────────────────
        detection = self._guard.detect(request.question)
        if detection.is_injection:
            elapsed = (time.perf_counter() - start) * 1000
            self._rejected_queries += 1
            logger.warning("Injection detected: %s", detection.explanation)
            return RejectedResponse(
                reason=RejectionReason.INJECTION_DETECTED,
                message=(
                    "Your query appears to contain instructions that attempt to "
                    "alter or override this assistant's behaviour.  Please ask a "
                    "genuine question about the knowledge base."
                ),
                response_time_ms=elapsed,
            )

        # ── 2. Semantic cache lookup ────────────────────────────────────
        query_vec = self._emb_index.embed_query(request.question)
        cached = self._cache.get(query_vec)
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            self._total_response_time_ms += elapsed
            cached.response_time_ms = elapsed
            cached.cached = True
            return cached

        # ── 3. Hybrid retrieval ─────────────────────────────────────────
        if not self.is_ready:
            return RejectedResponse(
                reason=RejectionReason.OUT_OF_SCOPE,
                message="Knowledge base index is not loaded.  Contact the administrator.",
                response_time_ms=(time.perf_counter() - start) * 1000,
            )

        raw_dense = self._emb_index.search(request.question, top_k=1)
        top_raw_score = raw_dense[0][1] if raw_dense else 0.0

        chunks = self._retriever.retrieve(
            request.question,
            top_k_dense=self._settings.top_k_dense,
            top_k_sparse=self._settings.top_k_sparse,
            top_k_final=request.top_k,
        )

        # ── 4. Domain relevance check ───────────────────────────────────
        if top_raw_score < self._settings.domain_similarity_threshold:
            elapsed = (time.perf_counter() - start) * 1000
            self._rejected_queries += 1
            logger.info(
                "Out-of-scope query rejected (top_score=%.3f < threshold=%.3f)",
                top_raw_score, self._settings.domain_similarity_threshold,
            )
            return RejectedResponse(
                reason=RejectionReason.OUT_OF_SCOPE,
                message=(
                    "Your question does not appear to be related to the available "
                    "knowledge base.  Please ask questions about AI, LLMs, RAG, "
                    "embeddings, prompt engineering, or LLM evaluation."
                ),
                response_time_ms=elapsed,
            )

        # ── 5. Multi-model routing + generation ────────────────────────
        routed = await self._router.generate(request.question, chunks)
        self._total_tokens += routed.tokens
        if routed.used_fallback:
            self._fallback_used += 1
            logger.warning("Fallback model used for query: %s", request.question[:60])

        # ── 6. Build response ───────────────────────────────────────────
        elapsed = (time.perf_counter() - start) * 1000
        self._total_response_time_ms += elapsed

        response = QueryResponse(
            answer=routed.answer,
            sources=[
                SourceChunk(content=c.content[:300], source=c.source, score=round(s, 3))
                for c, s in chunks
            ],
            response_time_ms=round(elapsed, 1),
            cached=False,
            tokens_used=routed.tokens,
            model_used=routed.model_used,
            routing_tier=routed.tier,
            used_fallback=routed.used_fallback,
        )

        # ── 7. Update cache ─────────────────────────────────────────────
        self._cache.put(query_vec, response)
        return response

    # ------------------------------------------------------------------
    # Health & metrics
    # ------------------------------------------------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok" if self.is_ready else "degraded",
            version=self._settings.app_version,
            documents_indexed=len(
                set(c.source for c in self._emb_index._chunks)
            ) if self._emb_index._chunks else 0,
            chunks_indexed=self._emb_index.total_chunks,
            index_ready=self.is_ready,
        )

    def metrics(self) -> MetricsSummary:
        answered = self._total_queries - self._rejected_queries
        return MetricsSummary(
            total_queries=self._total_queries,
            rejected_queries=self._rejected_queries,
            cache_hits=self._cache.hits,
            avg_response_time_ms=round(
                self._total_response_time_ms / max(1, self._total_queries), 1
            ),
            avg_tokens_used=round(
                self._total_tokens / max(1, answered), 1
            ),
        )
