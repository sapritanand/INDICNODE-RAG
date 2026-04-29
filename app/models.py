from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of source chunks to return")


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float = Field(description="Retrieval relevance score (0-1)")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    response_time_ms: float
    cached: bool = False
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None      # which model generated the answer
    routing_tier: Optional[str] = None    # "fast" | "smart"
    used_fallback: bool = False


class RejectionReason(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    INJECTION_DETECTED = "injection_detected"


class RejectedResponse(BaseModel):
    rejected: bool = True
    reason: RejectionReason
    message: str
    response_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
    chunks_indexed: int
    index_ready: bool


class MetricsSummary(BaseModel):
    total_queries: int
    rejected_queries: int
    cache_hits: int
    avg_response_time_ms: float
    avg_tokens_used: float
