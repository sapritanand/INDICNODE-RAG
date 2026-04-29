"""
FastAPI application entry point.

Startup: loads persisted FAISS and BM25 indexes from disk.
         If indexes are missing, the /health endpoint returns "degraded"
         and /query returns a clear error.  Run scripts/ingest.py first.

Endpoints
---------
  POST /query          Main RAG Q&A endpoint
  GET  /health         Index readiness + document count
  GET  /metrics        Query analytics (total, rejected, cache hits, latency)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import (
    HealthResponse,
    MetricsSummary,
    QueryRequest,
    QueryResponse,
    RejectedResponse,
)
from app.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Global pipeline instance (initialised in lifespan)
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    logger.info("Starting %s v%s …", settings.app_name, settings.app_version)
    _pipeline = RAGPipeline(settings)
    _pipeline.load()
    yield
    logger.info("Shutting down.")


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Domain-specific RAG assistant with prompt injection protection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=Union[QueryResponse, RejectedResponse],
    summary="Submit a question to the RAG assistant",
)
async def query(request: QueryRequest) -> Union[QueryResponse, RejectedResponse]:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    return await _pipeline.query(request)


@app.get("/health", response_model=HealthResponse, summary="Index readiness check")
def health() -> HealthResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    return _pipeline.health()


@app.get("/metrics", response_model=MetricsSummary, summary="Query analytics")
def metrics() -> MetricsSummary:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    return _pipeline.metrics()


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
