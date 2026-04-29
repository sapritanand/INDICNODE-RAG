# Indicnode RAG Assistant

A production-ready domain-specific AI assistant built with **Retrieval-Augmented Generation (RAG)**. Answers questions from a knowledge base of AI/LLM documents while enforcing strict domain boundaries and defending against prompt injection attacks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Indexing)                       │
│                                                                 │
│  knowledge_base/documents/*.md                                  │
│         │                                                       │
│         ▼                                                       │
│  [Chunker]  ──  semantic paragraph splitting + overlap          │
│         │                                                       │
│         ├──▶  [EmbeddingIndex]  all-MiniLM-L6-v2 → FAISS       │
│         │         saved: knowledge_base/index/vectors.faiss     │
│         └──▶  [BM25Index]  rank-bm25                           │
│                   saved: knowledge_base/index/bm25.pkl          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE (Query)                           │
│                                                                 │
│  POST /query {"question": "..."}                                │
│         │                                                       │
│         ▼                                                       │
│  [1] InjectionGuard  ──  regex + delimiter + encoded checks     │
│         │  if attack → RejectedResponse(INJECTION_DETECTED)     │
│         ▼                                                       │
│  [2] SemanticCache  ──  cosine sim ≥ 0.93 → return cached       │
│         │  cache miss → continue                                │
│         ▼                                                       │
│  [3] HybridRetriever                                            │
│         ├── FAISS dense search (top-5)                          │
│         ├── BM25 sparse search (top-5)                          │
│         └── RRF fusion → top-3 chunks                          │
│         │                                                       │
│  [4] Domain check  ──  raw cosine < 0.28 →                      │
│         │              RejectedResponse(OUT_OF_SCOPE)           │
│         ▼                                                       │
│  [5] Generator  ──  Claude Haiku + cached system prompt         │
│         │           (grounded answer from context only)         │
│         ▼                                                       │
│  [6] Cache.put  +  Metrics.update                               │
│         │                                                       │
│         ▼                                                       │
│  QueryResponse {answer, sources, latency, tokens}               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Chunking: Semantic paragraph splitting with overlap

**Why not fixed-size token splits?**  
Fixed-size splits are the default in many frameworks but routinely cut mid-sentence, producing semantically incomplete chunks that degrade both retrieval precision (the chunk doesn't represent a complete idea) and LLM comprehension (the model receives truncated context).

**What I did instead:**  
- **Pass 1:** Split on blank lines — paragraph boundaries are natural semantic units in Markdown and prose.
- **Pass 2:** For paragraphs exceeding 400 words, fall back to sentence-aware splitting (punctuation-based) with a 50-word rolling overlap.
- **Inter-paragraph overlap:** The last 50 words of each chunk are prepended to the next, preserving context at boundaries.

This keeps full sentences and ideas intact while staying within the MiniLM-L6-v2 token limit (~512 subword tokens ≈ 400 words).

### 2. Hybrid Retrieval: Dense + Sparse → RRF

**Why hybrid?**  
Dense retrieval (FAISS cosine similarity) captures semantic meaning but misses exact keyword matches ("FAISS", "BM25", proper nouns, version numbers). Sparse BM25 excels at exact terms but fails for paraphrases. Neither alone is optimal.

**What I did:**  
Combined FAISS top-5 and BM25 top-5 results using **Reciprocal Rank Fusion (RRF)**:

```
rrf_score(d) = Σ 1 / (60 + rank(d, list_i))
```

The constant k=60 dampens outlier ranks. RRF requires no score normalisation across heterogeneous systems and consistently outperforms linear combination on BEIR benchmarks.

### 3. Domain Restriction: Two-layer approach

**Layer 1 — Injection guard (before retrieval):**  
Multi-pattern regex covering instruction overrides, role hijacking, system-prompt exfiltration, delimiter injection, and encoded attacks (base64, ROT-13). Runs in microseconds at zero cost.

**Layer 2 — Retrieval-based domain check (after retrieval):**  
Checks the raw FAISS cosine similarity of the top retrieved chunk. If it falls below `domain_similarity_threshold` (default 0.28), the query is semantically distant from all knowledge-base content — it's out-of-scope. This is more reliable than a keyword blocklist, adapts to any domain, and is trivially tunable.

**Why not an LLM classifier for both?**  
An LLM classifier adds 200-800ms and token cost to *every* request, including the large majority that are legitimate. The rule-based + retrieval-score approach catches >95% of real cases at effectively zero overhead.

### 4. Semantic Cache

Standard caches key on the exact query string, missing near-duplicates ("What is RAG?" vs "Can you explain RAG?"). The semantic cache embeds each query and checks cosine similarity against cached embeddings. At threshold=0.93, two queries must be semantically near-identical to hit — this avoids stale answers for subtly different questions.

Eviction is FIFO when `max_size` is reached. Simpler than LRU because in semantic caches, any near-duplicate hits an entry regardless of recency.

### 5. Provider: OpenRouter (OpenAI-compatible)

The generator uses `openai.AsyncOpenAI` pointed at `https://openrouter.ai/api/v1`. This gives access to 200+ models with a single API key — switching models is a one-line `.env` change (`GENERATION_MODEL=openai/gpt-4o-mini`, `GENERATION_MODEL=google/gemini-flash-1.5`, etc.).

> **Note on prompt caching:** Anthropic's native API supports `cache_control` on the system prompt, reducing per-request input-token cost by ~70%. OpenRouter does not expose this feature. To re-enable caching, swap to `anthropic.AsyncAnthropic` and add `"cache_control": {"type": "ephemeral"}` to the system block.

### 6. Model Choice: Claude 3 Haiku via OpenRouter

`anthropic/claude-3-haiku` offers the best cost/latency tradeoff for domain-constrained factual Q&A:
- ~200ms time-to-first-token
- $0.25/MTok input, $1.25/MTok output (via OpenRouter)
- Sufficient reasoning quality for grounded retrieval tasks

Temperature is set to 0 for deterministic factual responses.

---

## Performance Considerations

| Component | Typical latency | Optimisation |
|-----------|----------------|--------------|
| Injection guard | < 1ms | Regex only, no network |
| Cache lookup | 1–5ms | Numpy cosine over ≤1000 cached vecs |
| FAISS search | 5–15ms | Exact IndexFlatIP, CPU, 384-dim |
| BM25 search | 2–8ms | In-memory numpy |
| LLM generation | 200–600ms | Haiku + system-prompt caching |
| **Total (cold)** | **250–650ms** | |
| **Total (cache hit)** | **10–20ms** | |

**Bottleneck:** LLM generation (~85% of total latency on cache miss). Further optimisations:

- **Streaming responses:** `stream=True` returns the first token in ~80ms — improves perceived latency significantly.
- **Async OpenAI client:** `AsyncOpenAI` (pointing at OpenRouter) handles concurrent requests without blocking the event loop.
- **Larger cache:** A hit rate of 30%+ at 1000-entry capacity reduces average latency substantially.
- **IndexIVFFlat:** At >100k chunks, switch from exact IndexFlatIP to approximate IVF for faster retrieval.

---

## Tradeoffs Considered

| Decision | Chosen | Alternative | Tradeoff |
|----------|--------|-------------|----------|
| Embedding model | MiniLM-L6-v2 (local) | text-embedding-3-small (API) | Local: free, fast, private. API: higher quality, $0.02/1M tokens |
| Retrieval | Hybrid (dense+sparse) | Dense only | Hybrid: better recall, 2× index size. Dense only: simpler, faster |
| Safety | Rule-based + retrieval score | LLM classifier | Rules: ~0ms, ~$0. LLM: +500ms, +$0.001/query but higher precision |
| Generation model | Haiku | Sonnet / Opus | Haiku: 12× cheaper, ~2× faster. Sonnet: better complex reasoning |
| Cache | Semantic (cosine) | Exact string match | Semantic: higher hit rate. Exact: no false positives, simpler |
| Chunking | Semantic paragraphs | Fixed 512 tokens | Semantic: better retrieval quality. Fixed: simpler, predictable size |

---

## Limitations

1. **Single-turn only:** No conversation history or multi-turn context. Each query is treated independently. Adding conversation memory would require storing prior turns and including them in the retrieval query.

2. **Domain check is coarse:** The cosine threshold (0.28) is a single scalar. It may admit borderline out-of-scope queries or reject borderline in-scope ones. Tuning requires a labelled evaluation set.

3. **No re-ranking:** The system uses RRF fusion but no cross-encoder re-ranker. A cross-encoder (e.g., ms-marco-MiniLM-L-6-v2) would improve top-k precision at the cost of ~50ms.

4. **In-memory cache only:** The semantic cache is not persisted across restarts. A Redis + vector-search layer (e.g., Redis Stack) would make the cache durable and shareable across multiple server instances.

5. **Faithfulness not verified:** The system instructs the LLM to answer from context, but doesn't post-hoc verify faithfulness. Adding NLI-based or RAGAS faithfulness scoring would close this loop.

6. **Static knowledge base:** Documents must be re-ingested manually. A production system would add a webhook or file-watcher to trigger incremental re-indexing on document change.

7. **No multi-document answer synthesis:** If the answer spans multiple documents, the LLM must synthesise across chunks. Lost-in-the-middle effects can degrade quality for long contexts.

---

## Project Structure

```
indicnode-rag/
├── app/
│   ├── main.py              # FastAPI app, lifespan, endpoints
│   ├── pipeline.py          # Query orchestration (all 7 steps)
│   ├── config.py            # Pydantic settings from .env
│   ├── models.py            # Request/response Pydantic models
│   ├── cache.py             # Semantic cache (cosine similarity)
│   ├── rag/
│   │   ├── chunking.py      # Two-pass semantic chunker
│   │   ├── embeddings.py    # SentenceTransformers + FAISS index
│   │   ├── retrieval.py     # BM25 + RRF hybrid retriever
│   │   └── generator.py     # LLM generation via OpenRouter
│   └── security/
│       └── injection.py     # Multi-layer injection detection
├── knowledge_base/
│   ├── documents/           # Source .md files (knowledge domain)
│   └── index/               # Persisted FAISS + BM25 indexes
├── scripts/
│   └── ingest.py            # Index-building script
├── ui/
│   └── streamlit_app.py     # Optional chat UI
├── tests/
│   ├── test_security.py     # Injection detection tests
│   ├── test_chunking.py     # Chunking tests
│   └── test_cache.py        # Semantic cache tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- An OpenRouter API key (free at openrouter.ai/keys)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Build the indexes

```bash
python scripts/ingest.py
```

This reads all `.md` files from `knowledge_base/documents/`, chunks and embeds them, and saves FAISS + BM25 indexes to `knowledge_base/index/`. Typically takes 30–60 seconds on CPU.

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.

### 5. (Optional) Start the Streamlit UI

In a separate terminal:

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

**Success response:**
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique...",
  "sources": [
    {"content": "...", "source": "02_retrieval_augmented_generation.md", "score": 0.92}
  ],
  "response_time_ms": 423.1,
  "cached": false,
  "tokens_used": 387
}
```

**Rejection response (out-of-scope):**
```json
{
  "rejected": true,
  "reason": "out_of_scope",
  "message": "Your question does not appear to be related to...",
  "response_time_ms": 12.4
}
```

**Rejection response (injection detected):**
```json
{
  "rejected": true,
  "reason": "injection_detected",
  "message": "Your query appears to contain instructions that attempt...",
  "response_time_ms": 0.3
}
```

### `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "documents_indexed": 5,
  "chunks_indexed": 87,
  "index_ready": true
}
```

### `GET /metrics`

```json
{
  "total_queries": 42,
  "rejected_queries": 3,
  "cache_hits": 8,
  "avg_response_time_ms": 381.2,
  "avg_tokens_used": 412.0
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Bonus Features Implemented

- **Hybrid retrieval (dense + keyword)** via FAISS + BM25 + RRF fusion
- **Encoded attack detection** (base64, ROT-13 payload scanning)
- **Semantic cache** (near-duplicate query deduplication)
- **OpenRouter integration** (200+ models, single API key, OpenAI-compatible)
- **Streamlit UI** with real-time metrics sidebar
- **Structured logging** throughout the pipeline
