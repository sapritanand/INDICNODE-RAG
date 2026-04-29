# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base and injecting them into the model's context at inference time. It was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al.

RAG addresses two fundamental LLM limitations: knowledge cutoff (models can't know facts after training) and hallucination (models fabricate facts not in their training data). By grounding responses in retrieved documents, RAG makes LLMs more factual, up-to-date, and verifiable.

## RAG Pipeline Components

A typical RAG pipeline has three phases:

### 1. Indexing (Offline)
Documents are loaded, chunked into smaller pieces, embedded into vector representations, and stored in a vector database. This is done once (or when the knowledge base is updated).

Steps:
- **Document loading:** Ingest PDFs, Markdown, text files, HTML, etc.
- **Chunking:** Split documents into overlapping chunks of 200-500 words. Chunk size is a key hyperparameter — too small loses context, too large dilutes relevance signals.
- **Embedding:** Each chunk is encoded by an embedding model into a dense vector (e.g., 384 or 1536 dimensions).
- **Indexing:** Vectors are stored in a vector database (FAISS, Pinecone, Chroma, Weaviate) for fast approximate nearest-neighbour search.

### 2. Retrieval (Online)
When a user query arrives, it is embedded using the same model, and the top-k most similar chunks are retrieved from the index.

Types of retrieval:
- **Dense retrieval:** Uses embedding cosine similarity. Good for semantic matching.
- **Sparse retrieval (BM25/TF-IDF):** Keyword-based. Good for exact term matching.
- **Hybrid retrieval:** Combines dense and sparse signals (e.g., via Reciprocal Rank Fusion). Consistently outperforms either alone.

### 3. Generation (Online)
Retrieved chunks are assembled into a context string and injected into the LLM prompt alongside the user's question. The LLM is instructed to answer using only the provided context.

## Chunking Strategies

**Fixed-size chunking:** Split by fixed token count with overlap (e.g., 512 tokens, 50-token overlap). Simple but cuts mid-sentence.

**Semantic/paragraph chunking:** Split on paragraph boundaries (blank lines, section headers). Keeps semantic units whole, more expensive to implement but better retrieval quality.

**Sentence-window chunking:** Retrieve at the sentence level but return surrounding sentences as context. Good for precision.

**Hierarchical chunking (RAPTOR):** Build a tree of progressively summarised chunks. Allows answering both specific and broad questions.

The overlap between chunks ensures that information at chunk boundaries is not lost — the end of one chunk is repeated at the start of the next.

## Embedding Models

Embedding quality is critical to retrieval accuracy. Common choices:

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ~14ms/query CPU | Good for English |
| all-mpnet-base-v2 | 768 | ~25ms CPU | Better quality |
| text-embedding-3-small (OpenAI) | 1536 | API latency | Very high quality |
| text-embedding-ada-002 (OpenAI) | 1536 | API latency | High quality |

Local models (MiniLM, MPNet) avoid API costs and latency but are less powerful than hosted models. The choice depends on latency budget and cost constraints.

## Reciprocal Rank Fusion (RRF)

RRF is a fusion algorithm that combines multiple ranked lists into a single ranking without requiring score normalisation:

```
rrf_score(document) = Σ 1 / (k + rank(document, list_i))
```

Where k=60 is an empirically optimal constant. RRF is robust to differences in score magnitude between dense and sparse retrievers. It consistently outperforms simple score combination on BEIR and MS-MARCO benchmarks.

## Advanced RAG Techniques

**Query expansion:** Generate multiple paraphrases of the user query and retrieve for each, then deduplicate. Improves recall for ambiguous queries.

**Hypothetical Document Embeddings (HyDE):** Ask the LLM to generate a hypothetical answer to the query, embed that answer, and use it for retrieval. Often improves precision.

**Re-ranking:** After initial retrieval, apply a cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2) to re-score the top-k chunks. Cross-encoders are slower but more accurate than bi-encoders.

**Self-RAG:** The LLM decides whether to retrieve at all, what to retrieve, and whether retrieved content is relevant. More compute-intensive but reduces unnecessary retrieval.

## RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge update | Instant (update index) | Requires retraining |
| Cost | Retrieval + inference | Large training cost |
| Hallucination | Lower (grounded) | Can still hallucinate |
| Factual precision | High (citable sources) | Depends on training data |
| Generalisation | Limited to KB | Better generalisation |

For domain-specific factual Q&A over a changing knowledge base, RAG is almost always preferred over fine-tuning.

## Common RAG Failure Modes

- **Retrieval failure:** The correct chunk is not retrieved (low recall). Causes: poor embedding model, bad chunking, query-document vocabulary mismatch.
- **Context dilution:** Retrieved chunks are relevant but the LLM ignores them in favour of parametric knowledge.
- **Faithfulness failure:** The LLM generates content not supported by retrieved chunks (hallucination despite RAG).
- **Lost in the middle:** LLMs attend poorly to information in the middle of long context windows. Keep the most relevant chunk first.
- **Chunk boundary issues:** Key information split across two chunks; neither chunk alone answers the query.
