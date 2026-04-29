# Vector Embeddings and Similarity Search

## What are Vector Embeddings?

A vector embedding is a dense numerical representation of data (text, images, audio) in a high-dimensional space, where semantically similar items are mapped to nearby points. Embeddings are learned by neural networks trained on large corpora.

For text, an embedding model takes a string and returns a fixed-length float vector (e.g., 384 or 1536 dimensions). The geometry of this vector space encodes semantic relationships: "king" - "man" + "woman" ≈ "queen" is a classical example from Word2Vec.

Modern sentence embedding models (like all-MiniLM-L6-v2 from the Sentence Transformers library) produce fixed-size representations for entire sentences or paragraphs, not just words. They are trained using contrastive learning objectives (e.g., SimCSE) on pairs of semantically similar and dissimilar sentences.

## Cosine Similarity

The most common similarity metric for text embeddings is **cosine similarity**:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

Cosine similarity measures the angle between two vectors, not their magnitude. It ranges from -1 (opposite) to +1 (identical). For normalised vectors (unit length), cosine similarity equals the dot product, enabling extremely efficient computation.

In practice, embeddings are L2-normalised before indexing, so dot product search equals cosine search — this is what FAISS `IndexFlatIP` (inner product) implements.

Other similarity metrics include Euclidean distance and Manhattan distance, but cosine similarity dominates for NLP because it is scale-invariant.

## FAISS (Facebook AI Similarity Search)

FAISS is an open-source library from Meta for efficient similarity search over dense vectors. It provides multiple index types with different accuracy/speed tradeoffs:

### Index Types

**IndexFlatIP / IndexFlatL2 (Exact Search)**
- Brute-force comparison against all vectors
- 100% recall, no approximation
- Scales to ~1M vectors on CPU comfortably
- Used when the knowledge base is small (< 100k chunks)

**IndexIVFFlat (Inverted File Index)**
- Clusters vectors into `nlist` Voronoi cells
- At query time, only `nprobe` cells are searched
- O(nprobe × cluster_size) instead of O(n)
- Tradeoff: recall drops slightly (tunable with nprobe)
- 10-100x faster than brute-force at scale

**IndexHNSW (Hierarchical Navigable Small World)**
- Graph-based approximate nearest neighbour
- Excellent recall-speed tradeoff
- Memory-resident, no disk requirement
- Best for real-time serving at scale

**IndexIVFPQ (Product Quantization)**
- Compresses vectors using product quantization
- 4-16x memory reduction vs flat float32
- Slight accuracy loss; acceptable for large corpora
- Often combined with IVF: IndexIVFPQ

### Choosing an Index

For < 10k chunks: `IndexFlatIP` (exact, fast enough)
For 10k–1M chunks: `IndexIVFFlat` with nlist=sqrt(n), nprobe=10-20
For > 1M chunks or memory-constrained: `IndexIVFPQ`
For latency-critical serving: `IndexHNSW`

## BM25 (Best Match 25)

BM25 is a sparse retrieval algorithm based on term frequency-inverse document frequency (TF-IDF) with saturation and document length normalisation:

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:
- `f(qi, D)` = frequency of term qi in document D
- `|D|` = document length
- `avgdl` = average document length in corpus
- `k1` ≈ 1.2–2.0 (term frequency saturation)
- `b` ≈ 0.75 (length normalisation)

BM25 is highly effective for keyword search and exact phrase matching. It handles rare terms well (high IDF) and normalises for document length. It is the backbone of Elasticsearch and many production search engines.

BM25 weaknesses: no semantic understanding, vocabulary mismatch (synonym problem), sensitive to typos.

## Vector Databases

Vector databases are purpose-built for storing, indexing, and querying high-dimensional vectors at scale. They add metadata filtering, persistence, replication, and multi-tenancy on top of ANN libraries like FAISS.

**FAISS** — Meta's library. Low-level, no persistence, requires wrapper code. Best for research and embedded use.

**Chroma** — Lightweight, open-source, great for development. In-memory or SQLite persistence.

**Pinecone** — Fully managed cloud service. High throughput, automatic scaling, paid.

**Weaviate** — Open-source, graph-structured, supports hybrid search natively, module-based architecture.

**Qdrant** — Rust-based, high performance, supports payload filtering, payload indexing.

**Milvus** — Distributed, cloud-native, designed for billion-scale vector search.

## Embedding Model Selection Tradeoffs

| Factor | Small local model | Large API model |
|--------|-------------------|-----------------|
| Latency | ~10–30ms (CPU) | 100–500ms (API) |
| Cost | Free (compute only) | $0.02–$0.13/1M tokens |
| Privacy | Data stays local | Data sent to provider |
| Quality | Good | Excellent |
| Consistency | Fixed model | API may change |

For production RAG systems, `all-MiniLM-L6-v2` is a strong default: 80MB, 14ms/query on CPU, good MTEB benchmark scores, Apache-2.0 license.

## Approximate Nearest Neighbour (ANN) Search

Exact nearest neighbour search in high dimensions is O(n×d) and does not scale. ANN algorithms trade a small amount of accuracy (recall) for large speed gains.

**Key metrics:**
- **Recall@k:** Fraction of true top-k results returned. Typically target 95%+.
- **Queries per second (QPS):** Throughput at a given latency percentile.
- **Build time:** Time to construct the index (one-time cost).
- **Memory footprint:** RAM required to hold the index.

The ANN-benchmarks project provides standardised recall-vs-QPS Pareto curves for popular algorithms and datasets.
