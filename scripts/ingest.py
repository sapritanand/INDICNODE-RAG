"""
Document Ingestion Script
--------------------------
Usage:
    python scripts/ingest.py [--kb-dir PATH] [--index-dir PATH]

Reads all .txt and .md files from the knowledge base directory, chunks them,
builds FAISS and BM25 indexes, and saves them to the index directory.

Re-run this script whenever the knowledge base is updated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.rag.chunking import chunk_document
from app.rag.embeddings import EmbeddingIndex
from app.rag.retrieval import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_documents(kb_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, text) for all .txt and .md files."""
    docs = []
    for ext in ("*.txt", "*.md"):
        for path in sorted(kb_dir.glob(ext)):
            text = path.read_text(encoding="utf-8")
            if text.strip():
                docs.append((path.name, text))
                logger.info("Loaded %s (%d chars)", path.name, len(text))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG indexes from documents.")
    parser.add_argument("--kb-dir", default=None, help="Knowledge base documents directory")
    parser.add_argument("--index-dir", default=None, help="Index output directory")
    args = parser.parse_args()

    settings = get_settings()
    kb_dir = Path(args.kb_dir) if args.kb_dir else Path(settings.knowledge_base_dir)
    index_dir = Path(args.index_dir) if args.index_dir else Path(settings.index_dir)

    if not kb_dir.exists():
        logger.error("Knowledge base directory not found: %s", kb_dir)
        sys.exit(1)

    # ── Load documents ──────────────────────────────────────────────────
    docs = load_documents(kb_dir)
    if not docs:
        logger.error("No .txt or .md files found in %s", kb_dir)
        sys.exit(1)
    logger.info("Loaded %d documents.", len(docs))

    # ── Chunk ───────────────────────────────────────────────────────────
    all_chunks = []
    for filename, text in docs:
        chunks = chunk_document(
            text=text,
            source=filename,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)
        logger.info("  %s → %d chunks", filename, len(chunks))

    logger.info("Total chunks: %d", len(all_chunks))

    # ── Build FAISS index ───────────────────────────────────────────────
    logger.info("Building FAISS index (this may take a minute) …")
    emb_index = EmbeddingIndex(
        model_name=settings.embedding_model,
        index_dir=str(index_dir),
        embedding_dim=settings.embedding_dim,
    )
    emb_index.build(all_chunks)

    # ── Build BM25 index ────────────────────────────────────────────────
    logger.info("Building BM25 index …")
    retriever = HybridRetriever(embedding_index=emb_index, index_dir=str(index_dir))
    retriever.build(all_chunks)

    logger.info("Ingestion complete.  Indexes saved to %s", index_dir)
    logger.info("You can now start the server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
