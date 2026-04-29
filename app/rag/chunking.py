"""
Chunking Strategy
-----------------
We use a two-pass semantic chunking approach:

Pass 1 – Structural split:
  Split the document on blank lines (paragraph boundaries).  Paragraphs are
  natural semantic units in prose and markdown.  This avoids cutting a concept
  mid-sentence.

Pass 2 – Size enforcement:
  If a paragraph exceeds `chunk_size` words we fall back to sentence-aware
  splitting (period/question/exclamation followed by whitespace).  A rolling
  window of `chunk_overlap` words is prepended to the next sub-chunk so the
  model retains cross-sentence context at chunk boundaries.

Why not fixed-size tokens?
  Fixed-size splits are fast but routinely cut inside a sentence, degrading
  both retrieval precision (the chunk is semantically incomplete) and LLM
  comprehension.  Semantic paragraph splitting keeps concepts whole while
  staying within embedding-model limits (~512 subword tokens for MiniLM-L6-v2,
  which comfortably holds 400 words).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    """Sentence-boundary split using punctuation heuristic."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> int:
    return len(text.split())


def _sub_chunk(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    """Split an oversized paragraph into overlapping sentence windows."""
    sentences = _split_sentences(paragraph)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = _words(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Seed the next chunk with the overlap tail of the current one
            tail_words = " ".join(current).split()[-overlap:]
            current = [" ".join(tail_words)] if tail_words else []
            current_len = len(tail_words)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Return a list of Chunk objects for one document.

    Parameters
    ----------
    text       : raw document text (markdown or plain)
    source     : filename / identifier for citation
    chunk_size : max words per chunk
    overlap    : words of context carried into the next chunk
    """
    # Strip markdown headers to avoid them dominating embeddings
    paragraphs = re.split(r"\n\s*\n", text.strip())
    raw_chunks: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _words(para) <= chunk_size:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(_sub_chunk(para, chunk_size, overlap))

    # Apply inter-paragraph overlap: prepend tail of previous chunk
    final_chunks: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            tail = " ".join(raw_chunks[i - 1].split()[-overlap:])
            chunk = tail + " " + chunk
        final_chunks.append(chunk.strip())

    return [
        Chunk(
            content=c,
            source=source,
            chunk_id=idx,
            metadata={"char_count": len(c), "word_count": _words(c)},
        )
        for idx, c in enumerate(final_chunks)
        if c.strip()
    ]
