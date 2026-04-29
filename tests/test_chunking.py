"""Tests for the chunking module."""

import pytest
from app.rag.chunking import Chunk, chunk_document


SAMPLE_TEXT = """\
## Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses
by retrieving relevant documents at inference time. It was introduced in 2020.

## How It Works

The RAG pipeline has three phases: indexing, retrieval, and generation.
During indexing, documents are chunked and embedded into vectors.
During retrieval, the query is embedded and similar chunks are fetched.
During generation, the LLM synthesises an answer from the retrieved chunks.

## Benefits

RAG reduces hallucination because the model is grounded in retrieved facts.
It allows knowledge to be updated without retraining the model.
Sources can be cited, making responses verifiable.
"""


class TestChunkDocument:
    def test_returns_chunks(self):
        chunks = chunk_document(SAMPLE_TEXT, source="test.md")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_source(self):
        chunks = chunk_document(SAMPLE_TEXT, source="my_doc.md")
        for chunk in chunks:
            assert chunk.source == "my_doc.md"

    def test_chunk_ids_are_sequential(self):
        chunks = chunk_document(SAMPLE_TEXT, source="test.md")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i

    def test_metadata_populated(self):
        chunks = chunk_document(SAMPLE_TEXT, source="test.md")
        for chunk in chunks:
            assert "word_count" in chunk.metadata
            assert "char_count" in chunk.metadata
            assert chunk.metadata["word_count"] > 0

    def test_chunk_size_respected(self):
        chunks = chunk_document(SAMPLE_TEXT, source="test.md", chunk_size=50, overlap=5)
        for chunk in chunks:
            assert chunk.metadata["word_count"] <= 60, (
                f"Chunk too large: {chunk.metadata['word_count']} words"
            )

    def test_overlap_creates_context_continuity(self):
        long_text = " ".join([f"sentence{i} about topic{i}." for i in range(200)])
        chunks = chunk_document(long_text, source="long.md", chunk_size=50, overlap=10)
        assert len(chunks) > 1
        # Each chunk after the first should share some words with the previous
        for i in range(1, len(chunks)):
            prev_tail = set(chunks[i - 1].content.split()[-10:])
            curr_head = set(chunks[i].content.split()[:15])
            assert prev_tail & curr_head, "No overlap between consecutive chunks"

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_document("", source="empty.md")
        assert chunks == []

    def test_single_paragraph_creates_one_chunk(self):
        text = "This is a single paragraph with fewer than four hundred words."
        chunks = chunk_document(text, source="single.md")
        assert len(chunks) == 1
        assert "single paragraph" in chunks[0].content
