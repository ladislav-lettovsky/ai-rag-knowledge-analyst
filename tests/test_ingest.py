"""Tests for PDF ingestion — NO API key needed."""

from __future__ import annotations

from rag_analyst.ingest import chunk_documents, load_pdf


class TestLoadPdf:
    def test_load_pdf_returns_pages(self):
        pages = load_pdf()
        assert len(pages) > 0, "PDF should produce at least one page"

    def test_pages_have_content(self):
        pages = load_pdf()
        for page in pages:
            assert page.page_content, "Page should have content"


class TestChunkDocuments:
    def test_chunk_documents_returns_chunks(self):
        chunks = chunk_documents()
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunks_have_content(self):
        chunks = chunk_documents()
        for chunk in chunks:
            assert chunk.page_content, "Each chunk should have content"

    def test_chunk_size_reasonable(self):
        """No chunk should exceed a reasonable token count."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunks = chunk_documents()
        for chunk in chunks:
            token_count = len(enc.encode(chunk.page_content))
            # Allow some slack above chunk_size (256) due to splitter behavior
            assert token_count < 512, (
                f"Chunk has {token_count} tokens, expected < 512"
            )
