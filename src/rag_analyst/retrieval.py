"""Retrieval: similarity search with first-chunk injection."""

from __future__ import annotations

from langchain_community.vectorstores import Chroma

from .config import SIMILARITY_K


def retrieve_context(
    query: str,
    vectorstore: Chroma,
    document_chunks: list,
    k: int | None = None,
) -> str:
    """Retrieve relevant document chunks and return combined context string.

    Always includes the first chunk (title page with author/publisher metadata)
    since semantic search often misses metadata-heavy content.
    """
    k = SIMILARITY_K if k is None else k
    relevant_chunks = vectorstore.similarity_search(query, k=k)
    context_list = [d.page_content for d in relevant_chunks]

    # First-chunk injection for metadata coverage
    first_chunk_content = document_chunks[0].page_content
    if first_chunk_content not in context_list:
        context_list = [first_chunk_content] + context_list

    return ". ".join(context_list)
