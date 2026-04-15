"""PDF ingestion: load, chunk, embed, and build vector store."""

from __future__ import annotations

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    ENCODING_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PDF_FILENAME,
)


def load_pdf() -> list:
    """Load and return page-level Document objects from the PDF."""
    pdf_path = DATA_DIR / PDF_FILENAME
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def chunk_documents(pages: list | None = None) -> list:
    """Load PDF and split into token-sized chunks.

    Parameters
    ----------
    pages:
        Pre-loaded page Documents (e.g. from :func:`load_pdf`).  When
        *None* the PDF is loaded from disk automatically.  Passing an
        already-loaded list avoids reading the file a second time.
    """
    if pages is None:
        pdf_path = DATA_DIR / PDF_FILENAME
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(pages)


def build_vectorstore(document_chunks: list | None = None) -> Chroma:
    """Build a ChromaDB vector store from document chunks."""
    if document_chunks is None:
        document_chunks = chunk_documents()

    embedding_model = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    return Chroma.from_documents(document_chunks, embedding_model)
