"""Environment-based configuration and logging setup."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Silence noisy third-party libraries before they are imported elsewhere.
os.environ.setdefault("TQDM_DISABLE", "1")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data"))
RESULTS_DIR: Path = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent.parent / "results"))

PDF_FILENAME: str = os.environ.get("PDF_FILENAME", "HBR_How_Apple_Is_Organized_For_Innovation.pdf")

# ---------------------------------------------------------------------------
# API / model settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    warnings.warn(
        "OPENAI_API_KEY is not set — all API calls will fail with an authentication error.",
        stacklevel=2,
    )
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL") or None

MODEL_GENERATION: str = os.environ.get("MODEL_GENERATION", "gpt-4o-mini")
MODEL_EVALUATION: str = os.environ.get("MODEL_EVALUATION", "gpt-4o")

TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.75"))
TOP_P: float = float(os.environ.get("TOP_P", "0.95"))
MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "1000"))

# Chunking settings
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "256"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "20"))
ENCODING_NAME: str = os.environ.get("ENCODING_NAME", "cl100k_base")

# Retrieval settings
SIMILARITY_K: int = int(os.environ.get("SIMILARITY_K", "3"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "WARNING")

NOISY_LOGGERS: list[str] = [
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langsmith",
    "chromadb",
]


def _configure_logging() -> None:
    """Set root log level and suppress noisy third-party loggers."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()
