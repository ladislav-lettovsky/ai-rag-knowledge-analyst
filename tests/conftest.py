"""Shared test fixtures."""

from __future__ import annotations

import os

# Disable LangSmith tracing during tests
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
