# Contributing

Thank you for your interest in contributing to the AI RAG Knowledge Analyst.

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/ai-rag-knowledge-analyst.git
   cd ai-rag-knowledge-analyst
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync --extra dev
   ```

3. **Set up your environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY
   ```

4. **Install pre-commit hooks** (once per clone):
   ```bash
   just install-hooks
   ```
   This registers the 10 hooks defined in `.pre-commit-config.yaml` so they
   run automatically on every `git commit`. They include ruff, ruff-format,
   `ty` type checking, and a guard against committing directly to `main`.

5. **Create a feature branch**:
   ```bash
   git switch -c feat/your-feature-name
   ```

## Development Workflow

- Write code following existing patterns in `src/rag_analyst/`.
- Add or update tests in `tests/`.
- Run the full quality gate before committing:
  ```bash
  just check
  ```
  This is the same command CI runs — pre-commit hooks + `ty check` + `pytest`.

## Project Layout

| Directory | Purpose |
|-----------|---------|
| `src/rag_analyst/` | Main package source code |
| `src/rag_analyst/reporting/` | JSON and terminal output formatters |
| `tests/` | Pytest test suite |
| `data/` | PDF document for RAG pipeline |
| `results/` | Generated JSON results (git-ignored) |
| `.github/workflows/` | CI configuration |
| `.cursor/rules/` | Cursor AI rules (RAG conventions, test conventions) |
| `.claude/` | Claude Code tool permissions |
| `.scratch/` | Ephemeral local work (git-ignored except `.gitkeep`) |

## Branch Conventions

- `main` — production-ready code (direct commits are blocked by pre-commit)
- `feat/*` — new features
- `fix/*` — bug fixes
- `refactor/*` — internal restructuring
- `chore/*` — tooling, dependencies, CI
- `docs/*` — documentation changes
- `test/*` — test-only changes

## Running Tests

Deterministic tests only (no API key — what CI runs):
```bash
just test
```

All tests including API-gated (requires `OPENAI_API_KEY`):
```bash
OPENAI_API_KEY=your-key just test
```

## Test Suite Overview

| File | Coverage | API Key Required |
|------|----------|-----------------|
| `test_ingest.py` | PDF loading, chunking, token size | No |
| `test_retrieval.py` | Vector store, similarity search, first-chunk injection | Yes (auto-skip) |
| `test_response.py` | Raw LLM, engineered, RAG responses | Yes (auto-skip) |
| `test_evaluation.py` | GPT-4o evaluation scoring | Yes (auto-skip) |

API-gated tests use `pytest.mark.skipif` so they skip cleanly when the key
is not set — never hard-fail.

## Key Architecture Rules

See `AGENTS.md` for the full list. Quick version:

1. **Three response modes stay comparable** — do not drift per-mode sampling.
2. **First-chunk injection in `retrieve_context`** — deliberate, covers
   metadata queries that similarity search misses.
3. **Lazy client initialization** — never instantiate `OpenAI` / `ChatOpenAI`
   at module top level.
4. **Configuration from environment** — no hardcoded API keys, model names,
   or base URLs.
5. **Logging over printing** — `print` only in `runner.py` and `reporting/`.

## Submitting Changes

1. Commit with a Conventional Commits-style message
   (`feat:`, `fix:`, `refactor:`, `chore:`, etc.).
2. Push to your fork and open a Pull Request against `main`.
3. Describe **what changed and why** in the PR description.
4. CI must pass — the `just check` job is required.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
