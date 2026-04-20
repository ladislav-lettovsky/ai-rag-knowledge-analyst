# ai-rag-knowledge-analyst — AI Agent Memory

## What this is
A RAG (retrieval-augmented generation) pipeline that compares **three answering
strategies** side-by-side on the same question against a source document (Harvard
Business Review article on Apple's innovation organization):

1. **`raw_response`** — the LLM answers from parametric knowledge only
2. **`eng_response`** — same LLM with a carefully prompt-engineered system message
3. **`rag_response`** — the LLM receives retrieved chunks from a Chroma vector store
   (with deterministic first-chunk injection for metadata coverage)

A separate LLM-as-judge (`evaluation.py`) scores each response on **groundedness**
and **precision**, producing comparable metrics across the three modes. Results
can be emitted to the terminal or written to timestamped JSON for longitudinal
analysis.

## Stack
- Python 3.12+, `uv` for dependency management, `just` as the task runner
- LangChain 1.x (`langchain`, `langchain-classic`, `langchain-community`,
  `langchain-openai`, `langchain-text-splitters`)
- ChromaDB for the vector store
- PyMuPDF for PDF ingestion; `tiktoken` for token-aware chunking; `rank-bm25`
  available for lexical fallback
- OpenAI GPT-4o-mini (generator) and GPT-4o (evaluator)
- Pydantic, python-dotenv
- Testing: pytest 8 with API-gated tests auto-skipping via `pytest.mark.skipif`
- Linting: ruff; Type checking: `ty` (Astral); Pre-commit hooks enabled (10 hooks)

## Commands you can run without asking
- `just fmt` — format code (ruff format)
- `just lint` / `just lint-fix` — ruff check, optionally with --fix
- `just type` — `uv run ty check`
- `just test` — full pytest run (API-gated tests skip without `OPENAI_API_KEY`)
- `just pre-commit` — run all 10 pre-commit hooks against every file
- `just check` — the quality gate: `pre-commit` + `ty check` + `pytest`
  (identical to what CI runs)
- `just install-hooks` — one-time setup after cloning
- `uv sync`, `uv sync --extra dev`
- `uv run python -m rag_analyst [--flags]`
- `uv run rag-analyst --mode {raw_response,eng_response,rag_response,all_responses} [-q "..."]`
- Read-only git: `git status`, `git diff`, `git log`, `git branch`

## Commands with preconditions
- `git commit` is allowed on a non-`main` branch **only after `just check`
  passes with no errors**. A pre-commit hook (`no-commit-to-branch`) blocks
  direct commits to `main`/`master`; escape only with explicit
  `SKIP=no-commit-to-branch` and a good reason.

## Commands that need explicit approval
- `uv add`, `uv remove` (dependency changes)
- `git push`, `git reset --hard`
- `gh pr create`, `gh pr merge`
- Anything touching `.env`, `.github/workflows/`, or `data/`
- Any change to the **retrieval contract** (see invariants below)

## Architectural invariants (do not violate without explicit discussion)

1. **Three-mode comparison is the product.** The value of this repo is the
   apples-to-apples diff between `raw_response`, `eng_response`, and
   `rag_response` on identical questions. Never collapse modes or change one
   in a way that would make them non-comparable (e.g., different
   temperatures, different models, different system prompts across modes
   unless the mode's definition requires it).

2. **First-chunk injection in retrieval is deliberate**
   (`retrieval.py::retrieve_context`). Cover-page metadata (authors,
   publisher) is semantically dissimilar to most queries, so similarity
   search consistently misses it. We always prepend chunk 0 if similarity
   search didn't return it. Removing this silently breaks attribution
   questions — never do so without updating tests and the retrieval contract.

3. **Evaluation uses a separate, more capable model** (`MODEL_EVALUATION`
   = GPT-4o vs `MODEL_GENERATION` = GPT-4o-mini). LLM-as-judge is only
   meaningful when the judge is at least as capable as the generator. Don't
   point them at the same model without explicit discussion.

4. **Token-aware chunking with overlap** (`ingest.py::chunk_documents`)
   uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with
   `CHUNK_SIZE=256` tokens and `CHUNK_OVERLAP=20`. Test
   (`test_ingest.py::test_chunks_under_max_tokens`) asserts chunks stay
   under 512 tokens. If you change sizes, update both `config.py` and the
   test bound; don't loosen the bound silently.

5. **LLM clients are lazy-initialized inside functions**
   (`response.py::_get_client`, `evaluation.py::_get_evaluate_llm`). Never
   instantiate `OpenAI(...)` or `ChatOpenAI(...)` at module import time —
   it breaks tests (mocks) and triggers API-key checks on `import rag_analyst`.

6. **Configuration comes from environment variables** — model names, API
   keys, base URLs, chunking params, temperature all live in `.env` and
   are read via `rag_analyst.config`. Never hardcode these.

7. **Logging, not printing**, in library code. `logger = logging.getLogger(__name__)`
   at the top of every module. The `reporting/` package (`terminal.py`,
   `json_writer.py`) and the top-level `runner.py` are the only exceptions
   — that code legitimately writes to stdout for end-user display.

## Where things live
- `src/rag_analyst/` — production package (src layout)
  - `config.py` — centralized settings from env vars (paths, models, chunking,
    sampling, API keys/base URL)
  - `ingest.py` — `load_pdf`, `chunk_documents`, `build_vectorstore`
  - `retrieval.py` — `retrieve_context` (similarity + first-chunk injection)
  - `prompts.py` — system prompts for engineered and RAG modes
  - `response.py` — three response modes (`llm_response`, `eng_response`,
    `rag_response`)
  - `evaluation.py` — LLM-as-judge on groundedness + precision
  - `runner.py` — CLI entry point, `TEST_QUESTIONS`, `_run_modes` orchestration
  - `reporting/` — `terminal.py` (pretty printing), `json_writer.py`
    (timestamped JSON file per run in `results/`)
  - `__main__.py` — enables `python -m rag_analyst`
- `data/` — source PDF(s); checked in because the HBR article we use is
  publicly permissioned for this project
- `tests/` — pytest suite: `test_ingest.py` (deterministic, no API),
  `test_retrieval.py` / `test_response.py` / `test_evaluation.py`
  (API-gated, auto-skip without `OPENAI_API_KEY`)
- `results/` — JSON output per run (git-ignored)
- `.scratch/` — ephemeral work zone (git-ignored except `.gitkeep`)

## RAG conventions for this repo
- **Retrieval returns a single joined string** (not a list of Documents).
  Callers in `response.py` inject it directly into the RAG prompt. If you
  want structured retrieval output, add a new function — don't change the
  contract of `retrieve_context`.
- **Chroma is in-memory, per-run.** Re-ingestion happens every CLI
  invocation. This is fine for a ~20-page document; do not add persistence
  without discussion (it changes the test surface materially).
- **LangChain dynamic kwargs** (`base_url`, `api_key` on `ChatOpenAI` /
  `OpenAIEmbeddings`) are not in LangChain's exported type stubs. Use
  `# ty: ignore[unknown-argument]` with a brief comment — don't
  try to rewrite around them.

## Testing conventions
- Deterministic tests (no API) live in `tests/test_ingest.py` and run in CI.
- API-gated tests (`test_retrieval.py`, `test_response.py`,
  `test_evaluation.py`) set
  `pytestmark = pytest.mark.skipif(not HAS_API_KEY, reason="OPENAI_API_KEY not set")`.
  They are skipped in CI deliberately — CI is deterministic, not a billing
  line item.
- Run locally with a real key to validate end-to-end behavior before merging
  any change to `ingest.py`, `retrieval.py`, `response.py`, or
  `evaluation.py`.
- New features require at least one deterministic test when possible. For
  LLM-backed paths, mock `invoke()` / `.chat.completions.create()` returns.

## Type checking with `ty`
- To suppress a finding: `# ty: ignore[<rule-name>]` — the rule name comes
  from the ty diagnostic header (e.g., `unknown-argument`,
  `invalid-argument-type`, `unresolved-attribute`).
- Legitimate suppressions in this repo:
  - LangChain `ChatOpenAI` / `OpenAIEmbeddings` accept `api_key` and
    `base_url` dynamically but the exported stubs don't declare them.
  - `.content` on `AIMessage` is `str | list[str | dict]` — flatten to `str`
    at the call site (see `evaluation.py` for the pattern).

## Ephemeral / scratch work
Use `.scratch/` at the repo root for any exploratory, diagnostic, or
throwaway work — quick Python snippets, draft prompts, debug traces,
retrieval experiments, or scratch notes. The directory is git-ignored
(except `.gitkeep`), so nothing here is ever committed.

- Create on demand: `mkdir -p .scratch`
- Preferred file names: `<topic>.py`, `<topic>.md`, `<topic>.json`, etc.
- Do NOT place exploratory files at the repo root — always use `.scratch/`
- Clean up periodically (nothing persists beyond your working session)

Examples of good `.scratch/` use:
- `.scratch/try_new_chunking.py` — testing a different splitter config
- `.scratch/inspect_retrieval.py` — dumping top-k for a query to eyeball ranking
- `.scratch/prompt_v3.md` — drafting a new system-prompt variant before promoting

## Before saying "done"
1. `just check` passes (pre-commit + ty + pytest; 5 passed, 7 skipped is the
   expected baseline without an API key)
2. Any new public function has a test and a type-annotated signature
3. No new `print()` calls in library code (`reporting/`, `runner.py` excepted)
4. If the change affects behavior, `README.md` reviewed and updated
5. Diff against `main` looks like what you'd want in a PR review
