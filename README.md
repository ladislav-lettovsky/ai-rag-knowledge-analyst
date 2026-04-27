# AI-Powered RAG Business Report Analyst

[![CI](https://github.com/ladislav-lettovsky/ai-rag-knowledge-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/ladislav-lettovsky/ai-rag-knowledge-analyst/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ladislav-lettovsky/ai-rag-knowledge-analyst)

A Retrieval-Augmented Generation (RAG) application that enables business analysts to extract key insights from lengthy reports through natural-language queries — built as a capstone project for the **UT Austin Postgraduate Program in AI / ML (Agentic AI specialization)**.

## Architecture

```
PDF → Load (PyMuPDF) → Chunk (256 tokens) → Embed (OpenAI) → ChromaDB → Retrieve (top-k) → LLM (GPT-4o-mini) → Response
```

## Three Response Modes

The application implements and compares three approaches to demonstrate the value of RAG:

1. **Raw LLM** — No context, no system prompt (baseline)
2. **Prompt-Engineered LLM** — System prompt but no retrieved context (max 100 tokens)
3. **Full RAG** — System prompt + retrieved document context (grounded)

## Evaluation Methodology

Responses are scored by GPT-4o on two dimensions (0.0–1.0 scale):

- **Groundedness**: How well the response is factually supported by the source document
- **Precision**: How directly and accurately the response addresses the query

## Results

| Question | Raw LLM | Prompt-Engineered | RAG |
|----------|---------|-------------------|-----|
| Q1: Authors & publisher | 0.0 / 0.0 | 0.0 / 0.0 | **1.0 / 1.0** |
| Q2: Leadership characteristics | 0.0 / 0.0 | 0.0 / 0.0 | **1.0 / 1.0** |
| Q3: Innovation examples | 0.2 / 0.3 | 0.2 / 0.3 | **0.7 / 0.6** |

(Format: Groundedness / Precision)

### Key Findings

- RAG achieves **perfect scores** on factual/metadata questions where plain LLM scores zero
- Prompt engineering alone does not fix hallucination — retrieved context is essential
- Complex inferential questions score lower (0.7/0.6), suggesting hybrid retrieval and larger k values as next steps

## Quick Start

```bash
git clone https://github.com/ladislav-lettovsky/ai-rag-knowledge-analyst.git
cd ai-rag-knowledge-analyst

# Create environment and install (with dev extras)
uv venv .venv
source .venv/bin/activate
uv sync --extra dev

# One-time: install pre-commit hooks (ruff + ty + guardrails)
just install-hooks

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## CLI Usage

```bash
# Run all 3 modes on built-in questions
uv run rag-analyst

# Run a custom query
uv run rag-analyst -q "Who are the authors?"

# Run only RAG mode with evaluation
uv run rag-analyst --mode rag_response --evaluate

# All options
uv run rag-analyst --mode {raw_response,eng_response,rag_response,all_responses} \
                   --query "Your question" \
                   --evaluate \
                   --k 6 \
                   --json-output results/ \
                   --no-json \
                   --report \
                   --verbose
```

## Development

The full quality gate — lint + type-check + tests — is one command:

```bash
just check
```

It runs the same 10 pre-commit hooks + `ty check` + `pytest` that CI runs.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## Project Structure

```
ai-rag-knowledge-analyst/
├── src/
│   └── rag_analyst/                   # Production Python package (src layout)
│       ├── config.py                  # Environment-based configuration
│       ├── ingest.py                  # PDF load, chunking, vector store build
│       ├── retrieval.py               # Similarity search + first-chunk injection
│       ├── prompts.py                 # System prompts (engineered & RAG)
│       ├── response.py                # Three response modes
│       ├── evaluation.py              # GPT-4o LLM-as-judge scoring
│       ├── reporting/
│       │   ├── json_writer.py         # Structured JSON output
│       │   └── terminal.py            # Comparison table printer
│       ├── runner.py                  # CLI entry point (rag-analyst)
│       └── __main__.py                # python -m rag_analyst entry
├── tests/                             # 12 pytest tests (5 deterministic + 7 API-gated)
├── data/                              # Source PDF (publicly permissioned)
│   └── HBR_How_Apple_Is_Organized_For_Innovation.pdf
├── results/                           # CLI JSON output (git-ignored)
├── .scratch/                          # Sanctioned scratchpad for AI agents (git-kept, .gitignored contents)
├── .claude/                           # Claude Code project config
│   └── settings.json
├── .cursor/                           # Cursor IDE rules
│   └── rules/
│       ├── 00-always.mdc              # Always-on invariants + check gate
│       ├── rag.mdc                    # RAG pipeline conventions (scoped)
│       ├── tests.mdc                  # Pytest conventions (scoped)
│       └── writing-rules.mdc          # Meta-guide for rule authoring
├── .github/workflows/ci.yml           # GitHub Actions CI (just check)
├── AGENTS.md                          # AI agent memory — invariants, architecture, pitfalls
├── CLAUDE.md                          # Claude Code entry point → AGENTS.md
├── CONTRIBUTING.md                    # Contribution guide
├── LICENSE                            # MIT (source code only; see README Acknowledgments for data)
├── README.md                          # You are here
├── justfile                           # Task runner — `just check` = full quality gate
├── pyproject.toml                     # Project metadata, deps, ruff/ty/pytest config
├── uv.lock                            # Reproducible dependency lockfile
├── .pre-commit-config.yaml            # Ruff + ty + hygiene pre-commit hooks (10)
├── .env.example                       # Environment variable template
└── .gitignore
```

## Tech Stack

- **RAG Framework**: LangChain 1.x (`langchain`, `langchain-classic`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`)
- **Vector Store**: ChromaDB
- **LLMs**: OpenAI GPT-4o-mini (generation), GPT-4o (evaluation)
- **PDF Processing**: PyMuPDF
- **Tokenization**: tiktoken (`cl100k_base`)
- **Hybrid Retrieval**: `rank-bm25` (available for lexical fallback experiments)
- **Runtime**: Python 3.12+
- **Tooling**: `uv` (deps), `just` (tasks), `ruff` (lint/format), `ty` (types), `pre-commit` (hooks)

## License & Acknowledgments

### Source code
The source code in this repository is released under the [MIT License](LICENSE).
Copyright (c) 2026 Ladislav Lettovsky.

### Data
The file under `data/` — `HBR_How_Apple_Is_Organized_For_Innovation.pdf`
("How Apple is Organized for Innovation" by Joel M. Podolny and Morten T. Hansen,
Harvard Business Review, Nov–Dec 2020) — is a third-party article included
solely to make the RAG demo reproducible. It is **not** redistributed under
the MIT License and is **not** covered by the copyright notice above. All
rights remain with the original authors and publisher.

### Built with
- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration framework
- [ChromaDB](https://github.com/chroma-core/chroma) — embedding vector store
- [OpenAI](https://openai.com/) — underlying LLMs
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) — PDF text extraction

## Author

**Ladislav Lettovsky** — [github.com/ladislav-lettovsky](https://github.com/ladislav-lettovsky)
