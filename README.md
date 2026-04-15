# AI-Powered RAG Business Report Analyst

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

# Create environment and install
uv venv .venv
source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## CLI Usage

```bash
# Run all 3 modes on built-in questions
uv run python -m rag_analyst

# Run a custom query
uv run python -m rag_analyst -q "Who are the authors?"

# Run only RAG mode with evaluation
uv run python -m rag_analyst --mode rag --evaluate

# Show comparison report
uv run python -m rag_analyst --report

# All options
uv run python -m rag_analyst --mode {raw,engineered,rag,all} \
                       --query "Your question" \
                       --evaluate \
                       --k 6 \
                       --json-output results/ \
                       --no-json \
                       --report \
                       --verbose
```

## Project Structure

```
ai-rag-knowledge-analyst/
├── src/rag_analyst/
│   ├── __init__.py              # Package version
│   ├── config.py                # Environment-based configuration
│   ├── ingest.py                # PDF loading, chunking, vector store
│   ├── retrieval.py             # Similarity search + first-chunk injection
│   ├── prompts.py               # System prompts (LLM and RAG)
│   ├── response.py              # Three response modes
│   ├── evaluation.py            # GPT-4o scoring
│   ├── reporting/
│   │   ├── json_writer.py       # Structured JSON results
│   │   └── terminal.py          # Comparison table printer
│   └── runner.py                # CLI entry point
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_ingest.py           # PDF load & chunk tests (no API key)
│   ├── test_retrieval.py        # Retrieval tests (needs API key)
│   ├── test_response.py         # Response tests (needs API key)
│   └── test_evaluation.py       # Evaluation tests (needs API key)
├── data/
│   └── HBR_How_Apple_Is_Organized_For_Innovation.pdf
├── .github/workflows/ci.yml
├── pyproject.toml
├── .env.example
├── README.md
└── CONTRIBUTING.md
```

## Data Source

- **Document**: "How Apple is Organized for Innovation" (HBR, Nov–Dec 2020)
- **Authors**: Joel M. Podolny and Morten T. Hansen
- **Format**: PDF, 11 pages

## Tech Stack

- **RAG Framework**: LangChain, LangChain Community, LangChain Text Splitters
- **Vector Store**: ChromaDB
- **LLMs**: OpenAI GPT-4o-mini (generation), GPT-4o (evaluation)
- **PDF Processing**: PyMuPDF
- **Tokenization**: tiktoken (`cl100k_base`)
- **Hybrid Retrieval**: BM25Retriever, EnsembleRetriever, rank-bm25
- **Runtime**: Python 3.12+

## License

MIT
