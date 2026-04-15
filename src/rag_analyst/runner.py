"""CLI entry point for RAG Analyst."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import RESULTS_DIR, SIMILARITY_K
from .evaluation import response_evaluation
from .ingest import build_vectorstore, chunk_documents, load_pdf
from .reporting.json_writer import write_json_results
from .reporting.terminal import print_comparison
from .response import eng_response, llm_response, rag_response

logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    {"question": "Who are the authors of this article and who published this article?", "k": 6},
    {"question": "List down the three leadership characteristics in bulleted points and explain each one of the characteristics under two lines.", "k": 3},
    {"question": "Can you explain specific examples from the article where Apple's approach to leadership has led to successful innovations?", "k": 3},
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-analyst",
        description="AI-powered RAG Business Report Analyst",
    )
    parser.add_argument(
        "--mode",
        choices=["raw_response", "eng_response", "rag_response", "all_responses"],
        default="all_responses",
        help="Response mode (default: all — runs all 3 for comparison)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom query (default: run built-in 3 questions)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run GPT-4o evaluation after responses",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=SIMILARITY_K,
        help=f"Number of retrieved chunks (default: {SIMILARITY_K})",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help=f"Directory for JSON results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show terminal comparison report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Set log level to DEBUG",
    )
    return parser


def _run_modes(
    question: str,
    mode: str,
    vectorstore,
    document_chunks: list,
    content: str,
    k: int,
    evaluate: bool,
) -> list[dict]:
    """Run selected mode(s) for a single question and return results."""
    results: list[dict] = []
    modes = ["raw_response", "eng_response", "rag_response"] if mode == "all_responses" else [mode]

    for m in modes:
        logger.info("Running %s mode for: %s", m, question)

        if m == "raw_response":
            resp = llm_response(question)
        elif m == "eng_response":
            resp = eng_response(question)
        else:
            resp = rag_response(question, vectorstore, document_chunks, k=k)

        entry: dict = {"question": question, "mode": m, "response": resp}

        if evaluate:
            entry["evaluation"] = response_evaluation(content, question, resp)

        results.append(entry)

    return results


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Ingest ---
    print("Loading and chunking PDF...")
    pages = load_pdf()
    # Reuse already-loaded pages to avoid reading the PDF twice.
    document_chunks = chunk_documents(pages)
    # Flatten Document list to a plain string for evaluation context.
    content = " ".join(doc.page_content for doc in pages)

    print("Building vector store...")
    vectorstore = build_vectorstore(document_chunks)

    # --- Questions ---
    if args.query:
        questions = [{"question": args.query, "k": args.k}]
    else:
        questions = TEST_QUESTIONS

    # --- Run ---
    all_results: list[dict] = []
    for q in questions:
        k = q.get("k", args.k)
        results = _run_modes(
            question=q["question"],
            mode=args.mode,
            vectorstore=vectorstore,
            document_chunks=document_chunks,
            content=content,
            k=k,
            evaluate=args.evaluate,
        )
        all_results.extend(results)

        # Print inline
        for r in results:
            print(f"\n[{r['mode'].upper()}] {r['question']}...")
            print(r["response"])
            if "evaluation" in r:
                print(f"Evaluation: {r['evaluation']}")

    # --- Report ---
    if args.report:
        print_comparison(all_results)

    # --- JSON output ---
    if not args.no_json:
        output_dir = Path(args.json_output) if args.json_output else None
        filepath = write_json_results(all_results, output_dir=output_dir)
        print(f"\nResults written to {filepath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
