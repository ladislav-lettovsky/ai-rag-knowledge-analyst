"""Response evaluation using GPT-4o scoring."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .config import MODEL_EVALUATION, OPENAI_BASE_URL

_evaluate_llm: ChatOpenAI | None = None


def _get_evaluate_llm() -> ChatOpenAI:
    """Lazy-init evaluation LLM."""
    global _evaluate_llm
    if _evaluate_llm is None:
        _evaluate_llm = ChatOpenAI(
            model_name=MODEL_EVALUATION,
            base_url=OPENAI_BASE_URL,
        )
    return _evaluate_llm


def response_evaluation(content: str, question: str, response: str) -> str:
    """Evaluate response accuracy using GPT-4o on groundedness and precision."""
    evaluation_prompt = f"""
    Evaluate the assistant's response to a user's query using the provided context.

    Context: {content}
    Query: {question}
    Response: {response}

    Instructions:
    1. **Groundedness (0.0 to 1.0)**: Score based on how well the response is factually supported by the context.
                                    - Score closer to 1 if all facts are accurate and derived from the context.
                                    - Score closer to 0 if there is hallucination, guesswork, or any fabricated information.

    2. **Precision (0.0 to 1.0)**: Score based on how directly and accurately the assistant addresses the query.
                                    - Score closer to 1 if the response is concise, focused, and answers the exact user query.
                                    - Score closer to 0 if it includes irrelevant details or misses the main point.

    Output format:
      groundedness: float between 0 and 1 ,
      precision: float between 0 and 1

    """
    llm = _get_evaluate_llm()
    return llm.invoke([HumanMessage(content=evaluation_prompt)]).content.strip()
