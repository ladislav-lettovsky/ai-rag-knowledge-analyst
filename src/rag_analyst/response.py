"""Three response modes: raw LLM, prompt-engineered, and RAG."""

from __future__ import annotations

from openai import OpenAI

from .config import MAX_TOKENS, MODEL_GENERATION, OPENAI_BASE_URL, TEMPERATURE, TOP_P
from .prompts import LLM_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from .retrieval import retrieve_context

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-init OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL)
    return _client


def llm_response(
    user_prompt: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Raw LLM response — no context, no system prompt."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_GENERATION,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content or ""


def eng_response(
    user_prompt: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Prompt-engineered LLM response — system prompt, no retrieval."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_GENERATION,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content or ""


def rag_response(
    user_prompt: str,
    vectorstore,
    document_chunks: list,
    k: int | None = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Full RAG response — system prompt + retrieved document context."""
    context = retrieve_context(user_prompt, vectorstore, document_chunks, k=k)

    rag_user_prompt = f"""
    ###Context
    Here are some excerpts from literature and their sources that are relevant to the question mentioned below:
    {context}

    ###Question
    {user_prompt}
    """

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_GENERATION,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": rag_user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return (response.choices[0].message.content or "").strip()
