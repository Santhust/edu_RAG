"""Grounded answer generation: retrieved excerpts in, cited answer out."""

from __future__ import annotations

from dataclasses import dataclass

from . import llm
from .config import Settings
from .retrieve import Retriever, ScoredChunk

NOT_FOUND = "I don't know based on the book."

SYSTEM_WITH_CONTEXT = """You are a careful research assistant answering questions about the book
"Noble Warrior: A Life of the Buddha" (Ṭhānissaro Bhikkhu & Khematto Bhikkhu).

Rules:
- Use ONLY the numbered excerpts supplied by the user. Never use outside knowledge.
- CITATIONS ARE MANDATORY. End EVERY sentence with a citation copied exactly from an
  excerpt header, like this example: The seer was Asita [p. 19, Birth].
  A citation on only the final sentence is not enough.
- If the excerpts do not contain the answer, reply exactly: "I don't know based on the book."
- If excerpts conflict, say so briefly.
- Be concise: at most 150 words. Plain prose, no markdown headings."""

SYSTEM_NO_CONTEXT = """Answer the question briefly and directly from your own knowledge.
Keep it under 120 words. This is a baseline used to compare against a RAG system."""


@dataclass
class Answer:
    question: str
    text: str
    passages: list[ScoredChunk]
    response: llm.LLMResponse


def format_context(passages: list[ScoredChunk]) -> str:
    blocks = []
    for number, item in enumerate(passages, start=1):
        chunk = item.chunk
        blocks.append(
            f"[{number}] [{chunk.citation}] (section: {chunk.section}, id: {chunk.id})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def prepare(
    question: str,
    retriever: Retriever,
    settings: Settings,
    k: int | None = None,
    chapter: str | None = None,
    mode: str | None = None,
    with_context: bool = True,
) -> tuple[list[dict], list[ScoredChunk]]:
    passages: list[ScoredChunk] = []
    if with_context:
        passages = retriever.search(question, k=k, chapter=chapter, mode=mode)
    if with_context and not passages:
        passages = []
    if with_context:
        messages = [
            {"role": "system", "content": SYSTEM_WITH_CONTEXT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n{format_context(passages)}",
            },
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_NO_CONTEXT},
            {"role": "user", "content": question},
        ]
    return messages, passages


def finalize(
    question: str, text: str, passages: list[ScoredChunk], response: llm.LLMResponse
) -> Answer:
    return Answer(question=question, text=text, passages=passages, response=response)


def answer(
    question: str,
    retriever: Retriever,
    settings: Settings,
    k: int | None = None,
    chapter: str | None = None,
    mode: str | None = None,
    with_context: bool = True,
) -> Answer:
    messages, passages = prepare(
        question, retriever, settings, k=k, chapter=chapter, mode=mode, with_context=with_context
    )
    response = llm.chat(settings, messages)
    return finalize(question, response.text, passages, response)
