"""Evaluation harness: does retrieval find the right passage, and does the
answer stay grounded in it?"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from . import llm
from .config import EVAL_DIR, Settings
from .generate import NOT_FOUND, answer
from .retrieve import Retriever

_CITATION = re.compile(r"p{1,2}\.?\.?\s*\d+", re.IGNORECASE)
_PAGE_TOLERANCE = 2


@dataclass
class EvalRow:
    id: str
    question: str
    expected_page: int
    page_rank: int | None = None
    chapter_hit: bool = False
    answer: str = ""
    citations: bool = False
    keyword_score: float = 0.0
    grounded: bool | None = None
    error: str = ""


def load_golden() -> list[dict[str, Any]]:
    path = EVAL_DIR / "golden_qa.json"
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def _judge(question: str, excerpts: str, candidate: str, settings: Settings) -> bool | None:
    from .generate import format_context

    messages = [
        {
            "role": "system",
            "content": "You grade grounding. Answer with exactly YES or NO. YES only if every factual claim in the ANSWER is supported by the EXCERPTS and its citations point to plausible excerpts.",
        },
        {
            "role": "user",
            "content": f"QUESTION: {question}\n\nEXCERPTS:\n{excerpts}\n\nANSWER:\n{candidate}",
        },
    ]
    try:
        response = llm.chat(settings, messages)
    except llm.LLMError:
        return None
    return response.text.strip().upper().startswith("YES")


def run_eval(
    settings: Settings,
    k: int | None = None,
    dry_run: bool = False,
    judge: bool = False,
    tag: str = "last",
) -> list[EvalRow]:
    retriever = Retriever(settings)
    rows: list[EvalRow] = []
    for item in load_golden():
        row = EvalRow(
            id=item["id"],
            question=item["question"],
            expected_page=int(item["expected_page"]),
        )
        results = retriever.search(item["question"], k=k or max(settings.top_k, 5))
        for rank, scored in enumerate(results, start=1):
            chunk = scored.chunk
            if abs(chunk.page_start - row.expected_page) <= _PAGE_TOLERANCE or (
                chunk.page_start <= row.expected_page <= chunk.page_end
            ):
                row.page_rank = rank
                break
        row.chapter_hit = any(
            str(item["expected_chapter"]).casefold()
            in f"{scored.chunk.chapter} {scored.chunk.section}".casefold()
            for scored in results
        )

        if not dry_run:
            try:
                result = answer(item["question"], retriever, settings, k=k)
                row.answer = result.text
                row.citations = bool(_CITATION.search(result.text))
                keywords = [str(word).casefold() for word in item.get("keywords", [])]
                if keywords:
                    row.keyword_score = sum(
                        word in result.text.casefold() for word in keywords
                    ) / len(keywords)
                if NOT_FOUND.casefold() in result.text.casefold():
                    row.keyword_score = 0.0
                if judge:
                    from .generate import format_context

                    row.grounded = _judge(
                        item["question"],
                        format_context(result.passages),
                        result.text,
                        settings,
                    )
            except Exception as exc:  # keep the run going, report the failure
                row.error = str(exc)[:160]
        rows.append(row)
        # Save after every question so an interrupted run keeps its results.
        save_report(rows, settings, tag)
    return rows


def print_report(rows: list[EvalRow], dry_run: bool) -> None:
    console = Console()
    table = Table(title="RAG evaluation", show_lines=False)
    table.add_column("id")
    table.add_column("Q", max_width=44)
    table.add_column("page@rank", justify="center")
    table.add_column("chap", justify="center")
    if not dry_run:
        table.add_column("cite", justify="center")
        table.add_column("kw", justify="center")
        table.add_column("ground", justify="center")

    for row in rows:
        values = [
            row.id,
            row.question,
            "-" if row.page_rank is None else str(row.page_rank),
            "Y" if row.chapter_hit else "-",
        ]
        if not dry_run:
            values.extend(
                [
                    "Y" if row.citations else "-",
                    f"{row.keyword_score:.2f}",
                    "?" if row.grounded is None else ("Y" if row.grounded else "N"),
                ]
            )
        table.add_row(*values)
    console.print(table)

    n = len(rows) or 1
    found = [row for row in rows if row.page_rank is not None]
    recall = len(found) / n
    mrr = sum(1 / row.page_rank for row in found if row.page_rank) / n
    summary = [
        f"questions: {len(rows)}",
        f"page recall: {recall:.0%}",
        f"MRR: {mrr:.2f}",
        f"chapter accuracy: {sum(row.chapter_hit for row in rows) / n:.0%}",
    ]
    if not dry_run:
        answered = [row for row in rows if not row.error and not row.answer.startswith("I don't know")]
        summary.extend(
            [
                f"citation rate: {sum(row.citations for row in answered) / (len(answered) or 1):.0%}",
                f"avg keyword score: {sum(row.keyword_score for row in answered) / (len(answered) or 1):.2f}",
            ]
        )
        if any(row.grounded is not None for row in rows):
            graded = [row for row in rows if row.grounded is not None]
            summary.append(f"grounded: {sum(bool(row.grounded) for row in graded) / (len(graded) or 1):.0%}")
    errors = [row for row in rows if row.error]
    if errors:
        summary.append(f"errors: {len(errors)} ({errors[0].error})")
    console.print("\n" + " | ".join(summary))


def save_report(rows: list[EvalRow], settings: Settings, tag: str = "last") -> str:
    path = settings.cache_dir / f"eval-{tag}.json"
    path.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
