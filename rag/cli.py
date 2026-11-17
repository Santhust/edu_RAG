"""Terminal interface: one-shot subcommands plus an interactive REPL."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import llm
from .chunk import build_chunks
from .config import CONFIG_PATH, PROVIDER_PRESETS, Settings, save_public_config
from .evaluate import print_report, run_eval, save_report
from .generate import Answer, answer, finalize, prepare
from .index import DenseIndex, LexicalIndex
from .ingest import build_pages
from .retrieve import Retriever, ScoredChunk

console = Console()


# --------------------------------------------------------------------------- helpers
def _snippet(text: str, width: int = 170) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width] + "…"


def print_passages(passages: list[ScoredChunk], full: bool = False) -> None:
    if not passages:
        console.print("[yellow]No passages retrieved.[/yellow]")
        return
    table = Table(title="Retrieved passages", expand=False)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("text")
    for rank, item in enumerate(passages, start=1):
        chunk = item.chunk
        dense = "-" if item.dense is None else f"{item.dense:.3f}"
        table.add_row(
            str(rank),
            f"{item.score:.3f} (lex {item.lexical:.3f} / dense {dense})",
            f"[bold]{escape(chunk.citation)}[/bold]\n{escape(chunk.section)}\n{chunk.id}",
            escape(chunk.text if full else _snippet(chunk.text)),
        )
    console.print(table)


def print_answer(result: Answer) -> None:
    console.print(
        Panel(
            Text(result.text),
            title=f"Answer — {result.response.backend}/{result.response.model}"
            + (" [cached]" if result.response.cached else ""),
            subtitle=f"{len(result.passages)} passages",
            border_style="green",
        )
    )
    if result.passages:
        console.print(
            "Sources: " + " ".join(f"[{item.chunk.id}, {item.chunk.citation}]" for item in result.passages),
            markup=False,
        )


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.load()
    return base.overrides(
        chunk_size=args.chunk_size if hasattr(args, "chunk_size") else None,
        overlap=args.overlap if hasattr(args, "overlap") else None,
        top_k=args.k if hasattr(args, "k") else None,
        llm_backend=args.backend if hasattr(args, "backend") else None,
        llm_model=args.model if hasattr(args, "model") else None,
        base_url=args.base_url if hasattr(args, "base_url") else None,
        embed_backend=args.embed_backend if hasattr(args, "embed_backend") else None,
        embed_model=args.embed_model if hasattr(args, "embed_model") else None,
        hybrid_weight=args.hybrid_weight if hasattr(args, "hybrid_weight") else None,
        max_tokens=args.max_tokens if hasattr(args, "max_tokens") else None,
        think=args.think if hasattr(args, "think") else None,
        request_timeout=args.timeout if hasattr(args, "timeout") else None,
        show_context=args.show_context if hasattr(args, "show_context") else None,
    )


# --------------------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace) -> None:
    settings = _settings(args)
    console.print(f"[dim]Reading {settings.pdf_path}[/dim]")
    pages = build_pages(settings, force=args.force)
    chunks = build_chunks(settings, force=True, pages=pages)
    LexicalIndex(settings).build(chunks, force=args.force)
    if settings.embed_backend != "tfidf":
        console.print(f"[dim]Building dense vectors with {settings.embed_backend}...[/dim]")
        DenseIndex(settings).build(chunks, force=args.force)

    from collections import Counter

    counts = Counter(chunk.chapter for chunk in chunks)
    table = Table(title="Ingestion complete")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in [
        ("pages", len(pages)),
        ("chunks", len(chunks)),
        ("avg chunk chars", round(sum(len(c.text) for c in chunks) / len(chunks))),
        ("chapters", len(counts)),
        ("index", str(settings.index_dir)),
    ]:
        table.add_row(label, str(value))
    console.print(table)
    console.print("Top chapters: " + ", ".join(f"{name} ({n})" for name, n in counts.most_common(6)))
    console.print("[dim]Tip: `search \"a question\"` to inspect retrieval without an LLM.[/dim]")


def cmd_search(args: argparse.Namespace) -> None:
    settings = _settings(args)
    retriever = Retriever(settings)
    results = retriever.search(args.query, k=args.k, chapter=args.chapter, mode=args.mode)
    retriever.warn_if_needed()
    print_passages(results, full=args.full)


def _stream_answer(settings: Settings, messages: list[dict], passages: list[ScoredChunk], question: str) -> Answer:
    stream = llm.chat(settings, messages, stream=True)
    parts: list[str] = []
    for delta in stream:
        parts.append(delta)
        console.print(delta, end="")
    console.print()
    return finalize(
        question,
        "".join(parts),
        passages,
        llm.LLMResponse("".join(parts), backend=settings.llm_backend, model=settings.llm_model),
    )


def _ask(settings: Settings, question: str, k: int | None, chapter: str | None, mode: str | None,
         with_context: bool, show_context: bool, stream: bool) -> Answer:
    retriever = Retriever(settings)
    messages, passages = prepare(
        question, retriever, settings, k=k, chapter=chapter, mode=mode, with_context=with_context
    )
    retriever.warn_if_needed()
    if show_context and passages:
        from .generate import format_context

        console.print(Panel(Text(format_context(passages)), title="Context sent to the model", border_style="yellow"))
    if stream:
        result = _stream_answer(settings, messages, passages, question)
    else:
        response = llm.chat(settings, messages)
        result = finalize(question, response.text, passages, response)
    print_answer(result)
    return result


def cmd_ask(args: argparse.Namespace) -> None:
    _ask(
        _settings(args),
        args.query,
        args.k,
        args.chapter,
        args.mode,
        with_context=not args.no_context,
        show_context=args.show_context,
        stream=args.stream,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    settings = _settings(args)
    rag = _ask(settings, args.query, args.k, args.chapter, args.mode, True, False, False)
    baseline = _ask(settings, args.query, args.k, args.chapter, args.mode, False, False, False)
    console.print(Panel(baseline.text, title="Baseline (no context)", border_style="red"))
    console.print(
        "RAG answer added "
        f"{len(rag.passages)} cited passages; baseline cited: "
        + ("yes" if "[" in baseline.text else "no")
    )


def cmd_eval(args: argparse.Namespace) -> None:
    settings = _settings(args)
    rows = run_eval(settings, k=args.k, dry_run=args.dry_run, judge=args.judge, tag=args.tag)
    print_report(rows, dry_run=args.dry_run)
    console.print(f"[dim]Saved: {save_report(rows, settings, args.tag)}[/dim]")


def cmd_capstone(args: argparse.Namespace) -> None:
    from .capstone import chroma_demo, llamaindex_demo

    settings = _settings(args)
    if args.tool == "chroma":
        chroma_demo(settings, args.query, args.k or settings.top_k)
    else:
        llamaindex_demo(settings, args.query, args.k or settings.top_k)


def cmd_config(args: argparse.Namespace) -> None:
    settings = Settings.load()
    console.print(json.dumps(settings.public_dict(), indent=2, ensure_ascii=False))
    console.print(f"\nKey from environment: {'yes' if settings.api_key else 'no'}")
    console.print(f"Config file: {CONFIG_PATH}")
    console.print(
        "\nSet RAG_LLM_BACKEND (ollama|openrouter|groq|gemini|custom), "
        "RAG_LLM_MODEL, RAG_API_KEY and optionally RAG_BASE_URL.\n"
        "Examples:\n"
        "  RAG_API_KEY=... RAG_LLM_BACKEND=groq RAG_LLM_MODEL=llama-3.3-70b-versatile python -m rag.cli ask \"...\"\n"
        "  RAG_LLM_BACKEND=ollama python -m rag.cli ask \"...\""
    )
    if args.save:
        console.print(f"Saved non-secret defaults to {save_public_config(settings)}")


# --------------------------------------------------------------------------- REPL
_HELP = """Commands:
  <question>            ask with retrieved context
  search <query>        show ranked passages only (no LLM)
  k <N>                 change top-k
  mode lexical|hybrid|dense   change retrieval mode
  context on|off        show/hide the context sent to the model
  chapter <name>        filter retrieval to a chapter/section
  clear chapter         remove the filter
  backend <name>        ollama | openrouter | groq | gemini | custom
  model <name>          change generation model
  think on|off          enable/disable Ollama thinking (default: off)
  timeout <seconds>     change request timeout (default: 300)
  eval                  run the evaluation set (retrieval only)
  help, quit, exit      ..."""


def cmd_repl(args: argparse.Namespace) -> None:
    settings = _settings(args)
    state: dict[str, Any] = {"chapter": None, "mode": None, "stream": True}
    console.print(
        Panel(
            f"[bold]edu_RAG[/bold] — {settings.llm_backend}/{settings.llm_model}\n"
            f"retrieval: {settings.embed_backend} | top-k: {settings.top_k}\n"
            f"thinking: {'on' if settings.think else 'off'} | timeout: {settings.request_timeout}s\n"
            "Type [bold]help[/bold] for commands, [bold]quit[/bold] to exit.",
            title="Interactive RAG",
            border_style="cyan",
        )
    )
    from rich.prompt import Prompt

    while True:
        try:
            text = Prompt.ask("[bold cyan]rag>[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        low = text.lower()
        if low in {"quit", "exit"}:
            break
        if low == "help":
            console.print(_HELP)
            continue
        try:
            if low.startswith("search "):
                query = text.split(maxsplit=1)[1]
                retriever = Retriever(settings)
                results = retriever.search(query, k=settings.top_k, chapter=state["chapter"], mode=state["mode"])
                retriever.warn_if_needed()
                print_passages(results)
                continue
            if low.startswith("k "):
                settings.top_k = max(1, int(text.split()[1]))
                console.print(f"top-k = {settings.top_k}")
                continue
            if low.startswith("mode "):
                state["mode"] = text.split(maxsplit=1)[1].strip()
                console.print(f"mode = {state['mode']}")
                continue
            if low.startswith("context "):
                settings.show_context = text.split()[1].lower() in {"on", "true", "yes"}
                console.print(f"context = {'on' if settings.show_context else 'off'}")
                continue
            if low.startswith("chapter "):
                state["chapter"] = text.split(maxsplit=1)[1].strip()
                console.print(f"chapter = {state['chapter']}")
                continue
            if low == "clear chapter":
                state["chapter"] = None
                console.print("chapter filter cleared")
                continue
            if low.startswith("backend "):
                new_backend = text.split(maxsplit=1)[1].strip()
                preset_model = PROVIDER_PRESETS.get(new_backend, {}).get("model")
                changes = {"llm_backend": new_backend}
                if preset_model:
                    changes["llm_model"] = str(preset_model)
                settings = settings.overrides(**changes)
                if not preset_model:
                    console.print("[yellow]Set a model with `model <name>` for this backend.[/yellow]")
                console.print(f"backend = {settings.llm_backend}, model = {settings.llm_model}")
                continue
            if low.startswith("model "):
                settings = settings.overrides(llm_model=text.split(maxsplit=1)[1].strip())
                console.print(f"model = {settings.llm_model}")
                continue
            if low.startswith("think "):
                value = text.split(maxsplit=1)[1].strip().lower()
                if value not in {"on", "off", "true", "false", "yes", "no", "1", "0"}:
                    console.print("[red]Use: think on | think off[/red]")
                    continue
                settings = settings.overrides(think=value in {"on", "true", "yes", "1"})
                console.print(f"thinking = {'on' if settings.think else 'off'}")
                continue
            if low.startswith("timeout "):
                settings = settings.overrides(request_timeout=max(1, int(text.split()[1])))
                console.print(f"timeout = {settings.request_timeout}s")
                continue
            if low == "eval":
                rows = run_eval(settings, k=settings.top_k, dry_run=True)
                print_report(rows, dry_run=True)
                continue
            _ask(
                settings,
                text,
                settings.top_k,
                state["chapter"],
                state["mode"],
                with_context=True,
                show_context=settings.show_context,
                stream=state["stream"],
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="A small, hands-on RAG terminal app")
    subs = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, with_query: bool = False) -> None:
        p.add_argument("--backend", "--llm-backend", dest="backend")
        p.add_argument("--model", "--llm-model", dest="model")
        p.add_argument("--base-url", dest="base_url")
        p.add_argument("-k", "--top-k", dest="k", type=int)
        p.add_argument("--embed-backend", choices=["tfidf", "fastembed", "openai"])
        p.add_argument("--embed-model")
        p.add_argument("--hybrid-weight", type=float)
        p.add_argument("--max-tokens", type=int)
        p.add_argument(
            "--think",
            dest="think",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="enable/disable thinking for Ollama models (default: off)",
        )
        p.add_argument("--timeout", type=int, help="request timeout in seconds (default: 300)")
        if with_query:
            p.add_argument("query")

    ingest = subs.add_parser("ingest", help="extract, clean, chunk and index the book")
    common(ingest)
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--overlap", type=int)
    ingest.add_argument("--force", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    search = subs.add_parser("search", help="retrieve passages without calling an LLM")
    common(search, with_query=True)
    search.add_argument("--chapter")
    search.add_argument("--mode", choices=["lexical", "dense", "hybrid"])
    search.add_argument("--full", action="store_true")
    search.set_defaults(func=cmd_search)

    ask = subs.add_parser("ask", help="answer a question with retrieved, cited context")
    common(ask, with_query=True)
    ask.add_argument("--chapter")
    ask.add_argument("--mode", choices=["lexical", "dense", "hybrid"])
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--no-context", action="store_true")
    ask.add_argument("--stream", action="store_true")
    ask.set_defaults(func=cmd_ask)

    compare = subs.add_parser("compare", help="compare RAG answer with a no-context baseline")
    common(compare, with_query=True)
    compare.add_argument("--chapter")
    compare.add_argument("--mode", choices=["lexical", "dense", "hybrid"])
    compare.set_defaults(func=cmd_compare)

    ev = subs.add_parser("eval", help="run the golden question set")
    common(ev)
    ev.add_argument("--dry-run", action="store_true", help="retrieval metrics only, no API calls")
    ev.add_argument("--judge", action="store_true", help="also grade groundedness with the LLM")
    ev.add_argument("--tag", default="last")
    ev.set_defaults(func=cmd_eval)

    capstone = subs.add_parser("capstone", help="optional ChromaDB / LlamaIndex comparison")
    common(capstone, with_query=True)
    capstone.add_argument("--tool", choices=["chroma", "llamaindex"], default="chroma")
    capstone.set_defaults(func=cmd_capstone)

    config = subs.add_parser("config", help="show effective configuration")
    config.add_argument("--save", action="store_true", help="save non-secret defaults")
    config.set_defaults(func=cmd_config)

    repl = subs.add_parser("repl", help="interactive session")
    common(repl)
    repl.set_defaults(func=cmd_repl)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, llm.LLMError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
