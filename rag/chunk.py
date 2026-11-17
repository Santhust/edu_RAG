"""Structure-aware chunking with page and chapter provenance."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .config import Settings
from .ingest import Page, load_pages


@dataclass
class Chunk:
    id: str
    text: str
    chapter: str
    section: str
    page_start: int
    page_end: int

    @property
    def citation(self) -> str:
        page = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{page}, {self.chapter}"


def _tidy_lines(page: Page) -> list[str]:
    lines: list[str] = []
    blank = False
    for line in page.lines:
        if not line.strip():
            if not blank and lines:
                lines.append("")
            blank = True
        else:
            lines.append(line)
            blank = False
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _tail_overlap(buffer: list[tuple[str, Page]], overlap: int) -> list[tuple[str, Page]]:
    if overlap <= 0 or len(buffer) < 2:
        return []
    total = 0
    tail: list[tuple[str, Page]] = []
    for line, page in reversed(buffer):
        if not line.strip():
            continue
        total += len(line) + 1
        tail.append((line, page))
        if total >= overlap:
            break
    tail.reverse()
    # Keep at most half a chunk so overlap cannot dominate a chunk.
    return tail[: max(1, len(buffer) // 2)]


def chunk_pages(
    pages: list[Page], chunk_size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    min_size = max(200, chunk_size // 3)
    chunks: list[Chunk] = []
    buffer: list[tuple[str, Page]] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        lines = [line for line, _ in buffer]
        text = "\n".join(lines).strip()
        if not text:
            buffer, size = [], 0
            return
        source_pages = [page for _, page in buffer]
        chunks.append(
            Chunk(
                id=f"c{len(chunks) + 1:04d}",
                text=text,
                chapter=source_pages[0].chapter,
                section=source_pages[0].section,
                page_start=min(page.printed_page for page in source_pages),
                page_end=max(page.printed_page for page in source_pages),
            )
        )

    for page in pages:
        lines = _tidy_lines(page)
        for line in lines:
            line_size = len(line) + 1
            chapter_change = bool(buffer) and buffer[-1][1].chapter != page.chapter
            too_long = size + line_size > chunk_size and size >= min_size
            if chapter_change or too_long:
                previous = buffer
                flush()
                # Never carry overlap across a chapter boundary: the chunk's
                # chapter label is taken from its first line.
                buffer = [] if chapter_change else _tail_overlap(previous, overlap)
                size = sum(len(line) + 1 for line, _ in buffer)
            buffer.append((line, page))
            size += line_size
    flush()

    # Merge a tiny trailing chunk back into its predecessor (same chapter only).
    if len(chunks) >= 2:
        last = chunks[-1]
        if len(last.text) < min_size and last.chapter == chunks[-2].chapter:
            prev = chunks[-2]
            prev.text = prev.text.rstrip() + "\n\n" + last.text.strip()
            prev.page_end = max(prev.page_end, last.page_end)
            chunks.pop()

    for chunk in chunks:
        chunk.text = re.sub(r"\n{3,}", "\n\n", chunk.text).strip()
    return chunks


def build_chunks(
    settings: Settings, force: bool = False, pages: list[Page] | None = None
) -> list[Chunk]:
    if settings.chunks_path.exists() and not force and pages is None:
        return load_chunks(settings)
    pages = pages or load_pages(settings)
    chunks = chunk_pages(pages, settings.chunk_size, settings.overlap)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text(
        "\n".join(json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    return chunks


def load_chunks(settings: Settings) -> list[Chunk]:
    if not settings.chunks_path.exists():
        raise RuntimeError("Run `python -m rag.cli ingest` first.")
    return [
        Chunk(**json.loads(line))
        for line in settings.chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
