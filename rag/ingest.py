"""PDF -> clean, labelled pages."""

from __future__ import annotations

import bisect
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass

from .clean import clean_page, split_pages
from .config import Settings

_TOC_LINE = re.compile(r"^(\s*)(.+?)\s{2,}(\d{1,3})\s*$")


@dataclass
class Page:
    pdf_index: int  # 1-based PDF page
    printed_page: int  # page number printed in the book
    text: str
    chapter: str = "Front matter"
    section: str = "Front matter"

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


@dataclass(frozen=True)
class TocEntry:
    title: str
    page: int
    top_level: bool


def extract_raw_pages(pdf_path) -> list[str]:
    """Prefer pdftotext -layout (best for this verse-heavy book)."""
    if shutil.which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return split_pages(proc.stdout)

    # Fallback: pypdf (slower, less faithful layout).
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pdftotext or pypdf to extract the PDF") from exc
    return [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]


def parse_toc(pages: list[str]) -> list[TocEntry]:
    """The table of contents is printed near the end of this Calibre PDF."""
    starts = [i for i, page in enumerate(pages) if "Table of Contents" in page]
    if not starts:
        return []
    start = starts[0]
    entries: list[TocEntry] = []
    for page in pages[start : start + 4]:
        for line in page.splitlines():
            match = _TOC_LINE.match(line.rstrip())
            if not match:
                continue
            indent, title, number = match.groups()
            title = title.strip()
            if len(title) < 3 or title.lower() in {"titlepage", "copyright"}:
                continue
            entries.append(TocEntry(title=title, page=int(number), top_level=not indent))
    return sorted(entries, key=lambda item: item.page)


def label_page(page: int, toc: list[TocEntry]) -> tuple[str, str]:
    if not toc:
        return "Unknown", "Unknown"
    starts = [entry.page for entry in toc]
    pos = bisect.bisect_right(starts, page) - 1
    if pos < 0:
        return "Front matter", toc[0].title
    chapter = "Front matter"
    for entry in toc[: pos + 1]:
        if entry.top_level:
            chapter = entry.title
    return chapter, toc[pos].title


def build_pages(settings: Settings, force: bool = False) -> list[Page]:
    """Extract, clean and label every page; cache the result as JSON."""
    cache_path = settings.pages_path
    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Page(**item) for item in data]

    if not settings.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {settings.pdf_path}")

    raw_pages = extract_raw_pages(settings.pdf_path)
    toc = parse_toc(raw_pages)
    pages: list[Page] = []
    for pdf_index, raw in enumerate(raw_pages, start=1):
        text, printed = clean_page(raw, pdf_index)
        if not text.strip():
            continue
        chapter, section = label_page(printed, toc)
        pages.append(
            Page(
                pdf_index=pdf_index,
                printed_page=printed,
                text=text,
                chapter=chapter,
                section=section,
            )
        )
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(page) for page in pages], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return pages


def load_pages(settings: Settings) -> list[Page]:
    if not settings.pages_path.exists():
        raise RuntimeError("Run `python -m rag.cli ingest` first.")
    data = json.loads(settings.pages_path.read_text(encoding="utf-8"))
    return [Page(**item) for item in data]
