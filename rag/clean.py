"""Text cleanup for this particular PDF.

The book was produced from fonts whose ligature glyphs land in Unicode's
Private Use Area.  pdftotext preserves those code points, so words such as
"The" and "after" come out as "\\ue062e" and "a\\ue09der".  Fixing them before
indexing makes a large difference to retrieval quality.
"""

from __future__ import annotations

import re
import unicodedata

# Private-use glyphs used by the book's fonts.
PUA_LIGATURES = {
    "\ue062": "Th",
    "\ue0bb": "Th",
    "\ue09d": "ft",
    "\ue117": "ft",
}

# Standard Unicode ligatures.
UNICODE_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# str.translate() keys must be Unicode ordinals, not characters.
TRANSLATION = {ord(key): value for key, value in {**PUA_LIGATURES, **UNICODE_LIGATURES, "\u00a0": " "}.items()}

_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


def clean_line(line: str) -> str:
    line = line.translate(TRANSLATION)
    # Keep deliberate indentation in verse; only remove trailing whitespace.
    return line.rstrip()


def clean_text(text: str) -> str:
    """Normalise a block of extracted text while retaining line breaks."""
    text = unicodedata.normalize("NFC", text)
    lines = [clean_line(line) for line in text.splitlines()]
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 2:
                out.append("")
            continue
        blank_run = 0
        out.append(line)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def split_pages(raw: str) -> list[str]:
    pages = raw.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


def clean_page(raw_page: str, pdf_index: int) -> tuple[str, int]:
    """Clean one PDF page and return (text, printed_page_number).

    In this book the printed page number is always PDF index minus one.
    The number also appears as a header/footer line, which we strip.
    """
    text = clean_text(raw_page)
    lines = text.splitlines()
    if len(lines) >= 2:
        for position in (0, -1):
            if _PAGE_NUMBER.match(lines[position]):
                lines.pop(position)
    # Title page repeats the running head; drop it.
    if lines and lines[0].strip() == "Noble Warrior":
        lines.pop(0)
    text = clean_text("\n".join(lines))
    return text, max(0, pdf_index - 1)
