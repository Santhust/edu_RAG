"""Build the static learning site served by GitHub Pages from docs/.

Reads each site/*.html, expands <!--demo:STEM--> markers into escaped source
and captured output from demos/STEM.py and demos/STEM.txt, and writes the
result into docs/ alongside site/style.css. Stdlib only.

Run from the repo root:
    .venv/bin/python tools/build_site.py
"""

from __future__ import annotations

import builtins
import html
import io
import keyword
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"
DOCS_DIR = ROOT / "docs"
DEMOS_DIR = ROOT / "demos"
REPO_URL = "https://github.com/Santhust/edu_RAG"
MARKER = re.compile(r"<!--\s*demo:([A-Za-z0-9_.-]+)\s*-->")
STEP_HEADER = re.compile(r"^(== step \d+:.*==)$", re.MULTILINE)
LONG_STEP = 800
BUILTINS = set(dir(builtins))


def highlight_python(source: str) -> str:
    """Wrap Python tokens in spans carrying CSS classes.

    Args:
        source: raw text of a demo's .py file.

    Returns:
        Escaped HTML whose text content is identical to the source.
    """
    lines = source.splitlines(keepends=True)
    parts: list[str] = []
    previous = ""
    row, col = 1, 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        start_row, start_col = token.start
        while row < start_row:
            parts.append(html.escape(lines[row - 1][col:]))
            row += 1
            col = 0
        if start_col > col:
            parts.append(html.escape(lines[row - 1][col:start_col]))
            col = start_col
        value = token.string
        css = ""
        if token.type == tokenize.COMMENT:
            css = "tok-comment"
        elif token.type == tokenize.STRING:
            css = "tok-string"
        elif token.type == tokenize.NUMBER:
            css = "tok-number"
        elif token.type == tokenize.NAME:
            if keyword.iskeyword(value):
                css = "tok-keyword"
            elif previous in {"def", "class"}:
                css = "tok-def"
            elif value in BUILTINS:
                css = "tok-builtin"
        if css:
            parts.append(f'<span class="{css}">{html.escape(value)}</span>')
        else:
            parts.append(html.escape(value))
        row, col = token.end
        if value.strip():
            previous = value
    return "".join(parts)


def render_output(text: str) -> str:
    """Split captured stdout into expandable per-step blocks.

    Args:
        text: raw captured stdout from a demo's paired .txt file.

    Returns:
        HTML details elements, one per printed step; long steps start closed.
    """
    parts = STEP_HEADER.split(text)
    blocks: list[str] = []
    if parts[0].strip():
        blocks.append(f"<pre>{html.escape(parts[0].strip())}</pre>")
    for index in range(1, len(parts) - 1, 2):
        summary = html.escape(parts[index].strip().strip("=").strip())
        body = parts[index + 1].strip("\n")
        opened = "" if len(body) > LONG_STEP else " open"
        blocks.append(
            f'<details class="trace-step"{opened}>\n'
            f"  <summary>{summary}</summary>\n"
            f"  <pre>{html.escape(body)}</pre>\n"
            "</details>\n"
        )
    if not blocks:
        blocks.append(f"<pre>{html.escape(text)}</pre>")
    return "\n".join(blocks)


def expand_demo(match: re.Match[str]) -> str:
    """Replace one marker with an escaped source block and output block.

    Args:
        match: regex match whose first group is the demo stem.

    Returns:
        HTML section containing the code, its captured stdout and links.
    """
    stem = match.group(1)
    source = DEMOS_DIR / f"{stem}.py"
    output = DEMOS_DIR / f"{stem}.txt"
    if not source.exists() or not output.exists():
        raise FileNotFoundError(
            f"Demo {stem!r} needs both {source.name} and {output.name} in {DEMOS_DIR}"
        )
    code = highlight_python(source.read_text(encoding="utf-8"))
    text = output.read_text(encoding="utf-8")
    blob = f"{REPO_URL}/blob/main/demos/{stem}"
    return (
        '<section class="demo">\n'
        f"  <h2>{html.escape(stem)}</h2>\n"
        '  <details class="demo-source">\n'
        f"    <summary>Source: {html.escape(source.name)} (expand)</summary>\n"
        f'    <pre><code class="language-python">{code}</code></pre>\n'
        "  </details>\n"
        "  <h3>Captured output — expand a step</h3>\n"
        f"  {render_output(text)}\n"
        '  <p class="demo-links">'
        f'<a href="{blob}.py">source on GitHub</a> · '
        f'<a href="{blob}.txt">output on GitHub</a>'
        "</p>\n"
        "</section>\n"
    )


def build() -> list[Path]:
    """Render every site source page into docs/ and copy the stylesheet.

    Returns:
        Paths of the generated HTML pages, in sorted order.
    """
    if not SITE_DIR.is_dir():
        raise FileNotFoundError(f"Missing site source directory: {SITE_DIR}")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for source in sorted(SITE_DIR.glob("*.html")):
        page = MARKER.sub(expand_demo, source.read_text(encoding="utf-8"))
        target = DOCS_DIR / source.name
        target.write_text(page, encoding="utf-8")
        written.append(target)

    css = SITE_DIR / "style.css"
    if css.exists():
        (DOCS_DIR / css.name).write_text(css.read_text(encoding="utf-8"), encoding="utf-8")

    return written


if __name__ == "__main__":
    pages = build()
    for page in pages:
        print(f"wrote {page.relative_to(ROOT)}")
