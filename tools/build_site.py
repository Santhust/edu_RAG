"""Build the static learning site served by GitHub Pages from docs/.

Reads each site/*.html, expands <!--demo:STEM--> markers into escaped source
and captured output from demos/STEM.py and demos/STEM.txt, and writes the
result into docs/ alongside site/style.css. Stdlib only.

Run from the repo root:
    .venv/bin/python tools/build_site.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"
DOCS_DIR = ROOT / "docs"
DEMOS_DIR = ROOT / "demos"
REPO_URL = "https://github.com/Santhust/edu_RAG"
MARKER = re.compile(r"<!--\s*demo:([A-Za-z0-9_.-]+)\s*-->")


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
    code = html.escape(source.read_text(encoding="utf-8"))
    text = html.escape(output.read_text(encoding="utf-8"))
    blob = f"{REPO_URL}/blob/main/demos/{stem}"
    return (
        '<section class="demo">\n'
        f"  <h2>{html.escape(stem)}</h2>\n"
        "  <h3>Source</h3>\n"
        f'  <pre><code class="language-python">{code}</code></pre>\n'
        "  <h3>Captured output</h3>\n"
        f"  <pre>{text}</pre>\n"
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
