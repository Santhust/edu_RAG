"""Inspect dense embeddings instead of only their scores.

Step 1: load the chunks and the 1404 x 384 dense matrix.
Step 2: choose labelled chunks from several chapters plus the demo-01 hits.
Step 3: print a vector sample and write full vectors to a JSON asset.
Step 4: compute cosine neighbours inside the selected set.
Step 5: project to 3D with PCA and write an isometric SVG of arrows.
Step 6: state what the plot shows and what it does not mean.

Run from the repo root:
    .venv/bin/python demos/04_embeddings.py

Writes docs/assets/embedding-vectors.json and docs/assets/embeddings-3d.svg.
No chat model is called. All numbers print at 2 decimals.
"""

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from rag.config import Settings
from rag.index import DenseIndex
from rag.retrieve import Retriever
from rag.chunk import load_chunks

QUESTION = "Where was the bodhisatta born?"
ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
PALETTE = ["#2f6b4f", "#8b3a62", "#a27200", "#285f8a", "#6b3fa0"]


def arrow_svg(points: np.ndarray, labels: list[str], colors: list[str]) -> str:
    """Draw labelled 3D arrows using an isometric projection.

    Args:
        points: selected rows of the PCA result, shape (n, 3).
        labels: one short label per row.
        colors: one hex colour per row.

    Returns:
        A complete SVG document as a string.
    """
    scales = np.maximum(np.max(np.abs(points), axis=0), 1e-9)
    scaled = points / scales
    cx, cy, factor = 470, 350, 180

    def project(x: float, y: float, z: float) -> tuple[float, float]:
        px = cx + factor * (x - z) * 0.866
        py = cy - factor * (y * 0.8 + (x + z) * 0.45)
        return px, py

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700" width="900" height="700">',
        '<rect width="900" height="700" fill="#f7f5f0"/>',
        '<text x="32" y="42" font-family="Georgia,serif" font-size="22" fill="#1f2621">Embeddings in three dimensions</text>',
        '<text x="32" y="66" font-family="monospace" font-size="13" fill="#5b665d">3D PCA projection of 384-dimensional vectors; arrows are directions, not book positions</text>',
    ]
    for number, colour in enumerate(dict.fromkeys(colors)):
        lines.append(
            f'<defs><marker id="arrow{number}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker></defs>'
        )
    marker_id = {colour: number for number, colour in enumerate(dict.fromkeys(colors))}

    for axis, name in enumerate(("PC1", "PC2", "PC3")):
        unit = np.zeros(3)
        unit[axis] = 1.0
        x, y = project(*unit)
        lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            'stroke="#c9c5ba" stroke-dasharray="5 5"/>'
        )
        lines.append(
            f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" font-family="monospace" '
            f'font-size="13" fill="#7a837a">{name}</text>'
        )

    for index, (point, label, colour) in enumerate(zip(scaled, labels, colors)):
        x, y = project(*point)
        lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2.5" '
            f'marker-end="url(#arrow{marker_id[colour]})"/>'
        )
        anchor = "end" if x > 740 else "start"
        dx = -9 if x > 740 else 9
        lines.append(
            f'<text x="{x + dx:.1f}" y="{y - 8:.1f}" text-anchor="{anchor}" '
            f'font-family="monospace" font-size="13" fill="{colour}">'
            f"{html.escape(label)}</text>"
        )

    lines.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#1f2621"/>')
    lines.append(
        '<text x="32" y="680" font-family="monospace" font-size="12" fill="#5b665d">'
        'PCA preserves large variance directions only; distance in this plot is a sketch, not the full 384D geometry.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


settings = Settings.load()
dense_files = sorted(settings.index_dir.glob("dense-*.npy"))
if not dense_files:
    raise RuntimeError("No dense index found; run ingest --embed-backend fastembed")
backend = dense_files[0].stem.removeprefix("dense-").split("-")[0]
settings = settings.overrides(embed_backend=backend)

chunks = load_chunks(settings)
dense = DenseIndex(settings).load()
matrix = dense.matrix
retriever = Retriever(settings, chunks=chunks)
hits = retriever.search(QUESTION, k=4, mode="hybrid")

print("== step 1: load the index ==")
print(f"  chunks: {len(chunks)} | dense matrix: {matrix.shape} | dtype: {matrix.dtype}")
print(f"  each row is one chunk vector; every row has length {np.linalg.norm(matrix[0]):.2f}")

chosen: list[int] = []
for chapter in ("The Noble Search", "The Awakening", "Dwelling near Sāvatthī", "The Last Year"):
    candidates = [
        i for i, chunk in enumerate(chunks)
        if chunk.chapter == chapter and len(chunk.text) > 600
    ]
    if candidates:
        chosen.append(candidates[0])
for item in hits:
    index = next(i for i, chunk in enumerate(chunks) if chunk.id == item.chunk.id)
    if index not in chosen:
        chosen.append(index)

print("\n== step 2: selected chunks ==")
for index in chosen:
    chunk = chunks[index]
    print(f"  {chunk.id} [{chunk.citation}] section={chunk.section}")
    print(f"    {' '.join(chunk.text.split())[:110]}...")

ASSETS.mkdir(parents=True, exist_ok=True)
vectors = []
print("\n== step 3: vector sample and full vectors ==")
for index in chosen:
    chunk = chunks[index]
    row = matrix[index]
    vectors.append({
        "id": chunk.id,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "citation": chunk.citation,
        "vector": [round(float(value), 6) for value in row],
    })
    head = ", ".join(f"{value:.2f}" for value in row[:12])
    print(f"  {chunk.id} first 12 of {len(row)}: {head}")
json_path = ASSETS / "embedding-vectors.json"
json_path.write_text(
    json.dumps(vectors, ensure_ascii=False, indent=1), encoding="utf-8"
)
print(f"  full 384-value vectors written to {json_path.relative_to(ASSETS.parent.parent)}")

selected = matrix[chosen]
print("\n== step 4: cosine neighbours inside the selected set ==")
similarity = selected @ selected.T
for position, index in enumerate(chosen):
    others = [i for i in range(len(chosen)) if i != position]
    top = sorted(others, key=lambda i: -similarity[position, i])[:3]
    names = ", ".join(
        f"{chunks[chosen[i]].id}={similarity[position, i]:.2f}" for i in top
    )
    print(f"  {chunks[index].id}: {names}")

pca = PCA(n_components=3)
projected = pca.fit_transform(selected)
print("\n== step 5: PCA projection ==")
print("  explained variance: " + ", ".join(f"{v:.2f}" for v in pca.explained_variance_ratio_))
labels = [f"{chunks[index].id} {chunks[index].chapter[:18]}" for index in chosen]
chapter_colours = {
    chapter: PALETTE[i % len(PALETTE)]
    for i, chapter in enumerate(dict.fromkeys(chunks[index].chapter for index in chosen))
}
colors = [chapter_colours[chunks[index].chapter] for index in chosen]
svg_path = ASSETS / "embeddings-3d.svg"
svg_path.write_text(arrow_svg(projected, labels, colors), encoding="utf-8")
print(f"  wrote {svg_path.relative_to(ASSETS.parent.parent)}")
print(f"  {len(chosen)} arrows: one per selected chunk, from the origin")

print("\n== step 6: how to read the plot ==")
print("  Arrows point in the direction of each chunk's embedding after PCA.")
print("  Similar directions mean similar meaning in the model's space.")
print("  PCA keeps the three largest variance directions of 384, so it discards detail.")
