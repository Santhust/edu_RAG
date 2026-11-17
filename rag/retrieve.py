"""Retrieval orchestration: filtering, scoring and sparse/dense fusion."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .chunk import Chunk, load_chunks
from .config import Settings
from .index import DenseIndex, LexicalIndex


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    lexical: float
    dense: float | None


def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.ones_like(values) * 0.5
    return (values - low) / (high - low)


class Retriever:
    def __init__(self, settings: Settings, chunks: list[Chunk] | None = None):
        self.settings = settings
        self.chunks = chunks if chunks is not None else load_chunks(settings)
        self.lexical = LexicalIndex(settings)
        self.dense: DenseIndex | None = None
        self._warning: str | None = None

    @property
    def default_mode(self) -> str:
        return "lexical" if self.settings.embed_backend == "tfidf" else "hybrid"

    def _ensure(self, mode: str) -> None:
        if not self.settings.lexical_path.exists():
            raise RuntimeError("Run `python -m rag.cli ingest` first.")
        self.lexical.load()
        wants_dense = mode in {"dense", "hybrid"}
        if wants_dense and self.settings.embed_backend == "tfidf":
            self._warning = (
                "embed_backend is 'tfidf'; falling back to lexical retrieval. "
                "Use --embed-backend fastembed (or openai) after ingest for dense vectors."
            )
            return
        if wants_dense:
            self.dense = DenseIndex(self.settings)
            try:
                self.dense.load()
            except RuntimeError as exc:
                self._warning = f"{exc} Falling back to lexical retrieval."
                self.dense = None

    def search(
        self,
        query: str,
        k: int | None = None,
        chapter: str | None = None,
        mode: str | None = None,
    ) -> list[ScoredChunk]:
        mode = mode or self.default_mode
        k = k or self.settings.top_k
        self._warning = None
        self._ensure(mode)

        lex_scores = self.lexical.scores(query)
        dense_scores = None
        use_dense = mode == "dense" or (mode == "hybrid" and self.dense is not None)
        if use_dense and self.dense is not None:
            dense_scores = self.dense.scores(query)

        if mode == "dense" and dense_scores is None:
            dense_scores = lex_scores  # already warned; keep results usable

        weight = self.settings.hybrid_weight
        if dense_scores is None:
            final = lex_scores
        elif mode == "dense":
            final = dense_scores
        else:
            final = (1 - weight) * _minmax(lex_scores) + weight * _minmax(dense_scores)

        pool = np.arange(len(self.chunks))
        if chapter:
            needle = chapter.casefold()
            filtered = [
                i
                for i, chunk in enumerate(self.chunks)
                if needle in chunk.chapter.casefold() or needle in chunk.section.casefold()
            ]
            if filtered:
                pool = np.array(filtered)

        order = pool[np.argsort(-final[pool])][:k]
        results = []
        for index in order:
            results.append(
                ScoredChunk(
                    chunk=self.chunks[index],
                    score=float(final[index]),
                    lexical=float(lex_scores[index]),
                    dense=float(dense_scores[index]) if dense_scores is not None else None,
                )
            )
        return results

    @property
    def warning(self) -> str | None:
        return self._warning

    def warn_if_needed(self) -> None:
        if self._warning:
            print(f"[warning] {self._warning}", file=sys.stderr)
