"""Sparse (TF-IDF) and optional dense vector indexes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunk import Chunk
from .config import Settings


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class LexicalIndex:
    """TF-IDF: a sparse vector index. No model download, fully explainable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vectorizer = None
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk], force: bool = False) -> "LexicalIndex":
        path = self.settings.lexical_path
        if path.exists() and not force:
            return self.load()
        from joblib import dump
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            max_features=80_000,
        )
        sparse = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        self.matrix = sparse.toarray().astype(np.float32)
        self.matrix = _normalise(self.matrix)
        dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, path)
        return self

    def load(self) -> "LexicalIndex":
        from joblib import load

        data = load(self.settings.lexical_path)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        return self

    @property
    def ready(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None

    def scores(self, query: str) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Lexical index is not built. Run `ingest` first.")
        vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        return (self.matrix @ _normalise(vector)[0]).astype(np.float32)


class DenseIndex:
    """Neural embeddings: fastembed locally, or an OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.matrix: np.ndarray | None = None
        self._model = None

    @property
    def path(self) -> Path:
        return self.settings.dense_path()

    def _get_fastembed(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "Dense retrieval needs fastembed: pip install fastembed"
                ) from exc
            self._model = TextEmbedding(model_name=self.settings.embed_model)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        backend = self.settings.embed_backend
        if backend == "fastembed":
            model = self._get_fastembed()
            vectors = np.asarray(list(model.embed(texts, batch_size=32)), dtype=np.float32)
        elif backend == "openai":
            from .llm import embed_texts

            vectors = np.asarray(embed_texts(self.settings, texts), dtype=np.float32)
        else:
            raise RuntimeError(f"Unknown embedding backend: {backend}")
        return _normalise(vectors)

    def build(self, chunks: list[Chunk], force: bool = False) -> "DenseIndex":
        if self.path.exists() and not force:
            self.matrix = np.load(self.path)
            return self
        self.matrix = self._embed([chunk.text for chunk in chunks])
        np.save(self.path, self.matrix)
        return self

    def load(self) -> "DenseIndex":
        if not self.path.exists():
            raise RuntimeError(
                f"No dense index at {self.path}. Run `python -m rag.cli ingest --embed-backend "
                f"{self.settings.embed_backend}` to build it."
            )
        self.matrix = np.load(self.path)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self.matrix is None:
            self.load()
        vector = self._embed([query])[0]
        return (self.matrix @ vector).astype(np.float32)
