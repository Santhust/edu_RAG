"""Optional capstone: the same pipeline expressed with ChromaDB / LlamaIndex.

These imports are deliberately lazy. Install only if you want the comparison:

    pip install chromadb
    pip install llama-index-core llama-index-llms-ollama

The LlamaIndex fastembed integration still pins fastembed<0.2 and does not
install on Python 3.13, so below we wrap fastembed ourselves in ~15 lines.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from .chunk import load_chunks
from .config import Settings

console = Console()


def chroma_demo(settings: Settings, query: str, k: int = 4) -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install first: pip install chromadb") from exc

    chunks = load_chunks(settings)
    client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
    collection = client.get_or_create_collection(
        name="noble_warrior", metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                }
                for chunk in chunks
            ],
        )
        console.print(f"[dim]Indexed {len(chunks)} chunks into ChromaDB[/dim]")

    result = collection.query(query_texts=[query], n_results=k)
    console.print(Panel(f"query: {query}", title="ChromaDB (vector store)", border_style="magenta"))
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        console.print(
            f"[bold]p. {metadata['page_start']} — {metadata['chapter']}[/bold] "
            f"(distance {distance:.3f})\n{' '.join(document.split())[:220]}…\n"
        )


_FASTEMBED_MODELS: dict = {}


def _fastembed_model(name: str):
    if name not in _FASTEMBED_MODELS:
        from fastembed import TextEmbedding

        _FASTEMBED_MODELS[name] = TextEmbedding(model_name=name)
    return _FASTEMBED_MODELS[name]


def _local_embedder_class():
    from llama_index.core.embeddings import BaseEmbedding

    class LocalFastEmbed(BaseEmbedding):
        """Minimal adapter: LlamaIndex embedding API over our local fastembed."""

        model_name: str

        def _get_text_embedding(self, text: str) -> list[float]:
            return list(_fastembed_model(self.model_name).embed([text]))[0].tolist()

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return LocalFastEmbed


def llamaindex_demo(settings: Settings, query: str, k: int = 4) -> None:
    try:
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core import Settings as LISettings
        from llama_index.llms.ollama import Ollama
    except ImportError as exc:
        raise RuntimeError(
            "Install first: pip install llama-index-core llama-index-llms-ollama"
        ) from exc

    chunks = load_chunks(settings)
    documents = [
        Document(
            text=chunk.text,
            metadata={
                "chapter": chunk.chapter,
                "section": chunk.section,
                "page": chunk.page_start,
            },
        )
        for chunk in chunks
    ]
    Embedder = _local_embedder_class()
    LISettings.embed_model = Embedder(model_name=settings.embed_model)
    LISettings.llm = Ollama(model=settings.llm_model, request_timeout=settings.request_timeout)
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine(similarity_top_k=k)
    response = engine.query(query)
    console.print(
        Panel(
            str(response),
            title=f"LlamaIndex query engine — {settings.llm_model}",
            border_style="blue",
        )
    )
