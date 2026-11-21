"""See LangChain compose the same pipeline with LCEL and a Chroma retriever.

Step 1: state the question, the model and the embedding adapter.
Step 2: wrap the chunks as LangChain Documents with metadata.
Step 3: build a fresh Chroma store with our bge-small embeddings.
Step 4: retrieve nodes and scores, then locate the Lumbini chunk deeper down.
Step 5: compare with the from-scratch hybrid retriever.
Step 6: run an LCEL chain on the top four excerpts.
Step 7: run the same chain with twelve excerpts.
Step 8: name what the framework owns and mark both answers as snapshots.

Run from the repo root:
    .venv/bin/python demos/07_langchain_chain.py > demos/07_langchain_chain.txt

The store is built once in batches of 64 to bound memory, then loaded from
disk on later runs. The captured .txt shows that first build; later runs print
a load line instead. Steps 6 and 7 use qwen2.5:3b because lfm3 returned empty
content under this prompt through langchain-ollama even with reasoning=False.
Both answers are snapshots: LangChain does not use this repo's response cache.
All numbers print at 2 decimals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.chunk import load_chunks
from rag.config import Settings
from rag.retrieve import Retriever

QUESTION = "Where was the bodhisatta born?"


class LocalBge(Embeddings):
    """Expose our cached fastembed model through LangChain's interface."""

    def __init__(self, model_name: str) -> None:
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()


settings = Settings.load()
print("== step 1: question and wiring ==")
print(f"  question: {QUESTION}")
print(f"  generation: {settings.llm_backend} / qwen2.5:3b (stable for this chain)")
print(f"  embedding: {settings.embed_model}, through a LangChain adapter")
print("  reasoning: off, via ChatOllama's reasoning=False parameter")

chunks = load_chunks(settings)
documents = [
    Document(
        page_content=chunk.text,
        metadata={
            "id": chunk.id,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "page_start": chunk.page_start,
        },
    )
    for chunk in chunks
]
print("\n== step 2: chunks become Documents ==")
print(f"  Documents: {len(documents)} from {len(chunks)} chunks")
print(f"  first metadata: {dict(sorted(documents[0].metadata.items()))}")

embedding = LocalBge(settings.embed_model)
store_path = settings.data_dir / "chroma_langchain"
if store_path.exists():
    store = Chroma(persist_directory=str(store_path), embedding_function=embedding)
    store_state = "loaded from disk, no embeddings recomputed"
else:
    store = Chroma(persist_directory=str(store_path), embedding_function=embedding)
    for start in range(0, len(documents), 64):
        store.add_documents(documents[start : start + 64])
    store_state = "built in batches of 64 to bound memory on this machine"
print("\n== step 3: build the vector store ==")
print(f"  entries: {store._collection.count()}")
print(f"  persisted at: {store_path.relative_to(settings.data_dir.parent)}")
print(f"  store: {store_state}")

retriever = store.as_retriever(search_kwargs={"k": 4})
chosen = retriever.invoke(QUESTION)
scored = store.similarity_search_with_score(QUESTION, k=4)
print("\n== step 4: retrieve without an LLM call ==")
print("  rank  id       distance  page  chapter")
for rank, (document, distance) in enumerate(scored, start=1):
    print(
        f"  {rank:>2}.   {document.metadata['id']:<7} {distance:.2f}      "
        f"{document.metadata['page_start']:>3}  {document.metadata['chapter']}"
    )

wide = store.similarity_search_with_score(QUESTION, k=12)
lumbini_rank = next(
    (
        rank
        for rank, (document, _) in enumerate(wide, start=1)
        if "Lumbini" in document.page_content
    ),
    None,
)
print(f"  c0035 with Lumbini: rank {lumbini_rank} of {len(wide)} retrieved")
if lumbini_rank is not None:
    document, distance = wide[lumbini_rank - 1]
    flat = " ".join(document.page_content.split())
    position = flat.find("Lumbini")
    print(f"  c0035 distance: {distance:.2f}, page {document.metadata['page_start']}")
    print(f"  book says: ...{flat[max(0, position - 95) : position + 30]}...")

dense_files = sorted(settings.index_dir.glob("dense-*.npy"))
if not dense_files:
    raise RuntimeError("No dense index found; run ingest --embed-backend fastembed")
backend = dense_files[0].stem.removeprefix("dense-").split("-")[0]
ours = Retriever(settings.overrides(embed_backend=backend), chunks=chunks).search(
    QUESTION, k=4, mode="hybrid"
)
print("\n== step 5: compare with the from-scratch retriever ==")
print("  rank  LangChain |  ours     fused")
for rank in range(1, 5):
    item = ours[rank - 1]
    print(
        f"  {rank:>2}.   {chosen[rank - 1].metadata['id']:<7} |  "
        f"{item.chunk.id:<7} {item.score:.2f}"
    )
overlap = {d.metadata["id"] for d in chosen} & {item.chunk.id for item in ours}
print(f"  overlapping ids: {len(overlap)} of 4")
print("  The store and the scoring rule changed; the embedding model did not.")

context4 = "\n\n".join(
    f"[{document.metadata['id']}] {document.page_content}" for document in chosen
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the numbered excerpts, in at most three sentences. "
            "Cite each sentence with its [id]. If the answer is not in the "
            "excerpts, say \"I don't know based on the book.\"",
        ),
        ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
    ]
)
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=settings.temperature,
    reasoning=False,
    num_predict=200,
)
chain = prompt | llm | StrOutputParser()
print("\n== step 6: LCEL chain with four excerpts (live local call) ==")
print(f"  excerpts: {len(chosen)} (no Lumbini chunk)")
answer4 = chain.invoke({"context": context4, "question": QUESTION})
print(f"  answer: {' '.join(answer4.split())}")
print("  This line is a captured snapshot, not a reproducible score.")

context12 = "\n\n".join(
    f"[{document.metadata['id']}] {document.page_content}"
    for document, _ in wide
)
print("\n== step 7: same chain with twelve excerpts (live local call) ==")
print(f"  excerpts: {len(wide)} (includes c0035 at rank {lumbini_rank})")
answer12 = chain.invoke({"context": context12, "question": QUESTION})
print(f"  answer: {' '.join(answer12.split())}")
print("  This line is also a captured snapshot.")

print("\n== step 8: what the framework owns ==")
print("  Document + Chroma vector store + retriever + LCEL chain are framework code.")
print("  The vector model is still ours; the LLM is still Ollama.")
print("  Steps 6 and 7 differ only in how many excerpts the retriever passed in.")
print("  No response cache: a rerun can produce different wording.")
