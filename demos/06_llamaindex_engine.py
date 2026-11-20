"""See LlamaIndex wrap retrieval and generation into one query engine.

Step 1: state the question, the local model and the embedding adapter.
Step 2: turn the same chunks into LlamaIndex Documents with metadata.
Step 3: build a VectorStoreIndex using our local fastembed adapter.
Step 4: retrieve nodes without calling the LLM.
Step 5: ask the query engine, which retrieves, prompts and calls Ollama.
Step 6: name what the framework replaced and where the boundary is.

Run from the repo root:
    .venv/bin/python demos/06_llamaindex_engine.py > demos/06_llamaindex_engine.txt

This is the code path behind `python -m rag.cli capstone --tool llamaindex`.
The step 5 answer is a snapshot: LlamaIndex does not use this repo's response
cache, so a rerun can produce different wording. All numbers print at 2 decimals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import Document, VectorStoreIndex
from llama_index.core import Settings as LISettings
from llama_index.llms.ollama import Ollama

from rag.capstone import _local_embedder_class
from rag.chunk import load_chunks
from rag.config import Settings

QUESTION = "Where was the bodhisatta born?"


settings = Settings.load()
print("== step 1: question and wiring ==")
print(f"  question: {QUESTION}")
print(f"  generation: {settings.llm_backend} / {settings.llm_model}")
print(f"  embedding: {settings.embed_model}, through the capstone adapter")
print("  thinking: off, to keep the local model's output bounded")

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
first = documents[0]
print("\n== step 2: chunks become Documents ==")
print(f"  Documents: {len(documents)} from {len(chunks)} chunks")
print(f"  first metadata: {first.metadata}")
print(f"  first text: {' '.join(first.text.split())[:110]}...")

Embedder = _local_embedder_class()
LISettings.embed_model = Embedder(model_name=settings.embed_model)
LISettings.llm = Ollama(
    model=settings.llm_model,
    request_timeout=settings.request_timeout,
    thinking=False,
    additional_kwargs={"num_predict": 500},
)
index = VectorStoreIndex.from_documents(documents)
print("\n== step 3: build the index ==")
print(f"  index type: {type(index).__name__}")
print("  VectorStoreIndex embedded every Document with the local adapter.")

engine = index.as_query_engine(similarity_top_k=4)
nodes = engine.retrieve(QUESTION)
print("\n== step 4: retrieve without an LLM call ==")
print("  rank  score  page  chapter")
for rank, item in enumerate(nodes, start=1):
    assert item.score is not None
    node = item.node
    print(
        f"  {rank:>2}.   {item.score:.2f}   {node.metadata['page']:>3}  "
        f"{node.metadata['chapter']}"
    )

print("\n== step 5: query the engine (live local call) ==")
response = engine.query(QUESTION)
print(f"  answer: {' '.join(str(response).split())}")
print("  This line is a captured snapshot, not a reproducible score.")

print("\n== step 6: what the framework owns ==")
print("  Document + Index + QueryEngine replaced our chunk/index/prompt/LLM wiring.")
print("  The vector model is still ours; the LLM is still Ollama.")
print("  Unlike rag/llm.py, this path has no response cache; no answer is reused.")
