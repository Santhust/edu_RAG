"""See a chunk become a record in a vector database.

Step 1: state the question and the two retrieval paths being compared.
Step 2: load the chunks and open the persistent Chroma collection.
Step 3: inspect one chunk as a stored record with vector and metadata.
Step 4: query Chroma with its own default embedding, not our saved vectors.
Step 5: compare the result with the from-scratch hybrid retriever.
Step 6: filter by metadata and state where this demo stops.

Run from the repo root:
    .venv/bin/python demos/05_chroma_store.py > demos/05_chroma_store.txt

This is the code path behind `python -m rag.cli capstone --tool chroma`.
No chat model is called. All numbers print at 2 decimals.

The captured .txt shows a first run that builds the collection. Later runs
reuse the persistent store, so the "entries before" line will differ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from rag.chunk import load_chunks
from rag.config import Settings
from rag.retrieve import Retriever

QUESTION = "Where was the bodhisatta born?"


settings = Settings.load()
dense_files = sorted(settings.index_dir.glob("dense-*.npy"))
if not dense_files:
    raise RuntimeError("No dense index found; run ingest --embed-backend fastembed")
backend = dense_files[0].stem.removeprefix("dense-").split("-")[0]
settings = settings.overrides(embed_backend=backend)

print("== step 1: two retrieval paths ==")
print(f"  question: {QUESTION}")
print(f"  path A: rag/retrieve.py hybrid — TF-IDF + {backend} bge-small vectors")
print("  path B: Chroma vector store — Chroma's own default embedding")
print("  Path B does not load our saved .npy or .joblib files.")

chunks = load_chunks(settings)
client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
collection = client.get_or_create_collection(
    name="noble_warrior", metadata={"hnsw:space": "cosine"}
)
print("\n== step 2: the persistent collection ==")
print(f"  chunks in book: {len(chunks)}")
print(f"  collection: {collection.name} | entries before: {collection.count()}")
print(f"  persisted at: {(settings.data_dir / 'chroma').relative_to(settings.data_dir.parent)}")
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
print(f"  entries after: {collection.count()}")

result = collection.query(query_texts=[QUESTION], n_results=4)
top_ids = result["ids"][0]
distances = result["distances"]
result_metadatas = result["metadatas"]
assert distances is not None and result_metadatas is not None
record = collection.get(
    ids=[top_ids[0]], include=["metadatas", "documents", "embeddings"]
)
print("\n== step 3: one chunk as a database entry ==")
print(f"  id: {record['ids'][0]}")
metadatas = record["metadatas"]
embeddings = record["embeddings"]
documents = record["documents"]
assert isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], dict)
assert embeddings is not None and len(embeddings) > 0
assert isinstance(documents, list) and documents and isinstance(documents[0], str)
record_metadata = dict(sorted(metadatas[0].items()))
print(f"  metadata: {record_metadata}")
print(f"  vector dimensions: {len(embeddings[0])}")
print(f"  text: {' '.join(documents[0].split())[:140]}...")
print("  An entry is id + vector + text + metadata, not just a row of numbers.")

print("\n== step 4: query the vector store ==")
print("  Chroma embeds the question with its own default model and searches by cosine distance.")
print("  rank  id       distance  chapter")
for rank, (chunk_id, distance, metadata) in enumerate(
    zip(top_ids, distances[0], result_metadatas[0]), start=1
):
    print(f"  {rank:>2}.   {chunk_id:<7} {distance:.2f}      {metadata['chapter']}")

retriever = Retriever(settings, chunks=chunks)
ours = retriever.search(QUESTION, k=4, mode="hybrid")
print("\n== step 5: compare with the from-scratch retriever ==")
print("  rank  Chroma   distance  |  ours     fused  lexical  dense")
ours_by_rank = {rank: item for rank, item in enumerate(ours, start=1)}
for rank in range(1, 5):
    item = ours_by_rank[rank]
    print(
        f"  {rank:>2}.   {top_ids[rank - 1]:<7} {distances[0][rank - 1]:.2f}      |  "
        f"{item.chunk.id:<7} {item.score:.2f}     {item.lexical:.2f}     {item.dense:.2f}"
    )
overlap = set(top_ids) & {item.chunk.id for item in ours}
print(f"  overlapping ids: {len(overlap)} of 4")
print("  Three things changed at once: store, embedding model and score type.")
print("  A difference here is not evidence that a vector store improves recall.")

filtered = collection.query(
    query_texts=[QUESTION], n_results=3, where={"chapter": "Birth"}
)
filtered_distances = filtered["distances"]
assert filtered_distances is not None
print("\n== step 6: metadata filtering and boundary ==")
print("  Entries carry chapter metadata, so filtering is a database query:")
for rank, (chunk_id, distance) in enumerate(
    zip(filtered["ids"][0], filtered_distances[0]), start=1
):
    print(f"  {rank}. {chunk_id} distance={distance:.2f} chapter=Birth")
print("  No LLM is called here. This demo stops at retrieved records;")
print("  rag/llm.py and rag/generate.py perform the answer step separately.")
