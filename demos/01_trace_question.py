"""Trace one question through retrieval without calling an LLM.

Step 1: read saved Settings, then find the dense index actually on disk.
Step 2: load the chunks that every score refers back to.
Step 3: score the question with TF-IDF (sparse, lexical).
Step 4: score it with fastembed (dense, semantic) when a dense index exists.
Step 5: min-max both score lists, then fuse with hybrid_weight.
Step 6: rank top-k; print lexical-only beside the mode the generator gets.
Step 7: print the exact numbered excerpt block that becomes the user message.

Run from the repo root:
    .venv/bin/python demos/01_trace_question.py
    .venv/bin/python demos/01_trace_question.py "What did Asita see?"

Scores are cosine similarities rounded to 2 decimals for readability.
Fused scores are relative ranks, not probabilities. No chat model is called.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag.config import Settings
from rag.generate import SYSTEM_WITH_CONTEXT, format_context
from rag.index import DenseIndex, LexicalIndex
from rag.retrieve import Retriever, _minmax

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "Where was the bodhisatta born?"

settings = Settings.load()
print("== step 1: configuration ==")
print(
    f"  saved defaults: llm={settings.llm_backend}/{settings.llm_model} | "
    f"embed={settings.embed_backend} | top_k={settings.top_k} | "
    f"hybrid_weight={settings.hybrid_weight:.2f}"
)

dense_files = sorted(settings.index_dir.glob("dense-*.npy"))
has_dense = bool(dense_files)
if has_dense:
    backend = dense_files[0].stem.removeprefix("dense-").split("-")[0]
    print(f"  dense file on disk: {dense_files[0].name} -> backend={backend}")
    settings = settings.overrides(embed_backend=backend)
    print(f"  demo uses embed={settings.embed_backend} so dense_path() resolves to that file")
else:
    print("  no dense-*.npy found: this demo will run lexical only")

retriever = Retriever(settings)
chunks = retriever.chunks
print("\n== step 2: chunks ==")
print(f"  loaded {len(chunks)} chunks from {settings.chunks_path.name}")
print(f"  example: {chunks[0].id} [{chunks[0].citation}]")

lexical = LexicalIndex(settings).load()
lex_scores = lexical.scores(QUESTION)
best_lex = int(lex_scores.argmax())
print("\n== step 3: lexical scores (TF-IDF cosine) ==")
print(
    f"  vector shape: {lex_scores.shape} | min={lex_scores.min():.2f} | "
    f"max={lex_scores.max():.2f} | nonzero={int((lex_scores > 0).sum())}"
)
print(f"  best chunk: {chunks[best_lex].id} [{chunks[best_lex].citation}]")

if has_dense:
    dense = DenseIndex(settings).load()
    dense_scores = dense.scores(QUESTION)
    best_dense = int(dense_scores.argmax())
    print("\n== step 4: dense scores (embedding cosine) ==")
    print(
        f"  vector shape: {dense_scores.shape} | min={dense_scores.min():.2f} | "
        f"max={dense_scores.max():.2f}"
    )
    print(f"  best chunk: {chunks[best_dense].id} [{chunks[best_dense].citation}]")
else:
    dense_scores = None
    print("\n== step 4: dense scores ==")
    print("  skipped: no dense index file for this embed backend")

k = settings.top_k
if dense_scores is None:
    final = lex_scores
    print("\n== step 5: no fusion; lexical scores are the final scores ==")
else:
    lex_norm = _minmax(lex_scores)
    dense_norm = _minmax(dense_scores)
    weight = settings.hybrid_weight
    final = (1 - weight) * lex_norm + weight * dense_norm
    print("\n== step 5: min-max fusion ==")
    print(
        f"  lex_norm: {lex_norm.min():.2f}..{lex_norm.max():.2f} | "
        f"dense_norm: {dense_norm.min():.2f}..{dense_norm.max():.2f}"
    )
    print(f"  final = {1 - weight:.2f} * lex_norm + {weight:.2f} * dense_norm")
print(f"  fused top-{k} indices (manual argsort): {np.argsort(-final)[:k].tolist()}")

lexical_results = retriever.search(QUESTION, k=k, mode="lexical")
mode = "hybrid" if has_dense else "lexical"
results = retriever.search(QUESTION, k=k, mode=mode)
print(f"\n== step 6: top-{k} ranking for \"{QUESTION}\" ==")
print("  lexical only:")
for rank, item in enumerate(lexical_results, start=1):
    print(f"    {rank}. {item.chunk.id} score={item.score:.2f} [{item.chunk.citation}]")
print(f"  {mode} (this is what the generator receives):")
for rank, item in enumerate(results, start=1):
    dense_text = "-" if item.dense is None else f"{item.dense:.2f}"
    print(
        f"    {rank}. {item.chunk.id} score={item.score:.2f} "
        f"(lex={item.lexical:.2f}, dense={dense_text}) [{item.chunk.citation}]"
    )

context = format_context(results)
user_message = f"Question: {QUESTION}\n\nExcerpts:\n{context}"
print("\n== step 7: exact context block in the user message ==")
print(f"  system prompt: {len(SYSTEM_WITH_CONTEXT)} chars | user message: {len(user_message)} chars")
print("  --- begin context ---")
print(context)
print("  --- end context ---")
print("  No LLM call here. llm.chat() would POST these two messages next.")
