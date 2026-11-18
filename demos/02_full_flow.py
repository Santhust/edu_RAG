"""Follow one question all the way to an Answer object, LLM call included.

Step 1: load Settings and detect the dense backend, as in demo 01.
Step 2: prepare() -> messages + passages; the retrieval-to-generation hand-off.
Step 3: compute the response-cache key and report hit or miss.
Step 4: llm.chat() -> the real HTTP call to Ollama (or an OpenAI-compatible API).
Step 5: finalize() -> Answer; print its fields and sources.
Step 6: show the side effects: cache file and the usage log line.
Step 7: call llm.chat() again to demonstrate the free cache hit.

Run from the repo root:
    .venv/bin/python demos/02_full_flow.py
    .venv/bin/python demos/02_full_flow.py "What did Asita see?"

The .txt beside this file is a snapshot: answer text depends on the model
and on cache state, so a rerun after a cache miss will differ. The demo caps
max_tokens at 300 to keep the CPU-only run short; that also changes the cache
key, so this entry is separate from a normal `ask`. Numbers print at 2 decimals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import llm
from rag.config import Settings
from rag.generate import finalize, prepare
from rag.retrieve import Retriever

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "Where was the bodhisatta born?"

settings = Settings.load()
dense_files = sorted(settings.index_dir.glob("dense-*.npy"))
if dense_files:
    backend = dense_files[0].stem.removeprefix("dense-").split("-")[0]
    settings = settings.overrides(embed_backend=backend, max_tokens=300)

print("== step 1: configuration ==")
print(
    f"  llm={settings.llm_backend}/{settings.llm_model} | "
    f"embed={settings.embed_backend} | top_k={settings.top_k} | "
    f"max_tokens={settings.max_tokens} | think={settings.think}"
)
print(f"  question: {QUESTION}")

retriever = Retriever(settings)
messages, passages = prepare(QUESTION, retriever, settings)
print("\n== step 2: prepare() ==")
print(f"  messages: {len(messages)} -> {[m['role'] for m in messages]}")
print(
    f"  system chars: {len(messages[0]['content'])} | "
    f"user chars: {len(messages[1]['content'])}"
)
print(f"  passages: {len(passages)} -> {[p.chunk.id for p in passages]}")
print("  full excerpt block: see demo 01, step 7")

key = llm._cache_key(settings, messages)
cache_file = llm._cache_file(settings, key)
print("\n== step 3: response cache key ==")
print(f"  key: {key}")
print(f"  file: {cache_file.name} | exists={cache_file.exists()}")

print("\n== step 4: llm.chat() ==")
response = llm.chat(settings, messages)
assert isinstance(response, llm.LLMResponse)  # stream=False returns LLMResponse, not deltas
usage = response.usage
print(
    f"  backend={response.backend} | model={response.model} | "
    f"cached={response.cached}"
)
print(
    f"  prompt_tokens={usage.get('prompt_tokens', 0)} | "
    f"completion_tokens={usage.get('completion_tokens', 0)} | "
    f"total_duration_ms={usage.get('total_duration_ms', 0)}"
)
print("  --- answer text ---")
print(response.text)
print("  --- end answer text ---")

result = finalize(QUESTION, response.text, passages, response)
print("\n== step 5: finalize() -> Answer ==")
print(f"  Answer.question: {result.question}")
print(f"  Answer.text: {len(result.text)} chars")
print(f"  Answer.passages: {len(result.passages)}")
print("  sources: " + " ".join(f"[{p.chunk.id}, {p.chunk.citation}]" for p in result.passages))

print("\n== step 6: side effects ==")
print(f"  cache file now exists: {cache_file.exists()}")
usage_path = settings.cache_dir / "usage.jsonl"
last_line = usage_path.read_text(encoding="utf-8").splitlines()[-1]
print(f"  last usage line: {last_line}")
if response.cached:
    print("  cached responses skip usage logging; that line is from an earlier call")

again = llm.chat(settings, messages)
assert isinstance(again, llm.LLMResponse)
print("\n== step 7: second identical call ==")
print(f"  cached={again.cached} | same_text={again.text == response.text}")
print("  This is why repeated experiments with the same prompt are free.")
