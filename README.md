# edu_RAG — learn RAG by building one

A terminal app that turns one real PDF — *Noble Warrior: A Life of the Buddha*
(652 pages) — into a small retrieval-augmented question-answering system.
The point is to see every gear, so the core uses only Python plus
scikit-learn / numpy / rich, with Ollama or any OpenAI-compatible cloud API
for generation.

## Quick start

```bash
cd /home/sanrpi5/Desktop/learning/edu_RAG

# 1. extract, clean, chunk, index (no network, no API key)
python -m rag.cli ingest

# 2. look at retrieval alone — no LLM involved
python -m rag.cli search "Where was the bodhisatta born?" -k 5

# 3. ask with retrieved, cited context (local Ollama by default)
python -m rag.cli ask "Where was the bodhisatta born?" --show-context

# 4. compare against a no-context baseline
python -m rag.cli compare "What are the Buddha's three main accomplishments?"

# 5. measure retrieval quality without spending tokens
python -m rag.cli eval --dry-run

# 6. interactive session
python -m rag.cli repl
```

Data written by `ingest` lives in `data/` (ignored by version control).
The source PDF is only read.

## Configure a cloud model

Secrets are read from environment variables and never written to project files.
Non-secret defaults can be saved with `python -m rag.cli config --save`.

```bash
# OpenRouter
export RAG_LLM_BACKEND=openrouter
export RAG_API_KEY=sk-or-...
export RAG_LLM_MODEL=openai/gpt-4o-mini       # any model your key can access

# Groq
export RAG_LLM_BACKEND=groq
export RAG_API_KEY=gsk_...
export RAG_LLM_MODEL=llama-3.3-70b-versatile

# Gemini (OpenAI-compatible endpoint)
export RAG_LLM_BACKEND=gemini
export RAG_API_KEY=...
export RAG_LLM_MODEL=gemini-2.5-flash

# back to local, for comparison
export RAG_LLM_BACKEND=ollama
export RAG_LLM_MODEL=qwen2.5:3b
```

Any OpenAI-compatible service works with `RAG_LLM_BACKEND=custom` plus
`RAG_BASE_URL` and `RAG_LLM_MODEL`. CLI flags override the environment for a
single command, e.g. `--backend ollama --model qwen2.5:3b`.

## Ollama thinking models (e.g. `lfm3`)

`lfm3:latest` is a *thinking* model. Ollama normally spends the token budget on
private reasoning and returns an empty `content`, which also causes request
timeouts. Thinking is therefore **off by default** for Ollama requests.

```bash
# works: thinking disabled (the default)
python -m rag.cli ask "Where was Buddha born?" --backend ollama --model lfm3:latest --no-think -k 4

# deliberately enable it (slower; may need more tokens/time)
python -m rag.cli ask "Where was Buddha born?" --backend ollama --model lfm3:latest --think --max-tokens 800 --timeout 600

# persistent default via environment
export RAG_THINK=1          # or 0 to disable
```

In the REPL use `think on` / `think off` and `timeout 600`. The `think` setting
is part of the response cache key, so toggling it always generates a fresh
answer. `--think` only affects Ollama; OpenAI-compatible providers have their
own reasoning controls.

LLM responses are cached in `data/cache/llm/`, so repeated experiments with the
same prompt/model are free. Usage is appended to `data/cache/usage.jsonl`.

## Learning path — what to look at and what to try

| Stage | Code | Command | Lesson |
|---|---|---|---|
| 1. Ingest | `clean.py`, `ingest.py` | `ingest` | PDF text is dirty. This book's fonts emit private-use ligature glyphs (`Th`, `ft`) that must be mapped before indexing. Page numbers and the table of contents become metadata. |
| 2. Chunk | `chunk.py` | `ingest --chunk-size 800 --overlap 150` | Chunk size and overlap decide what evidence can ever be retrieved. Watch how chapter boundaries are respected. |
| 3. Retrieve | `index.py`, `retrieve.py` | `search ... --mode lexical` | TF-IDF is already a vector search — just sparse and lexical. Inspect scores, not just answers. |
| 4. Generate | `llm.py`, `generate.py` | `ask`, `compare` | The prompt forces citations to `[p. N, Chapter]` and an explicit "I don't know". Compare with the uncited baseline. |
| 5. Dense / hybrid | `index.py` | below | Dense embeddings add semantic matching; fusion combines both score distributions. |
| 6. Evaluate | `evaluate.py`, `eval/golden_qa.json` | `eval --dry-run`, `eval` | Measure recall@k / MRR / citation rate instead of trusting a few impressive demos. |
| 7. Capstone | `capstone.py` | below | Rebuild the same thing with ChromaDB and LlamaIndex to see what those frameworks abstract away. |

### Enable dense retrieval

This machine's system Python is externally managed (PEP 668), so use the
project venv. It is created with `--system-site-packages`, so the existing
numpy / scikit-learn / rich are reused:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install fastembed

.venv/bin/python -m rag.cli ingest --embed-backend fastembed
.venv/bin/python -m rag.cli search "the seer who wept for the newborn prince" --mode hybrid
.venv/bin/python -m rag.cli search "..." --mode hybrid --hybrid-weight 0.8
```

`--embed-backend openai` uses the configured cloud provider's `/embeddings`
endpoint (set `RAG_EMBED_MODEL` / `dense_model_for_openai`). Ollama's embedding
endpoint is disabled on this machine, so native Ollama embeddings are not used.

### Experiments worth running

- `search "old age sickness death mendicant"` vs `search "why did the prince leave the palace"` — lexical vs semantic failure modes.
- `ask ... --no-context` vs `ask ...` — hallucination reduction.
- `eval --dry-run -k 1` vs `-k 8` — precision/recall trade-off. Full `eval`
  generates 14 answers; on this CPU-only machine that is slow with `qwen2.5:3b`.
  Use `--dry-run`, a smaller local model (`--model gemma3:1b`), a cloud backend,
  or cap output with `--max-tokens 200`. Partial results are saved after every
  question in `data/cache/eval-<tag>.json`.
- Change `chunk_size` and re-run `eval --dry-run`; keep the config that maximises recall.
- `ask "What did the Buddha have for breakfast?"` — the model should say it doesn't know.

### Results measured here

Retrieval on the 14 golden questions at `k=8`:

| mode | page recall | MRR | chapter accuracy |
|---|---:|---:|---:|
| TF-IDF only | 71% | 0.36 | 79% |
| hybrid (TF-IDF + fastembed) | 86% | 0.48 | 93% |

With hybrid retrieval at `k=4`, a full 14-question answer run gave
`qwen2.5:3b` a 50% citation rate and 0.79 average keyword score (the run was
capped at 160 output tokens, which truncated some citations); with the
default 700-token cap in `ask`, citations appear reliably. `gemma3:1b`
scored 0% citations and produced off-topic answers on several questions.
Model capability matters as much as retrieval: a cloud model usually gives
the cleanest grounding behaviour; the local models are kept for comparison.

### Capstone (optional)

```bash
.venv/bin/pip install chromadb
.venv/bin/python -m rag.cli capstone "What did Asita see?" --tool chroma

.venv/bin/pip install llama-index-core llama-index-llms-ollama
.venv/bin/python -m rag.cli capstone "What did Asita see?" --tool llamaindex

# llama-index-embeddings-fastembed pins fastembed<0.2 and cannot install on
# Python 3.13; capstone.py wraps fastembed directly instead.
```

Compare `capstone.py` with `chunk.py` + `retrieve.py`: Chroma replaces the
vector store and metadata filter; LlamaIndex replaces the document/chunk/index/
query-engine wiring.

## Module map

```
rag/config.py     settings, provider presets, key handling
rag/clean.py      ligature + page cleanup
rag/ingest.py     pdftotext -> labelled pages (TOC -> chapter/section)
rag/chunk.py      overlapping, structure-aware chunks
rag/index.py      TF-IDF + optional dense vectors
rag/retrieve.py   filtering, score fusion, top-k
rag/llm.py        Ollama / OpenAI-compatible chat + embeddings + cache
rag/generate.py   grounded prompt, citations, answer objects
rag/evaluate.py   golden QA metrics
rag/cli.py        subcommands + REPL
rag/capstone.py   framework comparison
```

## License

SPDX-License-Identifier: GPL-3.0-or-later

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

