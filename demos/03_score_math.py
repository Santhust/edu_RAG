"""Work TF-IDF and cosine similarity by hand on a three-document toy corpus.

Step 1: tokenise the tiny corpus.
Step 2: count term frequency in each document.
Step 3: compute document frequency and IDF for every term.
Step 4: multiply TF by IDF to get raw weights.
Step 5: L2-normalise each document vector.
Step 6: build and normalise the query vector the same way.
Step 7: dot product equals cosine similarity; rank the documents.

Run from the repo root:
    .venv/bin/python demos/03_score_math.py

No index file, no model and no network. The real vectorizer in rag/index.py
adds smoothing; this toy uses plain ln(n/df) so the numbers are easy to check.
All numbers print at 2 decimals.
"""

import math
from collections import Counter

DOCS = {
    "d1": "the seer saw the child and wept",
    "d2": "the seer saw one lion and the prince",
    "d3": "the prince left the palace and the seer stayed",
}
QUERY = "seer wept prince"


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Args:
        text: a short string of prose.

    Returns:
        One lowercase token per word.
    """
    return text.lower().split()


def show_vector(title: str, vector: dict[str, float], terms: list[str]) -> None:
    """Print a full vector, one term per line.

    Args:
        title: heading for the block.
        vector: term-to-weight mapping.
        terms: ordered term list so the columns line up.

    Returns:
        Nothing; prints to stdout.
    """
    print(f"  {title}")
    for term in terms:
        print(f"    {term:<9} {vector[term]:>6.2f}")


tokens = {name: tokenize(text) for name, text in DOCS.items()}
all_terms = sorted({term for values in tokens.values() for term in values})
print("== step 1: tokenise the toy corpus ==")
for name, values in tokens.items():
    print(f"  {name}: {values}")
print(f"  vocabulary ({len(all_terms)} terms): {all_terms}")

tf = {name: Counter(values) for name, values in tokens.items()}
print("\n== step 2: term frequency ==")
print("  term        d1  d2  d3")
for term in all_terms:
    print(f"  {term:<10} {tf['d1'][term]:>2}  {tf['d2'][term]:>2}  {tf['d3'][term]:>2}")

df = {term: sum(term in values for values in tf.values()) for term in all_terms}
idf = {term: math.log(len(DOCS) / df[term]) for term in all_terms}
print(f"\n== step 3: document frequency and IDF (n={len(DOCS)}) ==")
print("  term          df   idf = ln(n/df)")
for term in all_terms:
    print(f"  {term:<12} {df[term]:>2}    {idf[term]:>6.2f}")
print("  Terms in every document get IDF 0.00: that is the stopword effect.")

weights = {
    name: {term: tf[name][term] * idf[term] for term in all_terms}
    for name in DOCS
}
print("\n== step 4: raw TF-IDF weights ==")
for name in DOCS:
    show_vector(f"{name}:", weights[name], all_terms)


def normalise(vector: dict[str, float]) -> dict[str, float]:
    """Scale a vector to length one.

    Args:
        vector: term-to-weight mapping.

    Returns:
        The same mapping divided by its L2 norm.
    """
    length = math.sqrt(sum(value * value for value in vector.values()))
    if length == 0:
        return dict(vector)
    return {term: value / length for term, value in vector.items()}


unit = {name: normalise(weights[name]) for name in DOCS}
print("\n== step 5: L2-normalised document vectors ==")
for name in DOCS:
    show_vector(f"{name}:", unit[name], all_terms)

query_tokens = tokenize(QUERY)
query_weights = {term: query_tokens.count(term) * idf.get(term, 0.0) for term in all_terms}
query_unit = normalise(query_weights)
print("\n== step 6: query vector ==")
print(f"  query: {query_tokens}")
show_vector("normalised query:", query_unit, all_terms)
print("  'seer' appears in all three documents, so its IDF is zero and it vanishes.")

print("\n== step 7: cosine similarity and ranking ==")
scores = {
    name: sum(unit[name][term] * query_unit[term] for term in all_terms)
    for name in DOCS
}
for name, score in scores.items():
    print(f"  {name}: dot(normalised document, normalised query) = {score:.2f}")
for rank, (name, score) in enumerate(sorted(scores.items(), key=lambda item: -item[1]), 1):
    print(f"  {rank}. {name}  score={score:.2f}")
print("  Both vectors have length one, so the dot product IS the cosine similarity.")
