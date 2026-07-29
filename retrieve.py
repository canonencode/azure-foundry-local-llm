# Week 3 - Retrieval function: embed a query and find the most relevant stored chunks
# .\venv\Scripts\Activate.ps1

import sys
import sqlite3
import json
import math
import re
from collections import Counter
from foundry_local_sdk import Configuration, FoundryLocalManager

# See main.py for why this is needed - non-Latin document content would
# otherwise crash on print() with UnicodeEncodeError on Windows' default
# console codepage.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cosine_similarity(a, b):
    if len(a) != len(b):
        # zip() would otherwise silently truncate to the shorter vector and
        # return a plausible-looking but meaningless number - fail loudly
        # instead, since mismatched dimensions mean something upstream (a
        # corrupted row, a mixed embedding model) is already wrong.
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Standalone bootstrap for this file's own __main__ smoke test below - not
# used by main.py/app.py, which load their own clients via build_clients().
def get_embedding_client():
    config = Configuration(app_name="azure-foundry-local-llm-retrieve")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()
    return embedding_model.get_embedding_client()


# Embeddings match on meaning, which is what lets a question find a passage
# that shares no words with it. The weakness is the mirror image: a question
# and a passage can be about the same broad topic and score highly while the
# passage does not actually answer it. Measured on 50 labelled questions, the
# embedding alone ranked the right passage first 74% of the time, and the
# misses were mostly this failure - "What does cosine similarity ignore?"
# returned a general passage about cosine similarity instead of the one
# containing the words "ignoring their length".
#
# BM25 is the counterweight: a plain keyword score that rewards a passage for
# containing the rarer words of the question. Blending the two lifted that to
# 82%, and put the right passage inside the top 3 for all 50 questions.
#
# DENSE_WEIGHT is how much of the blend comes from meaning rather than
# keywords. It is not a delicate constant: every value tested from 0.5 to 0.9
# beat the embedding alone, so 0.7 is a comfortable middle rather than a peak
# balanced on a knife edge.
DENSE_WEIGHT = 0.7


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query, tokenized_corpus, k1=1.5, b=0.75):
    """Score every document against the query with BM25.

    Written out by hand for the same reason cosine_similarity() above is:
    the arithmetic is short enough to read, and pulling in a search library
    for 39 chunks would add a dependency to install and keep working for no
    measurable gain. k1 and b are the standard defaults.
    """
    total_docs = len(tokenized_corpus)
    if not total_docs:
        return []

    average_length = sum(len(tokens) for tokens in tokenized_corpus) / total_docs
    document_frequency = Counter()
    for tokens in tokenized_corpus:
        document_frequency.update(set(tokens))

    query_terms = tokenize(query)
    scores = []
    for tokens in tokenized_corpus:
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            containing = document_frequency.get(term, 0)
            if not containing:
                # A word in the question that appears nowhere in the corpus
                # carries no evidence either way, so it contributes nothing.
                continue
            # Rare words say more about relevance than common ones, so a term
            # in few documents is weighted more heavily than one in many.
            inverse_document_frequency = math.log(
                (total_docs - containing + 0.5) / (containing + 0.5) + 1
            )
            frequency = term_frequency[term]
            # Repeated occurrences count for progressively less, and longer
            # documents are discounted so they can't win on length alone.
            score += inverse_document_frequency * (frequency * (k1 + 1)) / (
                frequency + k1 * (1 - b + b * len(tokens) / average_length)
            )
        scores.append(score)
    return scores


def normalize(scores):
    """Rescale scores to 0..1 so cosine and BM25 can be blended.

    They are otherwise not comparable: cosine is bounded near 0..1 while BM25
    is unbounded and depends on corpus statistics, so adding them raw would
    let BM25 dominate purely because its numbers are bigger.
    """
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    spread = highest - lowest
    if spread == 0:
        # Every document scored the same, so this signal has nothing to say
        # about ranking. A flat 0.5 lets the other signal decide alone.
        return [0.5] * len(scores)
    return [(score - lowest) / spread for score in scores]


def get_top_chunks(query, embedding_client, k=3):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    # timeout=5 makes SQLite retry for up to 5s on "database is locked"
    # instead of failing instantly, in case ingest.py has a write in flight.
    conn = sqlite3.connect("knowledge.db", timeout=5)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT content, embedding FROM documents")
        rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "knowledge.db has no 'documents' table yet - run 'python ingest.py' first."
        ) from exc
    finally:
        conn.close()

    contents = []
    similarities = []
    for content, embedding_str in rows:
        # A single corrupted row (NULL/malformed embedding JSON, wrong
        # dimensions from a partial write) shouldn't take down retrieval
        # for every other document - skip just that row and keep going.
        try:
            embedding = json.loads(embedding_str)
            score = cosine_similarity(query_embedding, embedding)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping corrupted row ({content[:50]!r}...): {exc}")
            continue
        contents.append(content)
        similarities.append(score)

    if not contents:
        return []

    # Rank by the blend of meaning and keywords, but return the plain cosine
    # score alongside each chunk. Callers gate on that number against
    # RELEVANCE_THRESHOLD and display it as "similarity 0.88", so it has to
    # keep meaning exactly what it meant before: a cosine similarity. The
    # blended score decides ordering only and is deliberately not returned.
    keyword_scores = bm25_scores(query, [tokenize(text) for text in contents])
    dense_normalized = normalize(similarities)
    keyword_normalized = normalize(keyword_scores)
    ranked = sorted(
        range(len(contents)),
        key=lambda i: DENSE_WEIGHT * dense_normalized[i]
        + (1 - DENSE_WEIGHT) * keyword_normalized[i],
        reverse=True,
    )
    return [(similarities[i], contents[i]) for i in ranked[:k]]


# Week 3-era smoke test: a quick manual check that retrieval finds
# sensible chunks, predating evaluate.py's proper test harness (Week 5).
# Kept as-is rather than removed - still a fast way to eyeball retrieval.
def main():
    embedding_client = get_embedding_client()

    test_queries = [
        "What does Foundry Local do?",
        "How does cosine similarity work?",
        "What is the World Cup?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        for score, content in get_top_chunks(query, embedding_client):
            print(f"  {score:.4f} - {content}")


if __name__ == "__main__":
    main()
