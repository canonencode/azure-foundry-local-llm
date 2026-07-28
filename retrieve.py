# Week 3 - Retrieval function: embed a query and find the most relevant stored chunks
# .\venv\Scripts\Activate.ps1

import sys
import sqlite3
import json
import math
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


def get_top_chunks(query, embedding_client, k=3):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    conn = sqlite3.connect("knowledge.db")
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

    scored = []
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
        scored.append((score, content))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


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
