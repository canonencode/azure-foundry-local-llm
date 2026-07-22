# Week 3 - Retrieval function: embed a query and find the most relevant stored chunks
# .\venv\Scripts\Activate.ps1

import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


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
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    scored = []
    for content, embedding_str in rows:
        embedding = json.loads(embedding_str)
        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, content))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


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
