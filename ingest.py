# Week 3 - Real Ingestion pipeline: embed documents, store in SQLite (with update-if-exists logic)
# .\venv\Scripts\Activate.ps1

import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

documents = [
    "Foundry Local is a tool that was created by Microsoft to run LLM models locally on your computer.",
    "RAG uses a databank and retrieves a chunk relevant to your question. Then it augments that prompt and creates the final prompt for your LLM model. Then the LLM receives the prompt and gives the answer.",
    "We use embedding to turn words into vector values. Then we check the vector values with cosine similarity to find relevance between words.",
    "Cosine similarity means that you have two vectors. It checks the angle between them, not the magnitude, and finds the relevance amongst them.",
    "SQLite is a tool which allows us to store data on a single file.",
    "System prompt is the behavior we want to get from the model (the limits, rules). User prompt is the question, the request of the user.",
    "phi-3-mini-4k is a small LLM model which runs smoothly on Foundry Local, and qwen3 allows us to embed words into vectors.",
    "Small LLM models sometimes ignore the system prompt (the instructions).",
]

def main():
    # Set up SQLite connection + table
    conn = sqlite3.connect("knowledge.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            doc_index INTEGER UNIQUE,
            content TEXT,
            embedding TEXT
        )
    """)

    config = Configuration(app_name = "azure-foundry-local-llm-ingest")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    response = embedding_client.generate_embeddings(documents)
    doc_embeddings = [item.embedding for item in response.data]

    for doc_index, content in enumerate(documents):
        embedding = doc_embeddings[doc_index]
        embedding_str = json.dumps(embedding)
        cursor.execute("SELECT id FROM documents WHERE doc_index = ?", (doc_index,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                 "UPDATE documents SET content = ?, embedding = ? WHERE doc_index = ?",
                (content, embedding_str, doc_index)    
            )
            

        else:
            cursor.execute(
                "INSERT INTO documents (doc_index, content, embedding) VALUES (?, ?, ?)",
                (doc_index, content, embedding_str)
            )
            
    conn.commit()
    conn.close()
    print(f"Ingestion complete. {len(documents)} documents processed.")


if __name__ == "__main__":
    main()