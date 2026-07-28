# Week 3 - Real Ingestion pipeline: chunk documents, embed each chunk, store in SQLite (with update-if-exists logic)
# .\venv\Scripts\Activate.ps1

import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

# Each entry is a source document - can be one short fact (as below) or a
# longer, multi-paragraph text. chunk_documents() below splits these into
# passage-sized pieces before embedding, so adding longer documents here
# doesn't require any other changes to this file.
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


def chunk_text(text, max_chars=500):
    """Split text into paragraph-sized chunks (blank-line-separated), so RAG
    retrieval operates on passage-level pieces rather than a whole document.
    A paragraph longer than max_chars gets split further on sentence
    boundaries, so no single chunk is too large to be a useful, specific
    match for a query.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        # Split on sentence boundaries first. A "sentence" can still exceed
        # max_chars if there's no nearby period (e.g. no punctuation at
        # all) - hard-split it by character count as a last resort so no
        # chunk is ever unboundedly large.
        pieces = []
        for sentence in paragraph.split(". "):
            while len(sentence) > max_chars:
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                pieces.append(sentence)

        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)

    return chunks


def chunk_documents(documents):
    chunks = []
    for document in documents:
        chunks.extend(chunk_text(document))
    return chunks


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

    config = Configuration(app_name="azure-foundry-local-llm-ingest")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chunks = chunk_documents(documents)

    response = embedding_client.generate_embeddings(chunks)
    chunk_embeddings = [item.embedding for item in response.data]

    # Keyed on the UNIQUE doc_index column so rerunning this script updates
    # existing rows in place instead of inserting duplicates.
    for doc_index, content in enumerate(chunks):
        embedding = chunk_embeddings[doc_index]
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

    # Chunks are keyed by their position in the flattened list above (via
    # enumerate), so if a document was removed or now produces fewer chunks,
    # any row at or past the new shorter length is orphaned - delete it
    # rather than leaving a stale chunk retrievable forever.
    cursor.execute("DELETE FROM documents WHERE doc_index >= ?", (len(chunks),))

    conn.commit()
    conn.close()
    print(f"Ingestion complete. {len(documents)} documents -> {len(chunks)} chunks processed.")


if __name__ == "__main__":
    main()
