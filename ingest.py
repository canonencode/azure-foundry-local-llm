# Week 3 - Real Ingestion pipeline: chunk documents, embed each chunk, store in SQLite (with update-if-exists logic)
# .\venv\Scripts\Activate.ps1

import sys
import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

# See main.py for why this is needed - documents containing non-Latin text
# would otherwise crash on print() with UnicodeEncodeError on Windows'
# default console codepage.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    if max_chars <= 0:
        # The hard character-split below (sentence[:max_chars]) never
        # shrinks the remaining string when max_chars <= 0 - sentence[:0] is
        # "" and sentence[0:] is unchanged, so it would loop forever.
        raise ValueError("max_chars must be a positive integer")

    # Normalize Windows line endings first - this project is Windows-only,
    # and "\r\n\r\n" (e.g. text pasted from Notepad/Word) wouldn't match the
    # "\n\n" paragraph-break check below otherwise, silently treating a
    # multi-paragraph document as one giant unsplit paragraph.
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        # Split on sentence boundaries first. str.split(". ") discards the
        # ". " delimiter, so restore the period on every piece except the
        # last (which already ends with the paragraph's own punctuation) -
        # otherwise rejoining pieces below silently drops internal periods.
        raw_sentences = paragraph.split(". ")
        sentences = [
            s + "." if i < len(raw_sentences) - 1 else s
            for i, s in enumerate(raw_sentences)
        ]

        # A "sentence" can still exceed max_chars if there's no nearby
        # period (e.g. no punctuation at all) - hard-split it by character
        # count as a last resort so no chunk is ever unboundedly large.
        pieces = []
        for sentence in sentences:
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
    if isinstance(documents, str):
        # A bare string is technically iterable (character by character),
        # so without this check a single string passed by mistake would
        # silently get "chunked" one character at a time instead of failing.
        raise TypeError("chunk_documents expects a list of documents, not a single string")
    chunks = []
    for document in documents:
        chunks.extend(chunk_text(document))
    return chunks


def main():
    # timeout=5 makes SQLite retry for up to 5s on "database is locked"
    # instead of failing instantly, in case app.py has a read in flight.
    conn = sqlite3.connect("knowledge.db", timeout=5)
    try:
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

        # If the API ever returned a different count than requested (partial
        # failure, filtering), assume the order is unreliable too and fail
        # loudly - silently zipping mismatched lists by position would pair
        # the wrong embedding with the wrong chunk, permanently, with no way
        # to detect it later.
        if len(chunk_embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: requested {len(chunks)} chunks, "
                f"got {len(chunk_embeddings)} embeddings back. Aborting rather "
                f"than risk storing mismatched content/embedding pairs."
            )

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
        # rather than leaving a stale chunk retrievable forever. Warn loudly
        # first, since this is a destructive, unconfirmed delete - if the
        # count looks surprisingly large, that's worth noticing before it's
        # too late to stop it.
        cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_index >= ?", (len(chunks),))
        stale_count = cursor.fetchone()[0]
        if stale_count:
            print(f"Removing {stale_count} stale row(s) no longer produced by the current documents list.")
        cursor.execute("DELETE FROM documents WHERE doc_index >= ?", (len(chunks),))

        conn.commit()
        print(f"Ingestion complete. {len(documents)} documents -> {len(chunks)} chunks processed.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
