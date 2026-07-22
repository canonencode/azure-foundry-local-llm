# Local RAG Assistant with Microsoft Foundry Local

A local, offline Q&A assistant built using Microsoft Foundry Local and Python, 
following the Retrieval-Augmented Generation (RAG) pattern. Everything runs 
on-device — no internet connection required after initial model downloads.

## Project Goal

Build a chatbot that answers questions about a small document collection by 
retrieving relevant content locally (via embeddings + SQLite) and feeding it 
to a local LLM for grounded, source-based answers — with zero cloud dependency.

## Tech Stack

- **Microsoft Foundry Local** — on-device LLM runtime (chat + embedding models)
- **Python** — `foundry-local-sdk-winml`
- **SQLite** (`sqlite3`, built-in) — local storage for document chunks + embeddings
- Models used: `phi-3-mini-4k` (chat), `qwen3-embedding-0.6b` (embeddings)

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Progress Log

### Week 1 — Foundations: Setup & First Local Inference ✅
- Installed Foundry Local on Windows; resolved OpenVINO execution provider 
  download issue on first run
- Built `main.py` — loads `phi-3-mini-4k` and streams a chat response
- Fixed an `IndexError` on the final stream chunk (empty `choices` list) by 
  guarding with `if chunk.choices:`
- Cleaned up dependencies into an isolated `venv` + scoped `requirements.txt`

### Week 2 — Embeddings, SQLite, and Prompt Engineering ✅
- `embedding_test.py` — generated embeddings with `qwen3-embedding-0.6b`, 
  computed cosine similarity between a query and sample sentences; correctly 
  matched a Windows-related query to the right sentence (0.79 vs ~0.30 for 
  unrelated sentences)
- `sqlite_test.py` — practiced SQLite basics: created a `documents` table 
  (`id`, `content`, `embedding`), inserted rows safely with `?` placeholders, 
  queried and fetched results
  - Discovered: `CREATE TABLE IF NOT EXISTS` does not prevent duplicate data 
    on repeated script runs — each run re-inserts the same rows
- `prompt_test.py` — tested system-prompt-based context grounding
  - Control test: model correctly answered a question covered by the context
  - Failure test: asked a question unrelated to the context — model ignored 
    instructions and answered from its own training knowledge instead of 
    declining, even after strengthening the prompt wording
  - **Lesson learned:** relying only on the model to respect prompt 
    instructions is not sufficient. A code-level relevance filter (e.g. a 
    cosine similarity threshold) is needed before the model is called, 
    planned for Week 3

### Week 3 — Data Ingestion & Retrieval Pipeline ✅
- `ingest.py` — chunks the knowledge base, embeds each chunk with
  `qwen3-embedding-0.6b`, and stores `(doc_index, content, embedding)` in
  `knowledge.db`, updating existing rows instead of duplicating them on rerun
- `retrieve.py` — `get_top_chunks(query, embedding_client, k)` embeds a query,
  computes cosine similarity against every stored embedding, and returns the
  top-K matching chunks; tested against on-topic and off-topic queries
  (relevant queries scored 0.70–0.87, an unrelated query topped out at 0.34)

## Notes / Known Issues
- Document's example model `phi-1.5` is not in the actual Foundry Local 
  catalog; using `phi-3-mini-4k` instead
- Document's SQLite reference link is for C#/.NET (Windows app dev), not 
  Python — using Python's built-in `sqlite3` module instead