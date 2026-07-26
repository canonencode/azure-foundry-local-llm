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
  queried and fetched results, then queried a single row by `id` and filtered
  rows with a `LIKE` keyword match (added during the Week 5 audit below —
  the original version only did `SELECT *`, missing the plan's explicit
  "query by id or filter by keyword" instruction)
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

### Week 4 — LLM Integration & Application Assembly ✅
- `main.py` — assembled the end-to-end CLI: `answer_query()` retrieves the
  top chunks, gates the call to the chat model with a relevance threshold
  (`0.5`) before ever invoking it, and streams a grounded answer from
  `phi-3-mini-4k` using a system prompt with the retrieved context
- Implemented the code-level relevance filter flagged as a gap in Week 2 —
  since the model itself doesn't reliably refuse out-of-context questions,
  off-topic queries are now short-circuited to "I don't have that
  information" before the LLM is called at all
- Verified end-to-end: an in-scope question ("What is Foundry Local?") got a
  grounded, source-based answer; an out-of-scope question ("What is the
  capital of France?") correctly triggered the fallback
- Code build-out is done; the plan's optional stretch item (source citations,
  "according to Document X...") was deliberately skipped as non-required
- **Still open:** the plan's Week 4 instruction to log retrieved chunks for
  verification (page 10-11: "ensure the retrieval is happening... log
  retrieved chunks for verification") was never implemented in `main.py` —
  a real gap, not optional
- Tutoring status: `ingest.py` and `main.py` have been walked through
  line-by-line with the user (imports, SQL, embeddings, the relevance gate,
  streaming). `retrieve.py`'s `cosine_similarity()` has been explained
  conceptually (why divide by `norm_a * norm_b`) but not yet walked line-by-
  line the way the other two files were
- **Next up:** either finish `retrieve.py`'s line-by-line walkthrough, or
  build the retrieval-logging fix (good candidate for the scaffolded
  "user writes it" technique — see the "Session continuity" note below), or
  move to Week 5 (System Testing & Evaluation) once Week 4 is fully closed out

### Week 5 — System Testing & Evaluation ✅

**Functional testing**
- Closed the Week 4 gap: `answer_query()` in `main.py` now logs retrieved
  `(score, content)` chunks before answering, and returns the answer text
  instead of only printing it (needed so tests can check it programmatically)
- `evaluate.py` — functional test harness covering three categories the plan
  calls for: answerable questions, off-topic/unanswerable questions, and
  edge cases. Edge cases include the plan's own named examples — empty
  query input, and a general question ("Can you help me with something?")
  — plus one self-devised case (a gibberish string), added because it
  exposed a real bug (below), not because the plan asked for it specifically
- Two real bugs found and fixed during testing:
  1. A gibberish query scored 0.69 similarity — above the 0.5
     `RELEVANCE_THRESHOLD` — by chance token overlap with a stored chunk,
     so it reached the LLM instead of getting rejected. Fixed with
     `is_gibberish()` in `main.py`: rejects any word with 5+ consecutive
     consonants (treating `y` as vowel-like) before the question is ever
     embedded. (First tried a 4-consonant threshold; that false-positived
     on the real word "Foundry", so raised to 5.)
  2. Calling `answer_query("", ...)` directly crashed with
     `ValueError('Input must be a non-empty string.')` — the CLI's blank-input
     skip in `main()` masked this, but the function itself wasn't safe.
     Fixed by rejecting `not question.strip()` the same way as gibberish,
     before any embedding call.
- The general-question edge case ("Can you help me with something?") also
  scored above threshold (0.59) and reached the LLM, same as the gibberish
  case — but the model asked for clarification instead of fabricating an
  answer, so this one was left alone; it's a real question, not garbage
  input, and graceful clarification is reasonable behavior here
- Final result: 10/10 test cases pass

**Performance & Debugging** — checked against the plan's three named
optimizations specifically, not just "discussed" in the abstract:
- *"Retrieving fewer chunks"* — tested directly: `get_top_chunks` with
  `k=2` vs `k=3` measured within noise of each other (~350-425ms either
  way), because it computes cosine similarity against all 8 stored
  embeddings regardless of `k` — `k` only slices the result afterward. So
  this optimization would not help here; verified, not assumed
- *"Using a smaller model"* — already satisfied since Week 4: `phi-3-mini-4k`
  was chosen specifically for its small size, matching the plan's own
  suggestion ("Phi-3.5 Mini or similar," picked for speed)
- *"Caching embeddings instead of recomputing them"* — already satisfied
  since Week 3: `ingest.py` stores document embeddings in `knowledge.db`
  and only recomputes on rerun if content changed
- No incorrect retrieval or formatting issues found; every test question's
  top-scored chunk was the actually-relevant one

**Evaluation and Improvement** — the plan's own example for fixing
long/repetitive answers is "adjust the prompt format," so that's what got
applied (not a token-length cap, which was tried first and reverted — see
below):
- Self-critique found two answers (RAG explanation, cosine-similarity
  explanation) read long and repetitive against a "concise" instruction
- Fix: reworded `SYSTEM_PROMPT` in `main.py` to explicitly cap answers at
  2 sentences and forbid repeating the question
- Verified with real before/after timing on the same two questions: RAG
  explanation 11.2s -> 7.2s, cosine-similarity explanation 15.1s -> 6.3s,
  both now 2 clean sentences with no repetition
- **Dead end, kept for the record:** first tried capping
  `chat_client.settings.max_tokens` instead of touching the prompt. At 150
  tokens it cut the cosine-similarity answer off mid-sentence with no
  closing punctuation — a real regression (a truncated answer looks broken,
  not concise). Raised to 220 to stop the truncation, but once the prompt
  fix above was tested and shown to solve both conciseness and speed more
  effectively on its own, the token cap was removed entirely rather than
  stacking both fixes

## Session continuity
Working style established with this user: tutoring, not vibe-coding — explain
concepts before code, small chunks, ask the user to explain things back, and
when the user needs to write code themselves, use a scaffolded technique
(sketch the module map, give each small piece a contract, user writes 3-8
lines at a time, test in isolation before integrating). Git commits should
not include a Co-Authored-By trailer. Full detail lives in this session's
chat history, but the essentials above are what a fresh conversation needs to
pick this project back up without re-deriving any of it.

## Notes / Known Issues
- Document's example model `phi-1.5` is not in the actual Foundry Local 
  catalog; using `phi-3-mini-4k` instead
- Document's SQLite reference link is for C#/.NET (Windows app dev), not 
  Python — using Python's built-in `sqlite3` module instead