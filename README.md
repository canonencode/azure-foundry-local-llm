# Local RAG Assistant with Microsoft Foundry Local

**Project is over.**

A local, offline Q&A assistant I built using Microsoft Foundry Local and Python,
following the Retrieval-Augmented Generation (RAG) pattern. Everything runs
on-device — no internet connection required after initial model downloads.

## Problem Statement

Generic chat models either don't know about a specific, private document
collection at all, or hallucinate an answer that sounds plausible but isn't
grounded in that collection. Sending that data to a cloud API isn't always
acceptable either. The scenario I targeted with this project: someone with a
small set of documents (course notes, FAQs, internal docs) who wants accurate,
source-grounded answers to questions about them — entirely on their own
machine, with no internet dependency and no data leaving the device.

## Project Goal

Build a chatbot that answers questions about a small document collection by
retrieving relevant content locally (via embeddings + SQLite) and feeding it
to a local LLM for grounded, source-based answers — with zero cloud dependency.

## Tech Stack

- **Microsoft Foundry Local** — on-device LLM runtime (chat + embedding models)
- **Python** — `foundry-local-sdk-winml`
- **SQLite** (`sqlite3`, built-in) — local storage for document chunks + embeddings
- Models used: `phi-3-mini-4k` (chat), `qwen3-embedding-0.6b` (embeddings)

## Architecture

The two gates below (gibberish check, relevance threshold) are what make this
pipeline different from a plain "embed, retrieve, answer" loop — I added both
because my testing showed the chat model won't reliably refuse on its own.

```
  question (CLI in main.py, or Streamlit in app.py)
      |
      v
  is_gibberish(question)?  --- yes --> "I don't have that information." (main.py)
      | no
      v
  qwen3-embedding-0.6b.embed(question)                       (Foundry Local)
      |
      v
  get_top_chunks(): cosine_similarity() against every row     (retrieve.py)
  in knowledge.db  [doc_index | content | embedding]          (ingest.py builds
      |                                                        this table via
      v                                                        chunk_text())
  best score >= RELEVANCE_THRESHOLD (0.5)?
      |                        |
      | no                     | yes
      v                        v
  "I don't have               phi-3-mini-4k.complete_streaming_chat()
   that information."          (context = joined top chunks)   (Foundry Local)
      |                        |
      +----------+-------------+
                 v
         streamed back to the CLI / Streamlit UI
```

## Resources

Tutorials and docs I actually referenced while building this:
- [Building Your First Local RAG Application with Foundry Local](https://azurefeeds.com/2026/03/30/building-your-first-local-rag-application-with-foundry-local/) — the Tech Community blog post I modeled this whole project after (retrieve → augment → generate, embeddings + SQLite + a local chat model)
- [What is Foundry Local?](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) — overview I used during Week 1 setup
- [Get started with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/get-started?tabs=windows&pivots=programming-language-python) — install steps and chat-completions API usage (Python), covering both the plan's "Get started" and "Quickstart" resource mentions since Foundry Local's docs don't have those as two separate current pages
- [Tutorial: Build a RAG app with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app?tabs=windows) — the official walkthrough my ingestion/retrieval pipeline follows the shape of
- [SQLite](https://sqlite.org/index.html) — official docs, which I used instead of the internship plan document's own SQLite reference link, since that one pointed to C#/.NET Windows app-dev docs rather than Python (see Notes / Known Issues)
- [Prompt engineering techniques](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering) — basics of system/user prompt construction, which I referenced while designing `SYSTEM_PROMPT` in `main.py` and the Week 2 prompt experiment
- [Streamlit documentation](https://docs.streamlit.io/) — third-party, used for `app.py`'s chat interface, session state, and layout components (Option B from the plan)

## Version Control

I committed each week's work as its own milestone (`Week N - Completed`),
with smaller commits for individual exercises/scripts within a week where it
made sense to split them up — see `git log` for the full history. This keeps
each week's state inspectable on its own rather than squashed into one final
commit, and it let me point at exactly what changed and why at several points
during the project (e.g. the Week 5/6 bug fixes reference specific
before/after behavior).

## Setup

**Prerequisites:** Windows 10/11 with Python 3.10+ on PATH. This project
depends on `foundry-local-sdk-winml`/`foundry-local-core-winml`, which bundle
Windows ML native binaries — it is Windows-only, not cross-platform. An
internet connection is needed for the first run only, to download the two
models below; every run after that is fully offline.

**Quick setup (automated):**
```powershell
.\setup.ps1
```
This creates the venv, installs dependencies, and runs `ingest.py` to build
`knowledge.db`. It's safe to re-run. If PowerShell blocks it with an
execution-policy error, run `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
instead. The script only activates the venv for its own commands — activate
it yourself afterward for your own shell (see below).

**Manual setup**, or if you want to see each step:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ingest.py
```

`requirements.txt` is UTF-16 encoded — preserve that encoding if editing it
by hand or regenerating it (`pip freeze | Out-File -FilePath requirements.txt
-Encoding unicode` in PowerShell), or pip may fail to parse it.

`knowledge.db` and `test.db` are gitignored and not tracked in the repo — on
a fresh clone there is no database file, so `ingest.py` must run once (done
automatically by `setup.ps1`, or manually as shown above) before `main.py` or
`app.py` return anything useful.

**Running the app**, once setup is done and the venv is activated:
```powershell
python main.py          # CLI Q&A loop
streamlit run app.py    # web UI
python check-db.py      # inspect knowledge.db contents (debugging)
```

## Progress Log

### Week 1 — Foundations: Setup & First Local Inference ✅
- Installed Foundry Local on Windows; resolved an OpenVINO execution provider
  download issue on first run
- Built `main.py` — loads `phi-3-mini-4k` and streams a chat response
- Fixed an `IndexError` on the final stream chunk (empty `choices` list) by
  guarding with `if chunk.choices:`
- Cleaned up dependencies into an isolated `venv` + scoped `requirements.txt`

### Week 2 — Embeddings, SQLite, and Prompt Engineering ✅
- `embedding_test.py` — generated embeddings with `qwen3-embedding-0.6b`,
  computed cosine similarity between a query and sample sentences; it
  correctly matched a Windows-related query to the right sentence (0.79 vs
  ~0.30 for unrelated sentences)
- `sqlite_test.py` — practiced SQLite basics: created a `documents` table
  (`id`, `content`, `embedding`), inserted rows safely with `?` placeholders,
  queried and fetched results, then queried a single row by `id` and filtered
  rows with a `LIKE` keyword match (I added this during the Week 5 audit below
  — my original version only did `SELECT *`, missing the plan's explicit
  "query by id or filter by keyword" instruction)
  - I found that `CREATE TABLE IF NOT EXISTS` does not prevent duplicate data
    on repeated script runs — each run re-inserts the same rows
- `prompt_test.py` — tested system-prompt-based context grounding
  - Control test: the model correctly answered a question covered by the context
  - Failure test: I asked a question unrelated to the context — the model
    ignored my instructions and answered from its own training knowledge
    instead of declining, even after I strengthened the prompt wording
  - **Lesson learned:** relying only on the model to respect prompt
    instructions is not sufficient. A code-level relevance filter (e.g. a
    cosine similarity threshold) is needed before the model is called, which
    I planned for Week 3

### Week 3 — Data Ingestion & Retrieval Pipeline ✅
- `ingest.py` — `chunk_text()` splits each entry in the `documents` list on
  blank-line paragraph breaks, further splitting any paragraph over 500
  characters on sentence boundaries; `chunk_documents()` flattens this
  across the whole list. The resulting chunks are embedded with
  `qwen3-embedding-0.6b` and stored as `(doc_index, content, embedding)` in
  `knowledge.db`, updating existing rows instead of duplicating them on
  rerun. (I added this after the fact, because I wanted to support a bigger/
  longer document list later — my original 8 short facts didn't need
  splitting on their own, so I initially skipped it and documented it as a
  limitation before implementing it. I verified it with unit tests on
  synthetic multi-paragraph input, a regression test confirming the
  existing 8 facts still produce exactly 8 unchanged chunks, and a live
  test showing a 3-paragraph document correctly split into 3 independently
  retrievable rows, with a query about one paragraph's topic correctly
  scoring that specific chunk highest)
- `retrieve.py` — `get_top_chunks(query, embedding_client, k)` embeds a query,
  computes cosine similarity against every stored embedding, and returns the
  top-K matching chunks; I tested it against on-topic and off-topic queries
  (relevant queries scored 0.70–0.87, an unrelated query topped out at 0.34)

### Week 4 — LLM Integration & Application Assembly ✅
- `main.py` — assembled the end-to-end CLI: `answer_query()` retrieves the
  top chunks, gates the call to the chat model with a relevance threshold
  (`0.5`) before ever invoking it, and streams a grounded answer from
  `phi-3-mini-4k` using a system prompt with the retrieved context
- Implemented the code-level relevance filter I flagged as a gap in Week 2 —
  since the model itself doesn't reliably refuse out-of-context questions,
  off-topic queries are now short-circuited to "I don't have that
  information" before the LLM is called at all
- Verified end-to-end: an in-scope question ("What is Foundry Local?") got a
  grounded, source-based answer; an out-of-scope question ("What is the
  capital of France?") correctly triggered the fallback
- Code build-out is done; I deliberately skipped the plan's optional stretch
  item (source citations, "according to Document X...") as non-required
- **Still open:** the plan's Week 4 instruction to log retrieved chunks for
  verification (page 10-11: "ensure the retrieval is happening... log
  retrieved chunks for verification") was never implemented in `main.py` —
  a real gap, not optional
- **Next up:** either build the retrieval-logging fix above, or move to
  Week 5 (System Testing & Evaluation) once Week 4 is fully closed out

### Week 5 — System Testing & Evaluation ✅

**Functional testing**
- Closed the Week 4 gap: `answer_query()` in `main.py` now logs retrieved
  `(score, content)` chunks before answering, and returns the answer text
  instead of only printing it (I needed this so tests could check it
  programmatically)
- `evaluate.py` — functional test harness covering the three categories the
  plan calls for: answerable questions, off-topic/unanswerable questions, and
  edge cases. Edge cases include the plan's own named examples — empty
  query input, and a general question ("Can you help me with something?")
  — plus one case I came up with myself (a gibberish string), which I added
  because it exposed a real bug (below), not because the plan asked for it
- I found and fixed two real bugs while testing:
  1. A gibberish query scored 0.69 similarity — above the 0.5
     `RELEVANCE_THRESHOLD` — by chance token overlap with a stored chunk,
     so it reached the LLM instead of getting rejected. I fixed it with
     `is_gibberish()` in `main.py`: it rejects any word with 5+ consecutive
     consonants (treating `y` as vowel-like) before the question is ever
     embedded. (I first tried a 4-consonant threshold; that false-positived
     on the real word "Foundry", so I raised it to 5.)
  2. Calling `answer_query("", ...)` directly crashed with
     `ValueError('Input must be a non-empty string.')` — the CLI's blank-input
     skip in `main()` masked this, but the function itself wasn't safe.
     I fixed it by rejecting `not question.strip()` the same way as gibberish,
     before any embedding call.
- The general-question edge case ("Can you help me with something?") also
  scored above threshold (0.59) and reached the LLM, same as the gibberish
  case — but the model asked for clarification instead of fabricating an
  answer, so I left this one alone; it's a real question, not garbage
  input, and graceful clarification is reasonable behavior here
- Final result: 10/10 test cases pass

**Performance & Debugging** — I checked against the plan's three named
optimizations specifically, rather than just discussing them in the abstract:
- *"Retrieving fewer chunks"* — I tested this directly: `get_top_chunks` with
  `k=2` vs `k=3` measured within noise of each other (~350-425ms either
  way), because it computes cosine similarity against all 8 stored
  embeddings regardless of `k` — `k` only slices the result afterward. So
  this optimization would not help here; I verified that rather than assuming it
- *"Using a smaller model"* — already satisfied since Week 4: I chose
  `phi-3-mini-4k` specifically for its small size, matching the plan's own
  suggestion ("Phi-3.5 Mini or similar," picked for speed)
- *"Caching embeddings instead of recomputing them"* — already satisfied
  since Week 3: `ingest.py` stores document embeddings in `knowledge.db`
  and only recomputes on rerun if content changed
- I found no incorrect retrieval or formatting issues; every test question's
  top-scored chunk was the actually-relevant one

**Evaluation and Improvement** — the plan's own example for fixing
long/repetitive answers is "adjust the prompt format," so that's what I
applied (not a token-length cap, which I tried first and reverted — see
below):
- My self-critique found two answers (RAG explanation, cosine-similarity
  explanation) read long and repetitive against a "concise" instruction
- Fix: I reworded `SYSTEM_PROMPT` in `main.py` to explicitly cap answers at
  2 sentences and forbid repeating the question
- I verified this with real before/after timing on the same two questions: RAG
  explanation 11.2s -> 7.2s, cosine-similarity explanation 15.1s -> 6.3s,
  both now 2 clean sentences with no repetition
- **Dead end, kept for the record:** I first tried capping
  `chat_client.settings.max_tokens` instead of touching the prompt. At 150
  tokens it cut the cosine-similarity answer off mid-sentence with no
  closing punctuation — a real regression (a truncated answer looks broken,
  not concise). I raised it to 220 to stop the truncation, but once I'd
  tested the prompt fix above and shown it solved both conciseness and speed
  more effectively on its own, I removed the token cap entirely rather than
  stacking both fixes

### Added GUI — Streamlit Web Interface
- `app.py` — a Streamlit chat-style front end (Option B from the plan) layered
  over the existing pipeline. It reuses `build_clients()`, `answer_query()`,
  `is_gibberish()`, and `RELEVANCE_THRESHOLD` from `main.py` and
  `get_top_chunks()` from `retrieve.py` unchanged — no modification to the
  underlying RAG logic
- Chat interface via `st.chat_message`/`st.chat_input`; conversation history
  kept in `st.session_state`. Each answer shows an expander labeled either
  "Chunks used for this answer" or "No chunk cleared the relevance threshold
  (0.5)...", so retrieval stays visible and verifiable rather than hidden
  behind the UI
- Delete controls: a per-message `✕` button (no confirmation needed — just
  ask again if you delete one by mistake) and a sidebar "Clear all history"
  button gated behind a confirmation popover (irreversible, so it asks first)
- Custom dark theme (`.streamlit/config.toml` + injected CSS): warm amber
  accent on charcoal, Fraunces serif for headings, IBM Plex Sans/Mono for
  body and technical values (model names, scores); custom SVG favicon and
  user avatar (`assets/`) replacing Streamlit's defaults, after I decided the
  emoji icons looked unprofessional for a product-style interface
- Real bugs I found and fixed while building this, not just cosmetic tweaks:
  - Hiding Streamlit's toolbar (to remove the "Deploy" button) also hid the
    sidebar's reopen arrow, which lives in the same container — this made
    the sidebar permanently unreachable once collapsed. Fixed by relying on
    `config.toml`'s `toolbarMode="minimal"` instead of a blanket CSS hide
  - A global `white-space: nowrap` I added to stop one button's text from
    wrapping broke the longer example-question buttons, which relied on
    wrapping to fit their column, causing them to overlap. I reverted the
    global rule and replaced the delete button's label with a single
    non-wrapping glyph (`✕`) instead of fighting column width
  - The delete button's red hover color silently failed to apply at first —
    my CSS selector assumed the wrong DOM structure. Streamlit wraps
    `st.columns()` output in an extra `stLayoutWrapper` layer that isn't
    visible from the plan/docs; I traced the actual structure directly in the
    browser and rewrote the selector to match
- Verified end-to-end in the browser: grounded answers, the relevance-gate
  fallback, gibberish rejection, delete/clear history, and sidebar
  collapse/reopen all confirmed working together, not just individually

### Week 6 — Code Cleanup & Comments
I ran a full audit across every file in the codebase, which surfaced mostly
cosmetic issues plus three real bugs. I triaged them one at a time — fix now,
or accept and document as a trade-off — and decided all three were worth
fixing, stress-testing the trickiest fix before writing it and verifying each
one after it landed:

- **`main.py`'s `is_gibberish()` false-positived on real words** with long
  consonant runs — "strengths", "twelfths", "catchphrase" were all
  incorrectly rejected as gibberish, and since the check is `any()` over
  words, this misfired on a whole legitimate question if just one word
  triggered it (e.g. "What are the strengths of this approach?"). I proved by
  hand that no single `min_run` threshold can fix this without reopening the
  original Week 5 bug ("catchphrase"'s consonant run is *longer* than the
  known gibberish test case's). Fix: collapse English digraphs (`ch, sh, th,
  ph, gh, wh, ck, ng` — two letters, one sound) into a single unit before
  counting the run, keeping `min_run=5`. I stress-tested this against ~120
  words — it holds up, with one accepted, bounded trade-off: keyboard-mash
  strings that happen to contain a digraph at the exact run-length-5 boundary
  (e.g. `sdfgh`) can now slip past this specific check. I documented that
  rather than hiding it — `RELEVANCE_THRESHOLD` is the backstop, since
  garbage text essentially never scores >= 0.5 similarity against real
  content. Verified with 8 new direct unit checks in `evaluate.py`
  (`check_is_gibberish()`, run before `build_clients()` since it needs no
  models) — all pass, and the existing 10-case end-to-end suite still reports
  10/10 unchanged
- **`ingest.py` never deleted stale rows** — removing a document from the
  hardcoded list left its old row in `knowledge.db` forever, still
  retrievable. Fix: delete any row where `doc_index >= len(documents)` after
  the upsert loop. I verified this live: temporarily shrank the list from 8 to
  7 documents, ran `ingest.py`, confirmed the orphaned row was gone
  (`check-db.py` showed exactly 7 rows, no stale entry), then restored the
  full list and re-ran to confirm `knowledge.db` was back to normal
- **`app.py` called `get_top_chunks()` twice per question** — once for its
  own "chunks used" display, once again inside `answer_query()` — the same
  query re-embedded and every document re-scored twice. Fix: I added an
  optional `top_chunks=None` parameter to `answer_query()`; when provided,
  it skips the internal call. Fully backward compatible (`main.py`'s CLI
  loop and `evaluate.py` never pass it). I verified this live with temporary
  print instrumentation in `retrieve.py`: submitted one question through the
  Streamlit UI, confirmed `get_top_chunks` logged exactly once, then removed
  the instrumentation
- Added comments explaining non-obvious "why"s: `VOWELS` including `y`
  (main.py), the digraph trade-off (main.py), the upsert-by-`doc_index`
  rationale (ingest.py), `retrieve.py`'s standalone smoke-test entry point
  (predates `evaluate.py`, not dead code), and `prompt-test.py`'s empirical
  link to the `RELEVANCE_THRESHOLD` design decision it motivated
- Normalized style across `ingest.py`, `check-db.py`, and all three
  `test-files-week2/` sandbox scripts: consistent keyword-arg spacing,
  comment formatting, trailing whitespace removed, a stray format-spec bug
  fixed (`embedding-test.py` was printing an extra leading space before
  scores), and `check-db.py` brought in line with every other script's
  `def main(): ... if __name__ == "__main__":` structure
- I left two additional false-positive words I found during stress-testing
  (`postscript`, `thumbscrew` — a different, pre-existing class of false
  positive unrelated to digraphs) as a documented, out-of-scope limitation
  rather than fixing them — they're implausible as real query vocabulary for
  this project, and no adjustment can fix them without reopening other cases

### Extensive Bug Hunt & Hardening ✅
A deliberate paranoid pass: a fresh static security/correctness audit of every
file, plus live adversarial testing with real malicious/malformed input
against every non-trivial function. I verified every fix below either by a
live reproduction (a real corrupted DB row, a real missing `knowledge.db`, a
real Turkish sentence) or a regression test, not just by code review.

**Confirmed clean by the audit — no fix needed:**
- SQL injection: every query across the whole codebase uses `?` parameterized
  placeholders; none interpolate values via f-string/`.format()`/`%`
- XSS/HTML injection in `app.py`'s `unsafe_allow_html=True` blocks: none of
  them interpolate user-controlled data (question, answer, retrieved
  content) — only hardcoded CSS and trusted constants

**Critical bugs found and fixed:**
1. **Infinite loop in `chunk_text()`** on `max_chars <= 0` — confirmed live:
   a test call hung, and `Get-Process` showed 160+ seconds of real CPU time
   before I killed it. `sentence[:0]` is `""` and `sentence[0:]` is
   unchanged, so the loop that's supposed to shrink an oversized "sentence"
   never terminates. Fixed by validating `max_chars > 0` up front
2. **`is_gibberish()` flagged all non-Latin-script text as gibberish** —
   `VOWELS` is ASCII-only, so a Turkish/Cyrillic/Arabic/CJK sentence has zero
   recognized vowels and gets rejected outright. Fixed by skipping the
   consonant-run check entirely for any non-ASCII word (the heuristic only
   models English orthography to begin with)
3. **That same non-Latin text then crashed `print()` on Windows** —
   `UnicodeEncodeError` from the console's default `cp1252` codepage, which I
   discovered while adding a Turkish regression test. Without this fix,
   fix #2 above would have been undermined immediately: the app would
   correctly *accept* a Turkish question, then crash trying to echo it back.
   Fixed by forcing UTF-8 stdout (`sys.stdout.reconfigure`) in every
   CLI-facing script (`main.py`, `ingest.py`, `retrieve.py`, `check-db.py`,
   `evaluate.py`)
4. **One corrupted embedding row crashed retrieval for every question, not
   just questions related to that row** — `retrieve.py`'s
   `json.loads(embedding_str)` had no error isolation per row. Confirmed
   live: I inserted a real row with malformed JSON, watched `get_top_chunks()`
   fail entirely, fixed it to skip and log just the bad row, re-ran and
   confirmed the other 8 rows still returned correctly, then removed the
   test row
5. **Missing `knowledge.db`/table crashed `app.py`, `retrieve.py`, and
   `check-db.py` with raw tracebacks** — confirmed live by actually deleting
   `knowledge.db` and re-running each; all three now show a clear "run
   `python ingest.py` first" message instead. I backed up and restored the
   real database around this test
6. **No error handling around Foundry Local SDK calls** — a transient
   failure (service not running, no internet on first model download) on
   any single question crashed the entire CLI session or produced a raw
   Streamlit traceback. `build_clients()` now wraps SDK initialization with
   an actionable message; `main.py`'s CLI loop and `app.py`'s startup no
   longer die from one bad interaction

**Other real bugs found and fixed (lower severity, still real):**
7. `chunk_text()` silently dropped internal periods when a long paragraph
   got split on sentence boundaries and rejoined (`str.split(". ")` discards
   the delimiter) — I confirmed this by reconstructing chunked text and
   diffing it against the original
8. Windows line endings (`\r\n\r\n`) weren't recognized as paragraph breaks
   (only `\n\n` was), so a document pasted from Notepad/Word would silently
   fail to chunk by paragraph at all — this project is Windows-only, so this
   was a realistic gap, not a theoretical one
9. `chunk_documents("a string")` would silently iterate character-by-character
   instead of erroring, since strings are iterable — now raises `TypeError`
10. `cosine_similarity()` silently truncated to the shorter vector on a
    dimension mismatch via `zip()`, producing a plausible-looking but
    meaningless score instead of surfacing that something upstream (a
    corrupted row, mismatched embedding models) was already wrong
11. `ingest.py` had no check that `generate_embeddings()` returned one
    embedding per chunk in the right order — now aborts with a clear error
    instead of risking a silent, permanent, undetectable content/embedding
    mismatch in `knowledge.db`

**New permanent regression coverage I added to `evaluate.py`:** a Turkish
sentence in `GIBBERISH_CHECKS` (bug #2/#3), and a new `check_chunking()`
section covering the `max_chars` guard, period preservation, CRLF handling,
and the string-vs-list type check (bugs #1, #7, #8, #9). I re-ran the full
suite after every single fix: 10/10 end-to-end cases, 9/9 gibberish checks,
5/5 chunking checks, throughout.

**Follow-up: I went back and fixed four of the five lower-priority items
above too** — each verified live, not just reviewed:
- The destructive `DELETE` in `ingest.py` now prints `"Removing N stale
  row(s)..."` before it runs, whenever it's actually about to delete
  something — I confirmed this by temporarily shrinking the document list and
  watching the message appear, then confirming it stays silent on a normal
  re-run where nothing is stale
- Every `sqlite3.connect()` across `ingest.py`, `retrieve.py`, `app.py`,
  `check-db.py`, and `test-files-week2/sqlite-test.py` is now wrapped in
  `try/finally` so the connection always closes, even on an exception
- All of those connections now pass `timeout=5`, so a narrow
  `ingest.py`-vs-`app.py` locking race retries for 5s instead of failing
  instantly with "database is locked"
- `app.py`'s cached document count now uses `st.cache_data(ttl=60)` instead
  of caching forever, so it reflects a re-run of `ingest.py` within a
  minute instead of staying stale until the server restarts

**Left as-is, by my choice:** `doc_index` fragility (editing an early
document shifts every later chunk's index — not corrupting, just causes a
full rewrite of everything after it). A proper fix means keying rows by a
content hash instead of list position, which requires a `knowledge.db`
schema change. I checked what the plan document actually asks for here: it
only says ingestion should be "a simple setup script to re-run... if
documents are added or changed" (optional, Week 3) — nothing about how
rows should be keyed or about update efficiency. Since this isn't a
document requirement and doesn't cause any actual corruption, I deliberately
left it as a documented trade-off rather than fixing it.

## Lessons Learned

Pulling together the insights that are otherwise scattered across the weekly
log above:

- **A model's own instructions aren't a reliable safety mechanism.** Week 2's
  prompt experiment showed the chat model ignoring explicit, even
  emphatic ("DO NOT", "SAY I DON'T KNOW") instructions not to answer
  off-topic questions. I had to enforce relevance filtering in code
  (`RELEVANCE_THRESHOLD` in `main.py`) rather than trusting it to the prompt.
- **A similarity threshold is a probabilistic filter, not a guarantee.**
  Both the original Week 5 bug and the false positives I found in Week 6 came
  from the same root cause: short or unusual text can score above/below
  0.5 by chance, in either direction. The fix was never "pick a better
  number" — it was adding an earlier, cheaper filter (`is_gibberish()`) and
  accepting that even that has its own bounded edge cases, documented
  rather than pretending they don't exist.
- **A heuristic's threshold can be mathematically unfixable in isolation.**
  Week 6's gibberish-detector bug couldn't be solved by adjusting a single
  number — the known gibberish test case and a real false-positive word had
  overlapping consonant-run lengths, so any threshold that fixed one broke
  the other. The actual fix required changing what was being measured
  (collapsing digraphs first), not just retuning the existing measurement.
- **Testing the underlying function beats testing only the final output.**
  `evaluate.py` couldn't distinguish "correctly rejected as off-topic" from
  "incorrectly rejected as gibberish" by inspecting the answer text alone,
  since both produce the identical fallback message. Proving the Week 6 fix
  worked required calling `is_gibberish()` directly, not just running the
  full pipeline and checking the output.
- **A framework's internal DOM structure isn't documented and can't be
  assumed.** Streamlit's `st.columns()` output is wrapped in an
  `stLayoutWrapper` layer that isn't mentioned in its docs — a CSS fix that
  looked correct on paper silently failed until I inspected the actual
  structure directly in the browser.
- **Understanding and testing are not the same thing.** Being able to
  verify that code works (via `evaluate.py`, live browser testing, or a
  clean-machine setup check) is real and valuable, but it's a different
  skill from being able to explain or reproduce the code from scratch.

## Development notes

I built this with AI assistance (Claude Code). I set the scope and design
decisions, chose which issues to fix versus document as deliberate
trade-offs, and verified the work by testing rather than by review alone —
the relevance threshold, the gibberish filter, and every bug in the log above
came out of tests run against the running system, not from reading the code.
I worked through the early pipeline (`ingest.py`, `main.py`) line by line;
the later testing, UI, and hardening passes were more heavily AI-assisted.

## Notes / Known Issues
- The plan document's example model `phi-1.5` is not in the actual Foundry
  Local catalog; I used `phi-3-mini-4k` instead
- The plan document's SQLite reference link ([Windows Apps - SQLite data access](https://learn.microsoft.com/en-us/windows/apps/develop/data-access/sqlite-data-access))
  is for C#/.NET (Windows app dev), not Python — I used Python's built-in
  `sqlite3` module instead
- The plan's Week 2 prompt-engineering exercise calls for testing against
  "a public web AI (Bing Chat/ChatGPT)"; my `test-files-week2/prompt-test.py`
  tests against the local `phi-3-mini-4k` model instead, since that's the
  actual model this project's grounding behavior depends on — a more
  directly relevant test than an unrelated public chatbot
- **Resolved:** this used to note that `ingest.py` had no real chunking
  algorithm, just a hand-written list of already-short facts. I added a real
  paragraph/sentence-boundary chunker (`chunk_text()`/`chunk_documents()`)
  afterward to support longer documents going forward — see the Week 3 entry
  above for details and verification
