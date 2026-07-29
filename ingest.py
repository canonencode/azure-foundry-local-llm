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

    # Longer, multi-paragraph entries. The eight facts above are each short
    # enough to pass through chunk_text() untouched, so nothing in the corpus
    # exercised paragraph splitting, sentence splitting, or overlap until
    # these were added.
    """The retrieval step decides how good a RAG answer can possibly be. If the
passage containing the answer is never retrieved, no amount of prompt wording
will recover it, because the model simply never sees the information. This is
why retrieval quality is measured separately from answer quality: they fail for
different reasons and are fixed in different places.

Retrieval is usually tuned along two axes. The first is how many chunks to
return, often written as k. A larger k raises the chance the right passage is
included, but it also dilutes the context with less relevant text, and small
models are easily distracted by that. The second axis is how the documents were
split in the first place, since a chunk that is too large buries the relevant
sentence among unrelated ones, while a chunk that is too small can lose the
context that made it meaningful.""",

    """Chunking is the process of splitting a document into passage-sized pieces
before embedding them. The goal is for each stored piece to be self-contained
enough that retrieving it alone is useful, because a retrieved chunk is shown to
the model without the surrounding document.

Splitting on paragraph boundaries is a reasonable default, since a paragraph is
already a unit of meaning written by a human. Very long paragraphs still need a
second pass, usually splitting on sentence boundaries, so that no single chunk
grows large enough to swamp the context window.

Chunk overlap means repeating a small amount of text from the end of one chunk
at the start of the next. Without overlap, a fact whose subject and predicate
land on opposite sides of a boundary is stored as two halves, and neither half
answers a question about it on its own. Overlap costs a little storage and some
duplicated text in exchange for removing that failure mode.""",

    """An embedding is a list of numbers that represents the meaning of a piece
of text. Texts with similar meanings are placed near each other in that numeric
space, which is what makes semantic search possible: a query about a topic can
match a passage that never uses the same words.

The number of values in an embedding is called its dimensionality, and it is
fixed by the model that produced it. Two embeddings can only be compared if they
came from the same model, because different models place meaning in different
coordinate systems. Mixing them produces numbers that look like valid scores but
mean nothing at all.""",

    """Cosine similarity measures the angle between two vectors while ignoring
their length. It is the standard choice for comparing embeddings because the
length of an embedding often reflects how long the original text was, which is
not what a search is trying to measure. Two passages about the same subject
should score highly whether one is a sentence and the other a paragraph.

The score ranges from minus one to one. A score of one means the vectors point
in exactly the same direction, zero means they are unrelated, and negative
values mean they point in opposing directions. In practice, embeddings of real
text rarely produce negative scores, so the useful range is narrower than the
theoretical one, and a threshold has to be chosen by testing rather than assumed
from the mathematics.""",

    """A relevance threshold is a minimum similarity score a retrieved passage
must reach before it is used at all. If the best available passage scores below
the threshold, the system declines to answer instead of passing weak context to
the model.

This matters because language models are strongly biased toward producing an
answer. Given irrelevant context and a question, a model will often answer from
its own training data rather than admitting the context does not cover the
question, even when the instructions explicitly tell it to refuse. Enforcing the
check in code rather than in the prompt removes the model's discretion, and it
has the side benefit of skipping the generation call entirely, which makes
off-topic questions faster to reject than on-topic ones are to answer.""",

    """Running a language model locally means the model weights are stored and
executed on the same machine that asks the questions. Nothing is sent to a
remote service, so the data never leaves the device and the system keeps working
without a network connection once the weights have been downloaded.

The trade-off is capability against resources. Models small enough to run
comfortably on a laptop have less general knowledge and weaker reasoning than
large hosted models, and they follow complex instructions less reliably. For a
retrieval-augmented system this matters less than it might seem, because the
knowledge comes from the retrieved passages rather than from the model's own
memory. The model's job is narrower: read the supplied context and phrase an
answer from it.""",

    """SQLite stores an entire database in a single ordinary file, with no server
process to install, configure, or keep running. It ships as part of the Python
standard library, so a project can use it without adding a dependency at all.
That makes it a natural fit for a small local system where the alternative
would be operating a database service purely to hold a few thousand rows.

Embeddings have no native column type here, so each vector is serialized to
text when it is written and parsed back when it is read. The cost of that
choice is paid on every query: the rows have to be read and deserialized
before anything can be compared. For a corpus of a few dozen chunks the cost
is invisible, and the simplicity of one portable file is worth far more than
the microseconds saved by a specialized store.

Writing to the same database while another process reads it can raise a
"database is locked" error. Setting a busy timeout tells SQLite to retry for
a few seconds rather than failing the moment it encounters contention, which
is usually enough for a single-user application where writes are rare and
short.""",

    """A context window is the maximum amount of text a model can consider at
once, measured in tokens rather than words. Everything counts against it: the
system prompt, the retrieved passages, the user's question, and the answer the
model is still generating. A model advertised with a four thousand token
window is not offering four thousand tokens of input on top of its reply, it is
offering that much for all of it combined.

This is the practical limit on how much retrieved context a RAG system can
supply. Returning more chunks looks like a free improvement to recall, but each
one consumes budget that the answer itself needs, and an overrun is not a
graceful degradation: the request either fails or the earliest content is
silently dropped. It is also why chunk size matters beyond retrieval quality,
since a handful of oversized chunks can fill a window that a larger number of
focused ones would have fit inside comfortably.

Tokens are not words. Common words are usually a single token, while rare
words, long words, and words in languages the tokenizer was not primarily
trained on can split into several. A rough English estimate is about four
characters per token, but it is an estimate, not a rule to depend on.""",

    """Language models generate text by predicting what is likely to come next,
not by looking facts up. That is why a model can produce a confident, fluent,
well-formatted answer that is simply untrue: fluency and accuracy are produced
by the same mechanism, so the output gives no signal about which one you got.
The failure is often called hallucination, though the model is not doing
anything different from when it is correct.

Grounding is the countermeasure. Instead of asking the model what it knows, the
system supplies the relevant source text and asks it to answer from that. This
narrows the job from recall to reading comprehension, which small models handle
far better, and it makes answers checkable, because the passages used can be
shown alongside the answer.

Grounding reduces the problem rather than eliminating it. A model can still
misread a passage, blend supplied context with its own training data, or answer
a question the context does not actually cover. Retrieval quality and an
explicit refusal path matter for exactly this reason: the most reliable way to
avoid a fabricated answer is to not attempt one when the source material is
missing.""",

    """Streaming means the model's answer is displayed token by token as it is
produced, instead of appearing all at once when generation finishes. The total
time to a complete answer is unchanged, but the wait before anything appears
drops to a fraction of a second, which makes a noticeable difference to how
responsive a system feels.

It also changes what a user can do while waiting. Text that starts appearing
immediately can be read as it arrives, and an answer heading in the wrong
direction can be abandoned early rather than after the full generation has
completed. The trade-off is that the application has to accumulate the pieces
itself if it wants the finished answer, and it must handle chunks that carry no
text, which streaming APIs commonly emit at the end of a response.""",

    """Evaluating a RAG system means testing retrieval and generation separately,
because they fail for different reasons. If the correct passage was never
retrieved, no change to the prompt or the model can produce a correct answer.
If the passage was retrieved and the answer is still wrong, the retrieval step
is not the thing to fix. Judging only the final answer collapses these two into
one signal and makes the cause impossible to identify.

A useful test set covers three categories rather than one. Answerable questions
confirm the system retrieves and uses the right source. Unanswerable questions
confirm it declines rather than inventing something, which is the behavior most
likely to be quietly broken. Edge cases such as empty input, a single
punctuation mark, or meaningless text confirm the system fails safely instead
of crashing.

Identical output from different causes is a particular trap. A system that
refuses because nothing was relevant and one that refuses because the input was
rejected upstream produce the same message, so an assertion on the answer text
cannot tell them apart. Testing the underlying function directly is the only
way to know which path actually ran.""",

    """Comparing a query against every stored vector is called exact or brute
force search. It always returns the true nearest matches, and its cost grows in
direct proportion to the size of the collection. For thousands of vectors this
is fast enough to be unnoticeable, and it has the considerable advantage of
being a few lines of code with no index to build, tune, or keep in sync.

At a large enough scale, exact search stops being practical, and approximate
nearest neighbor methods take over. These build an index that narrows the
search to a promising subset instead of scanning everything, trading a small
chance of missing a true match for a very large speedup. The trade is usually
worth making at scale, but it introduces an index that must be maintained and
parameters that must be tuned, which is real complexity to adopt before the
size of the data actually demands it.""",
]


def overlap_tail(text, overlap):
    """The last `overlap` characters of text, trimmed forward to a word
    boundary so the next chunk never begins mid-word.
    """
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        # The cut landed inside a word - drop that leading partial word
        # rather than starting a chunk with something like "ilarity".
        _, separator, rest = tail.partition(" ")
        tail = rest if separator else ""
    return tail.strip()


def chunk_text(text, max_chars=500, overlap=50):
    """Split text into paragraph-sized chunks (blank-line-separated), so RAG
    retrieval operates on passage-level pieces rather than a whole document.
    A paragraph longer than max_chars gets split further on sentence
    boundaries, so no single chunk is too large to be a useful, specific
    match for a query.

    Consecutive chunks of a split paragraph repeat the last `overlap`
    characters of the previous chunk. Without that, a fact whose subject and
    predicate land either side of a boundary is retrievable only as two
    halves, neither of which answers the question on its own. Overlap does
    not span separate paragraphs or documents, which are already independent
    units of meaning.
    """
    if max_chars <= 0:
        # The hard character-split below (sentence[:max_chars]) never
        # shrinks the remaining string when max_chars <= 0 - sentence[:0] is
        # "" and sentence[0:] is unchanged, so it would loop forever.
        raise ValueError("max_chars must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_chars:
        # Every new chunk would start already full of carried-over text and
        # could never fit new content, so splitting would make no progress.
        raise ValueError("overlap must be smaller than max_chars")

    # Normalize Windows line endings first - this project is Windows-only,
    # and "\r\n\r\n" (e.g. text pasted from Notepad/Word) wouldn't match the
    # "\n\n" paragraph-break check below otherwise, silently treating a
    # multi-paragraph document as one giant unsplit paragraph.
    text = text.replace("\r\n", "\n")
    # Collapse whitespace runs inside a paragraph to single spaces. Source
    # documents are usually hard-wrapped, so without this every chunk carries
    # the original line breaks around and renders as ragged text wherever the
    # retrieved passage is displayed. Paragraph breaks ("\n\n") are split on
    # first, so they survive this.
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
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
                    # Carry over as much of the tail as still leaves room for
                    # this piece, rather than all-or-nothing: never exceeding
                    # max_chars is the one guarantee this function makes, so
                    # a long sentence simply gets less overlap (or none, if it
                    # nearly fills a chunk by itself) instead of losing it.
                    room = max_chars - len(piece) - 1
                    carry = overlap_tail(current, min(overlap, room)) if room > 0 else ""
                    current = f"{carry} {piece}" if carry else piece
                else:
                    current = piece
        if current:
            chunks.append(current)

    return chunks


def chunk_documents(documents, max_chars=500, overlap=50):
    if isinstance(documents, str):
        # A bare string is technically iterable (character by character),
        # so without this check a single string passed by mistake would
        # silently get "chunked" one character at a time instead of failing.
        raise TypeError("chunk_documents expects a list of documents, not a single string")
    chunks = []
    for document in documents:
        chunks.extend(chunk_text(document, max_chars=max_chars, overlap=overlap))
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
