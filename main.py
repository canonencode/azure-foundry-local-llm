# Week 4 - App assembly: retrieval-augmented Q&A CLI over the local knowledge base
# .\venv\Scripts\Activate.ps1

import sys
# Windows' default console codepage (cp1252/cp437) can't encode most
# non-Latin text (Cyrillic, Arabic, CJK, or Turkish "i"-with-no-dot) -
# printing a question/answer containing it would crash with
# UnicodeEncodeError otherwise, found while testing is_gibberish() below
# with real non-English input.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieve import get_top_chunks

# Below this score the best retrieved chunk is considered off-topic. Chat models
# don't reliably refuse to answer just because we ask nicely in the system prompt
# (see Week 2 findings) so we gate the call with this instead.
RELEVANCE_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "Answer the user's question ONLY by using the provided context. "
    "If the context does not contain enough information, say you don't know "
    "rather than guessing. Be polite and straightforward. "
    "Answer in at most 2 sentences - do not repeat yourself or restate the question."
)


def build_clients():
    try:
        config = Configuration(app_name="azure-foundry-local-llm")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

        chat_model = manager.catalog.get_model("phi-3-mini-4k")
        chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
        print()
        chat_model.load()
        chat_client = chat_model.get_chat_client()

        embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
        print()
        embedding_model.load()

        return chat_client, embedding_model.get_embedding_client()
    except Exception as exc:
        # Wrap whatever the SDK raised (service not running, no internet on
        # first download, model missing) with an actionable message instead
        # of a raw multi-frame SDK traceback. `from exc` keeps the original
        # cause attached, it isn't swallowed.
        raise RuntimeError(
            "Could not initialize Foundry Local models. Check that Foundry "
            "Local is installed and running, and that you have an internet "
            f"connection for the first-time model download. Underlying error: {exc}"
        ) from exc


# "y" counts as a vowel here to avoid false positives on real words like
# "shyly"/"rhythm"/"gypsy", where it's the only vowel sound in a syllable.
VOWELS = set("aeiouy")

# Two letters, one sound - collapsing these before counting a consonant run
# is why real words with long raw letter-runs ("strengths", "twelfths",
# "catchphrase") don't get misread as gibberish. Known trade-off: a
# keyboard-mash string that happens to contain one of these at the exact
# run-length-5 boundary (e.g. "sdfgh") can now slip past this check alone -
# RELEVANCE_THRESHOLD below is the backstop for those, since garbage text
# essentially never scores >= 0.5 similarity against real content.
DIGRAPHS = ("ch", "sh", "th", "ph", "gh", "wh", "ck", "ng")


def collapse_digraphs(word):
    word = word.lower()
    for digraph in DIGRAPHS:
        word = word.replace(digraph, "#")
    return word


def has_long_consonant_run(word, min_run=5):
    if not word.isascii():
        # VOWELS/DIGRAPHS only model English orthography. Applying this
        # check to non-ASCII scripts (Cyrillic, Arabic, CJK, or Latin text
        # with diacritics like Turkish "i"/"g"/"s"/"c"/"o"/"u" variants)
        # would flag ordinary foreign-language words as gibberish, since
        # none of those characters are in VOWELS - confirmed empirically:
        # a plain Turkish sentence was misflagged before this check existed.
        return False
    run = 0
    for ch in collapse_digraphs(word):
        if ch == "#" or (ch.isalpha() and ch not in VOWELS):
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


# A long consonant run on its own isn't enough to call a word gibberish:
# English compounds stack consonants at the seam where the two halves join
# ("post+script", "thumb+screw", "cork+screw", "down+stream", "wind+screen",
# "heart+strings"), and all of those were being rejected as gibberish. What
# separates them from keyboard mash is that a real word still carries vowels
# throughout, while mash is vowel-starved. Measured over 137 real words and
# 26 mash strings, the two groups separate cleanly in the gap between 0.167
# (the most vowel-rich mash) and 0.182 (the least vowel-rich real word).
MIN_VOWEL_RATIO = 0.18


def vowel_ratio(word):
    letters = [ch for ch in word.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch in VOWELS) / len(letters)


def is_gibberish(question):
    # Both conditions must hold, so this errs toward letting text through.
    # That's the deliberate direction: a false positive silently refuses a
    # real question, while a false negative just reaches RELEVANCE_THRESHOLD,
    # which garbage text essentially never clears anyway.
    return any(
        has_long_consonant_run(word) and vowel_ratio(word) < MIN_VOWEL_RATIO
        for word in question.split()
    )


# Attribution is built in code from what retrieval actually returned, rather
# than by asking the model to cite its sources in SYSTEM_PROMPT. Same reason
# RELEVANCE_THRESHOLD exists: phi-3-mini does not reliably follow prompt
# instructions (Week 2), so a model-authored citation could name a passage it
# didn't use, or invent one. Built this way it cannot be wrong - the numbers
# are the passages that were placed in the context, by construction.
SOURCE_SNIPPET_CHARS = 100


def format_sources(top_chunks, max_chars=SOURCE_SNIPPET_CHARS):
    lines = []
    for position, (score, content) in enumerate(top_chunks, start=1):
        snippet = content if len(content) <= max_chars else content[:max_chars].rstrip() + "..."
        lines.append(f"  [{position}] (similarity {score:.2f}) {snippet}")
    return "\n".join(lines)


def answer_query(question, chat_client, embedding_client, verbose=True, top_chunks=None):
    if not question.strip() or is_gibberish(question):
        answer = "I don't have that information."
        print(f"Answer: {answer}\n")
        return answer

    if top_chunks is None:
        top_chunks = get_top_chunks(question, embedding_client, k=3)

    if verbose:
        print("[retrieved chunks]")
        for score, content in top_chunks:
            print(f"  {score:.4f} - {content}")

    print("Answer: ", end="", flush=True)

    if not top_chunks or top_chunks[0][0] < RELEVANCE_THRESHOLD:
        answer = "I don't have that information."
        print(f"{answer}\n")
        return answer

    # Number the passages so the sources printed below line up with what the
    # model was actually given, in the same order.
    context = "\n".join(
        f"[{position}] {content}"
        for position, (_, content) in enumerate(top_chunks, start=1)
    )
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
        {"role": "user", "content": question},
    ]

    answer_parts = []
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)
    print()
    # Printed, deliberately not appended to the return value - callers
    # (evaluate.py) compare the returned answer against FALLBACK_TEXT, and
    # attribution text would break that comparison. Gated on verbose because
    # app.py renders its own sources in the UI and would otherwise duplicate
    # them into the server console.
    if verbose:
        print("\nSources:")
        print(format_sources(top_chunks))
    print()
    return "".join(answer_parts)


def main():
    chat_client, embedding_client = build_clients()

    print("Local RAG Assistant - ask a question (or type 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            answer_query(question, chat_client, embedding_client)
        except RuntimeError as exc:
            # A transient failure on one question (e.g. missing knowledge.db,
            # a dropped connection to the model) shouldn't kill the whole
            # session - report it and let the user try again or exit.
            print(f"\nSomething went wrong answering that: {exc}\n")


if __name__ == "__main__":
    main()
