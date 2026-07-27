# Week 4 - App assembly: retrieval-augmented Q&A CLI over the local knowledge base
# .\venv\Scripts\Activate.ps1

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
    run = 0
    for ch in collapse_digraphs(word):
        if ch == "#" or (ch.isalpha() and ch not in VOWELS):
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def is_gibberish(question):
    return any(has_long_consonant_run(word) for word in question.split())


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

    context = "\n".join(content for _, content in top_chunks)
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
    print("\n")
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
        answer_query(question, chat_client, embedding_client)


if __name__ == "__main__":
    main()
