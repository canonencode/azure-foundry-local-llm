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
    "rather than guessing. Be polite, straightforward, and concise."
)


def build_clients():
    config = Configuration(app_name="azure-foundry-local-llm")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    chat_model = manager.catalog.get_model("phi-3-mini-4k")
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
    print()
    chat_model.load()

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()

    return chat_model.get_chat_client(), embedding_model.get_embedding_client()


def answer_query(question, chat_client, embedding_client):
    top_chunks = get_top_chunks(question, embedding_client, k=3)

    print("Answer: ", end="", flush=True)

    if not top_chunks or top_chunks[0][0] < RELEVANCE_THRESHOLD:
        print("I don't have that information.\n")
        return

    context = "\n".join(content for _, content in top_chunks)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
        {"role": "user", "content": question},
    ]

    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
    print("\n")


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
