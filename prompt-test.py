#Week 2 Prompt Engineering
# .\venv\Scripts\Activate.ps1

from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    config = Configuration(app_name = "azure_foundry_prompt_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    chat_model = manager.catalog.get_model("phi-3-mini-4k")
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    system_prompt = (
    "Answer the user's question ONLY BY using only the provided context. "
    "If the context does not contain enough information, DO NOT answer and say you don't know."
    "IF you don't have enough information to answer,SAY I DON'T KNOW rather than guessing."
    "You are polite, straightforward, and easy to understand."
    )

    context = "Foundry Local runs AI models directly on your device without cloud connectivity."

    messages = [{
        "role": "system",
        "content": (
        f"{system_prompt}\n\n"
        f"Context:\n{context}"
    )
    },
    {"role": "user",
     "content": "What is the capital of Turkey?"
    }
    ]

    print("\nAnswer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()