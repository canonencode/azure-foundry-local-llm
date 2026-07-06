from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    config = Configuration(app_name="azure-foundry-local-llm")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    chat_model = manager.catalog.get_model("phi-3-mini-4k")
    chat_model.download(lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True))
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    messages = [{"role": "user", "content":"Hello World!"}]
    print()
    print("Answer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
    

    