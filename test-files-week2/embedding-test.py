# Week 2 embedding + cosine simularity sandbox
# .\venv\Scripts\Activate.ps1

import math
from foundry_local_sdk import Configuration, FoundryLocalManager

sentences = [
    "Microsoft created Windows.",
    "GTA San Andreas is the best game in the series.",
    "The World Cup happens every four years.",
    "Honda is the best car brand."
]
def main():
    config = Configuration(app_name = "azure-foundry-local-llm-embedding-test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True))
    
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()
    response = embedding_client.generate_embeddings(sentences)
    sentence_embeddings = [item.embedding for item in response.data]

    query = "Microsoft released Windows 7 on 2009"
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    results = []
    for sentence, embedding in zip(sentences, sentence_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        results.append((sentence, score))

    for sentence, score in results:
        print(f"{score:.4f} - {sentence}")

    best_sentence, best_score = results[0]
    for sentence, score in results:
        if score > best_score:
            best_sentence = sentence
            best_score = score

    print(f"\nBest match: {best_sentence} (score: {best_score: .4f})")


def cosine_similarity(a,b):
    dot = sum(x*y for x, y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


if __name__ == "__main__":
    main()