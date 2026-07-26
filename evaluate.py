# Week 5 - Functional testing & evaluation: run a fixed set of queries against the
# assembled RAG pipeline (answerable, unanswerable, and edge-case questions), time
# each one, and report whether the behavior matched what was expected.
# .\venv\Scripts\Activate.ps1

import time
from main import build_clients, answer_query

FALLBACK_TEXT = "I don't have that information."

# expect: "answer" -> should be grounded in the knowledge base, not the fallback
#         "fallback" -> off-topic, should trigger the relevance-gate fallback
#         "edge" -> not a correctness check, just confirms the app doesn't crash
TEST_CASES = [
    {"question": "What is Foundry Local?", "expect": "answer"},
    {"question": "What does RAG stand for and how does it work?", "expect": "answer"},
    {"question": "How does cosine similarity measure relevance?", "expect": "answer"},
    {"question": "What is SQLite used for in this project?", "expect": "answer"},
    {"question": "What is the capital of France?", "expect": "fallback"},
    {"question": "Who won the last World Cup?", "expect": "fallback"},
    {"question": "asdkjhaskjdh random gibberish query", "expect": "fallback"},
    {"question": "?", "expect": "edge"},
    {"question": "", "expect": "edge"},
    {"question": "Can you help me with something?", "expect": "edge"},
]


def check(question, answer, expect):
    is_fallback = answer.strip() == FALLBACK_TEXT
    if expect == "answer":
        return not is_fallback and len(answer.strip()) > 0
    if expect == "fallback":
        return is_fallback
    return True  # "edge" - ran without raising, that's the whole test


def main():
    chat_client, embedding_client = build_clients()

    results = []
    for case in TEST_CASES:
        question, expect = case["question"], case["expect"]
        print(f"\n=== Query: {question!r} (expect: {expect}) ===")

        start = time.perf_counter()
        try:
            answer = answer_query(question, chat_client, embedding_client, verbose=True)
            passed = check(question, answer, expect)
        except Exception as exc:
            answer = f"<exception: {exc}>"
            passed = False
        elapsed = time.perf_counter() - start

        results.append((question, expect, passed, elapsed))

    print("\n\n=== Summary ===")
    for question, expect, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] ({elapsed:5.2f}s) expect={expect:<8} {question!r}")

    total = len(results)
    passed_count = sum(1 for *_, passed, _ in results if passed)
    print(f"\n{passed_count}/{total} passed")


if __name__ == "__main__":
    main()
