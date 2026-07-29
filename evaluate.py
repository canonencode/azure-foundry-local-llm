# Week 5 - Functional testing & evaluation: run a fixed set of queries against the
# assembled RAG pipeline (answerable, unanswerable, and edge-case questions), time
# each one, and report whether the behavior matched what was expected.
# .\venv\Scripts\Activate.ps1

import sys
import time

# See main.py for why this is needed - printing non-Latin test input
# (e.g. the Turkish GIBBERISH_CHECKS case below) would otherwise crash
# with UnicodeEncodeError on Windows' default console codepage.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main import build_clients, answer_query, is_gibberish
from ingest import chunk_text, chunk_documents

FALLBACK_TEXT = "I don't have that information."

# Direct unit checks on is_gibberish() itself, run separately from the
# end-to-end TEST_CASES below. Reason: check() below can only inspect the
# final answer string, and "I don't have that information." is identical
# whether a question was rejected by is_gibberish() or by the downstream
# RELEVANCE_THRESHOLD - no end-to-end assertion can tell those two rejection
# paths apart, only calling is_gibberish() directly can.
# Known, accepted limitation not covered here: keyboard-mash strings that
# happen to contain a digraph at the exact run-length-5 boundary (e.g.
# "sdfgh"), and mash built from adjacent keyboard rows ("poiuytrewq") which
# is vowel-rich enough to look word-like. Both fall through to
# RELEVANCE_THRESHOLD, which garbage text essentially never clears.
GIBBERISH_CHECKS = [
    ("strengths", False),
    ("twelfths", False),
    ("catchphrase", False),
    ("Foundry", False),
    ("rhythm", False),
    ("asdkjhaskjdh", True),
    ("ajksnfkasnds", True),
    ("sadjknd", True),
    # Non-ASCII scripts (Cyrillic, Arabic, CJK, or Latin-with-diacritics like
    # Turkish) have no characters in VOWELS, so every word in them used to
    # get flagged as gibberish - found during a full adversarial test pass.
    ("Bu bir Türkçe cümledir ve gayet anlamlıdır", False),
    # English compounds stack consonants where the two halves join, which used
    # to trip the run check on its own. MIN_VOWEL_RATIO in main.py is what
    # keeps them out - every one of these was rejected as gibberish before it.
    ("postscript", False),
    ("postscripts", False),
    ("thumbscrew", False),
    ("corkscrew", False),
    ("downstream", False),
    ("windscreen", False),
    ("heartstrings", False),
    ("offsprings", False),
    # A whole question has to survive too: is_gibberish() is any() over words,
    # so one tripped word rejects the entire question.
    ("Is the downstream corkscrew postscript relevant?", False),
]


def check_is_gibberish():
    passed = 0
    for word, expected in GIBBERISH_CHECKS:
        got = is_gibberish(word)
        status = "PASS" if got == expected else "FAIL"
        if got == expected:
            passed += 1
        print(f"[{status}] is_gibberish({word!r}) = {got} (expected {expected})")
    print(f"{passed}/{len(GIBBERISH_CHECKS)} gibberish-detector checks passed\n")
    return passed


# Direct unit checks on ingest.py's chunk_text()/chunk_documents(), covering
# real bugs found during an adversarial test pass: an infinite loop on
# non-positive max_chars, silently dropped periods when a long paragraph
# gets split and rejoined, Windows line endings not being recognized as
# paragraph breaks, and a string being silently mis-chunked character-by-
# character instead of erroring when passed where a list was expected.
CHUNKING_CHECKS = [
    ("no positive max_chars", lambda: _expect_raises(ValueError, chunk_text, "text", max_chars=0)),
    ("negative max_chars", lambda: _expect_raises(ValueError, chunk_text, "text", max_chars=-5)),
    ("periods preserved through split+rejoin",
     lambda: chunk_text("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.", max_chars=25) and
             " ".join(chunk_text("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.", max_chars=25))
             == "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."),
    ("CRLF paragraph breaks recognized",
     lambda: len(chunk_text("Para one.\r\n\r\nPara two.\r\n\r\nPara three.")) == 3),
    ("chunk_documents rejects a bare string",
     lambda: _expect_raises(TypeError, chunk_documents, "not a list")),
]


def _expect_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return False
    except exc_type:
        return True


def check_chunking():
    passed = 0
    for name, test in CHUNKING_CHECKS:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            print(f"[FAIL] {name} -> unexpected exception: {exc!r}")
            continue
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {name}")
    print(f"{passed}/{len(CHUNKING_CHECKS)} chunking checks passed\n")
    return passed

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
    print("=== Gibberish detector unit checks ===")
    check_is_gibberish()

    print("=== Chunking unit checks ===")
    check_chunking()

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
