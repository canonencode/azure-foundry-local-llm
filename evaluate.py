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
    # Checked at overlap=0: with overlap on, adjacent chunks repeat text by
    # design, so exact rejoin is no longer the right invariant. The original
    # bug this guards against (str.split(". ") dropping the delimiter) is
    # independent of overlap.
    ("periods preserved through split+rejoin",
     lambda: " ".join(chunk_text("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.",
                                 max_chars=25, overlap=0))
             == "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."),
    ("CRLF paragraph breaks recognized",
     lambda: len(chunk_text("Para one.\r\n\r\nPara two.\r\n\r\nPara three.")) == 3),
    ("chunk_documents rejects a bare string",
     lambda: _expect_raises(TypeError, chunk_documents, "not a list")),
    ("overlap must be smaller than max_chars",
     lambda: _expect_raises(ValueError, chunk_text, "text", max_chars=50, overlap=50)),
    ("negative overlap rejected",
     lambda: _expect_raises(ValueError, chunk_text, "text", max_chars=50, overlap=-1)),
    # The point of overlap: consecutive chunks of a split paragraph must
    # actually share text, so a fact straddling a boundary survives in one
    # piece somewhere.
    ("consecutive chunks share text when a paragraph is split",
     lambda: _chunks_overlap(chunk_text(_LONG_PARAGRAPH, max_chars=200, overlap=50))),
    # No chunk may exceed max_chars, overlap included. Checked at a deliberately
    # tight max_chars, where sentences nearly fill a chunk on their own and the
    # carry has to shrink or drop rather than overflow.
    ("overlap never pushes a chunk past max_chars",
     lambda: all(len(c) <= 120 for c in chunk_text(_LONG_PARAGRAPH, max_chars=120, overlap=40))),
    # Short documents take the fast path and must be untouched by overlap.
    ("short documents are unaffected by overlap",
     lambda: chunk_text("One short fact.", overlap=50) == ["One short fact."]),
    # Hard-wrapped source text must not carry its line breaks into a chunk,
    # while genuine paragraph breaks still split.
    ("line wrapping inside a paragraph is normalized",
     lambda: chunk_text("A wrapped\nparagraph here.") == ["A wrapped paragraph here."]),
    ("paragraph breaks still split after normalization",
     lambda: chunk_text("Wrapped\nline one.\n\nWrapped\nline two.")
             == ["Wrapped line one.", "Wrapped line two."]),
]

_LONG_PARAGRAPH = (
    "Cosine similarity compares the angle between two vectors rather than "
    "their magnitude. That means a short sentence and a long passage about "
    "the same topic can still score highly against one another. The measure "
    "ranges from minus one to one, where one means the vectors point in "
    "exactly the same direction. In this project it is what ranks stored "
    "chunks against an incoming question."
)


def _chunks_overlap(chunks):
    """True if every adjacent pair of chunks shares at least one word."""
    if len(chunks) < 2:
        return False
    for earlier, later in zip(chunks, chunks[1:]):
        tail_words = set(earlier.lower().split())
        head_words = later.lower().split()[:8]
        if not any(word in tail_words for word in head_words):
            return False
    return True


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
