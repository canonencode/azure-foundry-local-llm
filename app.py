# Week 6 - Streamlit UI (Option B from the plan), redesigned as a polished,
# product-style front end. Still just a thin layer over the existing
# pipeline - build_clients()/answer_query() from main.py are unchanged.
# streamlit run app.py

import sqlite3

import streamlit as st
from main import build_clients, answer_query, is_gibberish, RELEVANCE_THRESHOLD
from retrieve import get_top_chunks

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme: dark, warm-amber-on-charcoal "offline terminal / private archive"
# look, matched to the base/primary/background colors set in
# .streamlit/config.toml. Fraunces (serif, display) for headings gives the
# app some personality; IBM Plex Sans/Mono for body and technical details
# (model names, scores) keeps it legible and ties into the "runs on your own
# hardware" theme. #MainMenu/footer hiding is cosmetic best-effort (internal
# Streamlit test-ids, may shift between versions) - toolbarMode="minimal" in
# config.toml is the more durable way the Deploy button gets removed.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;0,700;1,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Deploy button is already removed via toolbarMode="minimal" in
   .streamlit/config.toml. Do NOT hide [data-testid="stToolbar"] wholesale -
   the sidebar's reopen arrow lives in that same container, and hiding it
   makes the sidebar unreachable once collapsed. */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stStatusWidget"] {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 20% -10%, rgba(217,158,66,0.10), transparent 60%),
        repeating-linear-gradient(135deg, rgba(217,158,66,0.025) 0px, rgba(217,158,66,0.025) 1px, transparent 1px, transparent 28px),
        #0F1113;
}

h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: 0.2px;
}

[data-testid="stMarkdownContainer"] h1 {
    font-weight: 600;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: rgba(217,158,66,0.12) !important;
    color: #E8B968 !important;
    border-radius: 4px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161A 0%, #0F1113 100%);
    border-right: 1px solid rgba(217,158,66,0.15);
}

[data-testid="stChatMessage"] {
    border: 1px solid rgba(217,158,66,0.10);
    border-radius: 12px;
    animation: fadeInUp 0.4s ease-out both;
}

.stButton > button, [data-testid="stChatInput"] button {
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
    transition: transform 0.15s ease, filter 0.15s ease, background-color 0.15s ease, border-color 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.1);
}

/* Scopes the red hover to just the per-message delete (X) buttons: the
   marker div rendered immediately before each delete column lets us select
   only that column's button via :has() + a sibling combinator, without
   touching any other secondary button (example questions, etc.) which
   share the same generic Streamlit button classes. Streamlit wraps
   st.columns() output in stLayoutWrapper > stHorizontalBlock > stColumn -
   verified via DOM inspection, since Streamlit's internal structure isn't
   documented and has changed across versions before. */
[data-testid="stElementContainer"]:has(.delete-row-marker) + [data-testid="stLayoutWrapper"] [data-testid="stColumn"]:last-child button:hover {
    background-color: #C1443B !important;
    border-color: #C1443B !important;
    color: #fff !important;
}

[data-testid="stCaptionContainer"] {
    font-style: italic;
    opacity: 0.75;
}

[data-testid="stAlertContainer"] {
    background-color: rgba(217,158,66,0.10) !important;
    border: 1px solid rgba(217,158,66,0.35) !important;
    border-radius: 10px;
}
[data-testid="stAlertContainer"] p {
    color: #ECE6DA !important;
}

/* Only the "Confirm delete" button uses type="primary" anywhere in this
   app, so this override is safe to apply broadly - marks the one
   destructive, irreversible action in red rather than the theme's amber. */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background-color: #C1443B !important;
    border-color: #C1443B !important;
    color: #fff !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background-color: #A53931 !important;
    border-color: #A53931 !important;
}

.brand {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #ECE6DA;
    margin: 0 0 0.1rem 0;
}

.panel-label {
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #D99E42;
    opacity: 0.9;
    margin: 0 0 0.7rem 0;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(217,158,66,0.08);
    font-size: 0.85rem;
}
.spec-row:last-child { border-bottom: none; padding-bottom: 0; }
.spec-row span { color: #ECE6DA; opacity: 0.65; }
.spec-row code { font-size: 0.8rem; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.stApp > header, [data-testid="stSidebarUserContent"], [data-testid="stMainBlockContainer"] > div > div {
    animation: fadeInUp 0.5s ease-out both;
}
[data-testid="stSidebarUserContent"] > div:nth-child(1) { animation-delay: 0.05s; }
[data-testid="stSidebarUserContent"] > div:nth-child(2) { animation-delay: 0.15s; }
[data-testid="stSidebarUserContent"] > div:nth-child(3) { animation-delay: 0.25s; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_clients():
    return build_clients()


@st.cache_data(ttl=60)
def get_doc_count():
    # ttl=60 so this refreshes within a minute if ingest.py is re-run while
    # this server is live, rather than staying stale until the next restart.
    conn = sqlite3.connect("knowledge.db", timeout=5)
    try:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except sqlite3.OperationalError:
        # No 'documents' table yet (ingest.py was never run) - 0 is a
        # reasonable display value; the chat flow below surfaces a proper
        # error if the user tries to ask a question before that's fixed.
        return 0
    finally:
        conn.close()


EXAMPLE_QUESTIONS = [
    "What is Foundry Local?",
    "How does cosine similarity measure relevance?",
    "What is SQLite used for in this project?",
]

if "history" not in st.session_state:
    st.session_state.history = []

try:
    with st.spinner("Initializing local AI models (first run may take a moment)..."):
        chat_client, embedding_client = get_clients()
except RuntimeError as exc:
    st.error(f"Could not start: {exc}")
    st.stop()

# --- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="brand">Local RAG Assistant</p>', unsafe_allow_html=True)
    st.caption("Offline Q&A over a small local knowledge base.")

    with st.container(border=True):
        st.markdown('<p class="panel-label">System</p>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="spec-row"><span>Chat model</span><code>phi-3-mini-4k</code></div>
<div class="spec-row"><span>Embedding model</span><code>qwen3-embedding-0.6b</code></div>
<div class="spec-row"><span>Relevance threshold</span><code>{RELEVANCE_THRESHOLD}</code></div>
<div class="spec-row"><span>Knowledge base</span><code>{get_doc_count()} documents</code></div>
""", unsafe_allow_html=True)

    st.divider()

    if st.session_state.history:
        with st.popover("Clear all history", use_container_width=True):
            st.write("This will permanently delete the entire conversation.")
            if st.button("Confirm delete", type="primary", use_container_width=True):
                st.session_state.history = []
                st.toast("History cleared.")
                st.rerun()
    else:
        st.button("Clear all history", disabled=True, use_container_width=True)

# --- Main area -----------------------------------------------------------
st.title("Local RAG Assistant")
st.caption("Powered by Microsoft Foundry Local - runs entirely on-device, no internet required.")

if not st.session_state.history:
    st.info("No conversation yet. Ask a question below, or try one of these:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for i, entry in enumerate(st.session_state.history):
    with st.chat_message("user", avatar="assets/user-avatar.svg"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])

        chunks = entry["chunks"]
        if chunks:
            passed_gate = chunks[0][0] >= RELEVANCE_THRESHOLD
            label = (
                "Sources used for this answer"
                if passed_gate
                else f"No chunk cleared the relevance threshold ({RELEVANCE_THRESHOLD}) - "
                     "closest matches shown for reference, none were used"
            )
            with st.expander(label):
                # Numbered to match the order the passages were given to the
                # model, so [1] here is the same [1] it saw in its context.
                for position, (score, content) in enumerate(chunks, start=1):
                    st.write(f"**[{position}]** *(similarity {score:.2f})* {content}")

        st.markdown('<div class="delete-row-marker"></div>', unsafe_allow_html=True)
        _, delete_col = st.columns([9, 1])
        if delete_col.button("✕", key=f"delete_{i}", help="Delete this exchange"):
            st.session_state.history.pop(i)
            st.rerun()

prompt = st.chat_input("Ask a question...")
if not prompt and st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")

if prompt:
    with st.chat_message("user", avatar="assets/user-avatar.svg"):
        st.write(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                if not prompt.strip() or is_gibberish(prompt):
                    top_chunks = []
                else:
                    top_chunks = get_top_chunks(prompt, embedding_client, k=3)
                answer = answer_query(prompt, chat_client, embedding_client, verbose=False, top_chunks=top_chunks)
        except RuntimeError as exc:
            # e.g. knowledge.db has no data yet - show a clear message
            # instead of a raw traceback, and don't add a failed turn to history
            st.error(f"Couldn't answer that: {exc}")
        else:
            st.write(answer)
            st.session_state.history.append({
                "question": prompt,
                "answer": answer,
                "chunks": top_chunks,
            })
            st.rerun()
