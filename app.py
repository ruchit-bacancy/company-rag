"""
app.py — Minimal Streamlit web UI for the multi-document RAG chatbot.

This is a thin PRESENTATION layer over the exact same RAG pipeline used by the
command-line chat.py. It does not re-implement any retrieval or generation
logic — it just calls chat.answer_query() and renders the result in the browser.

Run it with:

    streamlit run app.py

(Run `python ingest.py` first to build the knowledge base.)
"""

import streamlit as st

from chat import answer_query, build_llm, calls_today, describe_error, strip_citations
from retrieval import Retriever

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")


# Streamlit reruns this whole script on every interaction. @st.cache_resource
# loads the knowledge base, the local models and the LLM once and reuses them.
@st.cache_resource
def load_pipeline():
    return Retriever(), build_llm()


st.title("📄 Document Q&A Chatbot")
st.caption(
    "Ask about the GPO Annual Report FY2018, the World Bank Global Economic "
    "Prospects (June 2026), the Apple iMac Environmental Report, or company policies."
)

# If the API key or the knowledge base is missing, the loaders raise
# SystemExit with a friendly message — show it and stop instead of crashing.
try:
    with st.spinner("Loading the knowledge base..."):
        retriever, llm = load_pipeline()
except SystemExit as error:
    st.error(str(error))
    st.stop()

# We keep the conversation on screen for readability, but note this is DISPLAY
# ONLY: each question is answered independently (no chat history is sent to
# Gemini), which keeps every question to at most one Gemini call.
if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        # strip_citations also cleans answers stored before markers were removed.
        st.markdown(strip_citations(message["content"]))

question = st.chat_input("Ask a question about the documents...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                result = answer_query(retriever, llm, question)
            except Exception as error:
                st.error(describe_error(error))
                st.stop()

        st.markdown(result["answer"])
        if result["cached"]:
            st.caption("Answered from cache: no Gemini call.")

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})

# Drawn last, so the counter already includes the question just answered.
with st.sidebar:
    st.metric("Gemini calls today", calls_today())
    st.caption(
        "The free tier allows only about 20 answers per day. Search runs "
        "locally; Gemini is called once per new question, never for questions "
        "the documents don't cover, and repeated questions come from the cache."
    )
