"""
app.py — Minimal Streamlit web UI for the Company Policy RAG chatbot.

This is a thin PRESENTATION layer over the exact same RAG pipeline used by the
command-line chat.py. It does not re-implement any retrieval or generation
logic — it just calls chat.answer_query() and renders the result in the browser.

Run it with:

    streamlit run app.py

(Run `python ingest.py` first to build the knowledge base.)
"""

import streamlit as st

from chat import connect_to_vectorstore, build_llm, answer_query

st.set_page_config(page_title="Company Policy RAG Chatbot", page_icon="📄")


# Streamlit reruns this whole script on every interaction. @st.cache_resource
# builds the vector store + LLM once and reuses them, so we don't reconnect to
# ChromaDB or recreate the model on every click.
@st.cache_resource
def load_pipeline():
    return connect_to_vectorstore(), build_llm()


st.title("📄 Company Policy RAG Chatbot")
st.caption("Ask about leave, work-from-home, reimbursement, or general company info.")

# If the API key or the vector database is missing, connect/build raise
# SystemExit with a friendly message — show it and stop instead of crashing.
try:
    vectorstore, llm = load_pipeline()
except SystemExit as error:
    st.error(str(error))
    st.stop()

# We keep the conversation on screen for readability, but note this is DISPLAY
# ONLY: each question is still answered independently (no chat history is sent
# to Gemini), exactly like the CLI.
if "history" not in st.session_state:
    st.session_state.history = []

# Replay the messages exchanged so far.
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") is not None:
            if message["sources"]:
                st.caption("Sources: " + ", ".join(message["sources"]))
            else:
                st.caption("Sources: No relevant source found.")

# Chat input pinned to the bottom of the page.
question = st.chat_input("Ask a question about company policies...")
if question:
    # Show the user's message and remember it.
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    # Run the RAG pipeline and render the answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching the company documents..."):
            try:
                result = answer_query(vectorstore, llm, question)
            except Exception as error:
                st.error(f"Could not get a response from Gemini: {error}")
                st.stop()

        st.markdown(result["answer"])

        # Source attribution — comes straight from the retrieved chunk metadata.
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
        else:
            st.caption("Sources: No relevant source found.")

        # A collapsed panel that reveals exactly what RAG retrieved. It stays
        # tucked away so the answer is clean, but it's one click to inspect the
        # chunks and their distances — great for learning how retrieval works.
        with st.expander("Show retrieved chunks"):
            if not result["chunks"]:
                st.write("No chunks passed the relevance threshold.")
            for chunk in result["chunks"]:
                st.markdown(f"**{chunk['source']}** · distance `{chunk['distance']:.3f}`")
                st.text(chunk["text"])

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
