"""
chat.py — Interactive Company Policy RAG chatbot (command-line).

This is the "retrieval + generation" half of RAG. For every question it runs
four clear steps:

    1. Turn the user's question into an embedding vector (Gemini)
    2. Search ChromaDB for the most similar chunks (semantic search)
    3. Build a context string from those chunks
    4. Ask Gemini to answer using ONLY that context, then show the sources

Run it with:

    python chat.py

(Run `python ingest.py` first to build the knowledge base.)
"""

import os

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    LLM_MODEL,
    TOP_K,
    MAX_DISTANCE,
    DEBUG,
    load_api_key,
    get_embeddings,
)

# ---------------------------------------------------------------------------
# The grounded prompt — the heart of RAG.
# It is kept here as a plain string so it is easy to find and edit.
# {context} is filled with the retrieved chunks; {question} with the user's text.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a company policy assistant.

Your job is to answer questions using ONLY the information provided in the
context below.

Rules:
1. Use only the provided context to answer.
2. Do not use your general knowledge to invent information.
3. If the answer is not present in the context, reply with exactly this text
   and nothing else: INFO_NOT_FOUND
4. Keep answers concise and easy to understand.
5. Do not mention that you are an AI unless necessary.
6. Do not fabricate policies, numbers, names, dates, or rules.

Context:
{context}

Question:
{question}
"""

# The model is told to return this exact token when the context does not
# contain the answer. We detect it and show a friendly message instead — and,
# importantly, we do NOT list any sources, because none actually answered it.
NOT_FOUND_SENTINEL = "INFO_NOT_FOUND"

NO_ANSWER_MESSAGE = (
    "I couldn't find information about that in the provided company documents."
)


def connect_to_vectorstore():
    """
    Reconnect to the ChromaDB that ingest.py created.

    We pass the SAME embedding model here as during ingestion (via
    get_embeddings). Queries and documents must be embedded the same way, or
    similarity search is meaningless.
    """
    if not os.path.isdir(CHROMA_DIR):
        raise SystemExit(
            "Error: vector database not found.\n"
            "Please run 'python ingest.py' first to build the knowledge base."
        )
    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    except Exception as error:
        raise SystemExit(f"Error: could not open the vector database: {error}")


def build_llm():
    """
    Create the Gemini chat model used to write the final answer.

    Shared by the CLI (chat.py) and the web UI (app.py). temperature=0 keeps
    answers factual and grounded rather than creative.
    """
    api_key = load_api_key()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


def retrieve(vectorstore, question):
    """
    STEPS 1 & 2 — Embed the question and find the most similar chunks.

    similarity_search_with_score() embeds the question with the same Gemini
    model, then returns the TOP_K closest chunks together with a distance score
    (smaller = more similar, because we configured cosine distance).

    We then drop any chunk that is further away than MAX_DISTANCE. This is a
    simple relevance guard: if even the best match is far away, the question is
    probably not covered by our documents, so we return nothing and let the
    chatbot say "I couldn't find this".
    """
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    relevant = [(doc, score) for doc, score in results if score <= MAX_DISTANCE]
    return relevant


def build_context(relevant):
    """
    STEP 3 — Build the context string sent to the LLM.

    Each chunk is labelled with its source filename so (a) the model can see
    where the text came from and (b) we can list the sources under the answer.
    """
    blocks = []
    for doc, _score in relevant:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def unique_sources(relevant):
    """Collect the distinct source filenames (no duplicates), preserving order."""
    sources = []
    for doc, _score in relevant:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)
    return sources


def answer_query(vectorstore, llm, question):
    """
    Run the full RAG pipeline for one question and RETURN the result.

    This function does no printing, so BOTH the command-line app (chat.py) and
    the web UI (app.py) can share it. It returns a plain dict:

        answer  : str          -> the text to show the user
        sources : list[str]    -> distinct source filenames (empty if none)
        found   : bool         -> whether an answer was found in the documents
        chunks  : list[dict]   -> the retrieved chunks, for optional debugging;
                                  each is {"source", "distance", "text"}

    It may raise if the Gemini call fails; each caller decides how to show that.
    """
    # STEPS 1 & 2 — embed the question and retrieve the most similar chunks.
    relevant = retrieve(vectorstore, question)

    # Package the retrieved chunks so a caller can display them (debug view).
    chunks = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "distance": score,
            "text": doc.page_content,
        }
        for doc, score in relevant
    ]

    # No chunk passed the relevance filter -> nothing to ground an answer on, so
    # we must not let the model invent one.
    if not relevant:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "found": False, "chunks": chunks}

    # STEPS 3 & 4 — build the grounded context and ask Gemini to answer.
    context = build_context(relevant)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    # .text collapses the model's response (a list of content blocks on newer
    # models) into a plain string.
    answer = response.text.strip()

    # The chunks were close enough to retrieve, but the model decided they don't
    # actually contain the answer. Treat this as "not found": no invented answer
    # and no misleading sources.
    if NOT_FOUND_SENTINEL in answer:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "found": False, "chunks": chunks}

    return {
        "answer": answer,
        "sources": unique_sources(relevant),
        "found": True,
        "chunks": chunks,
    }


def answer_question(vectorstore, llm, question):
    """Command-line wrapper: run the pipeline and PRINT the result."""
    try:
        result = answer_query(vectorstore, llm, question)
    except Exception as error:
        print(f"\nError: could not get a response from Gemini: {error}")
        return

    # Optional debug view: show exactly what RAG retrieved. Very useful while
    # learning; hidden by default (DEBUG = False) so the output stays clean.
    if DEBUG:
        print("\n[DEBUG] Retrieved context:")
        print("-" * 40)
        if not result["chunks"]:
            print("(no chunks passed the relevance threshold)")
        for chunk in result["chunks"]:
            print(f"Source: {chunk['source']}  (distance: {chunk['distance']:.3f})")
            print(chunk["text"])
            print("-" * 40)

    print(f"\nBot: {result['answer']}")
    print("\nSources:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("No relevant source found.")


HELP_TEXT = """Commands:
  help  - Show available commands
  exit  - Exit the chatbot
  quit  - Exit the chatbot"""


def main():
    # connect_to_vectorstore() and build_llm() both load the API key and will
    # fail early with a friendly message if it is missing.
    vectorstore = connect_to_vectorstore()
    llm = build_llm()

    print("=" * 40)
    print("      Company Policy RAG Chatbot")
    print("=" * 40)
    print("\nAsk a question about company policies.")
    print("Type 'help' for commands.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        command = question.lower()
        if command in ("exit", "quit"):
            print("\nGoodbye!")
            break
        if command == "help":
            print(HELP_TEXT)
            continue

        answer_question(vectorstore, llm, question)
        print()


if __name__ == "__main__":
    main()
