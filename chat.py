"""
chat.py — Interactive multi-document RAG chatbot (command-line).

One chatbot answers questions about every document in docs/: the GPO Annual
Report, the World Bank Global Economic Prospects (June 2026), the Apple iMac
Product Environmental Report and the company policy files.

For every question it runs these steps:

    1. Retrieve: hybrid search + reranking finds the best passages (retrieval.py)
    2. Guard:    if nothing relevant was found, say so WITHOUT calling Gemini
    3. Context:  expand the passages into numbered sources [S1], [S2], ...
    4. Generate: ask Gemini to answer from those sources only, citing them.
                 Answers are cached on disk, so asking again costs nothing.
    5. Show the answer (the source list is printed only when DEBUG is on)

Gemini is called at most once per question, which matters on the free tier.

Run it with:

    python chat.py

(Run `python ingest.py` first to build the knowledge base.)
"""

import datetime
import hashlib
import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    ANSWER_CACHE_FILE,
    CACHE_DIR,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_THINKING_LEVEL,
    LLM_TIMEOUT,
    DEBUG,
    load_api_key,
    doc_title,
)
from retrieval import Retriever, is_relevant

# ---------------------------------------------------------------------------
# The grounded prompt — the heart of RAG.
# {library} lists the documents, {context} holds the numbered sources and
# {question} is the user's text.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a research assistant for this library of documents:
{library}

Answer the question using ONLY the numbered sources below.

Rules:
1. Use only the sources. Do not use your general knowledge to add facts.
2. After each fact, cite its source in square brackets, e.g. [S1] or [S2][S3].
3. Sources may contain Markdown tables, chart labels and footnotes. A marker
   such as [^7] refers to the note "[^7]: ..." under "Notes" in the same
   source. Read table rows and columns carefully and quote numbers exactly,
   with their units and the year or column they belong to.
4. If the sources do not contain the answer, reply with exactly this text and
   nothing else: INFO_NOT_FOUND
5. If the sources answer only part of the question, answer that part and say
   what is missing.
6. If asked to summarise or explain a table or chart, give its main
   takeaways with numbers (the overall picture, the highest and lowest
   values, notable changes or revisions), not just a list of its columns.
7. Keep answers concise and easy to understand.

Sources:
{context}

Question:
{question}
"""

# The model is told to return this exact token when the context does not
# contain the answer. We detect it and show a friendly message instead — and
# we do NOT list any sources, because none actually answered the question.
NOT_FOUND_SENTINEL = "INFO_NOT_FOUND"

NO_ANSWER_MESSAGE = "I couldn't find information about that in the provided documents."

QUOTA_MESSAGE = (
    "The Gemini free-tier quota is used up for now. Please try again later "
    "(questions you have asked before are answered from the cache)."
)


def build_llm():
    """
    Create the Gemini chat model used to write the final answer.

    Shared by the CLI (chat.py), the web UI (app.py) and evaluate.py.
    temperature=0 keeps answers factual rather than creative. Current Gemini
    Flash models "think" before answering; reading facts out of the sources
    needs little of that, and a low thinking level makes answers faster.

    A short timeout without retries makes a stuck or rate-limited request fail
    fast; LangChain's with_fallbacks() then sends the same prompt once to the
    lighter LLM_FALLBACK_MODEL.
    """
    api_key = load_api_key()
    primary = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0,
        thinking_level=LLM_THINKING_LEVEL,
        max_retries=0,
        timeout=LLM_TIMEOUT,
    )
    fallback = ChatGoogleGenerativeAI(
        model=LLM_FALLBACK_MODEL,
        google_api_key=api_key,
        temperature=0,
        max_retries=0,
        timeout=LLM_TIMEOUT,
    )
    return primary.with_fallbacks([fallback])


def describe_error(error):
    """Turn a Gemini exception into a short, friendly message."""
    text = str(error)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return QUOTA_MESSAGE
    return f"Could not get a response from Gemini: {error}"


# ---------------------------------------------------------------------------
# Answer cache + daily usage counter (both small JSON files in .cache/)
# ---------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)


def calls_today():
    """How many Gemini calls this app has made today (on this machine)."""
    usage = _read_json(ANSWER_CACHE_FILE).get("_usage", {})
    return usage.get(datetime.date.today().isoformat(), 0)


def generate(llm, prompt):
    """
    Ask Gemini — unless this exact prompt was answered before.

    The cache key is a hash of the model names plus the full prompt (which
    includes the retrieved sources), so an answer is reused only when both
    the question and its context are identical.
    Returns (answer, model_that_answered, cached).
    """
    key_text = f"{LLM_MODEL}|{LLM_FALLBACK_MODEL}\n{prompt}"
    key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    cache = _read_json(ANSWER_CACHE_FILE)
    if key in cache:
        return cache[key]["answer"], cache[key]["model"], True

    response = llm.invoke(prompt)
    # .text collapses the model's response (a list of content blocks on newer
    # models) into a plain string.
    answer = response.text.strip()
    model = response.response_metadata.get("model_name", LLM_MODEL)

    cache = _read_json(ANSWER_CACHE_FILE)   # re-read: another process may have written
    cache[key] = {"answer": answer, "model": model}
    usage = cache.setdefault("_usage", {})
    today = datetime.date.today().isoformat()
    usage[today] = usage.get(today, 0) + 1
    _write_json(ANSWER_CACHE_FILE, cache)
    return answer, model, False


# ---------------------------------------------------------------------------
# The RAG pipeline
# ---------------------------------------------------------------------------
def library_description(retriever):
    """Bullet list of the documents in the knowledge base, for the prompt."""
    sources = sorted({chunk["metadata"]["source"] for chunk in retriever.chunks})
    return "\n".join(f"- {doc_title(source)}" for source in sources)


def cited_sources(answer, sources):
    """The sources the answer actually cites ([S1], [S2][S3], [S1, S4]); all if none."""
    cited = set(re.findall(r"S\d+", " ".join(re.findall(r"\[([^\]]+)\]", answer))))
    return [s for s in sources if s["id"] in cited] or sources


def strip_citations(answer):
    """
    Remove the [S1] / [S2][S3] / [S1, S4] markers from an answer.

    Gemini is still asked to cite (it keeps answers tied to the sources), but
    the source list isn't shown to users, so the markers would point nowhere.
    Footnote markers such as [^7] are left alone.
    """
    return re.sub(r"\s*\[S\d+(?:\s*,\s*S\d+)*\]", "", answer).strip()


def answer_query(retriever, llm, question):
    """
    Run the full RAG pipeline for one question and RETURN the result.

    This function does no printing, so the command-line app, the web UI and
    evaluate.py can all share it. It returns a plain dict:

        answer  : str         -> the text to show the user (no [S1] markers)
        sources : list[dict]  -> cited sources: {"id", "title", "section", "pages", ...}
                                 (not shown to users; used by evaluate.py and DEBUG)
        found   : bool        -> whether an answer was found in the documents
        cached  : bool        -> True if the answer came from the cache
        called  : bool        -> True if Gemini was actually called
        model   : str         -> the Gemini model that wrote the answer ("" if none)
        chunks  : list[dict]  -> the retrieved chunks, for optional debugging:
                                 {"source", "section", "pages", "score", "text"}

    It may raise if the Gemini call fails; each caller decides how to show that.
    """
    # STEP 1 — retrieve the best chunks (hybrid search + reranking, local).
    hits = retriever.search(question)
    chunks = [
        {
            "source": hit["metadata"]["source"],
            "section": hit["metadata"]["section"],
            "pages": hit["metadata"]["pages"],
            "score": hit["score"],
            "pinned": hit["pinned"],
            "text": hit["text"],
        }
        for hit in hits
    ]
    result = {"answer": NO_ANSWER_MESSAGE, "sources": [], "found": False,
              "cached": False, "called": False, "model": "", "chunks": chunks}

    # STEP 2 — relevance guard. If even the best chunk scores low, the
    # documents don't cover the question: answer "not found" without
    # spending a Gemini call. A table/figure the question names by number
    # always passes.
    relevant = [hit for hit in hits if is_relevant(hit)]
    if not relevant:
        return result

    # STEPS 3 & 4 — build the numbered context and ask Gemini (or the cache).
    context, sources = retriever.build_context(relevant)
    prompt = PROMPT_TEMPLATE.format(
        library=library_description(retriever), context=context, question=question
    )
    answer, model, cached = generate(llm, prompt)
    result.update(cached=cached, called=not cached, model=model)

    # The chunks were relevant enough to retrieve, but the model decided they
    # don't contain the answer: no invented answer and no misleading sources.
    if NOT_FOUND_SENTINEL in answer:
        return result

    result.update(answer=strip_citations(answer), sources=cited_sources(answer, sources),
                  found=True)
    return result


def format_source(source):
    """One line for a source list, e.g. '[S1] GPO Annual Report FY2018 — ... — p. 27 (PDF p. 58)'."""
    parts = [source["title"], source["section"], source["pages"]]
    return f"[{source['id']}] " + " — ".join(p for p in parts if p)


def answer_question(retriever, llm, question):
    """Command-line wrapper: run the pipeline and PRINT the result."""
    try:
        result = answer_query(retriever, llm, question)
    except Exception as error:
        print(f"\nError: {describe_error(error)}")
        return

    # Optional debug view: show exactly what retrieval found, with the
    # reranker scores. Hidden by default (DEBUG = False).
    if DEBUG:
        print("\n[DEBUG] Retrieved chunks:")
        print("-" * 40)
        for chunk in result["chunks"]:
            print(f"{chunk['source']} p.{chunk['pages']}  (score: {chunk['score']:.2f})")
            print(chunk["text"])
            print("-" * 40)

    print(f"\nBot: {result['answer']}")
    if DEBUG:
        for source in result["sources"]:
            print(f"  {format_source(source)}")

    if result["cached"]:
        print("\n(answered from cache: no Gemini call)")
    elif result["called"]:
        print(f"\n(answered by {result['model']} · Gemini calls today: {calls_today()})")


HELP_TEXT = """Commands:
  help  - Show available commands
  exit  - Exit the chatbot
  quit  - Exit the chatbot

You can ask about:
  - the GPO Annual Report FY2018 (finances, operations, audit)
  - the World Bank Global Economic Prospects, June 2026 (forecasts, regions, debt, fiscal policy)
  - the Apple iMac Product Environmental Report (carbon footprint, materials, energy use)
  - the company policies (leave, work from home, reimbursement)"""


def main():
    # Retriever() loads the knowledge base and local models; build_llm()
    # needs the API key. Both fail early with a friendly message.
    print("Loading the knowledge base...")
    retriever = Retriever()
    llm = build_llm()

    print("=" * 40)
    print("       Document Q&A RAG Chatbot")
    print("=" * 40)
    print("\nAsk a question about the GPO, World Bank, iMac or company documents.")
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

        answer_question(retriever, llm, question)
        print()


if __name__ == "__main__":
    main()
