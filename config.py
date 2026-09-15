"""
config.py — Central configuration for the Company Policy RAG chatbot.

Everything you might want to tweak lives here in ONE obvious place:
- which folder holds the Markdown documents
- where the local vector database is stored
- which Gemini models are used (embeddings + chat)
- how documents are chunked and how many chunks are retrieved

Both ingest.py and chat.py import from this file, which guarantees they use
exactly the same settings (very important for embeddings — see below).
"""

import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Quiet a noisy informational notice from the Google GenAI SDK about automatic
# function calling. We don't use tools/function calling, so it isn't relevant.
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_DIR = "docs"                    # folder containing the Markdown knowledge base
CHROMA_DIR = "chroma_db"             # folder where ChromaDB persists data on disk
COLLECTION_NAME = "company_policies" # name of the Chroma collection

# ---------------------------------------------------------------------------
# Models  (change a model here and it changes everywhere)
# ---------------------------------------------------------------------------
# The SAME embedding model must be used for documents (during ingestion) and for
# questions (during chat). Otherwise the two sets of vectors live in different
# "spaces" and similarity search returns nonsense. Keeping the name here, used
# by get_embeddings() below, is what enforces that.
EMBEDDING_MODEL = "models/gemini-embedding-001"

# The chat model that writes the final, natural-language answer.
# "gemini-flash-latest" is an alias that always points to the current Gemini
# Flash model, so this keeps working as Google rotates model versions. You can
# pin a specific version (e.g. "gemini-3.6-flash") here if you prefer.
LLM_MODEL = "gemini-flash-latest"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500     # characters per chunk (small, so each vector stays focused)
CHUNK_OVERLAP = 50   # characters shared between neighbouring chunks (keeps context)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 3            # how many similar chunks to retrieve per question

# We configure Chroma to use cosine DISTANCE (0 = identical, larger = less
# similar). If even the closest chunk is further away than this, we assume the
# question is not covered by the documents and answer "I couldn't find this".
# This is a rough, easy-to-understand guard — not a finely tuned value. Turn on
# DEBUG (below) to see the real distances and adjust if you like.
MAX_DISTANCE = 0.6

# When True, chat.py prints the retrieved chunks and their distances so you can
# see exactly what RAG fed to the model. Keep False for clean chatbot output.
DEBUG = False


def load_api_key():
    """
    Load GEMINI_API_KEY from the .env file.

    Fails with a clear, user-friendly message if the key is missing. We never
    print the key itself.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: GEMINI_API_KEY is not configured.\n"
            "Please add GEMINI_API_KEY to your .env file."
        )
    return api_key


def get_embeddings():
    """
    Build the Gemini embedding model.

    Both ingest.py and chat.py call this, so documents and queries are always
    embedded the exact same way.
    """
    api_key = load_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
