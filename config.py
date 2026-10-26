"""
config.py — Central configuration for the multi-document RAG chatbot.

Everything you might want to tweak lives here in ONE obvious place:
- which folder holds the source documents (Markdown + PDF)
- where the local vector database and search indexes are stored
- which models are used (local embeddings + reranker, Gemini for answers)
- how documents are chunked, how many chunks are retrieved, and how much
  context is sent to Gemini

ingest.py, retrieval.py, chat.py and evaluate.py all import from this file,
which guarantees they use exactly the same settings (very important for
embeddings — see below).

A note on the Gemini free tier
------------------------------
The free tier allows only ~20 answer generations per day, so this project
spends Gemini calls ONLY on writing the final answer — at most one call per
question, and none at all when retrieval finds nothing relevant. Embeddings
and reranking run locally on your machine (no quota, no API key needed), and
answers are cached on disk so asking the same question twice is free.
"""

import logging
import os

# Keep model downloads quiet (no Hugging Face login hints).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.embeddings import Embeddings

# Quiet a noisy informational notice from the Google GenAI SDK about automatic
# function calling. We don't use tools/function calling, so it isn't relevant.
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_DIR = "docs"                    # folder containing the knowledge base (*.md, *.pdf)
CHROMA_DIR = "chroma_db"             # folder where ChromaDB persists data on disk
COLLECTION_NAME = "knowledge_base"   # name of the Chroma collection
# Side files written by ingest.py next to the vector database:
CHUNKS_FILE = os.path.join(CHROMA_DIR, "chunks.json")    # child chunks (for BM25)
PARENTS_FILE = os.path.join(CHROMA_DIR, "parents.json")  # parent sections (for context)
# Local caches (git-ignored). They survive re-ingestion.
CACHE_DIR = ".cache"
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "models")          # downloaded local models
ANSWER_CACHE_FILE = os.path.join(CACHE_DIR, "answers.json")  # Gemini answers by prompt

# Friendly names shown in answers and source lists. A file that isn't listed
# here is shown under its filename.
DOC_TITLES = {
    "GPO-Anual-report.pdf": "GPO Annual Report FY2018",
    "GEP-Jun-2026.pdf": "World Bank Global Economic Prospects, June 2026",
    "iMac_PER_Oct2024.pdf": "Apple iMac Product Environmental Report (Oct 2024)",
}

# ---------------------------------------------------------------------------
# Models  (change a model here and it changes everywhere)
# ---------------------------------------------------------------------------
# Local embedding model (runs on your CPU through fastembed/ONNX; downloaded
# once on first use). The SAME model must embed the documents (ingest.py) and
# the questions (retrieval.py), otherwise the two sets of vectors live in
# different "spaces" and similarity search returns nonsense. If you change it,
# re-run `python ingest.py`.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Local cross-encoder that re-scores the retrieved candidates. It reads the
# question and a chunk TOGETHER, so it judges relevance much more precisely
# than embedding similarity alone.
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"

# The Gemini chat model that writes the final, natural-language answer.
# "gemini-flash-latest" is an alias that always points to the current Gemini
# Flash model, so this keeps working as Google rotates model versions.
LLM_MODEL = "gemini-flash-latest"
# How much the model "thinks" before answering: "minimal", "low", "medium" or
# "high". Answering from given sources needs little; "low" is fast.
LLM_THINKING_LEVEL = "low"
# If the main model fails (overloaded, timed out, or out of quota), the same
# prompt is retried once on this lighter model. Free-tier limits are set per
# model, so the fallback has its own daily allowance.
LLM_FALLBACK_MODEL = "gemini-flash-lite-latest"
LLM_TIMEOUT = 30     # seconds to wait for a model before giving up on it

# ---------------------------------------------------------------------------
# Chunking  (sizes are in characters; ~4 characters ≈ 1 token)
# ---------------------------------------------------------------------------
# Small "child" chunks are what we search over: small = focused vectors.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
# Larger "parent" blocks are what we hand to Gemini: once a small chunk
# matches, its surrounding parent block gives the model enough context.
PARENT_SIZE = 3000
# Footnote text appended to a chunk that references it (characters).
CHILD_NOTE_CHARS = 300    # in the searchable child chunk
PARENT_NOTE_CHARS = 1500  # in the context sent to Gemini
# Bibliographies ("References" sections) add hundreds of chunks of paper
# titles that crowd out real answers, so they are not indexed by default.
INDEX_REFERENCES = False

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_K = 30        # candidates from semantic (embedding) search
BM25_K = 30          # candidates from keyword (BM25) search
RERANK_K = 30        # how many fused candidates the cross-encoder re-scores
TOP_K = 6            # best child chunks kept after reranking

# Relevance guard on the reranker score (higher = more relevant). If even the
# best chunk scores below this, the question is treated as "not covered by
# the documents" and Gemini is NOT called at all — which also saves quota.
MIN_RERANK_SCORE = -4.0

# Upper bound on the context sent to Gemini per question, in characters.
MAX_CONTEXT_CHARS = 12000

# When True, chat.py prints the retrieved chunks and their scores so you can
# see exactly what RAG fed to the model. Keep False for clean chatbot output.
DEBUG = False


def load_api_key():
    """
    Load GEMINI_API_KEY from the .env file.

    Fails with a clear, user-friendly message if the key is missing. We never
    print the key itself. Only answer generation needs the key; ingestion and
    retrieval run fully offline.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: GEMINI_API_KEY is not configured.\n"
            "Please add GEMINI_API_KEY to your .env file."
        )
    return api_key


def doc_title(source):
    """Human-friendly title for a source filename."""
    return DOC_TITLES.get(source, source)


class LocalEmbeddings(Embeddings):
    """
    LangChain wrapper around a local fastembed model.

    Documents and questions are embedded slightly differently ("passage" vs
    "query"), as the BGE models expect.
    """

    def __init__(self, model_name):
        self.model = TextEmbedding(model_name, cache_dir=MODEL_CACHE_DIR)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


def get_embeddings():
    """
    Build the local embedding model.

    Both ingest.py and retrieval.py call this, so documents and queries are
    always embedded the exact same way. The model is downloaded on first use.
    """
    return LocalEmbeddings(EMBEDDING_MODEL)


def get_reranker():
    """Build the local cross-encoder used to re-score retrieved chunks."""
    return TextCrossEncoder(RERANKER_MODEL, cache_dir=MODEL_CACHE_DIR)