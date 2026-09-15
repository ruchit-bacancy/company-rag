"""
ingest.py — Build the knowledge base for the Company Policy RAG chatbot.

This is the "indexing" half of RAG. Run it whenever the documents change.
It performs four clear steps:

    1. Load the Markdown documents from docs/
    2. Split each document into small, overlapping chunks
    3. Turn every chunk into an embedding vector (Gemini)
    4. Store the chunks + vectors + source metadata in a local ChromaDB

Run it with:

    python ingest.py
"""

import glob
import os
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    DOCS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_embeddings,
)


def load_documents():
    """
    STEP 1 — Load the source documents.

    We discover every *.md file in docs/ dynamically, so if you add a new file
    later (e.g. docs/holiday-policy.md) it is picked up automatically with no
    code changes. For each document we store just the filename in metadata
    ("source"), which the chatbot later uses for source attribution.
    """
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(f"Error: docs directory was not found (looked for '{DOCS_DIR}/').")

    md_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not md_files:
        raise SystemExit(f"Error: No Markdown documents were found in {DOCS_DIR}/.")

    print("Loading documents...")
    print("Found:")
    for path in md_files:
        print(f"  - {os.path.basename(path)}")

    documents = []
    for path in md_files:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        # Wrap the raw text in a LangChain Document and tag it with its source
        # filename, so every chunk we create later can be traced back to its file.
        documents.append(
            Document(page_content=text, metadata={"source": os.path.basename(path)})
        )

    print(f"\nLoaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    """
    STEP 2 — Split documents into smaller chunks.

    Why chunk at all? Embedding a whole document as one vector would blur many
    different topics into a single point, making search inaccurate (and could
    exceed model limits). Small, overlapping chunks give us focused vectors, so
    similarity search can retrieve the exact relevant passage. The overlap keeps
    a sentence that straddles a boundary from losing its surrounding context.
    """
    print("\nSplitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # Each resulting chunk automatically keeps its parent document's metadata,
    # so every chunk still knows which .md file it came from.
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def store_chunks(chunks):
    """
    STEPS 3 & 4 — Embed the chunks and store them in ChromaDB.

    Chroma.from_documents() runs the Gemini embedding model on every chunk
    (step 3) and writes the text + vector + metadata into a local, persistent
    database folder (step 4). No separate database server is required.

    Re-run safety: we delete any existing chroma_db/ folder first, then rebuild
    from scratch. This is the simplest way to avoid piling up duplicate chunks
    every time you run ingestion.
    """
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print("\nGenerating embeddings and storing in ChromaDB...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            # Use cosine distance so similarity scores are easy to reason about.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as error:
        raise SystemExit(f"Error while embedding/storing documents: {error}")


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    store_chunks(chunks)
    print("\nIngestion completed successfully.")
    print(f"Stored {len(chunks)} chunks in ChromaDB (collection: '{COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
