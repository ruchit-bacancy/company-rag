"""
ingest.py — Build the knowledge base for the multi-document RAG chatbot.

This is the "indexing" half of RAG. Run it whenever the documents change.
It performs four clear steps:

    1. Load every document in docs/ (Markdown and PDF) into structured blocks
       that know their section, page, caption and footnotes (loaders.py)
    2. Chunk section by section:
         - small "child" chunks are what we SEARCH over
         - larger "parent" blocks are what we SEND to Gemini as context
    3. Turn every child chunk into an embedding vector (local model, no quota)
    4. Store the child chunks + vectors + metadata in a local ChromaDB, and
       save children and parents as JSON (for keyword search and context)

No Gemini calls are made here, so you can re-run it as often as you like.

Run it with:

    python ingest.py
"""

import glob
import itertools
import json
import os
import re
import shutil
import time

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DOCS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNKS_FILE,
    PARENTS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_SIZE,
    CHILD_NOTE_CHARS,
    PARENT_NOTE_CHARS,
    INDEX_REFERENCES,
    doc_title,
    get_embeddings,
)
from loaders import NOTE_REF_RE, is_reference_list, load_markdown, load_pdf, table_cells

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


# ---------------------------------------------------------------------------
# STEP 1 — Load
# ---------------------------------------------------------------------------
def load_documents():
    """
    Load every *.md and *.pdf file in docs/ as a list of blocks.

    Files are discovered dynamically, so dropping a new file into docs/ and
    re-running ingestion is all it takes to add it to the chatbot.
    """
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(f"Error: docs directory was not found (looked for '{DOCS_DIR}/').")

    paths = sorted(
        glob.glob(os.path.join(DOCS_DIR, "*.md")) + glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    )
    if not paths:
        raise SystemExit(f"Error: No Markdown or PDF documents were found in {DOCS_DIR}/.")

    print("Loading documents...")
    documents = {}
    for path in paths:
        source = os.path.basename(path)
        started = time.time()
        blocks = load_pdf(path) if path.lower().endswith(".pdf") else load_markdown(path)
        if not INDEX_REFERENCES:
            blocks = [b for b in blocks if not is_reference_list(b)]
        documents[source] = blocks
        kinds = {k: sum(b["type"] == k for b in blocks) for k in ("text", "table", "figure")}
        print(
            f"  - {source}: {len(blocks)} blocks "
            f"({kinds['text']} text, {kinds['table']} tables, {kinds['figure']} figures) "
            f"in {time.time() - started:.1f}s"
        )
    return documents


# ---------------------------------------------------------------------------
# STEP 2 — Chunk
# ---------------------------------------------------------------------------
def _is_data_row(row):
    """A table row whose first cell is a label and whose values are mostly numbers."""
    cells = [c.strip() for c in table_cells(row)]
    if not cells or not cells[0]:
        return False
    values = [c for c in cells[1:] if c]
    return not values or sum(bool(re.search(r"\d", c)) for c in values) >= len(values) / 2


def split_table(table, caption):
    """
    Split a Markdown table into row groups that each fit in one chunk.

    Every piece repeats the caption and the header rows, so a row like
    "|India|7.2|7.1|7.7|6.6|..." is never separated from the years it
    belongs to.
    """
    lines = table.splitlines()
    separator = next((i for i, line in enumerate(lines) if re.fullmatch(r"\|(---\|)+", line)), None)
    if separator is None:
        header, rows = [], lines
    else:
        end = separator + 1
        while end < len(lines) and end <= separator + 3 and not _is_data_row(lines[end]):
            end += 1
        header, rows = lines[:end], lines[end:]

    prefix = ([caption] if caption else []) + header
    budget = max(CHUNK_SIZE - len("\n".join(prefix)), 300)
    pieces, current = [], []
    for row in rows:
        if current and len("\n".join(current + [row])) > budget:
            pieces.append("\n".join(prefix + current))
            current = []
        current.append(row)
    if current or not pieces:
        pieces.append("\n".join(prefix + current))
    return pieces


def split_block(block):
    """Split one block into pieces no bigger than a child chunk."""
    if block["type"] == "table":
        texts = split_table(block["text"], block["caption"])
    elif len(block["text"]) <= CHUNK_SIZE:
        texts = [block["text"]]
    else:
        texts = TEXT_SPLITTER.split_text(block["text"])
    return [dict(block, text=text) for text in texts]


def pack(pieces, limit, same_kind=False):
    """
    Greedily group consecutive pieces into groups of at most `limit` characters.

    With same_kind=True a group never mixes text with tables or figures, or
    pieces under different captions, so each child chunk stays about one thing.
    """
    groups, current, size = [], [], 0
    for piece in pieces:
        kind = (piece["type"], piece["caption"])
        changed = same_kind and current and kind != (current[-1]["type"], current[-1]["caption"])
        if current and (size + len(piece["text"]) > limit or changed):
            groups.append(current)
            current, size = [], 0
        current.append(piece)
        size += len(piece["text"]) + 2
    if current:
        groups.append(current)
    return groups


def collect_notes(pieces, body, max_chars):
    """Footnote texts for the markers that appear in `body`, as '[^n]: text' lines."""
    notes = {}
    for piece in pieces:
        for number, text in piece["notes"].items():
            notes.setdefault(number, text)
    lines = []
    for number in dict.fromkeys(NOTE_REF_RE.findall(body)):
        if number in notes:
            text = notes[number]
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
            lines.append(f"[^{number}]: {text}")
    return lines


def page_info(pieces):
    """Sorted distinct PDF pages and their printed labels for a group of pieces."""
    pages = {}
    for piece in pieces:
        if piece["page"] is not None:
            pages.setdefault(piece["page"], piece["page_label"])
    return sorted(pages), [pages[p] for p in sorted(pages)]


def chunk_document(source, blocks):
    """
    Section-aware parent/child chunking for one document.

    Blocks are grouped by section, so a chunk never straddles two sections.
    Within a section, pieces are packed into parents (~PARENT_SIZE chars),
    and every parent is packed into children (~CHUNK_SIZE chars).

    Each child is stored with a short header naming its document and section
    ("contextual chunk header"). A sentence such as "Growth is projected to
    decelerate to 4.2 percent" only makes sense once you know it sits under
    "Major economies > China", and the header gives both the search and the
    model that context. Footnotes referenced in the chunk are appended too.
    """
    title = doc_title(source)
    children, parents = [], {}
    for section, group in itertools.groupby(blocks, key=lambda b: tuple(b["section"])):
        section_path = " > ".join(section)
        pieces = [piece for block in group for piece in split_block(block)]
        for parent_pieces in pack(pieces, PARENT_SIZE):
            parent_id = f"{source}::p{len(parents)}"
            body = "\n\n".join(p["text"] for p in parent_pieces)
            notes = collect_notes(parent_pieces, body, PARENT_NOTE_CHARS)
            pages, labels = page_info(parent_pieces)
            parents[parent_id] = {
                "text": body + ("\n\nNotes:\n" + "\n".join(notes) if notes else ""),
                "source": source,
                "section": section_path,
                "pages": pages,
                "page_labels": labels,
            }
            for child_pieces in pack(parent_pieces, CHUNK_SIZE, same_kind=True):
                first = child_pieces[0]
                body = "\n\n".join(p["text"] for p in child_pieces)
                header = f"{title} | {section_path}" if section_path else title
                if first["caption"] and not body.startswith(first["caption"]):
                    header += f" | {first['caption']}"
                notes = collect_notes(child_pieces, body, CHILD_NOTE_CHARS)
                pages, labels = page_info(child_pieces)
                children.append({
                    "id": f"{source}::c{len(children)}",
                    "text": "\n".join([header, body] + notes),
                    "metadata": {
                        "source": source,
                        "section": section_path,
                        "caption": first["caption"],
                        "content_type": first["type"],
                        "page": pages[0] if pages else -1,
                        "page_label": labels[0] if labels else "",
                        "pages": ",".join(map(str, pages)),
                        "page_labels": ",".join(labels),
                        "parent_id": parent_id,
                    },
                })
    return children, parents


# ---------------------------------------------------------------------------
# STEPS 3 & 4 — Embed and store
# ---------------------------------------------------------------------------
def store_chunks(children, parents):
    """
    Embed the child chunks and store everything on disk.

    Chroma.from_texts() runs the local embedding model on every child chunk
    and writes text + vector + metadata into chroma_db/. The same children are
    saved to chunks.json for keyword (BM25) search, and the parents to
    parents.json so retrieval can expand a matching child into its context.

    Re-run safety: chroma_db/ is deleted and rebuilt from scratch, so
    re-running never piles up duplicate chunks. Embeddings are local, so this
    costs no API quota.
    """
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print(f"\nEmbedding {len(children)} chunks locally and storing them in ChromaDB...")
    started = time.time()
    try:
        Chroma.from_texts(
            texts=[c["text"] for c in children],
            metadatas=[c["metadata"] for c in children],
            ids=[c["id"] for c in children],
            embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            # Use cosine distance so similarity scores are easy to reason about.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as error:
        raise SystemExit(f"Error while embedding/storing documents: {error}")
    print(f"Done in {time.time() - started:.0f}s.")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(children, file, ensure_ascii=False)
    with open(PARENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(parents, file, ensure_ascii=False)


def main():
    documents = load_documents()

    print("\nSplitting documents into section-aware chunks...")
    children, parents = [], {}
    for source, blocks in documents.items():
        doc_children, doc_parents = chunk_document(source, blocks)
        children += doc_children
        parents.update(doc_parents)
        print(f"  - {source}: {len(doc_children)} chunks in {len(doc_parents)} parent blocks")

    store_chunks(children, parents)
    print("\nIngestion completed successfully.")
    print(f"Stored {len(children)} chunks in ChromaDB (collection: '{COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
