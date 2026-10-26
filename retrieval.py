"""
retrieval.py — Find the most relevant passages for a question.

This is the "retrieval" half of RAG. Plain similarity search (what this
project started with) struggles on long reports full of numbers, so this
module runs a three-stage pipeline, entirely on your machine (no API quota):

    1. Hybrid search. Two searches run side by side:
         - semantic search (embeddings in ChromaDB) finds passages that MEAN
           the same as the question, even when the wording differs
         - keyword search (BM25) finds passages sharing exact terms, which
           matters for names, codes and numbers ("Table 1.1", "EM7", "scope 2")
       Their two rankings are merged with Reciprocal Rank Fusion (RRF).
    2. Reranking. A cross-encoder reads the question TOGETHER with each
       candidate and scores how well the candidate answers it. It is slower
       than embeddings, so it only re-scores the top candidates, but it is
       far more precise.
    3. Small-to-big context. Each winning child chunk is swapped for its
       larger parent block (the surrounding part of the same section), and
       parents are added until the context budget is reached.

One shortcut sits beside the pipeline: if the question names a table,
figure, box or annex by number ("summarise Table 2.12"), the chunks of that
item are fetched directly by their caption/section and always kept. Neither
semantic search nor the reranker understands that "2.12" is a label.

search(mode=...) can also stop after stage 1 ("hybrid") or use semantic
search alone ("vector", the original baseline); evaluate.py uses this to
measure what each stage adds.
"""

import json
import os
import re

from langchain_chroma import Chroma
from py_rust_stemmers import SnowballStemmer
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNKS_FILE,
    PARENTS_FILE,
    VECTOR_K,
    BM25_K,
    RERANK_K,
    TOP_K,
    MAX_CONTEXT_CHARS,
    MIN_RERANK_SCORE,
    doc_title,
    get_embeddings,
    get_reranker,
)

# Very common words carry no meaning for keyword search.
STOP_WORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why will with does did do".split()
)
_stemmer = SnowballStemmer("english")

# A numbered item named in a question: "Table 2.12", "figure B1.1.2", "Box 3.1".
REFERENCE_RE = re.compile(r"\b(table|figure|fig\.?|box|annex)\s+([a-z]?\d+(?:\.\d+)*)", re.I)


def is_relevant(hit):
    """True if a hit passes the relevance guard (or was named in the question)."""
    if hit.get("pinned"):
        return True
    return hit["score"] is not None and hit["score"] >= MIN_RERANK_SCORE


def tokenize(text):
    """
    Split text into keyword-search terms.

    Lower-case, keep numbers whole ("2.5", "1,018,063"), drop stop words and
    reduce words to their stem, so "forecasts" matches "forecast".
    """
    tokens = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [_stemmer.stem_word(t) for t in tokens if t not in STOP_WORDS]


def format_pages(pages, labels):
    """
    Human-friendly page reference, e.g. "p. 4 (PDF p. 26)".

    The printed page number is what a reader looks for in the document; the
    PDF page is what a PDF viewer's page box needs. When the two are equal
    only one is shown.
    """
    if not pages:
        return ""

    def span(values):
        return values[0] if len(values) == 1 else f"{values[0]}–{values[-1]}"

    pdf = [str(p) for p in pages]
    if all(labels) and len(labels) == len(pdf):
        printed = list(labels)
        if printed == pdf:
            return f"p. {span(pdf)}"
        return f"p. {span(printed)} (PDF p. {span(pdf)})"
    return f"PDF p. {span(pdf)}"


class Retriever:
    """Loads the knowledge base built by ingest.py and searches it."""

    def __init__(self):
        if not (os.path.isdir(CHROMA_DIR) and os.path.exists(CHUNKS_FILE)):
            raise SystemExit(
                "Error: knowledge base not found.\n"
                "Please run 'python ingest.py' first to build the knowledge base."
            )
        with open(CHUNKS_FILE, encoding="utf-8") as file:
            self.chunks = json.load(file)
        with open(PARENTS_FILE, encoding="utf-8") as file:
            self.parents = json.load(file)
        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}

        # Keyword index: built in memory at start-up (a second or two).
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])
        # Semantic index: the ChromaDB that ingest.py created. The SAME local
        # embedding model must be used for questions as for the documents.
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        self.reranker = get_reranker()

    # -- Stage 1: hybrid search ---------------------------------------------
    def vector_search(self, question, k=VECTOR_K):
        """Ids of the k chunks whose embeddings are closest to the question."""
        results = self.vectorstore.similarity_search_with_score(question, k=k)
        return [doc.id for doc, _distance in results]

    def keyword_search(self, question, k=BM25_K):
        """Ids of the k chunks with the best BM25 keyword score."""
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i]["id"] for i in best if scores[i] > 0]

    @staticmethod
    def fuse(rankings, k=60):
        """
        Reciprocal Rank Fusion: merge several ranked lists into one.

        Each list gives a chunk 1 / (k + rank) points. Chunks ranked high by
        either search, or fairly high by both, rise to the top. Only ranks
        are used, so the two searches' incomparable scores never need mixing.
        """
        scores = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def referenced_chunks(self, question):
        """
        Ids of the chunks of every table/figure/box/annex the question names.

        "Table 2.12" matches chunks whose caption or section starts that item,
        e.g. "TABLE 2.12 Sub-Saharan Africa economy forecasts", but not
        "TABLE 2.1" or "TABLE 2.120".
        """
        ids = []
        for kind, number in REFERENCE_RE.findall(question):
            kind = "figure" if kind.lower().startswith("fig") else kind
            label = re.compile(rf"\b{kind}\s+{re.escape(number)}(?![.\d])", re.I)
            for chunk in self.chunks:
                meta = chunk["metadata"]
                if label.search(meta["caption"]) or label.search(meta["section"]):
                    ids.append(chunk["id"])
        return list(dict.fromkeys(ids))

    # -- Stage 2: rerank ----------------------------------------------------
    def rerank(self, question, chunk_ids):
        """Re-score candidates with the cross-encoder; best first."""
        texts = [self.by_id[chunk_id]["text"] for chunk_id in chunk_ids]
        scores = [float(s) for s in self.reranker.rerank(question, texts)]
        return sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)

    def search(self, question, mode="rerank", top_k=TOP_K):
        """
        Return the top_k best chunks for a question, best first.

        mode = "vector"  semantic search only (the original baseline)
               "hybrid"  semantic + keyword search, fused with RRF
               "rerank"  hybrid, then cross-encoder reranking (the default,
                         and the only mode that also fetches named tables)

        Each hit is a dict: {"id", "score", "pinned", "text", "metadata"}.
        "score" is the reranker score in "rerank" mode (higher = more
        relevant), else None. "pinned" hits belong to a table/figure/box the
        question names; they come first, best first, followed by the top_k
        best other hits. (A box can span pages, so the part that best matches
        the question goes first and fits in the context budget.)
        """
        vector_ids = self.vector_search(question)
        pinned = []
        if mode == "vector":
            ranked = [(chunk_id, None) for chunk_id in vector_ids]
        else:
            fused = self.fuse([vector_ids, self.keyword_search(question)])
            if mode == "hybrid":
                ranked = [(chunk_id, None) for chunk_id in fused]
            else:
                pinned = self.referenced_chunks(question)
                candidates = list(dict.fromkeys(pinned + fused[:RERANK_K]))
                ranked = self.rerank(question, candidates)

        # ranked is already best-first, so splitting it keeps both parts in order.
        named = [(chunk_id, score) for chunk_id, score in ranked if chunk_id in pinned]
        others = [(chunk_id, score) for chunk_id, score in ranked if chunk_id not in pinned]
        hits = []
        for chunk_id, score in named + others[:top_k]:
            chunk = self.by_id[chunk_id]
            hits.append({"id": chunk_id, "score": score, "pinned": chunk_id in pinned,
                         "text": chunk["text"], "metadata": chunk["metadata"]})
        return hits

    # -- Stage 3: small-to-big context --------------------------------------
    def build_context(self, hits, max_chars=MAX_CONTEXT_CHARS):
        """
        Turn hits into the numbered context sent to Gemini, plus a source list.

        Each hit is replaced by its parent block (the surrounding text of the
        same section, with its footnotes). Parents are added best-first until
        max_chars is reached; a parent that no longer fits is replaced by the
        small chunk itself. Returns (context_text, sources), where sources is
        a list of {"id": "S1", "source", "title", "section", "pages"}.
        """
        blocks, sources, seen, used = [], [], set(), 0
        for hit in hits:
            meta = hit["metadata"]
            parent_id = meta["parent_id"]
            if parent_id in seen:
                continue
            parent = self.parents[parent_id]
            text = parent["text"]
            pages = format_pages(parent["pages"], parent["page_labels"])
            if used + len(text) > max_chars:
                text = hit["text"]
                child_pages = [int(p) for p in meta["pages"].split(",") if p]
                pages = format_pages(child_pages, meta["page_labels"].split(","))
                if used + len(text) > max_chars:
                    continue
            seen.add(parent_id)
            used += len(text)

            source = {
                "id": f"S{len(sources) + 1}",
                "source": parent["source"],
                "title": doc_title(parent["source"]),
                "section": parent["section"],
                "pages": pages,
            }
            sources.append(source)
            label = " — ".join(x for x in (source["title"], source["section"], pages) if x)
            blocks.append(f"[{source['id']}] {label}\n{text}")
        return "\n\n---\n\n".join(blocks), sources
