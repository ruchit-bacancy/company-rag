"""
evaluate.py — Measure how well the chatbot finds the right information.

Each test question in eval/questions.json says which document and PDF page(s)
hold the answer, and which strings the answer must contain, e.g.

    {"id": "imac-carbon-total",
     "question": "What is the total carbon footprint of the iMac ...?",
     "source": "iMac_PER_Oct2024.pdf", "pages": [10], "expected": ["346 kg"]}

An expected item can list alternatives separated by " | " ("52.6 | 52,624").
Questions with "source": null are out of scope: the bot should say it
doesn't know (the relevance guard should reject them without calling Gemini).

By default this script makes NO Gemini calls. It scores retrieval only:

    Doc@k    the right document is among the top-k chunks
    Page@k   a top-k chunk from the right document covers an expected page
    MRR      1 / rank of the first chunk on an expected page (higher = better)
    Facts    all expected strings appear in the context Gemini would receive

and compares three retrieval set-ups, to show what each technique adds:

    vector   semantic search only (the original baseline)
    hybrid   semantic + keyword (BM25) search, fused with RRF
    rerank   hybrid + cross-encoder reranking (what the chatbot uses)

Usage:

    python evaluate.py                 # compare the three set-ups
    python evaluate.py --verbose       # also show every question's result
    python evaluate.py --answers 3     # ALSO ask Gemini for 3 answers (uses quota;
                                       # answers are cached, re-runs are free)
    python evaluate.py --answers 3 --ids gpo-staff imac-sleep-power
"""

import argparse
import json
import re

from config import MIN_RERANK_SCORE, TOP_K
from retrieval import Retriever, is_relevant

QUESTIONS_FILE = "eval/questions.json"
MODES = ["vector", "hybrid", "rerank"]


def normalise(text):
    return re.sub(r"\s+", " ", text).lower()


def contains_all(text, expected):
    """True if every expected item (or one of its ' | ' alternatives) is in text."""
    text = normalise(text)
    return all(
        any(normalise(alt.strip()) in text for alt in item.split(" | "))
        for item in expected
    )


def first_page_hit(hits, question):
    """1-based rank of the first hit from the right document on an expected page."""
    for rank, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        if meta["source"] != question["source"]:
            continue
        pages = {int(p) for p in meta["pages"].split(",") if p}
        if not question.get("pages") or pages & set(question["pages"]):
            return rank
    return None


def evaluate_mode(retriever, questions, mode, verbose):
    """Score one retrieval set-up on all questions; returns a summary dict."""
    answerable = [q for q in questions if q["source"]]
    negatives = [q for q in questions if not q["source"]]
    doc_hits = page_hits = fact_hits = 0
    reciprocal_ranks = []
    rejected_wrongly = 0

    if verbose:
        print(f"\n--- {mode} ---")
    for q in answerable:
        hits = retriever.search(q["question"], mode=mode)
        if mode == "rerank":
            relevant = [h for h in hits if is_relevant(h)]
            rejected_wrongly += not relevant
        else:
            relevant = hits
        context, _sources = retriever.build_context(relevant)

        doc_ok = any(h["metadata"]["source"] == q["source"] for h in hits)
        rank = first_page_hit(hits, q)
        facts_ok = contains_all(context, q["expected"])
        doc_hits += doc_ok
        page_hits += rank is not None
        fact_hits += facts_ok
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        if verbose:
            top = hits[0]["metadata"] if hits else {}
            score = f" score={hits[0]['score']:.2f}" if hits and hits[0]["score"] is not None else ""
            status = "OK  " if facts_ok else "MISS"
            print(f"{status} {q['id']:28} page-rank={rank or '-':<2} "
                  f"top={top.get('source', '-')[:14]} p{top.get('pages', '')}{score}")

    negatives_ok = 0
    if mode == "rerank":
        for q in negatives:
            hits = retriever.search(q["question"], mode=mode)
            rejected = not any(is_relevant(h) for h in hits)
            negatives_ok += rejected
            if verbose:
                best = max((h["score"] for h in hits), default=float("-inf"))
                status = "OK  " if rejected else "MISS"
                print(f"{status} {q['id']:28} (should be rejected) best score={best:.2f}")

    n = len(answerable) or 1
    return {
        "Doc@k": doc_hits / n,
        "Page@k": page_hits / n,
        "MRR": sum(reciprocal_ranks) / n,
        "Facts": fact_hits / n,
        "rejected_wrongly": rejected_wrongly,
        "negatives_ok": negatives_ok,
        "negatives": len(negatives),
    }


def print_summary(results, n_answerable):
    print(f"\nRetrieval quality on {n_answerable} answerable questions (top-{TOP_K} chunks):\n")
    print(f"{'set-up':8} {'Doc@k':>7} {'Page@k':>7} {'MRR':>6} {'Facts':>7}")
    for mode, r in results.items():
        print(f"{mode:8} {r['Doc@k']:7.0%} {r['Page@k']:7.0%} {r['MRR']:6.2f} {r['Facts']:7.0%}")
    if "rerank" in results:
        r = results["rerank"]
        print(f"\nRelevance guard (MIN_RERANK_SCORE = {MIN_RERANK_SCORE}):")
        print(f"  out-of-scope questions rejected: {r['negatives_ok']}/{r['negatives']}")
        print(f"  answerable questions wrongly rejected: {r['rejected_wrongly']}/{n_answerable}")


def run_answers(retriever, questions, count, ids):
    """Generate real answers with Gemini (uses quota) and check expected strings."""
    from chat import answer_query, build_llm, describe_error, format_source

    chosen = [q for q in questions if (q["id"] in ids if ids else True)][:count]
    llm = build_llm()
    print(f"\nGenerating answers for {len(chosen)} question(s) "
          "(at most one Gemini call each; cached answers are free)...")
    correct = 0
    for q in chosen:
        try:
            result = answer_query(retriever, llm, q["question"])
        except Exception as error:
            print(f"\n[{q['id']}] {describe_error(error)}")
            break
        if q["source"]:
            ok = result["found"] and contains_all(result["answer"], q["expected"])
        else:
            ok = not result["found"]
        correct += ok
        if result["cached"]:
            note = f"cached answer from {result['model']}"
        elif result["called"]:
            note = f"1 Gemini call, answered by {result['model']}"
        else:
            note = "no Gemini call"
        print(f"\n[{q['id']}] {'PASS' if ok else 'FAIL'} ({note})")
        print(f"Q: {q['question']}")
        print(f"A: {result['answer']}")
        for source in result["sources"]:
            print(f"   {format_source(source)}")
    print(f"\nAnswer accuracy: {correct}/{len(chosen)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers).")
    parser.add_argument("--mode", choices=MODES, help="evaluate only this retrieval set-up")
    parser.add_argument("--verbose", "-v", action="store_true", help="show per-question results")
    parser.add_argument("--answers", type=int, default=0, metavar="N",
                        help="also generate N answers with Gemini (uses quota)")
    parser.add_argument("--ids", nargs="*", default=[], help="question ids for --answers")
    args = parser.parse_args()

    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        questions = json.load(file)
    retriever = Retriever()

    modes = [args.mode] if args.mode else MODES
    results = {mode: evaluate_mode(retriever, questions, mode, args.verbose) for mode in modes}
    print_summary(results, sum(1 for q in questions if q["source"]))

    if args.answers:
        run_answers(retriever, questions, args.answers, args.ids)


if __name__ == "__main__":
    main()
