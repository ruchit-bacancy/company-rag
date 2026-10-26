# Document Q&A RAG Chatbot

One Retrieval-Augmented Generation (RAG) chatbot that answers questions about
several long, complex documents, from the command line or a minimal web UI:

| Document | File | What makes it hard |
|---|---|---|
| GPO Annual Report FY2018 | `docs/GPO-Anual-report.pdf` | 82 pages, two reports glued together with two page-numbering schemes, 27 financial tables |
| World Bank Global Economic Prospects, June 2026 | `docs/GEP-Jun-2026.pdf` | 200 pages, chapters, boxes, 42 tables, 90 charts, per-page footnotes |
| Apple iMac Product Environmental Report (Oct 2024) | `docs/iMac_PER_Oct2024.pdf` | facts hidden in tables, a chart and 19 endnotes |
| Company policies | `docs/*.md` | the original demo documents |

It uses LangChain, a local ChromaDB vector database, local embedding and
reranking models, and Google's Gemini for the final answer. Every RAG stage is
in a small, commented file so you can see how the pieces fit together.

**Built for the Gemini free tier.** The free tier allows only about 20 answers
per day, so Gemini is used for one thing only: writing the final answer.
Parsing, embeddings, search and reranking all run locally, and the
evaluation script measures retrieval without any Gemini calls.

## Contents

- [What is RAG?](#what-is-rag)
- [Architecture](#architecture)
- [How the three tasks are solved](#how-the-three-tasks-are-solved)
- [Saving Gemini quota](#saving-gemini-quota)
- [How the code maps to the RAG stages](#how-the-code-maps-to-the-rag-stages)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Ingest documents](#ingest-documents)
- [Start the chatbot (CLI)](#start-the-chatbot-cli)
- [Optional: web UI](#optional-web-ui)
- [Evaluate](#evaluate)
- [Example questions](#example-questions)
- [Adding your own documents](#adding-your-own-documents)
- [Configuration knobs](#configuration-knobs-configpy)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Notes and limitations](#notes-and-limitations)

---

## What is RAG?

Retrieval-Augmented Generation combines information retrieval with an LLM.
The LLM has never seen our documents, so instead of asking it directly, the
application first **retrieves** the most relevant passages from a knowledge
base and gives them to the LLM as **context**. The LLM then writes an answer
grounded in those passages. Behind the scenes it cites the document and page
each fact came from (shown with `DEBUG = True` and in `evaluate.py`).

---

## Architecture

```text
INGESTION (python ingest.py, offline, no API calls)
  docs/*.pdf, docs/*.md
     ↓  loaders.py: PDF → Markdown per page (PyMuPDF4LLM): headings, tables,
     ↓  chart labels; headers/footers dropped; printed page numbers;
     ↓  section paths; captions; footnotes linked to their references
  structured blocks
     ↓  ingest.py: section-aware parent/child chunking
  child chunks (~900 chars, with "document | section" header)
     ↓  local embedding model (BAAI/bge-small-en-v1.5)
  ChromaDB (vectors) + chunks.json (for BM25) + parents.json (for context)

QUESTION TIME (python chat.py / streamlit run app.py)
  question
     ↓  retrieval.py: semantic search (Chroma) + keyword search (BM25)
     ↓  merged with Reciprocal Rank Fusion
     ↓  local cross-encoder reranker re-scores the top 30
     ↓  "Table 2.12" / "Figure 1.1" / "Box 1.1" named? → fetch it directly
     ↓  relevance guard: nothing relevant → "not found" (no Gemini call)
     ↓  each winning chunk → its parent block, up to a 12k-character budget
  numbered sources [S1] [S2] ...
     ↓  chat.py: Gemini writes a cited answer (cached on disk)
  answer (citation markers removed for display)
```

---

## How the three tasks are solved

### Task 1: a solid baseline (GPO Annual Report)

- **Ingestion:** PDFs are converted page by page to Markdown with
  [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/),
  which keeps headings and tables and drops running headers and footers. A
  few leftover footer lines are removed because they repeat on many pages.
- **Page numbers you can find:** the GPO PDF holds two reports with
  different numbering (PDF page 14 is printed page 12; PDF page 58 is printed
  page 27). `printed_page_labels()` detects the number printed on every page,
  so answers cite e.g. "p. 27 (PDF p. 58)".
- **Chunking, embeddings, retrieval, generation:** see the next sections.
  `evaluate.py` keeps the original vector-only search as a baseline, so every
  improvement is measured against it.

### Task 2: long, structured documents (World Bank GEP, 200 pages)

- **Section-aware chunking.** Every block knows its section path, built from
  the PDF bookmarks (chapters, regions) plus the detected headings, e.g.
  `Ch. 2 Regional Outlooks > South Asia > Outlook`. Boxes ("BOX 1.1 How much
  will AI affect global growth?") become their own sections, and
  "(continued)" pages are merged back in. A chunk never straddles two
  sections.
- **Metadata.** Each chunk stores its document, section, content type
  (text/table/figure), caption, PDF pages and printed pages.
- **Contextual chunk headers.** Each chunk is embedded with a first line such
  as `World Bank Global Economic Prospects, June 2026 | Ch. 1 Global Outlook >
  Major economies > China`, so a sentence like "Growth is projected to
  decelerate to 4.2 percent" is findable by "China".
- **Hybrid retrieval and reranking.** Semantic search finds paraphrases; BM25
  keyword search finds exact names and numbers ("EM7", "Table 1.1"). A
  cross-encoder then re-scores the merged candidates.
- **Context management.** Small chunks (~900 chars) are searched, but their
  larger parent blocks (~3,000 chars of the same section) are sent to Gemini,
  best first, until a 12,000-character budget is reached.
- **Text repair.** The report's body font turns the "Th" ligature into a bare
  "T" ("Te global economy"); `fix_th_ligature()` repairs these words, which
  matters for keyword search.

### Task 3: tables, charts and footnotes (Apple iMac report)

- **Tables** stay Markdown tables. Long tables are split into row groups that
  each repeat the caption and header rows, so "India | 7.2 | 7.1 | 7.7 | 6.6"
  never loses the years its numbers belong to.
- **Tables and figures by number.** Captions are recognised whether they
  are headings or bold text, and every piece of a table carries its caption.
  When a question names an item ("summarise Table 2.12", "what does Figure
  1.1 show?", "explain Box 1.1"), its chunks are fetched directly by caption
  or section and always kept. Neither semantic search nor the reranker
  understands that "2.12" is a label, and without this the bot answered
  "not found".
- **Charts** are vector graphics whose labels are real text. The labels are
  kept as `figure` chunks together with the figure caption and panel titles
  (e.g. the iMac chart "445 kg baseline → −99 kg → 346 kg"). No images are
  sent to an AI model, so this costs no quota.
- **Footnotes and endnotes.** Superscript markers become `[^n]`, and each
  marker is linked to its note: on the same page (World Bank footnotes, table
  notes) or in the "Endnotes" section (Apple). The note text is appended to
  the chunk that references it, so "iMac uses 58 percent less energy than
  ENERGY STAR[^18]" is retrieved together with endnote 18 and its power
  consumption table.

---

## Saving Gemini quota

| Where | What we do |
|---|---|
| Ingestion | No Gemini calls: PDF parsing and embeddings are local |
| Search | Local embeddings, BM25 and a local reranker: no API calls |
| Out-of-scope questions | If the best reranker score is below `MIN_RERANK_SCORE`, the bot answers "not found" without calling Gemini |
| Answers | At most one Gemini call per question; no chat history is sent; "low" thinking level |
| Main model down | If `LLM_MODEL` times out, is overloaded or is out of quota, the prompt goes once to `LLM_FALLBACK_MODEL` (Flash-Lite), which has its own free-tier limits |
| Repeats | Answers are cached in `.cache/answers.json`, keyed by model + full prompt; asking again is free |
| Evaluation | `evaluate.py` scores retrieval offline; `--answers N` spends at most N calls |

The CLI prints `Gemini calls today: N` after each new answer, and the web UI
shows it in the sidebar. This counts only this app's calls on this machine.

---

## How the code maps to the RAG stages

| RAG stage | File | Function |
|---|---|---|
| 1. Document loading (PDF + Markdown) | `loaders.py` | `load_pdf()`, `load_markdown()` |
| 2. Section-aware chunking | `ingest.py` | `chunk_document()` |
| 3. Embeddings (local) | `config.py` | `get_embeddings()` |
| 4. Vector + keyword storage | `ingest.py` | `store_chunks()` |
| 5. Hybrid retrieval | `retrieval.py` | `Retriever.search()` |
| 6. Reranking | `retrieval.py` | `Retriever.rerank()` |
| 7. Context construction | `retrieval.py` | `Retriever.build_context()` |
| 8. LLM generation (+ cache) | `chat.py` | `answer_query()`, `generate()` |
| 9. Source attribution (debug/eval only) | `chat.py` | `cited_sources()`, `strip_citations()` |
| 10. Evaluation | `evaluate.py` | `evaluate_mode()` |

---

## Project structure

```text
company-rag/
├── docs/                    # the knowledge base: 3 PDFs + 4 Markdown policies
├── eval/questions.json      # test questions with expected pages and answers
├── config.py                # models, paths, chunk sizes, top-k, thresholds
├── loaders.py               # PDF/Markdown → structured blocks
├── ingest.py                # blocks → chunks → embeddings → ChromaDB
├── retrieval.py             # hybrid search → rerank → context
├── chat.py                  # CLI chatbot + the shared answer pipeline
├── app.py                   # optional Streamlit web UI
├── evaluate.py              # retrieval (and optional answer) evaluation
├── requirements.txt
├── .env                     # your GEMINI_API_KEY (git-ignored)
├── chroma_db/               # vector DB + chunk files (built by ingest.py, git-ignored)
└── .cache/                  # local models + answer cache (git-ignored)
```

---

## Prerequisites

- **Python 3.10–3.14**, built natively for your machine. On an Apple Silicon
  Mac use an arm64 Python such as Homebrew's `/opt/homebrew/bin/python3`.
- **A Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey).
  The free tier is enough; see [Saving Gemini quota](#saving-gemini-quota).
- About **300 MB of disk** for the local models, downloaded automatically on
  first use into `.cache/models/`.

Tested with LangChain 1.x, `langchain-google-genai` 4.x, `chromadb` 1.5.x,
PyMuPDF 2.0, fastembed 0.9 on Python 3.14 (Apple Silicon).

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a file named `.env` in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

`.env` is git-ignored. Only answer generation needs the key; ingestion,
search and `evaluate.py` (without `--answers`) work without it.

---

## Ingest documents

Build (or rebuild) the knowledge base:

```bash
python ingest.py
```

Example output:

```text
Loading documents...
  - GEP-Jun-2026.pdf: 1348 blocks (1214 text, 44 tables, 90 figures) in 35s
  - GPO-Anual-report.pdf: 402 blocks (363 text, 27 tables, 12 figures) in 12s
  - iMac_PER_Oct2024.pdf: 93 blocks (89 text, 3 tables, 1 figures) in 3s
  ...
Splitting documents into section-aware chunks...
  - GEP-Jun-2026.pdf: 1124 chunks in 312 parent blocks
  ...
Embedding 1499 chunks locally and storing them in ChromaDB...
Ingestion completed successfully.
```

It takes a few minutes on a laptop CPU (mostly local embedding) and uses no
API quota. It is safe to run repeatedly: `chroma_db/` is rebuilt from scratch
each time. The answer cache in `.cache/` is kept; since its key includes the
retrieved context, changed documents automatically produce fresh answers.

---

## Start the chatbot (CLI)

```bash
python chat.py
```

Type `help` for commands and topics, `exit` or `quit` to leave.

---

## Optional: web UI

```bash
streamlit run app.py
```

A chat page (default `http://localhost:8501`) shows each answer, and today's
Gemini call count in the sidebar. To see what retrieval
found behind an answer, use the CLI with `DEBUG = True` or `python evaluate.py -v`.
Press `Ctrl+C` in the terminal to stop it.

---

## Evaluate

`eval/questions.json` holds test questions across all documents. Each one
names the document, the PDF page(s) with the answer and strings the answer
must contain; a few are out of scope and should be refused.

```bash
python evaluate.py              # compare retrieval set-ups (no Gemini calls)
python evaluate.py -v           # plus per-question results
python evaluate.py --answers 3  # also generate 3 real answers (≤ 3 Gemini calls, cached)
```

Results on the 32 answerable test questions (top 6 chunks), October 2026:

```text
set-up     Doc@k  Page@k    MRR   Facts
vector      100%     88%   0.60     88%     ← original baseline
hybrid      100%     97%   0.70     97%     ← + BM25 keyword search
rerank      100%    100%   0.83    100%     ← + cross-encoder + named items (the chatbot)

Relevance guard (MIN_RERANK_SCORE = -4.0):
  out-of-scope questions rejected: 4/4
  answerable questions wrongly rejected: 0/32
```

Out-of-scope questions score about −9 to −10 with the reranker, while
answerable ones score above −4, apart from questions that name a table or
figure, which pass because the item is fetched directly. A spot check of real
answers (`--answers 4` on one question per PDF plus one out-of-scope
question) passed 4/4 using 3 Gemini calls.

**Using the official sample questions:** add them to `eval/questions.json`
in the same format (the `pages` are PDF page numbers, as shown in a PDF
viewer's page box).

---

## Example questions

```text
What is the total carbon footprint of the iMac (two ports) with 256GB?
How much power does the iMac consume in sleep mode at 115V?
Which standard is used to verify Zero Waste to Landfill at iMac supplier sites?
What is India's real GDP growth forecast for 2026 in Table 1.1?
In Box 1.1, what do the two AI productivity scenarios assume?
How is China's debt measured in the Global Economic Prospects report?
What were GPO's total assets as of September 30, 2018?
Which firm audited GPO's FY 2018 financial statements?
How many paid leaves do employees get per year?
```

Out-of-scope questions are refused, usually without a Gemini call:

```text
You: Who won the 2018 FIFA World Cup?

Bot: I couldn't find information about that in the provided documents.
```

---

## Adding your own documents

Drop a `.pdf` or `.md` file into `docs/` and re-run `python ingest.py`. To
show a friendly name instead of the filename, add it to `DOC_TITLES` in
`config.py`. Scanned PDFs (images without a text layer) are not supported.

---

## Configuration knobs (config.py)

| Setting | Meaning |
|---|---|
| `EMBEDDING_MODEL` | Local embedding model (same for documents and questions) |
| `RERANKER_MODEL` | Local cross-encoder used for reranking |
| `LLM_MODEL` | Gemini model that writes the answer |
| `LLM_THINKING_LEVEL` | How much the model reasons first (`"low"`: fast) |
| `LLM_FALLBACK_MODEL` | Model tried once if `LLM_MODEL` fails |
| `LLM_TIMEOUT` | Seconds to wait for a model before falling back (30) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Child chunk size in characters (~900) |
| `PARENT_SIZE` | Parent block size sent as context (~3,000) |
| `VECTOR_K` / `BM25_K` | Candidates from each search (30 each) |
| `RERANK_K` | Candidates re-scored by the reranker (30) |
| `TOP_K` | Chunks kept after reranking (6) |
| `MIN_RERANK_SCORE` | Relevance guard; below it, no Gemini call |
| `MAX_CONTEXT_CHARS` | Context budget per question (12,000 characters) |
| `INDEX_REFERENCES` | Whether bibliographies are indexed (off) |
| `DEBUG` | Print retrieved chunks and scores in the CLI |

If you change `EMBEDDING_MODEL` or any chunking setting, re-run
`python ingest.py`. Run `python evaluate.py` after any change to see its effect.

---

## Troubleshooting

**`Error: GEMINI_API_KEY is not configured.`**
There's no `.env` file, or the variable is misspelled.

**`Error: knowledge base not found. Please run 'python ingest.py' first.`**
Run `python ingest.py`.

**"The Gemini free-tier quota is used up for now."**
Both models hit their free-tier daily or per-minute limit. Wait and retry;
questions asked before are still answered from the cache, and search keeps
working.

**Answers take 30+ seconds and say "answered by ...-flash-lite"**
The main model timed out (`504 DEADLINE_EXCEEDED`) and the fallback answered.
This happens when `gemini-flash-latest` is overloaded on the free tier. If it
persists, set `LLM_MODEL` to a model that responds (e.g.
`"gemini-flash-lite-latest"`) to skip the wait.

**`404 ... models/<name> is not found`**
`LLM_MODEL` isn't available for your key. List the models your key supports:

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for m in client.models.list():
    print(m.name, m.supported_actions)
```

**The first run is slow / downloads files.**
The local embedding and reranking models (~300 MB) are downloaded once into
`.cache/models/`.

**Answers say "not found" for something that IS in the documents.**
Run `python evaluate.py -v` or set `DEBUG = True` to see what was retrieved
and the reranker scores. If good chunks score just below `MIN_RERANK_SCORE`,
lower it slightly.

**`bad CPU type in executable`** (after moving from an Intel Mac to Apple Silicon)
Rebuild the virtual environment with a native arm64 Python:

```bash
rm -rf .venv __pycache__
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**`pip install` stops with `Connection interrupted while downloading`**
Re-run it, optionally with `--retries 10 --timeout 60`.

---

## Security

- Never commit your API key. `.env` (and every `.env.*`) is git-ignored.
- The key is read only via `python-dotenv`; it is never hardcoded or printed.
- `chroma_db/` and `.cache/` are local and git-ignored.
- Document text is sent to Gemini only as the context of a question.

---

## Notes and limitations

- **No conversation memory:** each question is answered independently, which
  keeps every question to one Gemini call. Ask follow-ups as full questions.
- **Charts** are searchable through their labels, captions and notes, not
  their visual shape. A question like "which line is highest in March?"
  can't be answered from labels alone.
- **Table headers** spanning several rows are sometimes garbled by the PDF
  parser; the data rows and the year row are usually intact.
- **Licensing:** PyMuPDF is AGPL-3.0 (or commercial). If that matters for
  your use, pdfplumber (MIT) is an alternative parser.
- The company policy documents are dummy data for demonstration only.
