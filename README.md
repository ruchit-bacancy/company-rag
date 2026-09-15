# Company Policy RAG Chatbot

A small, complete, **educational** Retrieval-Augmented Generation (RAG) chatbot.
It answers questions about a set of dummy company policy documents, from the
command line or a minimal web UI, using LangChain, Google's Gemini models, and a
local ChromaDB vector database.

This project is intentionally simple. Every RAG stage is easy to find in the
source code so you can learn how the pieces fit together.

## Contents

- [What is RAG?](#what-is-rag)
- [Architecture](#architecture)
- [How the code maps to the RAG stages](#how-the-code-maps-to-the-rag-stages)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Ingest documents](#ingest-documents)
- [Start the chatbot (CLI)](#start-the-chatbot-cli)
- [Optional: web UI](#optional-web-ui)
- [Adding your own documents](#adding-your-own-documents)
- [Example questions](#example-questions)
- [Debug mode](#debug-mode)
- [Configuration knobs](#configuration-knobs-configpy)
- [Models (and how to change them)](#models-and-how-to-change-them)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Notes and limitations](#notes-and-limitations)

---

## What is RAG?

Retrieval-Augmented Generation combines information retrieval with an LLM.

Instead of asking the LLM to answer directly (it does **not** know your private
company documents), the application first **retrieves** relevant information
from a knowledge base and then gives that information to the LLM as **context**.
The LLM then writes an answer grounded in that context.

**Why we need it:** the LLM has never seen our company documents.
**How it works:** we convert the documents into embeddings and store them in a
vector database.
**When a question arrives:** we embed the question and search for similar
document chunks.
**Then:** we hand those chunks to Gemini as context.
**Finally:** Gemini generates a natural-language answer based on that context.

---

## Architecture

```text
Documents (docs/*.md)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Gemini Embeddings
   ↓
ChromaDB (local, on disk)
   ↓
Query
   ↓
Similarity Search (top-k)
   ↓
Context (retrieved chunks + sources)
   ↓
Gemini LLM (grounded prompt)
   ↓
Answer + Sources
```

- `ingest.py` builds the knowledge base (the top half: documents → ChromaDB).
- `chat.py` answers questions (the bottom half: query → answer).
- `app.py` is an optional web UI that reuses `chat.py`'s pipeline.
- `config.py` holds all the settings both scripts share.

---

## How the code maps to the RAG stages

If you are reading the code to learn RAG, this is where each stage lives:

| RAG stage                      | File       | Function                       |
|--------------------------------|------------|--------------------------------|
| 1. Document loading            | `ingest.py`| `load_documents()`             |
| 2. Chunking                    | `ingest.py`| `split_documents()`            |
| 3. Embeddings (docs)           | `config.py`| `get_embeddings()`             |
| 4. Vector storage              | `ingest.py`| `store_chunks()`               |
| 5. Query embedding + retrieval | `chat.py`  | `retrieve()`                   |
| 6. Context construction        | `chat.py`  | `build_context()`              |
| 7. LLM generation              | `chat.py`  | `answer_query()`               |
| 8. Source attribution          | `chat.py`  | `unique_sources()`             |

The embedding model in `config.py` is used for **both** documents and queries —
that is essential, because the query and the documents must live in the same
vector space for similarity search to be meaningful.

---

## Project structure

```text
company-rag/
├── docs/                  # the Markdown knowledge base
│   ├── company.md
│   ├── leave-policy.md
│   ├── wfh-policy.md
│   └── reimbursement.md
├── config.py              # models, paths, chunk size, top-k, threshold, DEBUG
├── ingest.py              # load → chunk → embed → store
├── chat.py                # embed query → retrieve → build context → generate
├── app.py                 # optional Streamlit web UI (same pipeline as chat.py)
├── requirements.txt
├── .env                   # your GEMINI_API_KEY (git-ignored, never committed)
├── chroma_db/             # local vector database (created by ingest.py, git-ignored)
└── README.md
```

---

## Prerequisites

- **Python 3.10–3.13.** (Python 3.14 was too new at time of writing — some
  dependencies had no prebuilt wheels and failed to compile. If `pip install`
  fails while building `pandas`/`pkg_resources`, use 3.13. See
  [Troubleshooting](#troubleshooting).)
- **A Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
  (the free tier is enough for this demo, with limits — see
  [Troubleshooting](#troubleshooting)).

Tested with LangChain 1.x, `langchain-google-genai` 4.x, `langchain-chroma` 1.x,
`chromadb` 1.5.x on Python 3.13.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a file named `.env` in the project root with your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

`.env` is git-ignored, so your key is never committed. It is read with
`python-dotenv` and is never hardcoded or printed.

---

## Ingest documents

Build (or rebuild) the vector database:

```bash
python ingest.py
```

Example output:

```text
Loading documents...
Found:
  - company.md
  - leave-policy.md
  - reimbursement.md
  - wfh-policy.md

Loaded 4 documents.

Splitting documents...
Created 11 chunks.

Generating embeddings and storing in ChromaDB...

Ingestion completed successfully.
Stored 11 chunks in ChromaDB (collection: 'company_policies').
```

It is safe to run this multiple times — it resets the local collection first, so
you never accumulate duplicate chunks. (The exact chunk count depends on the
documents and the splitter settings.)

---

## Start the chatbot (CLI)

```bash
python chat.py
```

Then ask questions. Type `help` for commands, `exit` or `quit` to leave.

> Run `python ingest.py` **before** the first chat, or you'll see
> *"vector database not found."*

---

## Optional: web UI

A minimal Streamlit interface is included. It is a thin layer over the **same**
RAG pipeline — it imports and calls `answer_query()` from `chat.py`, so there is
no duplicated retrieval or generation logic.

```bash
streamlit run app.py
```

This opens a chat page (default `http://localhost:8501`) that shows each answer,
its source files, and a collapsible **"Show retrieved chunks"** panel (the chunks
and their similarity distances) so you can see exactly what retrieval fed to
Gemini.

To stop the server, press **`Ctrl+C`** in that terminal.

Note: the web UI makes it easy to send many questions quickly — keep the Gemini
free-tier limits in mind.

---

## Adding your own documents

The document loader discovers every `*.md` file in `docs/` automatically, so you
do not touch any code to extend the knowledge base:

1. Drop a new Markdown file into `docs/`, e.g. `docs/holiday-policy.md`.
2. Re-run `python ingest.py` to rebuild the vector database.
3. Ask away — answers will now cite `holiday-policy.md` when relevant.

Source attribution uses the filename, so give your files clear, descriptive
names.

---

## Example questions

Questions the documents **can** answer:

```text
How many paid leaves do employees get?
Can I work from home on Friday?
What are the working hours?
How much can I claim for meals?
How much internet reimbursement can I get?
How long does reimbursement take?
Can I carry unused leave to next year?
What happens if I take sick leave for more than two days?
```

Semantic search in action — "work remotely" finds the "work from home" policy
even though the exact words differ:

```text
You: Can I work remotely?

Bot: Yes. Employees who have completed probation can request work from home,
up to two days per week.

Sources:
- wfh-policy.md
```

Questions the documents do **not** cover — the bot does not make anything up:

```text
You: What is the resignation notice period?

Bot: I couldn't find information about that in the provided company documents.

Sources:
No relevant source found.
```

Other out-of-scope questions to try (should not be fabricated):

```text
Who is the CEO?
What is the company's stock price?
```

---

## Debug mode

Set `DEBUG = True` in `config.py` to print the retrieved chunks and their
similarity distances before each answer (the web UI always offers this via the
collapsible "Show retrieved chunks" panel). Example:

```text
Retrieved context:
----------------------------
Source: leave-policy.md  (distance: 0.360)
Every full-time employee receives 18 paid leaves per calendar year.
...
----------------------------
```

This is the best way to *see* how retrieval drives the answer. Keep it `False`
for clean CLI output.

---

## Configuration knobs (config.py)

| Setting           | Meaning                                              |
|-------------------|------------------------------------------------------|
| `EMBEDDING_MODEL` | Gemini embedding model (same for docs and queries)   |
| `LLM_MODEL`       | Gemini chat model used to write the answer           |
| `CHUNK_SIZE`      | Characters per chunk (~500)                          |
| `CHUNK_OVERLAP`   | Overlap between chunks (~50)                          |
| `TOP_K`           | How many chunks to retrieve (3)                      |
| `MAX_DISTANCE`    | Relevance guard; larger distance = less relevant     |
| `DEBUG`           | Show retrieved chunks + distances                    |

**Note on the relevance threshold:** `MAX_DISTANCE` is a simple cosine-distance
cutoff (0 = identical, larger = less similar), not a finely tuned value. If
relevant questions get rejected or irrelevant ones slip through, turn on `DEBUG`
to see the actual distances and adjust the number.

---

## Models (and how to change them)

Defaults (in `config.py`):

- `EMBEDDING_MODEL = "models/gemini-embedding-001"`
- `LLM_MODEL = "gemini-flash-latest"` (an alias that always points to the current
  Gemini Flash model, so it keeps working as Google rotates versions)

To use different models, edit those two lines. **If you change the embedding
model, re-run `python ingest.py`** — the stored document vectors and your query
vectors must come from the same model.

Not every model name is available on every key/API version. To list the models
**your** key supports:

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for m in client.models.list():
    print(m.name, m.supported_actions)
```

Look for `embedContent` (embedding models) and `generateContent` (chat models).

---

## Troubleshooting

**`Error: GEMINI_API_KEY is not configured.`**
There's no `.env` file, or the variable is misspelled. It must be exactly
`GEMINI_API_KEY=...` on its own line in a file named `.env` in the project root.

**`Error: vector database not found. Please run 'python ingest.py' first.`**
The `chroma_db/` folder doesn't exist yet. Run `python ingest.py`.

**`404 ... models/<name> is not found ... or is not supported`**
The model name in `config.py` isn't available for your key/API version. List the
models your key supports (see [Models](#models-and-how-to-change-them)) and set
`EMBEDDING_MODEL` / `LLM_MODEL` to a name from that list. Re-run `ingest.py` if
you changed the embedding model.

**`429 RESOURCE_EXHAUSTED` / "You exceeded your current quota"**
You've hit the Gemini **free-tier** limits (there are both per-minute and
per-day caps, and they can be small). The app handles this gracefully and keeps
running. Options: wait a minute and retry, space out your questions, or move to a
paid tier. Note ingestion embeddings and chat generation have separate quotas.

**`pip install` fails building `pandas` / `ModuleNotFoundError: No module named 'pkg_resources'`**
Your Python is likely too new for prebuilt wheels of a dependency. Use Python
3.10–3.13:

```bash
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Answers seem off, or it says "not found" for something that IS in the docs**
Turn on `DEBUG = True` in `config.py` to inspect what was retrieved and the
distances. If good chunks are being filtered out, raise `MAX_DISTANCE`; if junk
is getting through, lower it. You can also adjust `CHUNK_SIZE` / `TOP_K`.

---

## Security

- Never commit your API key. `.env` (and every `.env.*`) is git-ignored.
- The key is read only via `python-dotenv`; it is never hardcoded or printed.
- `chroma_db/` is local and git-ignored — it's rebuilt any time from `docs/`.
- All company data here is **dummy** data for demonstration only.

---

## Notes and limitations

- **No conversation memory:** each question is answered independently (the web UI
  shows past messages for readability, but they are not sent to Gemini).
- **No agents / LangGraph / async** — this is a deliberately minimal RAG demo.
- The relevance threshold is a simple distance cutoff, not a tuned classifier;
  the grounded prompt is the real safeguard against made-up answers.