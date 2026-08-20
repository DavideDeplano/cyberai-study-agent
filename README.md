# CyberAI Study Agent

> A retrieval-augmented study companion for the Master's in Cybersecurity and Artificial Intelligence at the University of Cagliari.

Ingests course PDFs (slides, lecture notes, papers) and answers natural-language questions, grounding every claim in the source material with page-level citations. Built end-to-end with free-tier services and open-source components.

---

## Table of contents

- [CyberAI Study Agent](#cyberai-study-agent)
  - [Table of contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Architecture](#architecture)
  - [Tech stack](#tech-stack)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Design notes](#design-notes)
  - [Project structure](#project-structure)
  - [Roadmap](#roadmap)
  - [License](#license)
  - [Author](#author)

---

## Motivation

A Master's programme generates a large corpus of study material — dozens of PDFs across cybersecurity, machine learning, and adversarial AI. Keyword search over that corpus is brittle, and re-reading it is slow. A retrieval-augmented question-answering agent turns the same material into an interactive knowledge base while providing a compact but realistic engineering exercise:

- A complete RAG pipeline: ingestion → chunking → embedding → vector search → generation.
- Local, cost-free embeddings via `sentence-transformers`.
- Persistent vector search via ChromaDB.
- LLM generation via Google Gemini on its free tier.
- A modular design that leaves clear extension points for quiz generation, spaced repetition, per-course collections, and a local LLM backend.

## Architecture

```
                ┌─────────────┐
     PDF  ────▶ │  ingest.py  │  ──▶ Chunk objects (text + source + page)
                └─────────────┘
                       │
                       ▼
                ┌───────────────┐
                │ embeddings.py │  ──▶ 768-dim vectors (multilingual-e5-base)
                └───────────────┘
                       │
                       ▼
                ┌────────────────┐
                │ vectorstore.py │  ──▶ ChromaDB (persistent, cosine)
                └────────────────┘
                       │
        query  ────────┤
                       ▼
                ┌─────────────┐        ┌──────────────┐
                │   agent.py  │  ────▶ │ Gemini 2.5   │  ──▶ cited answer
                │  (RAG loop) │        │    Flash     │
                └─────────────┘        └──────────────┘
                       │
                       ▼
                ┌─────────────┐
                │   cli.py    │  ──▶ `cyberai-agent {ingest,chat,stats}`
                └─────────────┘
```

Each module has a single responsibility and can be exercised in isolation via its `__main__` block during development.

## Tech stack

| Layer         | Choice                              | Rationale                                                    |
| ------------- | ----------------------------------- | ------------------------------------------------------------ |
| LLM           | Google Gemini 2.5 Flash             | Generous free tier (250 req/day), fast, low latency          |
| Embeddings    | `intfloat/multilingual-e5-base`     | Local, free, multilingual (IT + EN), 768-dim                 |
| Vector store  | ChromaDB (persistent client)        | Local, no separate server, cosine distance out of the box    |
| PDF parsing   | `pypdf`                             | Pure Python, adequate for text-based academic PDFs           |
| CLI           | Typer                               | Type-hint-driven, produces a clean `--help` for free         |
| Packaging     | `pyproject.toml` + setuptools       | Modern standard, `src/` layout                               |

## Installation

**Requirements**: Python 3.11 or newer, a free Google AI Studio API key ([aistudio.google.com](https://aistudio.google.com)).

```bash
# 1. Clone the repository
git clone https://github.com/DavideDeplano/cyberai-study-agent.git
cd cyberai-study-agent

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install the package in editable mode
pip install -e .

# 4. Configure the Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

The first ingestion or chat command will download the embedding model (~450 MB) from Hugging Face and cache it under `~/.cache/huggingface/`.

## Usage

Once installed, the package registers a `cyberai-agent` command with three subcommands.

**Ingest study material**

```bash
cyberai-agent ingest data/pdfs/lecture01.pdf
cyberai-agent ingest data/pdfs/                # every PDF in the directory
cyberai-agent ingest file1.pdf file2.pdf       # multiple explicit files
```

Ingestion is idempotent: re-running it on the same file overwrites the previously stored chunks rather than duplicating them.

**Ask questions interactively**

```bash
cyberai-agent chat
```

Opens a REPL. The embedding model and the Gemini client are initialised once at startup, so every subsequent question only pays for retrieval and generation. Type `exit`, `quit`, or an empty line to leave.

**Inspect the index**

```bash
cyberai-agent stats
```

## Design notes

- **Grounding is enforced by the system prompt**, which instructs the model to answer strictly from the retrieved excerpts and to cite every claim as `[source, page N]`. Combined with an explicit permission to refuse, this sharply reduces hallucination compared to unconstrained generation.
- **Chunking is page-aware**: text is split page by page rather than across the whole document, so every chunk retains an accurate page number that can be surfaced in citations.
- **The e5 embedding model uses asymmetric prefixes** (`passage: ` for documents, `query: ` for queries). Both are handled inside the `Embedder` wrapper so callers cannot forget them and degrade retrieval quality by accident.
- **Chunk identity is deterministic**: the store key is `f"{source}::{chunk_id}"`, which makes upsert-based re-ingestion trivial and keeps ids stable across runs.
- **The CLI amortises model loading**: `chat` builds the agent once and reuses it for every question, avoiding the 3–5-second embedding-model startup that dominates one-shot invocations used during development.
- **The vector store uses cosine distance** paired with L2-normalised embeddings, so distances are directly comparable across queries and lie in a predictable `[0, 2]` range.

## Project structure

```
cyberai-study-agent/
├── src/
│   └── cyberai_agent/
│       ├── __init__.py
│       ├── ingest.py          # PDF loading and chunking
│       ├── embeddings.py      # sentence-transformers wrapper
│       ├── vectorstore.py     # ChromaDB wrapper
│       ├── agent.py           # RAG loop + Gemini client
│       └── cli.py             # Typer CLI entry point
├── data/
│   ├── pdfs/                  # study PDFs (gitignored)
│   └── chroma/                # persistent vector DB (gitignored)
├── tests/
├── pyproject.toml
├── requirements.txt
├── .env                       # local secrets, gitignored
├── .gitignore
├── LICENSE
└── README.md
```

## Roadmap

Planned iterations, in rough priority order:

- [x] Conversational memory (multi-turn context within a session, with automatic query rewriting)
- [ ] Quiz and flashcard generation from ingested material
- [ ] Spaced-repetition tracking (SM-2 or FSRS)
- [ ] Per-course collections and metadata filtering at query time
- [ ] Better chunking (token-based; semantic splitters for structured slides)
- [ ] Optional local LLM backend via Ollama (fully offline mode)
- [ ] Minimal web UI (Streamlit or FastAPI + HTMX)
- [ ] Unit tests and CI via GitHub Actions

## License

Released under the [MIT License](LICENSE).

## Author

**[Davide Deplano](https://github.com/DavideDeplano)** — Master's student in Cybersecurity and Artificial Intelligence at the University of Cagliari.