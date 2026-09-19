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
                │   agent.py  │  ────▶ │              │  ──▶ cited answer
                │  (RAG loop) │        │  Gemini 2.5  │
                ├─────────────┤  ────▶ │     Flash    │  ──▶ quiz / flashcards
                │   quiz.py   │        │              │      (JSON)
                └─────────────┘        └──────────────┘
                       │
                       ▼
                ┌─────────────┐
                │   cli.py    │  ──▶ `cyberai-agent {ingest,chat,quiz,
                └─────────────┘                      flashcards,review,stats}`
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

Once installed, the package registers a `cyberai-agent` command with six subcommands.

**Ingest study material**

```bash
cyberai-agent ingest data/pdfs/lecture01.pdf
cyberai-agent ingest data/pdfs/                # every PDF in the directory
cyberai-agent ingest file1.pdf file2.pdf       # multiple explicit files
cyberai-agent ingest data/pdfs/ --course malware   # label the material
```

Ingestion is idempotent: re-running it on the same file overwrites the previously stored chunks rather than duplicating them.

**Ask questions interactively**

```bash
cyberai-agent chat
```

Opens a REPL. The embedding model and the Gemini client are initialised once at startup, so every subsequent question only pays for retrieval and generation. Type `/reset` to clear the conversation history; type `exit`, `quit`, or an empty line to leave.

**Generate a quiz**

```bash
cyberai-agent quiz "adversarial examples"
cyberai-agent quiz "malware detection" --n 10        # 10 questions (default 5)
cyberai-agent quiz "TLS handshake" --show-answers    # answers under each question
```

Multiple-choice questions built from the indexed material, each carrying
the source and page it came from. By default the answer key is printed
after the full question list, so the quiz can be attempted before
checking; `--show-answers` collapses each answer under its question,
which is the better layout for reviewing rather than self-testing.

**Generate flashcards**

```bash
cyberai-agent flashcards "malware detection"
cyberai-agent flashcards "adversarial ML" --n 15         # 15 cards (default 10)
cyberai-agent flashcards "network security" --csv deck.csv
```

`--csv` writes the deck as `front,back,source,page` — the column order
Anki imports without hand-editing.

Both commands accept `--top-k` (default 12) to control how many chunks
are retrieved as grounding. It is higher than the 5 used for answering
because a quiz should span a topic rather than rephrase a single passage
n times.

`--n` is a ceiling, not a guarantee: if the retrieved excerpts do not
support the requested number of items, fewer come back. That is
deliberate — padding a deck means inventing material the PDFs do not
cover, which is exactly what a study tool must not do.

**Review the deck**

```bash
cyberai-agent flashcards "malware detection" --save   # add cards to the deck
cyberai-agent review                                  # review what is due
cyberai-agent review --limit 10                       # cap the session
```

`--save` stores the cards in `data/review/deck.json`, skipping fronts
already in the deck. `review` then shows the cards that are due: press
Enter to reveal the answer, rate the recall 1-4 (Again, Hard, Good,
Easy), and FSRS schedules the next occurrence. The deck is written after
every card, so quitting with `q` never loses progress.

`--course` attaches a label to every chunk of those PDFs, which the
other commands can then filter on. Material ingested without it stays
searchable, but only when no filter is applied.

**Work on a single course**

```bash
cyberai-agent chat --course malware
cyberai-agent quiz "static analysis" --course malware
cyberai-agent flashcards "sandboxing" --course malware
```

`--course` restricts retrieval to the chunks carrying that label, so a
question is answered from one subject rather than from the whole
library. Filtering happens in the vector store via metadata, not by
post-filtering the results, so `--top-k` still returns that many chunks
from within the course.

**Inspect the index**

```bash
cyberai-agent stats
```

Prints the total number of indexed chunks and the breakdown per course,
with material ingested without a label grouped separately.

## Design notes

- **Grounding is enforced by the system prompt**, which instructs the model to answer strictly from the retrieved excerpts and to cite every claim as `[source, page N]`. Combined with an explicit permission to refuse, this sharply reduces hallucination compared to unconstrained generation.
- **Chunking is page-aware**: text is split page by page rather than across the whole document, so every chunk retains an accurate page number that can be surfaced in citations.
- **The e5 embedding model uses asymmetric prefixes** (`passage: ` for documents, `query: ` for queries). Both are handled inside the `Embedder` wrapper so callers cannot forget them and degrade retrieval quality by accident.
- **Chunk identity is deterministic**: the store key is `f"{source}::{chunk_id}"`, which makes upsert-based re-ingestion trivial and keeps ids stable across runs.
- **The CLI amortises model loading**: `chat` builds the agent once and reuses it for every question, avoiding the 3–5-second embedding-model startup that dominates one-shot invocations used during development.
- **Generated study items refuse to pad**: the quiz and flashcard prompts are told to return fewer items rather than invent material when the retrieved excerpts run thin. A hallucinated answer in a chat is a nuisance; a hallucinated quiz answer teaches the student something false and gets rehearsed.
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
│       ├── conversation.py    # multi-turn chat history
│       ├── quiz.py            # quiz and flashcard generation
│       ├── review.py          # FSRS spaced-repetition deck
│       └── cli.py             # Typer CLI entry point
├── data/
│   ├── pdfs/                  # study PDFs (gitignored)
│   └── chroma/                # persistent vector DB (gitignored)
│   └── review/                # saved flashcard deck (gitignored)
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
- [x] Quiz and flashcard generation from ingested material (CSV export for Anki)
- [x] Spaced-repetition tracking (FSRS, via the `fsrs` package)
- [x] Per-course collections and metadata filtering at query time
- [ ] Better chunking (token-based; semantic splitters for structured slides)
- [ ] Optional local LLM backend via Ollama (fully offline mode)
- [ ] Minimal web UI (Streamlit or FastAPI + HTMX)
- [ ] Unit tests and CI via GitHub Actions

## License

Released under the [MIT License](LICENSE).

## Author

Davide Deplano