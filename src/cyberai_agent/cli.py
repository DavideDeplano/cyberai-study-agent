"""Command-line interface for the CyberAI Study Agent.

This module wires the ingestion, storage and generation layers into a
small, user-facing CLI built with Typer. It is the single entry point
intended for day-to-day use, replacing the ad-hoc `python -m ...`
invocations that individual modules expose only for development.

Three subcommands are provided:

* `ingest`  — Parse one or more PDFs (or directories of PDFs), chunk
              their text, embed the chunks and upsert them into the
              persistent vector database.
* `chat`    — Start an interactive REPL. The embedding model and the
              Gemini client are loaded once at startup and reused for
              every question, avoiding the multi-second cold start that
              affects the one-shot scripts used during development.
* `stats`   — Print basic statistics about the current vector store,
              useful as a quick sanity check after ingestion.

Typer was chosen over `argparse` because it derives the CLI schema from
type hints, which keeps the command definitions close in style to the
rest of the codebase and produces a decent `--help` output for free.
"""

from pathlib import Path
import typer

from cyberai_agent.ingest import ingest_pdf
from cyberai_agent.vectorstore import VectorStore
from cyberai_agent.agent import StudyAgent


# Top-level Typer application. `add_completion=False` suppresses the
# shell-completion install prompts, which are noise for a personal tool.
app = typer.Typer(
    add_completion=False,
    help="AI study companion for LM Cybersecurity & AI coursework.",
)


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="One or more PDF files or directories containing PDFs.",
    ),
    chunk_size: int = typer.Option(
        500, help="Words per chunk during ingestion."
    ),
    overlap: int = typer.Option(
        50, help="Overlapping words between adjacent chunks."
    ),
):
    """Ingest one or more PDFs into the persistent vector store.

    Any argument that resolves to a directory is scanned non-recursively
    for `*.pdf` files; regular files must have a `.pdf` extension. Files
    with other extensions are skipped with a warning rather than aborting
    the whole run.

    Ingestion is idempotent: the underlying store uses upsert, so running
    this command twice on the same PDF simply overwrites the previously
    stored chunks instead of producing duplicates or raising an error.

    Args:
        paths: PDF files or directories to ingest, validated by Typer to
            exist and be readable before the function body runs.
        chunk_size: Forwarded to `ingest_pdf` (words per chunk).
        overlap: Forwarded to `ingest_pdf` (overlap in words).
    """
    # Flatten the mixed list of files and directories into a plain list
    # of PDF paths, warning on anything that clearly does not belong.
    pdf_files: list[Path] = []
    for p in paths:
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdf_files.append(p)
        else:
            typer.echo(f"Skipping non-PDF file: {p}", err=True)

    if not pdf_files:
        typer.echo("No PDF files found.", err=True)
        raise typer.Exit(code=1)

    # Construct the store once so the embedding model is loaded a single
    # time even when many PDFs are ingested in a single invocation.
    store = VectorStore()
    total_added = 0
    for pdf in pdf_files:
        typer.echo(f"Ingesting {pdf.name}...")
        chunks = ingest_pdf(pdf, chunk_size=chunk_size, overlap=overlap)
        added = store.add_chunks(chunks)
        total_added += added
        typer.echo(f"  {added} chunks added.")

    typer.echo(
        f"\nDone. {total_added} chunks added across {len(pdf_files)} file(s). "
        f"Total in DB: {store.count()}."
    )


@app.command()
def chat(
    top_k: int = typer.Option(
        5, help="Number of chunks retrieved per question."
    ),
):
    """Start an interactive question-answering session with the agent.

    A single `StudyAgent` is built at startup, which loads the embedding
    model and initialises the Gemini client. Every subsequent question
    then only pays for retrieval (fast, local) and LLM generation
    (network-bound), not for model loading.

    The loop terminates on an empty input, on the words `exit` or `quit`,
    or on Ctrl-C / Ctrl-D — all handled gracefully so the session never
    ends with a raw Python traceback.

    Args:
        top_k: Number of chunks to retrieve and pass to the model for
            each question. Higher values improve recall on broad
            questions but enlarge the prompt and can dilute focus.
    """
    typer.echo("Loading study agent (this can take a few seconds)...")
    agent = StudyAgent()

    # Refuse to enter the REPL against an empty store: the agent would
    # only ever produce its "no materials" fallback, which is confusing
    # in an interactive session and better surfaced up-front.
    if agent.vectorstore.count() == 0:
        typer.echo(
            "The vector store is empty. Run `cyberai-agent ingest <pdf>` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"Ready. {agent.vectorstore.count()} chunks indexed. "
        "Type your question (empty line or 'exit' to quit).\n"
    )

    while True:
        try:
            question = typer.prompt("You", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D on Unix, Ctrl-Z+Enter on Windows, or Ctrl-C at the
            # prompt: exit cleanly instead of surfacing the exception.
            typer.echo("\nBye.")
            break

        if not question or question.lower() in {"exit", "quit"}:
            typer.echo("Bye.")
            break

        answer = agent.ask(question, top_k=top_k)
        typer.echo(f"\nAgent:\n{answer}\n")


@app.command()
def stats():
    """Print basic statistics about the current vector store.

    Currently reports only the total number of stored chunks. Intended
    as a quick post-ingestion check; richer breakdowns (per source, per
    page range) can be added here without touching other modules.
    """
    store = VectorStore()
    typer.echo(f"Total chunks in DB: {store.count()}")


if __name__ == "__main__":
    # Allow running the CLI directly as `python -m cyberai_agent.cli ...`
    # in addition to the installed `cyberai-agent` script defined in
    # pyproject.toml.
    app()