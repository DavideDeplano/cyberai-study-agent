"""PDF ingestion pipeline: extract text from PDFs and split it into chunks.

This module is the first stage of the RAG pipeline. It converts raw PDF files
into a list of `Chunk` objects that carry both text and provenance metadata
(source file, page number, chunk id) so downstream retrieval can produce
citable answers.
"""

from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Chunk:
    """A single text fragment extracted from a document.

    Attributes:
        text: Raw text content of the chunk.
        source: Name of the source PDF file (e.g. "Cap3.1MalwareDetection.pdf").
        page: 1-indexed page number where the chunk originated.
        chunk_id: Progressive index of the chunk within the source document.
            Combined with `source` it forms a unique identifier in the vector
            store: `f"{source}::{chunk_id}"`.
    """
    text: str
    source: str
    page: int
    chunk_id: int


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract non-empty text from every page of a PDF.

    Uses pypdf's built-in text extractor. Pages that yield no text (e.g.
    image-only pages or blank separators) are silently skipped so they do
    not produce empty chunks downstream.

    Args:
        pdf_path: Filesystem path to a text-based PDF file. Scanned PDFs
            without an OCR layer will return an empty list.

    Returns:
        List of (page_number, page_text) tuples, with page numbers 1-indexed.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        # `extract_text()` may return None for image-only pages; coerce to "".
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a text into word-based chunks with a sliding overlap.

    Chunking keeps each fragment small enough to be embedded meaningfully
    by the sentence-transformer model (which has a limited context window).
    The overlap ensures that concepts spanning two chunks are preserved
    in at least one of them, reducing recall loss at chunk boundaries.

    Args:
        text: Text to split.
        chunk_size: Number of words per chunk. Default 500 ≈ ~650 tokens,
            comfortably under the 512-token limit of e5-base for most inputs.
        overlap: Number of overlapping words between consecutive chunks.
            Default 50 gives a 10% overlap.

    Returns:
        List of chunk strings in original order. Empty input yields an empty list.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap  # how many words to advance between chunks
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        # Stop once the current chunk has consumed the tail of the document,
        # to avoid emitting a nearly-duplicate final chunk.
        if start + chunk_size >= len(words):
            break
    return chunks


def ingest_pdf(
    pdf_path: Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Convert a PDF into a list of `Chunk` objects ready for embedding.

    Text is chunked page by page (not across the whole document) so that
    each chunk retains an accurate `page` field for citation purposes.
    The `chunk_id` field is a document-wide progressive counter.

    Args:
        pdf_path: Path to the PDF file to ingest.
        chunk_size: See `chunk_text`.
        overlap: See `chunk_text`.

    Returns:
        Flat list of Chunk objects in reading order.
    """
    source = pdf_path.name
    pages = extract_pages(pdf_path)

    all_chunks: list[Chunk] = []
    chunk_id = 0
    for page_num, page_text in pages:
        for piece in chunk_text(page_text, chunk_size, overlap):
            all_chunks.append(Chunk(
                text=piece,
                source=source,
                page=page_num,
                chunk_id=chunk_id,
            ))
            chunk_id += 1
    return all_chunks


if __name__ == "__main__":
    # Manual smoke test:
    #   python -m cyberai_agent.ingest data/pdfs/some_file.pdf
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m cyberai_agent.ingest <pdf_path>")
        sys.exit(1)

    chunks = ingest_pdf(Path(sys.argv[1]))
    print(f"Extracted {len(chunks)} chunks from {sys.argv[1]}")
    if chunks:
        print(f"\nFirst chunk (page {chunks[0].page}):")
        print(chunks[0].text[:200] + "...")