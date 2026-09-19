"""PDF ingestion pipeline: extract text from PDFs and split it into chunks.

This module is the first stage of the RAG pipeline. It converts raw PDF files
into a list of `Chunk` objects that carry both text and provenance metadata
(source file, page number, chunk id, course) so downstream retrieval can
produce citable answers.
"""

from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader

from cyberai_agent.embeddings import DEFAULT_MODEL

# Usable token budget per chunk. e5-base truncates at 512 tokens; the
# margin covers the two special tokens and the `passage: ` prefix added
# at embedding time.
MAX_TOKENS = 480

# Tokenizer cache: loading it is cheap but not free, and ingestion calls
# `chunk_text` once per page.
_TOKENIZER = None


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
        course: Short label of the course the document belongs to, used to
            filter retrieval to a single subject. Empty when unspecified,
            which is also the value carried by material ingested before
            courses existed.
    """
    text: str
    source: str
    page: int
    chunk_id: int
    course: str = ""


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


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """Return the tokenizer of the embedding model, loading it once.

    Only the tokenizer is loaded, not the model weights, so ingestion
    stays cheap: it is a few hundred kilobytes against the ~450 MB of
    the full sentence-transformer.

    Args:
        model_name: Hugging Face id of the embedding model whose
            tokenizer should be used. Must match the model used at
            embedding time, or the token counts will not correspond.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def chunk_text(
    text: str,
    chunk_size: int = MAX_TOKENS,
    overlap: int = 64,
    tokenizer=None,
) -> list[str]:
    """Split a text into token-based chunks with a sliding overlap.

    Chunks are measured in the embedding model's own tokens rather than
    in words. The previous word-based split used 500 words, which is
    roughly 650 tokens for this tokenizer: everything past the 512-token
    limit of e5-base was silently truncated at embedding time and never
    made it into the vector, so the tail of every long chunk was
    effectively unsearchable.

    The overlap ensures that concepts spanning two chunks are preserved
    in at least one of them, reducing recall loss at chunk boundaries.

    Args:
        text: Text to split.
        chunk_size: Number of tokens per chunk. Defaults to the model's
            usable budget, leaving room for the special tokens and the
            `passage: ` prefix that `Embedder` prepends.
        overlap: Number of overlapping tokens between consecutive
            chunks. Default 64 gives roughly a 14% overlap.
        tokenizer: Tokenizer to use. Defaults to the embedding model's
            own; injectable so tests need not download anything.

    Returns:
        List of chunk strings in original order. Empty input yields an
        empty list.
    """
    # Normalise the whitespace PDF extraction leaves behind before
    # counting tokens, so the count reflects real content.
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokenizer = tokenizer or get_tokenizer()
    # `add_special_tokens=False` keeps [CLS]/[SEP] out of the count:
    # they are added later, at embedding time, and are accounted for
    # by the margin baked into MAX_TOKENS.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(ids), step):
        window = ids[start:start + chunk_size]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
        # Stop once the current chunk has consumed the tail of the
        # document, to avoid emitting a nearly-duplicate final chunk.
        if start + chunk_size >= len(ids):
            break
    return chunks


def ingest_pdf(
    pdf_path: Path,
    chunk_size: int = MAX_TOKENS,
    overlap: int = 64,
    course: str = "",
) -> list[Chunk]:
    """Convert a PDF into a list of `Chunk` objects ready for embedding.

    Text is chunked page by page (not across the whole document) so that
    each chunk retains an accurate `page` field for citation purposes.
    The `chunk_id` field is a document-wide progressive counter.

    Args:
        pdf_path: Path to the PDF file to ingest.
        chunk_size: See `chunk_text`.
        overlap: See `chunk_text`.
        course: Course label attached to every chunk of this document.

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
                course=course,
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