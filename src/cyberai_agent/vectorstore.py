"""Persistent vector store built on ChromaDB.

This module is the middle stage of the RAG pipeline. It embeds `Chunk`
objects produced by `ingest.py` using the `Embedder` from `embeddings.py`,
persists them to disk, and exposes a similarity search API that returns
the most relevant chunks for a given query — together with the provenance
metadata needed by `agent.py` to generate cited answers.

ChromaDB is used in its `PersistentClient` mode, which stores collections
in a local directory (`data/chroma/` by default) as SQLite plus flat files.
No external database process is required.
"""

from pathlib import Path
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

from cyberai_agent.ingest import Chunk
from cyberai_agent.embeddings import Embedder


# Default on-disk location for the Chroma database. Kept under `data/`
# so it is easy to exclude from version control via .gitignore.
DEFAULT_DB_PATH = Path("data/chroma")

# All chunks live in a single named collection. Splitting by course or
# document is deferred to metadata filtering rather than multiple collections.
DEFAULT_COLLECTION = "study_materials"


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search.

    Extends the information carried by an ingested `Chunk` with the
    distance score computed by the vector store, useful for debugging
    retrieval quality and for downstream ranking heuristics.

    Attributes:
        text: The chunk's original text.
        source: Name of the source PDF file.
        page: 1-indexed page number in the source document.
        chunk_id: Progressive chunk index within the source document.
        course: Course label carried by the chunk, empty when the document
        was ingested without one.
        distance: Cosine distance to the query embedding, in [0, 2].
            Lower means more similar; values around 0.1–0.3 typically
            indicate strong topical relevance for e5-base embeddings.
    """
    text: str
    source: str
    page: int
    chunk_id: int
    course: str
    distance: float


class VectorStore:
    """Persistent ChromaDB store for embedded study material chunks.

    The store is created lazily on first use: the target directory is
    created if missing, and the collection is retrieved or created with
    the cosine distance metric configured.

    A single `Embedder` instance is held for the lifetime of the store,
    so callers should reuse one `VectorStore` across many operations
    to avoid reloading the embedding model on every call.
    """

    def __init__(
    self,
    db_path: Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedder: Embedder | None = None,
    ):
        """Open (or create) a persistent Chroma collection.

        The embedder is loaded lazily: it is not instantiated in the
        constructor, only on first access via the `embedder` property.
        This lets read-only callers (for example the `stats` CLI
        command) open the store without paying the multi-second cost
        of loading the sentence-transformer model.

        Args:
            db_path: Directory where Chroma will store its files.
            collection_name: Logical name of the collection inside the DB.
            embedder: Preconstructed embedder to use. If None, one is
                built the first time it is needed.
        """
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Cached embedder; None means "not yet loaded". Access via the
        # `embedder` property so loading is transparent to callers.
        self._embedder: Embedder | None = embedder

    @property
    def embedder(self) -> Embedder:
        """Return the embedder, loading it on first access."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder
    

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed a batch of chunks and upsert them into the collection.

        Uses `upsert` (not `add`) so that re-ingesting the same PDF simply
        overwrites the existing entries instead of raising on duplicate ids.
        This makes ingestion idempotent: running it twice is a no-op in
        terms of stored content.

        The chunk id used in the store is `f"{source}::{chunk_id}"`,
        which is unique across documents as long as filenames are unique.

        Args:
            chunks: Chunks to embed and store.

        Returns:
            Number of chunks written (equal to `len(chunks)` on success,
            or 0 if the input list was empty).
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        ids = [f"{c.source}::{c.chunk_id}" for c in chunks]
        metadatas = [
            {
                "source": c.source, 
                "page": c.page, 
                "chunk_id": c.chunk_id, 
                "course": c.course
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        course: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` chunks most similar to the query.

        The query is embedded with the query-side prefix expected by the
        e5 model (handled inside `Embedder.embed_query`), then compared
        to all stored embeddings via the collection's cosine metric.

        Args:
            query: Natural-language search string.
            top_k: How many chunks to return. Larger values give the LLM
                more context at the cost of prompt size and noise.
            course: Restrict the search to one course label. `None`
                searches everything, which is also what documents
                ingested without a course fall back to.

        Returns:
            List of `RetrievedChunk` sorted by ascending distance
            (most relevant first). Empty if the collection is empty or
            no chunk matches the filter.
        """
        query_embedding = self.embedder.embed_query(query)

        # ChromaDB's `query` accepts batched inputs; we always pass a
        # single query, so index [0] unwraps the singleton result set.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"course": course} if course else None,
        )

        retrieved: list[RetrievedChunk] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for text, meta, dist in zip(docs, metas, dists):
            retrieved.append(RetrievedChunk(
                text=text,
                source=meta["source"],
                page=meta["page"],
                chunk_id=meta["chunk_id"],
                # Material ingested before courses existed has no such
                # key: default it rather than crashing on old data.
                course=meta.get("course", ""),
                distance=dist,
            ))
        return retrieved

    def count(self, course: str | None = None) -> int:
        """Return the number of chunks stored, optionally for one course.

        Args:
            course: Course label to count. `None` counts everything.
        """
        if course is None:
            return self.collection.count()
        return len(self.collection.get(where={"course": course})["ids"])

    def courses(self) -> dict[str, int]:
        """Return the number of stored chunks per course label.

        Chunks ingested without a course are grouped under the empty
        string. Used by the `stats` command to show what is indexed.
        """
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for meta in metas:
            label = meta.get("course", "") or ""
            counts[label] = counts.get(label, 0) + 1
        return counts

if __name__ == "__main__":
    # Manual end-to-end smoke test:
    #   python -m cyberai_agent.vectorstore data/pdfs/some.pdf "query text"
    import sys
    from cyberai_agent.ingest import ingest_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m cyberai_agent.vectorstore <pdf_path> [query]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    query = sys.argv[2] if len(sys.argv) > 2 else "malware detection"

    store = VectorStore()
    chunks = ingest_pdf(pdf_path)
    added = store.add_chunks(chunks)
    print(f"Added {added} chunks. Total in DB: {store.count()}")

    print(f"\nSearching: '{query}'")
    results = store.search(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.source} p.{r.page} (dist={r.distance:.4f})")
        print(f"    {r.text[:200]}...")