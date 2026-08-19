"""Local text embeddings using sentence-transformers.

This module wraps a sentence-transformer model to convert text into
dense vector representations suitable for semantic similarity search.
It runs entirely locally on CPU, requires no API keys, and imposes
no per-call cost — making it the persistence-friendly counterpart to
the (rate-limited) LLM used in `agent.py`.

The chosen model, `intfloat/multilingual-e5-base`, is trained with an
asymmetric protocol that requires different prefixes for documents
being indexed (`"passage: "`) and for search queries (`"query: "`).
Omitting these prefixes measurably degrades retrieval quality, so the
two are exposed as separate methods.
"""

from sentence_transformers import SentenceTransformer


# Multilingual e5 model: strong retrieval quality across ~100 languages,
# including Italian and English (matches the mixed content of study
# materials plus academic papers). 768-dimensional output, ~280 MB on disk.
DEFAULT_MODEL = "intfloat/multilingual-e5-base"


class Embedder:
    """Wrapper around a sentence-transformers model for RAG-style retrieval.

    Instantiating this class loads the model into memory (a few seconds and
    a few hundred MB of RAM), so prefer reusing a single instance across
    many calls rather than constructing it per query.

    Attributes:
        model_name: Hugging Face identifier of the loaded model.
        model: The underlying `SentenceTransformer` instance.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Load the sentence-transformer model.

        Args:
            model_name: Hugging Face model id. Must be an e5-family model,
                otherwise the `passage: ` / `query: ` prefixes used below
                will be inappropriate.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing.

        Uses the e5 `"passage: "` prefix that the model was trained to
        associate with indexable content. Embeddings are L2-normalized so
        that cosine similarity reduces to a dot product downstream.

        Args:
            texts: Raw chunk texts to be embedded.

        Returns:
            One embedding per input text, each a list of floats of length
            equal to the model's embedding dimension (768 for e5-base).
        """
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True)
        # `.tolist()` converts numpy arrays to plain Python lists so the
        # output is JSON-serializable and consumable by ChromaDB as-is.
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for retrieval.

        Uses the e5 `"query: "` prefix, which maps the input to the region
        of the embedding space that best matches documents indexed with
        `embed_documents`.

        Args:
            text: The user's question or search string.

        Returns:
            A single embedding vector as a list of floats.
        """
        prefixed = f"query: {text}"
        vector = self.model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()


if __name__ == "__main__":
    # Manual smoke test: verifies that the model loads and produces
    # embeddings of the expected shape.
    embedder = Embedder()
    vec = embedder.embed_query("cos'è un malware?")
    print(f"Model: {embedder.model_name}")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")