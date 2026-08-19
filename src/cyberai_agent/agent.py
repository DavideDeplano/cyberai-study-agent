"""Retrieval-Augmented Generation agent powered by Google Gemini.

This module is the final stage of the RAG pipeline. It takes a natural-language
question, retrieves the most relevant chunks from the vector store, packages
them into a prompt with strict citation instructions, and sends everything to
a Gemini model for grounded answer generation.

The system prompt is deliberately restrictive: the model is told to answer
only from the provided excerpts and to cite every claim, which sharply
reduces hallucinations compared to unconstrained generation.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from cyberai_agent.vectorstore import VectorStore, RetrievedChunk


# Gemini 2.5 Flash: fast, cheap, and comfortably within the free tier
# quota (10 RPM / 250 RPD as of 2026). Sufficient for personal study use.
DEFAULT_MODEL = "gemini-2.5-flash"

# System instruction that governs the agent's behaviour for every request.
# The four directives, in order:
#   1. Ground answers strictly in the retrieved context (prevents
#      hallucination from the model's pretrained knowledge).
#   2. Force inline citations in a stable machine-readable format so the
#      student can verify every claim against the source PDF.
#   3. Grant explicit permission to refuse: without this, models tend to
#      fabricate rather than admit missing information.
#   4. Match the user's language, since study material is in Italian but
#      queries or supplementary sources may be in English.
SYSTEM_INSTRUCTION = """You are a study assistant for a Master's student in \
Cybersecurity and AI. Answer strictly based on the provided context excerpts \
from the student's study materials. Every factual claim MUST cite its source \
using the format [source, p.PAGE]. If the context does not contain enough \
information to answer, say so explicitly instead of guessing. Answer in the \
same language as the user's question."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a numbered context block for the prompt.

    Each chunk is prefixed with a bracketed index and a `Source: ..., page N`
    header so the model has an unambiguous handle to cite. Chunks are
    separated by a horizontal-rule line to make boundaries visually salient
    in the prompt (helps the model treat them as distinct sources).

    Args:
        chunks: Retrieved chunks in relevance order.

    Returns:
        A single string ready to be embedded into the user prompt.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c.source}, page {c.page}\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


class StudyAgent:
    """Retrieval-augmented study agent backed by Gemini and a local vector store.

    A single agent instance owns a Gemini client and a `VectorStore`. Both
    are relatively expensive to build (the vector store loads the embedding
    model on init), so the agent is designed to be constructed once and
    reused across many `ask` calls — for example inside a REPL loop.
    """

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialise the Gemini client and attach a vector store.

        The Gemini API key is read from the `GEMINI_API_KEY` environment
        variable, loaded transparently from a local `.env` file if present.

        Args:
            vectorstore: An existing vector store. If None, a default one
                is constructed, which triggers loading of the embedding
                model (a few seconds).
            model: Gemini model id to use for generation.

        Raises:
            RuntimeError: If `GEMINI_API_KEY` is not set.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.vectorstore = vectorstore or VectorStore()

    def ask(self, question: str, top_k: int = 5) -> str:
        """Answer a question using retrieval-augmented generation.

        Pipeline:
            1. Embed the question and retrieve the `top_k` most similar
               chunks from the vector store.
            2. Format those chunks into a context block.
            3. Send the context plus the question to Gemini, guarded by
               the module-level system instruction.

        Args:
            question: Natural-language question from the user.
            top_k: Number of chunks to include in the prompt. Higher values
                improve recall but enlarge the prompt and can dilute focus.

        Returns:
            The model's answer as plain text, with inline citations in the
            format specified by the system instruction. If the vector store
            is empty, returns a fixed message asking the user to ingest PDFs.
        """
        chunks = self.vectorstore.search(question, top_k=top_k)

        if not chunks:
            return "No study materials found in the database. Ingest some PDFs first."

        context = build_context(chunks)
        prompt = (
            f"Context excerpts from study materials:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        )

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
            contents=prompt,
        )
        return response.text


if __name__ == "__main__":
    # Manual smoke test:
    #   python -m cyberai_agent.agent "your question in any language"
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m cyberai_agent.agent "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = StudyAgent()
    answer = agent.ask(question)
    print(answer)