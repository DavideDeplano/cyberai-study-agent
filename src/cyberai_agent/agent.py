"""Retrieval-Augmented Generation agent powered by Google Gemini.

This module is the final stage of the RAG pipeline. It takes a natural-language
question, optionally rewrites it against the conversation history so that
elliptical follow-ups become self-contained, retrieves the most relevant
chunks from the vector store, packages them into a prompt with strict
citation instructions, and sends everything to a Gemini model for
grounded answer generation.

The system prompt is deliberately restrictive: the model is told to answer
only from the provided excerpts and to cite every claim, which sharply
reduces hallucinations compared to unconstrained generation.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from cyberai_agent.vectorstore import VectorStore, RetrievedChunk
from cyberai_agent.conversation import Conversation, Turn


# Gemini 2.5 Flash: fast, cheap, and comfortably within the free tier
# quota (10 RPM / 250 RPD as of 2026). Sufficient for personal study use.
DEFAULT_MODEL = "gemini-2.5-flash"

# System instruction that governs the agent's behaviour for every answer
# generation. The four directives, in order:
#   1. Ground answers strictly in the retrieved context (prevents
#      hallucination from the model's pretrained knowledge).
#   2. Force inline citations in a stable machine-readable format so the
#      student can verify every claim against the source PDF.
#   3. Grant explicit permission to refuse: without this, models tend to
#      fabricate rather than admit missing information.
#   4. Match the user's language, since study material is in Italian but
#      queries or supplementary sources may be in English.
ANSWER_SYSTEM_INSTRUCTION = """You are a study assistant for a Master's \
student in Cybersecurity and AI. Answer strictly based on the provided \
context excerpts from the student's study materials. Every factual claim \
MUST cite its source using the format [source, p.PAGE]. If the context \
does not contain enough information to answer, say so explicitly instead \
of guessing. Answer in the same language as the user's question."""

# System instruction for the auxiliary query-rewriting call. Its only
# job is to turn a possibly elliptical follow-up ("and its types?",
# "tell me more") into a self-contained question that retrieval can
# use as a semantic search key. The model must return ONLY the
# rewritten question, no explanation.
REWRITE_SYSTEM_INSTRUCTION = """You reformulate the user's latest \
question into a standalone question that can be understood without the \
conversation history. Resolve pronouns and elliptical references using \
the history. Keep the same language as the user. If the question is \
already self-contained, return it unchanged. Return ONLY the rewritten \
question, with no preamble, no quotes, no explanation."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a context block for the prompt.

    Each chunk is preceded by an explicit `Source: <filename>, page N`
    header — the same string the system prompt asks the model to use in
    its citations. Chunks are separated by a horizontal-rule line so the
    model treats them as distinct sources rather than one continuous
    passage. No numeric labels are added: earlier experiments showed the
    model would sometimes cite the label (e.g. `[1, p.10]`) instead of
    the source filename, so keeping only the filename in the header
    keeps citations grounded in the actual document.
    """
    parts = []
    for c in chunks:
        parts.append(
            f"Source: {c.source}, page {c.page}\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_history_block(conversation: Conversation) -> str:
    """Render the conversation transcript for inclusion in the answer prompt.

    Only past turns are rendered — the current user question is passed
    separately by the caller. The format mirrors the plain-text dump
    from `Conversation.as_plain_text` so that the model sees a familiar
    dialogue structure.
    """
    return conversation.as_plain_text()


class StudyAgent:
    """Retrieval-augmented study agent backed by Gemini and a local vector store.

    A single agent instance owns a Gemini client, a `VectorStore` and,
    optionally, a `Conversation` that tracks the multi-turn transcript.
    All three are relatively expensive to build (the vector store loads
    the embedding model on first use), so the agent is designed to be
    constructed once and reused across many `ask` calls — for example
    inside a REPL loop.
    """

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        conversation: Conversation | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialise the Gemini client, vector store and conversation.

        The Gemini API key is read from the `GEMINI_API_KEY` environment
        variable, loaded transparently from a local `.env` file if
        present.

        Args:
            vectorstore: An existing vector store. If None, a default
                one is constructed; note that its embedder is lazy, so
                no model is loaded until the first search.
            conversation: A conversation object holding the transcript.
                If None, a fresh empty one is created. Passing an
                existing conversation is how the CLI keeps history
                across turns in a session.
            model: Gemini model id to use for both answering and
                rewriting queries.

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
        self.conversation = conversation or Conversation()

    def _rewrite_query(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone one.

        Called only when the conversation already has previous turns.
        The rewrite runs on the same Gemini model as the answer, since
        the task is small and Flash handles it reliably. On any failure
        the original question is returned unchanged so the pipeline
        degrades gracefully rather than crashing mid-turn.
        """
        history = self.conversation.as_plain_text()
        prompt = (
            f"Conversation so far:\n\n{history}\n\n"
            f"---\n\nLatest question: {question}"
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=REWRITE_SYSTEM_INSTRUCTION,
                ),
                contents=prompt,
            )
            rewritten = (response.text or "").strip()
            return rewritten or question
        except Exception:
            # Never let a rewriting failure break the actual answer:
            # fall back to the original question and let retrieval do
            # its best with it.
            return question

    def ask(self, question: str, top_k: int = 5, course: str | None = None) -> str:
        """Answer a question using retrieval-augmented generation.

        Pipeline:
            1. If the conversation has prior turns, rewrite the question
               into a standalone form (extra LLM call) so retrieval is
               not confused by pronouns or elliptical phrasing.
            2. Embed the (rewritten) question and retrieve the `top_k`
               most similar chunks from the vector store.
            3. Format those chunks into a context block, prepend the
               conversation history, and send everything to Gemini
               guarded by the answer-time system instruction.
            4. Record both the original user question and the produced
               answer in the conversation transcript.

        Args:
            question: Natural-language question from the user, as
                originally typed.
            top_k: Number of chunks to include in the prompt. Higher
                values improve recall but enlarge the prompt and can
                dilute focus.
            course: Restrict retrieval to one course label. `None`
                searches the whole store.

        Returns:
            The model's answer as plain text, with inline citations in
            the format specified by the system instruction. If the
            vector store is empty, returns a fixed message asking the
            user to ingest PDFs.
        """
        # Step 1: query rewriting only when there is history to resolve
        # references against; on the first turn the original question
        # is already self-contained by definition.
        if self.conversation.is_empty():
            search_query = question
        else:
            search_query = self._rewrite_query(question)

        # Step 2: retrieve using the standalone form of the question.
        chunks = self.vectorstore.search(search_query, top_k=top_k, course=course)

        if not chunks:
            answer = (
                f"No study materials found for course '{course}'."
                if course
                else "No study materials found in the database. "
                    "Ingest some PDFs first."
            )
            # Still record the exchange so the transcript is faithful
            # even for degenerate turns.
            self.conversation.add_user(question)
            self.conversation.add_assistant(answer)
            return answer

        # Step 3: assemble the answer prompt. History is included
        # before the retrieved context so the model reads the dialogue
        # first and then treats the excerpts as reference material for
        # the current turn.
        context = build_context(chunks)
        history_block = build_history_block(self.conversation)

        if history_block:
            prompt = (
                f"Conversation so far:\n\n{history_block}\n\n"
                f"---\n\nContext excerpts from study materials:\n\n{context}\n\n"
                f"---\n\nCurrent question: {question}"
            )
        else:
            prompt = (
                f"Context excerpts from study materials:\n\n{context}\n\n"
                f"---\n\nQuestion: {question}"
            )

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_SYSTEM_INSTRUCTION,
            ),
            contents=prompt,
        )
        answer = response.text or ""

        # Step 4: append this turn to the transcript so the next call
        # can resolve references against it.
        self.conversation.add_user(question)
        self.conversation.add_assistant(answer)

        return answer


if __name__ == "__main__":
    # Manual smoke test:
    #   python -m cyberai_agent.agent "your question in any language"
    # Note: from the command line this only exercises a single-turn
    # exchange, since each invocation builds a fresh Conversation.
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m cyberai_agent.agent "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = StudyAgent()
    answer = agent.ask(question)
    print(answer)