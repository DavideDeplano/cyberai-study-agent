"""Quiz and flashcard generation from ingested study material.

This module reuses the retrieval half of the RAG pipeline but inverts
the generation task: instead of answering a question the student asks,
it asks the questions. Given a topic, it retrieves the most relevant
chunks from the vector store and prompts Gemini to turn them into
multiple-choice questions or flashcards.

Grounding matters even more here than in `agent.py`. A hallucinated
answer is annoying; a hallucinated quiz answer teaches the student
something false. The system instructions therefore forbid drawing on
pretrained knowledge and require every item to carry the source and
page it was built from, so any suspicious item can be checked against
the original PDF in seconds.

Structured output is requested via `response_mime_type="application/json"`
rather than parsed out of prose, because the items feed a rendering loop
that needs to know which option is correct.
"""

import json
from dataclasses import dataclass

from google.genai import types

from cyberai_agent.agent import StudyAgent, build_context
from cyberai_agent.vectorstore import RetrievedChunk


# Retrieval width for generation. Higher than the answering default of
# 5 because a quiz should span a topic rather than answer one narrow
# question: more chunks means the model can vary what it asks about
# instead of rephrasing the same passage n times.
DEFAULT_TOP_K = 12

QUIZ_SYSTEM_INSTRUCTION = """You write multiple-choice exam questions \
for a Master's student in Cybersecurity and AI, based strictly on the \
provided excerpts from the student's own study materials.

Rules:
- Build every question ONLY from the excerpts. Never use outside knowledge.
- Exactly one option is correct. The other three must be plausible and \
related to the topic, not obviously absurd.
- Test understanding, not verbatim recall of a sentence.
- The explanation must justify the correct answer using the excerpt.
- Each item must carry the source filename and page it came from.
- Write in the same language as the excerpts.
- If the excerpts do not support the requested number of questions, \
return fewer. Never invent material to reach a count.

Return ONLY a JSON array, with each element shaped as:
{"question": str, "options": [str, str, str, str], "correct_index": int, \
"explanation": str, "source": str, "page": int}"""

FLASHCARD_SYSTEM_INSTRUCTION = """You write study flashcards for a \
Master's student in Cybersecurity and AI, based strictly on the provided \
excerpts from the student's own study materials.

Rules:
- Build every card ONLY from the excerpts. Never use outside knowledge.
- The front is a short prompt: a term, a question, or "what happens when X".
- The back is a compact answer, two or three sentences at most.
- One idea per card. Split anything that would need "and also".
- Each card must carry the source filename and page it came from.
- Write in the same language as the excerpts.
- If the excerpts do not support the requested number of cards, return \
fewer. Never invent material to reach a count.

Return ONLY a JSON array, with each element shaped as:
{"front": str, "back": str, "source": str, "page": int}"""


@dataclass
class QuizQuestion:
    """A single multiple-choice question generated from the material.

    Attributes:
        question: The question stem.
        options: Answer choices, one of which is correct.
        correct_index: Index into `options` of the correct choice.
        explanation: Why that choice is correct, grounded in the excerpt.
        source: Filename of the PDF the item was built from.
        page: 1-indexed page in that PDF.
    """
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source: str
    page: int

    @property
    def correct_option(self) -> str:
        """Return the text of the correct answer."""
        return self.options[self.correct_index]

    def citation(self) -> str:
        """Return the provenance string in the same format the agent uses."""
        return f"[{self.source}, p.{self.page}]"


@dataclass
class Flashcard:
    """A single front/back study card generated from the material.

    Attributes:
        front: Prompt side — term, question, or scenario.
        back: Answer side, kept short enough to review at speed.
        source: Filename of the PDF the card was built from.
        page: 1-indexed page in that PDF.
    """
    front: str
    back: str
    source: str
    page: int

    def citation(self) -> str:
        """Return the provenance string in the same format the agent uses."""
        return f"[{self.source}, p.{self.page}]"


def parse_json_array(raw: str) -> list[dict]:
    """Parse a JSON array out of a model response.

    `response_mime_type="application/json"` makes fenced output unlikely,
    but not impossible: the fence still shows up occasionally when the
    model decides the answer is prose about JSON. Stripping it costs two
    lines and saves a crash mid-generation.

    Args:
        raw: Raw text of the model response.

    Returns:
        The decoded list, or an empty list if the response was not a
        JSON array.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def valid_page(value) -> int:
    """Coerce a page number coming back from the model into an int.

    The model is asked for an integer and usually complies, but it
    sometimes returns "12" or "p. 12". Anything unparseable becomes 0,
    which reads as "unknown page" in a citation rather than crashing the
    whole batch over one bad item.
    """
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip().lstrip("p. "))
    except ValueError:
        return 0


class QuizGenerator:
    """Generates quizzes and flashcards from the indexed study material.

    Shares the Gemini client and vector store with `StudyAgent` rather
    than building its own, so a CLI session that already paid the
    embedding-model startup cost does not pay it twice.
    """

    def __init__(self, agent: StudyAgent | None = None):
        """Wrap an existing agent, or build one if none is supplied.

        Args:
            agent: A constructed `StudyAgent` whose client and vector
                store will be reused. If None, a fresh one is built,
                which also validates that `GEMINI_API_KEY` is set.
        """
        self.agent = agent or StudyAgent()
        self.client = self.agent.client
        self.model = self.agent.model
        self.vectorstore = self.agent.vectorstore

    def retrieve(
        self, topic: str, top_k: int, course: str | None = None
    ) -> list[RetrievedChunk]:
        """Fetch the chunks a generation call will be grounded in."""
        return self.vectorstore.search(topic, top_k=top_k, course=course)

    def generate(self, system_instruction: str, prompt: str) -> list[dict]:
        """Run one JSON-constrained generation call and decode the result."""
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
            contents=prompt,
        )
        return parse_json_array(response.text or "")

    def generate_quiz(
        self,
        topic: str,
        n_questions: int = 5,
        top_k: int = DEFAULT_TOP_K,
        course: str | None = None,
    ) -> list[QuizQuestion]:
        """Build a multiple-choice quiz on a topic from the indexed material.

        Args:
            topic: What to quiz on, in natural language. Used as the
                retrieval query, so it should read like something that
                appears in the slides.
            n_questions: How many questions to request. The model may
                return fewer if the retrieved material does not support
                that many, which is the intended behaviour.
            top_k: How many chunks to retrieve as grounding.
            course: Restrict retrieval to one course label.

        Returns:
            The generated questions, possibly fewer than requested and
            possibly empty if the store holds nothing on the topic.
        """
        chunks = self.retrieve(topic, top_k, course)
        if not chunks:
            return []

        context = build_context(chunks)
        prompt = (
            f"Topic: {topic}\n\n"
            f"Write {n_questions} multiple-choice questions.\n\n"
            f"---\n\nExcerpts from study materials:\n\n{context}"
        )

        questions: list[QuizQuestion] = []
        for item in self.generate(QUIZ_SYSTEM_INSTRUCTION, prompt):
            options = item.get("options") or []
            index = item.get("correct_index")
            # Drop malformed items instead of surfacing a question whose
            # correct answer points nowhere — a broken quiz item is worse
            # than a shorter quiz.
            if not isinstance(index, int) or not 0 <= index < len(options):
                continue
            questions.append(QuizQuestion(
                question=item.get("question", ""),
                options=options,
                correct_index=index,
                explanation=item.get("explanation", ""),
                source=item.get("source", "unknown"),
                page=valid_page(item.get("page")),
            ))
        return questions

    def generate_flashcards(
        self,
        topic: str,
        n_cards: int = 10,
        top_k: int = DEFAULT_TOP_K,
        course: str | None = None,
    ) -> list[Flashcard]:
        """Build a flashcard deck on a topic from the indexed material.

        Args:
            topic: What to make cards about, used as the retrieval query.
            n_cards: How many cards to request; fewer may come back.
            top_k: How many chunks to retrieve as grounding.
            course: Restrict retrieval to one course label.
        Returns:
            The generated cards, possibly empty if nothing was retrieved.
        """
        chunks = self.retrieve(topic, top_k, course)
        if not chunks:
            return []

        context = build_context(chunks)
        prompt = (
            f"Topic: {topic}\n\n"
            f"Write {n_cards} flashcards.\n\n"
            f"---\n\nExcerpts from study materials:\n\n{context}"
        )

        cards: list[Flashcard] = []
        for item in self.generate(FLASHCARD_SYSTEM_INSTRUCTION, prompt):
            front = item.get("front", "").strip()
            back = item.get("back", "").strip()
            if not front or not back:
                continue
            cards.append(Flashcard(
                front=front,
                back=back,
                source=item.get("source", "unknown"),
                page=valid_page(item.get("page")),
            ))
        return cards


if __name__ == "__main__":
    # Manual smoke test:
    #   python -m cyberai_agent.quiz "adversarial examples"
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m cyberai_agent.quiz "topic"')
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    generator = QuizGenerator()

    for i, q in enumerate(generator.generate_quiz(topic, n_questions=3), 1):
        print(f"\n{i}. {q.question}")
        for j, opt in enumerate(q.options):
            print(f"   {chr(97 + j)}) {opt}")
        print(f"   -> {chr(97 + q.correct_index)}  {q.citation()}")
        print(f"   {q.explanation}")