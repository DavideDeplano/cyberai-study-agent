"""Spaced-repetition tracking for generated flashcards.

This module turns the one-shot decks produced by `quiz.py` into a
persistent review queue scheduled with FSRS (Free Spaced Repetition
Scheduler), via the `fsrs` package. It owns two concerns only:

- persistence: the deck lives in a single JSON file, so it can be
  inspected by hand, versioned, or deleted without extra tooling;
- scheduling: every review is delegated to `fsrs.Scheduler`, which
  decides when each card is due next.

No LLM or embedding model is involved here, which keeps the module fast
to import and fully testable offline.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fsrs import Card, Rating, ReviewLog, Scheduler

if TYPE_CHECKING:
    # Imported for type hints only: `quiz` pulls in the Gemini client and
    # the vector store, which this module does not need at runtime.
    from cyberai_agent.quiz import Flashcard

DEFAULT_DECK_PATH = Path("data/review/deck.json")


@dataclass
class DeckEntry:
    """A flashcard together with its FSRS scheduling state.

    Attributes:
        front: Prompt side of the card.
        back: Answer side of the card.
        source: Filename of the PDF the card was built from.
        page: 1-indexed page in that PDF.
        card: FSRS memory state (stability, difficulty, due date, ...).
        logs: Every review recorded for this card, oldest first. Kept so
            that FSRS parameters can later be optimised on real history.
    """
    front: str
    back: str
    source: str
    page: int
    card: Card = field(default_factory=Card)
    logs: list[ReviewLog] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        """Identity used to avoid storing the same card twice."""
        # Normalising case and whitespace catches the near-identical
        # fronts an LLM tends to produce when asked twice on one topic.
        return (" ".join(self.front.lower().split()), self.source)

    def citation(self) -> str:
        """Return the provenance string in the same format the agent uses."""
        return f"[{self.source}, p.{self.page}]"

    def to_dict(self) -> dict:
        return {
            "front": self.front,
            "back": self.back,
            "source": self.source,
            "page": self.page,
            "card": self.card.to_dict(),
            "logs": [log.to_dict() for log in self.logs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckEntry":
        return cls(
            front=data["front"],
            back=data["back"],
            source=data["source"],
            page=data["page"],
            card=Card.from_dict(data["card"]),
            logs=[ReviewLog.from_dict(log) for log in data.get("logs", [])],
        )


class ReviewDeck:
    """Persistent collection of flashcards scheduled with FSRS.

    Attributes:
        path: JSON file backing the deck.
        entries: Cards currently in the deck.
        scheduler: FSRS scheduler used for every review.
    """

    def __init__(self, path: Path = DEFAULT_DECK_PATH, scheduler: Scheduler | None = None):
        """Load the deck from `path`, or start empty if the file is missing.

        Args:
            path: JSON file backing the deck.
            scheduler: Custom FSRS scheduler. Defaults to the package's
                standard parameters, which are a sensible starting point
                until enough review history exists to optimise them.
        """
        self.path = Path(path)
        self.scheduler = scheduler or Scheduler()
        self.entries: list[DeckEntry] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [DeckEntry.from_dict(item) for item in data]

    def save(self) -> None:
        """Write the deck to disk.

        The file is written to a temporary sibling first and then renamed,
        so an interrupted save never leaves a truncated deck behind.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([e.to_dict() for e in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def add_flashcards(self, cards: "list[Flashcard]") -> int:
        """Add new cards to the deck, skipping ones already present.

        Args:
            cards: Flashcards produced by `QuizGenerator.generate_flashcards`.

        Returns:
            Number of cards actually added.
        """
        known = {e.key for e in self.entries}
        added = 0
        for c in cards:
            entry = DeckEntry(front=c.front, back=c.back, source=c.source, page=c.page)
            if entry.key in known:
                continue
            known.add(entry.key)
            self.entries.append(entry)
            added += 1
        return added

    def due(self, now: datetime | None = None) -> list[DeckEntry]:
        """Return the cards due for review, most overdue first.

        Args:
            now: Reference time (UTC). Defaults to the current time;
                exposed so tests can simulate the passing of days.
        """
        now = now or datetime.now(timezone.utc)
        return sorted(
            (e for e in self.entries if e.card.due <= now),
            key=lambda e: e.card.due,
        )

    def review(self, entry: DeckEntry, rating: Rating, now: datetime | None = None) -> None:
        """Record a review and reschedule the card.

        Args:
            entry: Card being reviewed; updated in place.
            rating: How well the answer was recalled (Again/Hard/Good/Easy).
            now: Review time (UTC). Defaults to the current time.
        """
        now = now or datetime.now(timezone.utc)
        entry.card, log = self.scheduler.review_card(entry.card, rating, review_datetime=now)
        entry.logs.append(log)