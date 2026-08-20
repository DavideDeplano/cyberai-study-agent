"""Conversation state for multi-turn interactions with the study agent.

This module owns the in-memory transcript of a chat session: a plain
ordered list of user and assistant turns. It is deliberately storage-
agnostic — persistence to disk or a database can be layered on later
without touching the agent logic.

The transcript is passed to the LLM at every turn so that follow-up
questions can be resolved against previous exchanges (pronoun
resolution, elliptical questions, "tell me more" style prompts).
"""

from dataclasses import dataclass, field
from typing import Literal


# A single turn is tagged with its speaker. Only two roles exist in
# this agent: the human user and the assistant's own replies. System
# instructions are not stored here — they are configured on the LLM
# client separately.
Role = Literal["user", "assistant"]


@dataclass
class Turn:
    """One message in the conversation transcript.

    Attributes:
        role: Who spoke — "user" or "assistant".
        content: The message text as it was sent or produced. For
            assistant turns this is the final answer, not the
            intermediate retrieved context.
    """
    role: Role
    content: str


@dataclass
class Conversation:
    """Ordered transcript of a chat session.

    The `turns` list grows in append-only fashion during a session and
    is cleared explicitly via `reset` when the user wants to start
    fresh. No automatic truncation is applied here: keeping the whole
    history is fine for the short, focused study sessions this agent
    targets, and any windowing policy is better decided at the point
    the transcript is rendered into a prompt.
    """
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        """Append a user turn to the transcript."""
        self.turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant turn to the transcript."""
        self.turns.append(Turn(role="assistant", content=content))

    def reset(self) -> None:
        """Drop all turns and start over."""
        self.turns.clear()

    def is_empty(self) -> bool:
        """True if no turns have been recorded yet."""
        return not self.turns

    def as_plain_text(self) -> str:
        """Render the transcript as a readable plain-text block.

        Used to feed the history into the query-rewriting LLM call as
        a single string. The format is intentionally minimal — one
        turn per paragraph, prefixed by an uppercase role label — so
        the model can parse it without ambiguity.
        """
        lines = []
        for t in self.turns:
            label = "USER" if t.role == "user" else "ASSISTANT"
            lines.append(f"{label}: {t.content}")
        return "\n\n".join(lines)