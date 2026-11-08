"""Provenance guard: the agent may only type or choose what the user actually gave it.

Allowed sources for any value typed, selected or ticked:
  * the user's profile,
  * the task the user selected or dictated,
  * answers the user gave during this task (ask_choice / ask_text).

Anything else (e.g. a hospital, a date or a wheelchair answer the model made up)
is refused with a message telling the agent to ask the user first. Some models
(seen with Amazon Nova Pro) invent plausible details even when told not to, so
this is enforced in code, not just in the prompt.
"""
from __future__ import annotations

import re
from typing import Any

from .profile import flatten_values

STOPWORDS = {
    "the", "and", "for", "with", "from", "near", "into", "this", "that", "your", "you", "are", "was",
    "have", "has", "not", "but", "all", "any", "can", "our", "out", "about", "please", "via", "per",
}
WORD = re.compile(r"[a-z0-9]+")


class AskFirst(Exception):
    """The value did not come from the user; the agent must ask before using it."""


def _words(s: str) -> set[str]:
    return {w for w in WORD.findall(s.lower()) if (len(w) >= 3 or w.isdigit()) and w not in STOPWORDS}


class ProvenanceGuard:
    def __init__(self, profile: dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        self.profile_values = flatten_values(profile)
        self.profile_words = set().union(*(_words(v) for v in self.profile_values)) if self.profile_values else set()
        self.goal = ""
        self.answers: list[str] = []

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.answers = []

    def add_answer(self, reply: Any) -> None:
        if isinstance(reply, str) and reply not in ("back", "APPROVED", "REJECTED"):
            self.answers.append(reply)

    # ---------------------------------------------------------------- checks
    def _known_words(self) -> set[str]:
        return self.profile_words | _words(self.goal) | set().union(*(_words(a) for a in self.answers or [""]))

    def check_value(self, value: str, field: str = "") -> None:
        """For typed text and dropdown choices."""
        if not self.enabled:
            return
        v = value.strip().lower()
        if not v or v in self.profile_values:
            return
        if any(v == a.strip().lower() for a in self.answers):
            return
        missing = sorted(_words(v) - self._known_words())
        if missing:
            raise AskFirst(
                f"'{value}'{f' for \"{field}\"' if field else ''} was not given by the user or their profile "
                f"(unknown: {', '.join(missing[:5])}). Do not guess: ask the user first - ask_choice with the "
                f"page's own options, or ask_text for free text - then use their exact answer."
            )

    def check_choice(self, label: str, question: str = "") -> None:
        """For radio buttons and ticking checkboxes: the label must match something the user said."""
        if not self.enabled:
            return
        lab = label.strip().lower()
        if not lab:
            return
        if lab in self.profile_values or any(lab == a.strip().lower() or lab in a.strip().lower() for a in self.answers):
            return
        words = _words(lab)
        if words and words <= (self.profile_words | _words(self.goal)):
            return
        raise AskFirst(
            f"Choosing '{label}'{f' for \"{question}\"' if question else ''} is the user's decision. "
            f"Ask them with ask_choice (offer the page's options) and then pick exactly what they chose."
        )
