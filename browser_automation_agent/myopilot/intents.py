"""The four commands that drive MyoPilot, and where they come from.

The commands are exactly the ones Suvrit's EMG command logic emits
(emg_data/commands.py), so there is no translation between the two pipelines:

    command       gesture (NinaPro DB5 E2)      what it does in a menu
    scroll_down   fist (6)                      move the highlight down
    scroll_up     spread fingers (5)            move the highlight up
    forward       wrist ulnar deviation (16)    choose the highlighted option / approve
    back          wrist radial deviation (15)   cancel / go back

Sources: the keyboard / buttons in the control panel, and Suvrit's classifier
(myopilot/emg_source.py). Both publish into the same IntentBus.
"""
from __future__ import annotations

import asyncio
from enum import Enum


class Intent(str, Enum):
    SCROLL_DOWN = "scroll_down"  # move the highlight down
    SCROLL_UP = "scroll_up"      # move the highlight up
    FORWARD = "forward"          # choose the highlighted option / approve
    BACK = "back"                # cancel / go back


COMMANDS = [i.value for i in Intent]


def to_intent(name: Intent | str) -> Intent:
    """'scroll_down' / 'scroll_up' / 'forward' / 'back' (any case) -> Intent."""
    if isinstance(name, Intent):
        return name
    try:
        return Intent(str(name).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown command {name!r}; use one of: {', '.join(COMMANDS)}") from exc


class IntentBus:
    """Single queue every gesture source publishes into."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[tuple[Intent, str]] = asyncio.Queue()
        self.on_publish = None  # optional callback(intent, source) for logging

    def publish(self, command: Intent | str, source: str = "keyboard") -> Intent:
        intent = to_intent(command)
        if self.on_publish:
            self.on_publish(intent, source)
        self._q.put_nowait((intent, source))
        return intent

    async def next(self) -> Intent:
        intent, _ = await self._q.get()
        return intent

    def drain(self) -> None:
        """Drop commands that arrived while the agent was busy (avoids surprise actions)."""
        while not self._q.empty():
            self._q.get_nowait()
