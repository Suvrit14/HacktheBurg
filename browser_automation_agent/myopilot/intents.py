"""The four commands that drive MyoPilot, and where they come from.

Today the source is the keyboard in the control panel. When Suvrit's EMG
classifier is ready, a ClassifierSource just calls `bus.publish(...)` with the
mapped intent - nothing else in the system changes.
"""
from __future__ import annotations

import asyncio
from enum import Enum


class Intent(str, Enum):
    NEXT = "NEXT"      # move highlight forward / scroll down
    PREV = "PREV"      # move highlight back / scroll up
    SELECT = "SELECT"  # choose the highlighted option / approve
    BACK = "BACK"      # cancel / go back


class IntentBus:
    """Single queue every gesture source publishes into."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[tuple[Intent, str]] = asyncio.Queue()
        self.on_publish = None  # optional callback(intent, source) for logging

    def publish(self, intent: Intent | str, source: str = "keyboard") -> None:
        intent = Intent(intent)
        if self.on_publish:
            self.on_publish(intent, source)
        self._q.put_nowait((intent, source))

    async def next(self) -> Intent:
        intent, _ = await self._q.get()
        return intent

    def drain(self) -> None:
        """Drop commands that arrived while the agent was busy (avoids surprise actions)."""
        while not self._q.empty():
            self._q.get_nowait()


class ClassifierSource:
    """Placeholder for the EMG classifier hookup (owned by Suvrit).

    Expected wiring, once the classifier exists:
        label -> intent mapping, e.g. {"hand_close": NEXT, "hand_open": PREV,
                                      "wrist_flex": SELECT, "wrist_extend": BACK}
        Debouncing (majority vote, confidence threshold, cooldown, rest = nothing)
        happens on the classifier side or here, then: bus.publish(intent, "emg").
    """

    def __init__(self, bus: IntentBus, mapping: dict[str, Intent] | None = None):
        self.bus = bus
        self.mapping = mapping or {}

    def on_prediction(self, label: str, confidence: float, threshold: float = 0.8) -> None:
        intent = self.mapping.get(label)
        if intent and confidence >= threshold:
            self.bus.publish(intent, source="emg")
