"""Switch-access controller: turns scroll_down/scroll_up/tab/click/back into menus, tasks and answers.

Screens (the `mode` sent to the panel):
  home     everyday tasks to start from
  browse   options the LLM proposes for the current page
  working  the agent is acting; live captions only
  ask      the agent asked a question with choices
  confirm  the agent wants approval before submitting
  text     free text needed (voice dictation in the panel)
  result   the agent's answer plus follow-up options
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from .browser import Browser
from .events import EventLog
from .graph import TaskAgent
from .intents import Intent, IntentBus
from .options import ALWAYS, HOME_TASKS, PAGE_EXTRAS, MenuItem, propose_options

HOME_URL = "https://www.nhs.uk/"
Screen = tuple  # ("home",) | ("browse",) | ("result", answer, follow_ups)


class Session:
    def __init__(self, *, browser: Browser, bus: IntentBus, log: EventLog, agent: TaskAgent,
                 options_llm: BaseChatModel | None, profile: dict[str, Any], model_name: str = ""):
        self.browser = browser
        self.bus = bus
        self.log = log
        self.agent = agent
        self.options_llm = options_llm
        self.profile = profile
        self.model_name = model_name
        self.text_q: asyncio.Queue[str] = asyncio.Queue()
        self.view: dict[str, Any] = {"mode": "starting"}
        self.listeners: list[Callable[[dict[str, Any]], Awaitable[None]]] = []
        self.recent: list[str] = []

    # ------------------------------------------------------------- panel I/O
    async def publish(self, **view: Any) -> None:
        self.view = {**view, "metrics": self._metrics()}
        for fn in list(self.listeners):
            try:
                await fn({"kind": "view", **self.view})
            except Exception:  # noqa: BLE001 - a dropped websocket must not stop the session
                self.listeners.remove(fn)

    def _metrics(self) -> dict[str, Any]:
        t = self.log.totals
        return {"browser_steps": t["browser_steps"], "llm_calls": t["llm_calls"],
                "tokens": t["input_tokens"] + t["output_tokens"], "elapsed_s": round(self.log.elapsed, 1),
                "model": self.model_name}

    async def menu(self, mode: str, title: str, labels: list[str], **extra: Any) -> int | None:
        """Show a menu and wait for gestures. Returns the selected index, or None on `back`."""
        cursor = 0
        self.bus.drain()
        await self.publish(mode=mode, title=title, items=labels, cursor=cursor, **extra)
        while True:
            intent = await self.bus.next()
            if intent in (Intent.SCROLL_DOWN, Intent.TAB):
                cursor = (cursor + 1) % len(labels)
            elif intent is Intent.SCROLL_UP:
                cursor = (cursor - 1) % len(labels)
            elif intent is Intent.CLICK:
                self.log.emit("choice", mode=mode, title=title, index=cursor, label=labels[cursor])
                return cursor
            elif intent is Intent.BACK:
                self.log.emit("choice", mode=mode, title=title, index=None, label="back")
                return None
            await self.publish(mode=mode, title=title, items=labels, cursor=cursor, **extra)

    async def ask_text(self, question: str) -> str | None:
        """Wait for dictated text from the panel, or `back`."""
        while not self.text_q.empty():
            self.text_q.get_nowait()
        self.bus.drain()
        await self.publish(mode="text", title=question, items=[], cursor=0)
        text_task = asyncio.ensure_future(self.text_q.get())
        while True:
            intent_task = asyncio.ensure_future(self.bus.next())
            done, _ = await asyncio.wait({text_task, intent_task}, return_when=asyncio.FIRST_COMPLETED)
            if text_task in done:
                intent_task.cancel()
                return text_task.result().strip() or None
            if intent_task.result() is Intent.BACK:
                text_task.cancel()
                return None

    # ------------------------------------------------------------- agent bridge
    async def on_interrupt(self, payload: dict[str, Any]) -> Any:
        kind, q = payload.get("kind"), payload.get("question", "")
        if kind == "choice":
            opts = payload.get("options") or ["Yes", "No"]
            i = await self.menu("ask", q, opts)
            return "back" if i is None else opts[i]
        if kind == "confirm":
            i = await self.menu("confirm", q, ["Approve", "Don't submit"])
            return "APPROVED" if i == 0 else "REJECTED"
        text = await self.ask_text(q)
        return text or "back"

    async def run_task(self, goal: str, start_url: str | None = None) -> Screen:
        await self.publish(mode="working", title=goal, items=[], cursor=0, caption="Starting…")
        try:
            summary = await self.agent.run(goal, self.on_interrupt, start_url=start_url)
        except Exception as exc:  # noqa: BLE001 - show the failure, keep the session alive
            self.log.emit("task_error", goal=goal, error=str(exc)[:500])
            summary = {"answer": f"Sorry, something went wrong: {exc}", "follow_ups": []}
        self.recent.append(goal)
        return ("result", summary.get("answer", ""), list(summary.get("follow_ups", [])))

    # ------------------------------------------------------------- screens
    # Each screen returns the next screen as a tuple: ("home",), ("browse",),
    # ("result", answer, follow_ups). run() loops over them, so the call stack stays flat.
    async def screen_home(self) -> Screen:
        items = HOME_TASKS + ALWAYS
        while True:
            i = await self.menu("home", "What would you like to do?", [m.label for m in items])
            if i is not None and (nxt := await self._handle(items[i])):
                return nxt

    async def screen_browse(self) -> Screen:
        await self.publish(mode="working", title="Reading this page…", items=[], cursor=0, caption="Thinking of options")
        snap = await self.browser.snapshot()
        proposed = await propose_options(self.options_llm, snap, self.profile, self.recent, self.log, self.model_name)
        self.log.emit("options_proposed", url=snap.url, options=[m.label for m in proposed])
        items = proposed + PAGE_EXTRAS + ALWAYS
        while True:
            i = await self.menu("browse", snap.title, [m.label for m in items])
            if i is None:
                return ("home",)
            if nxt := await self._handle(items[i]):
                return nxt

    async def screen_result(self, answer: str, follow_ups: list[str]) -> Screen:
        items = [MenuItem(f, f) for f in follow_ups] + PAGE_EXTRAS + ALWAYS
        while True:
            i = await self.menu("result", "Done", [m.label for m in items], answer=answer)
            if i is None:
                return ("browse",)
            if nxt := await self._handle(items[i]):
                return nxt

    async def _handle(self, item: MenuItem) -> Screen | None:
        """Run a menu item. Returns the next screen, or None to keep the current menu up."""
        if item.action == "scroll_down":
            await self.browser.scroll("down")
            return None
        if item.action == "scroll_up":
            await self.browser.scroll("up")
            return None
        if item.action == "home":
            return ("home",)
        if item.action == "voice":
            text = await self.ask_text("What would you like to do? (speak)")
            return await self.run_task(text) if text else None
        return await self.run_task(item.goal, item.start_url)

    async def run(self) -> None:
        await self.browser.goto(HOME_URL)
        screen: Screen = ("home",)
        while True:
            name, *args = screen
            screen = await getattr(self, f"screen_{name}")(*args)

    # live captions while the agent works
    async def on_event(self, ev: dict[str, Any]) -> None:
        if self.view.get("mode") == "working" and ev["type"] == "agent_step":
            await self.publish(**{**self.view, "caption": ev.get("explain") or ev.get("tool")})
        for fn in list(self.listeners):
            try:
                await fn({"kind": "event", **_slim(ev)})
            except Exception:  # noqa: BLE001
                self.listeners.remove(fn)


def _slim(ev: dict[str, Any]) -> dict[str, Any]:
    keep = ("type", "t", "tool", "explain", "ok", "error", "url", "latency_s", "input_tokens", "output_tokens",
            "model", "purpose", "question", "reply", "label", "mode", "reason", "intent", "source", "answer", "note",
            "command", "gesture", "confidence", "intended", "decoded", "suppressed_repeats")
    return {k: ev[k] for k in keep if k in ev}
