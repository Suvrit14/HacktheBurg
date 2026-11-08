"""Structured event log: one JSON object per line in runs/<id>/events.jsonl.

This is the contract the dashboard (and later the classifier side) reads.
Event types:
  run_start, run_end            run boundaries and totals
  intent                        a scroll_down / scroll_up / forward / back command (keyboard or EMG)
  options_proposed, choice      switch-access menus and what the user picked
  llm_call                      one model call: model, latency, input/output/cache tokens
  agent_step                    one browser tool call: tool, args, url, ok/error, latency
  ask, answer                   agent asked the user something (choice / free text) and the reply
  confirm, confirm_result       pre-submit approval requests and the decision
  safety_block                  an action the SafetyGuard refused
  result                        final answer of a task
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class EventLog:
    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._fh = self.path.open("a", buffering=1)
        self._t0 = time.monotonic()
        self._subscribers: list[Callable[[dict[str, Any]], Any]] = []
        self.totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                       "cache_read_tokens": 0, "browser_steps": 0, "errors": 0}

    def subscribe(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        self._subscribers.append(fn)

    def emit(self, type_: str, **data: Any) -> dict[str, Any]:
        ev = {"type": type_, "t": round(time.monotonic() - self._t0, 3),
              "ts": datetime.now().isoformat(timespec="milliseconds"), **data}
        if type_ == "llm_call":
            self.totals["llm_calls"] += 1
            self.totals["input_tokens"] += data.get("input_tokens", 0) or 0
            self.totals["output_tokens"] += data.get("output_tokens", 0) or 0
            self.totals["cache_read_tokens"] += data.get("cache_read_tokens", 0) or 0
        elif type_ == "agent_step":
            self.totals["browser_steps"] += 1
            if not data.get("ok", True):
                self.totals["errors"] += 1
        self._fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
        for fn in self._subscribers:
            try:
                res = fn(ev)
                if asyncio.iscoroutine(res):
                    asyncio.ensure_future(res)
            except Exception:  # noqa: BLE001 - a broken subscriber must not stop the agent
                pass
        return ev

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def close(self) -> None:
        self._fh.close()
