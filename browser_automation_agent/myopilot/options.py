"""Switch-access menus: what the user can pick with scroll_down / scroll_up / forward.

* On the home screen: a fixed list of everyday tasks (the research-backed set).
* On any other page: the LLM proposes 3-4 next actions for that page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .browser import Snapshot
from .events import EventLog


@dataclass
class MenuItem:
    label: str                 # what the user sees (short)
    goal: str = ""             # instruction for the agent; empty = built-in action
    start_url: str | None = None
    action: str = "task"       # task | scroll_down | scroll_up | back | home | voice


# Everyday tasks shown on the home screen. Keep in sync with the pitch research.
HOME_TASKS: list[MenuItem] = [
    MenuItem("Find a pharmacy near me",
             "Find the nearest pharmacy to my home postcode that is open now, and tell me its name, opening hours and phone number.",
             "https://www.nhs.uk/service-search/pharmacy/find-a-pharmacy"),
    MenuItem("Register with a GP",
             "Find a GP surgery near my home that accepts online registration and is not closed, then fill in its online "
             "registration form for me using my profile, asking me only what you must.",
             "https://www.nhs.uk/service-search/find-a-gp"),
    MenuItem("Check train times",
             "Find the next trains from London Euston to Manchester Piccadilly today and tell me the next three departures.",
             "https://www.nationalrail.co.uk/"),
    MenuItem("Today's weather",
             "Tell me today's weather forecast for London in two sentences.",
             "https://www.bbc.co.uk/weather"),
    MenuItem("Book community transport (demo form)",
             "Book accessible community transport for me for a hospital appointment. Fill the whole form using my "
             "profile, ask me only what you must, then check the answers and submit once I approve.",
             "http://localhost:8765/mock/transport"),
]

ALWAYS = [
    MenuItem("Something else (speak)", action="voice"),
]
PAGE_EXTRAS = [
    MenuItem("Scroll down", action="scroll_down"),
    MenuItem("Scroll up", action="scroll_up"),
    MenuItem("Home", action="home"),
]


class MenuOption(BaseModel):
    label: str = Field(description="2-6 words shown to the user, e.g. 'Opening times'")
    goal: str = Field(description="Full instruction for the browser agent to carry out this option")


class MenuOptions(BaseModel):
    options: list[MenuOption] = Field(description="3-4 most useful next actions on this page")


OPTIONS_PROMPT = """You help a person with limited hand movement use a website. They can only pick from a
short menu. Suggest the 3-4 most useful things they might want to do next on THIS page, most likely first.
Prefer concrete tasks the browser agent can finish (e.g. "Opening times", "Call this pharmacy - show number",
"Book an appointment"). Do not suggest scrolling or going back (those are always available).

User interests: {interests}
Recently done: {recent}

PAGE:
{page}"""


async def propose_options(llm: BaseChatModel | None, snap: Snapshot, profile: dict, recent: list[str],
                          log: EventLog, model_name: str = "") -> list[MenuItem]:
    if llm is None:  # --fake: no model, offer generic page actions
        return [MenuItem("Summarise this page", "Summarise this page for me in three sentences.")]
    prompt = OPTIONS_PROMPT.format(
        interests=", ".join(profile.get("interests", [])) or "none",
        recent="; ".join(recent[-3:]) or "nothing yet",
        page=snap.render(max_elements=60)[:6000],
    )
    t0 = time.monotonic()
    try:
        structured = llm.with_structured_output(MenuOptions, include_raw=True)
        out = await structured.ainvoke(prompt)
        parsed: MenuOptions | None = out.get("parsed")
        usage = getattr(out.get("raw"), "usage_metadata", None) or {}
        log.emit("llm_call", model=model_name, purpose="options", latency_s=round(time.monotonic() - t0, 3),
                 input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                 cache_read_tokens=(usage.get("input_token_details") or {}).get("cache_read", 0))
        items = [MenuItem(o.label.strip()[:40], o.goal.strip()) for o in (parsed.options if parsed else [])][:4]
    except Exception as exc:  # noqa: BLE001 - menu must still appear if the model fails
        log.emit("llm_error", purpose="options", error=str(exc)[:300])
        items = []
    return items or [MenuItem("Summarise this page", "Summarise this page for me in three sentences.")]
