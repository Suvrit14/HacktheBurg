"""Tool schemas the LLM sees. Execution lives in graph.py (it needs the browser,
the safety guard and the event log), so these functions are schema-only stubs."""
from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool
def navigate(url: str, explain: str) -> str:
    """Open a web address. Only allowed sites work. `explain` is a short plain-English
    caption shown to the user, e.g. 'Opening the NHS pharmacy finder'."""
    raise NotImplementedError


@tool
def click(ref: int, explain: str) -> str:
    """Click the element with this [ref] number (links, buttons, radio buttons, checkboxes)."""
    raise NotImplementedError


@tool
def type_text(ref: int, text: str, explain: str, press_enter: bool = False) -> str:
    """Replace the contents of a text field [ref] with `text`. Set press_enter to submit a search box."""
    raise NotImplementedError


@tool
def select_option(ref: int, option: str, explain: str) -> str:
    """Choose an option (by its visible text) in a dropdown [ref]."""
    raise NotImplementedError


@tool
def set_checkbox(ref: int, checked: bool, explain: str) -> str:
    """Tick (checked=true) or untick a checkbox [ref]."""
    raise NotImplementedError


@tool
def scroll(direction: Literal["up", "down"], explain: str) -> str:
    """Scroll the page to reveal off-screen elements."""
    raise NotImplementedError


@tool
def go_back(explain: str) -> str:
    """Go back to the previous page."""
    raise NotImplementedError


@tool
def ask_choice(question: str, options: list[str]) -> str:
    """Ask the user to pick one of 2-4 short options using their gesture controls.
    Use whenever the answer is not in the profile but can be offered as choices
    (e.g. which pharmacy, which time slot, yes/no). Returns the chosen option, or
    'back' if the user cancelled."""
    raise NotImplementedError


@tool
def ask_text(question: str) -> str:
    """Ask the user for free text they will dictate by voice. Use ONLY when the answer
    is genuinely free-form and cannot be offered as choices (e.g. describe a symptom)."""
    raise NotImplementedError


@tool
def confirm(summary: str) -> str:
    """Before any final submit/send/confirm button, show the user a short summary of
    what will be sent and ask for approval. Returns 'APPROVED' or 'REJECTED'."""
    raise NotImplementedError


@tool
def finish(answer: str, follow_ups: list[str]) -> str:
    """End the task. `answer` is the result for the user in 2-4 plain-English sentences and MUST
    include the actual facts requested (e.g. name, opening hours, phone number), not a summary
    saying they were found.
    `follow_ups` are 2-4 short next actions the user might want (shown as gesture options)."""
    raise NotImplementedError


BROWSER_TOOLS = {"navigate", "click", "type_text", "select_option", "set_checkbox", "scroll", "go_back"}
HUMAN_TOOLS = {"ask_choice", "ask_text", "confirm"}
ALL_TOOLS = [navigate, click, type_text, select_option, set_checkbox, scroll, go_back,
             ask_choice, ask_text, confirm, finish]
