"""SafetyGuard: hard rules enforced in code, independent of what the LLM decides."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from .browser import Snapshot

SECRET_AUTOCOMPLETE = re.compile(r"(current-password|new-password|one-time-code|cc-)", re.I)
SECRET_NAME = re.compile(r"(password|passcode|\bpin\b|card number|security code|cvv|cvc|sort code|account number)", re.I)
SUBMIT_NAME = re.compile(
    r"^(submit|send|confirm|register|apply|pay|place order|accept and send|confirm and send|submit (application|registration|form))\b",
    re.I,
)
REVIEW_PAGE = re.compile(r"(check your answers|check your details|review your|confirm your details|declaration)", re.I)


class SafetyBlock(Exception):
    """Raised when an action is refused. The message is shown to the LLM and the user."""


class SafetyGuard:
    def __init__(self, allowed_domains: tuple[str, ...], live_no_submit: bool):
        self.allowed_domains = allowed_domains
        self.live_no_submit = live_no_submit
        self.submit_approved = False  # set True only by a user CONFIRM, consumed by the next submit

    # ---- navigation ----
    def check_url(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise SafetyBlock(f"Not a valid web address: {url}")
        if not any(host == d or host.endswith("." + d) for d in self.allowed_domains):
            raise SafetyBlock(f"{host} is not on the list of allowed sites ({', '.join(self.allowed_domains)}).")

    # ---- typing ----
    def check_type(self, el: dict[str, Any] | None) -> None:
        if el is None:
            return
        if el.get("type") == "password" or SECRET_AUTOCOMPLETE.search(el.get("autocomplete", "")) or SECRET_NAME.search(el.get("name", "")):
            raise SafetyBlock(
                "This field asks for a password, PIN or payment detail. MyoPilot never types these; "
                "the user or their carer must enter it themselves."
            )

    # ---- clicking ----
    def is_submit(self, el: dict[str, Any] | None, snap: Snapshot | None) -> bool:
        if el is None or el.get("tag") not in ("button", "input") and el.get("role") != "button":
            return False
        name = (el.get("name") or "").strip()
        if SUBMIT_NAME.search(name):
            return True
        # "Continue" on a check-your-answers page is effectively the final submit.
        on_review = bool(snap and REVIEW_PAGE.search(" ".join(snap.headings) + " " + snap.title))
        return on_review and name.lower() in ("continue", "accept and continue", "next")

    def check_click(self, el: dict[str, Any] | None, snap: Snapshot | None) -> None:
        if not self.is_submit(el, snap):
            return
        host = (urlparse(snap.url).hostname or "") if snap else ""
        is_live = host not in ("localhost", "127.0.0.1")
        if self.live_no_submit and is_live:
            raise SafetyBlock(
                f"Demo safety: '{el.get('name')}' would submit this form to a real live service. "
                "MyoPilot stops here. Tell the user the form is filled and ready for them to review and submit themselves."
            )
        if not self.submit_approved:
            raise SafetyBlock(
                f"'{el.get('name')}' submits the form. Call confirm() first so the user can approve with a gesture."
            )
        self.submit_approved = False  # one approval = one submit
