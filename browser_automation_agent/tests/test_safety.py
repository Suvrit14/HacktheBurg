"""Unit tests for the SafetyGuard hard rules (no browser, no model)."""
import pytest

from myopilot.browser import Snapshot
from myopilot.config import DEFAULT_ALLOWED_DOMAINS
from myopilot.safety import SafetyBlock, SafetyGuard


def snap(url="https://gp-registration.nhs.uk/x", headings=("Who is registering?",), title="t"):
    return Snapshot(url=url, title=title, headings=list(headings), error_summary="", text="", elements=[], scroll={"y": 0, "max": 0})


def guard(no_submit=True):
    return SafetyGuard(DEFAULT_ALLOWED_DOMAINS, live_no_submit=no_submit)


@pytest.mark.parametrize("url", ["https://www.nhs.uk/a", "https://www.registertovote.service.gov.uk/start", "http://localhost:8000/f"])
def test_allowed_urls(url):
    guard().check_url(url)


@pytest.mark.parametrize("url", ["https://www.google.com/maps", "https://evilnhs.uk.attacker.com", "notaurl"])
def test_blocked_urls(url):
    with pytest.raises(SafetyBlock):
        guard().check_url(url)


@pytest.mark.parametrize("el", [
    {"tag": "input", "type": "password", "name": "Password"},
    {"tag": "input", "type": "text", "name": "Card number"},
    {"tag": "input", "type": "text", "name": "x", "autocomplete": "cc-number"},
])
def test_secret_fields_refused(el):
    with pytest.raises(SafetyBlock):
        guard().check_type(el)


def test_normal_field_allowed():
    guard().check_type({"tag": "input", "type": "text", "name": "Postcode"})


def test_continue_is_not_submit_on_normal_page():
    guard().check_click({"tag": "button", "name": "Continue"}, snap())


def test_submit_blocked_on_live_site_even_if_approved():
    g = guard(no_submit=True)
    g.submit_approved = True
    with pytest.raises(SafetyBlock, match="Demo safety"):
        g.check_click({"tag": "button", "name": "Submit registration"}, snap())


def test_continue_on_check_your_answers_is_submit():
    with pytest.raises(SafetyBlock):
        guard().check_click({"tag": "button", "name": "Continue"}, snap(headings=["Check your answers"]))


def test_local_submit_needs_approval_and_is_single_use():
    g = guard(no_submit=True)
    s = snap(url="http://localhost:8000/review")
    with pytest.raises(SafetyBlock, match="confirm"):
        g.check_click({"tag": "button", "name": "Submit"}, s)
    g.submit_approved = True
    g.check_click({"tag": "button", "name": "Submit"}, s)  # approved once
    with pytest.raises(SafetyBlock):
        g.check_click({"tag": "button", "name": "Submit"}, s)  # approval consumed
