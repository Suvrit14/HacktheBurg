"""The agent may only type/choose values from the profile, the task, or the user's answers."""
import json
from pathlib import Path

import pytest

from myopilot.profile import load_profile
from myopilot.provenance import AskFirst, ProvenanceGuard

PROFILE = load_profile(Path(__file__).parent.parent / "profiles" / "demo_user.json")


def guard(goal="Book community transport for a hospital appointment"):
    g = ProvenanceGuard(PROFILE)
    g.reset(goal)
    return g


@pytest.mark.parametrize("value", ["Alex Morgan", "NW1 2DB", "14", "03", "1990", "alex.morgan@example.com", "96 Euston Road"])
def test_profile_values_allowed(value):
    guard().check_value(value)


def test_goal_words_allowed():
    guard("Find the next trains from London Euston to Manchester Piccadilly").check_value("London Euston")


@pytest.mark.parametrize("value", [
    "Hospital appointment at St. Mary's Hospital, Praed Street, London, W2 1NY",  # invented place
    "Monday 09 November",                                                      # invented date
])
def test_invented_values_refused(value):
    with pytest.raises(AskFirst):
        guard().check_value(value)


def test_answers_unlock_values():
    g = guard()
    g.add_answer("Thursday 12 November")
    g.add_answer("Physiotherapy appointment at University College Hospital, 235 Euston Road")
    g.check_value("Thursday 12 November")
    g.check_value("Physiotherapy appointment at University College Hospital, 235 Euston Road")


def test_radio_needs_user_choice():
    g = guard()
    with pytest.raises(AskFirst):
        g.check_choice("No", "Do you use a wheelchair?")
    g.add_answer("No")
    g.check_choice("No", "Do you use a wheelchair?")


def test_radio_from_profile_allowed():
    guard().check_choice("England", "Where do you live?")


def test_disabled_guard_allows_everything():
    g = ProvenanceGuard(PROFILE, enabled=False)
    g.check_value("anything at all"); g.check_choice("Yes")
