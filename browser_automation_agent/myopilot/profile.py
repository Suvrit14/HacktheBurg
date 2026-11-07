"""User profile used for L1 autofill. The demo profile is fictional."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def profile_for_prompt(profile: dict[str, Any]) -> str:
    """Render the profile for the system prompt. Missing (null) values are listed
    explicitly so the agent knows to ask or pick an 'I don't know' option."""
    known, unknown = {}, []
    for k, v in profile.items():
        (unknown.append(k) if v in (None, "", []) else known.__setitem__(k, v))
    out = json.dumps(known, indent=1, ensure_ascii=False)
    if unknown:
        out += "\nNOT KNOWN (do not invent; ask the user or choose an \"I don't know\" option): " + ", ".join(unknown)
    return out


def flatten_values(profile: dict[str, Any]) -> set[str]:
    """All leaf string values, used to tag a typed value as 'from profile'."""
    vals: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif x not in (None, ""):
            s = str(x).strip().lower()
            vals.add(s)
            # ISO dates are typed into split Day / Month / Year boxes: count the parts too.
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                y, mth, d = s.split("-")
                vals.update({y, mth, d, mth.lstrip("0"), d.lstrip("0"), f"{d}/{mth}/{y}"})

    walk(profile)
    return vals
