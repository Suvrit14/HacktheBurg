"""Integration check (no LLM): walk the local demo form with the real browser layer
and SafetyGuard, including a validation error and the approve-then-submit rule."""
import asyncio, re

import uvicorn
from fastapi import FastAPI

from myopilot.browser import Browser
from myopilot.config import DEFAULT_ALLOWED_DOMAINS
from myopilot.mock_site import router
from myopilot.safety import SafetyBlock, SafetyGuard

BASE = "http://localhost:8766/mock/transport"


def ref(snap, pattern):
    for e in snap.elements:
        if re.search(pattern, e["name"], re.I):
            return e
    raise AssertionError(f"no element matching {pattern!r}: {[e['name'] for e in snap.elements]}")


async def main():
    app = FastAPI(); app.include_router(router)
    server = uvicorn.Server(uvicorn.Config(app, port=8766, log_level="warning"))
    st = asyncio.create_task(server.serve()); await asyncio.sleep(0.8)
    b = Browser(headless=True); await b.start()
    g = SafetyGuard(DEFAULT_ALLOWED_DOMAINS, live_no_submit=True)
    try:
        await b.goto(BASE); s = await b.snapshot()
        await b.click(ref(s, "^start now")["ref"]); s = await b.snapshot()
        assert "About you" in s.headings, s.headings
        day = next(e for e in s.elements if e["name"] == "Day")
        assert day.get("question") == "Date of birth", day
        # leave email empty -> expect validation error
        for pat, val in [("^full name", "Alex Morgan"), ("^day$", "14"), ("^month$", "3"), ("^year$", "1990"), ("mobile", "07700 900123")]:
            await b.type_text(ref(s, pat)["ref"], val)
        await b.click(ref(s, "^continue")["ref"]); s = await b.snapshot()
        assert "Enter your email" in s.error_summary, (s.url, s.headings, s.error_summary, s.text[:300])
        email = ref(s, "^email")
        assert "Enter your email" in email.get("error", ""), email
        print("validation error visible to agent:", s.error_summary[:80])
        await b.type_text(email["ref"], "alex.morgan@example.com")
        await b.click(ref(s, "^continue")["ref"]); s = await b.snapshot()
        for pat, val in [("line 1", "The British Library"), ("line 2", "96 Euston Road"), ("town", "London"), ("postcode", "NW1 2DB")]:
            await b.type_text(ref(s, pat)["ref"], val)
        await b.click(ref(s, "^continue")["ref"]); s = await b.snapshot()
        await b.click(ref(s, "^no$")["ref"]); await b.set_checked(ref(s, "help getting")["ref"], True)
        await b.click(ref(s, "^continue")["ref"]); s = await b.snapshot()
        day_sel = ref(s, "which day")
        await b.select_option(day_sel["ref"], day_sel["options"][1])
        await b.click(ref(s, "^morning")["ref"])
        await b.type_text(ref(s, "where do you need")["ref"], "Outpatient physiotherapy at the local hospital")
        await b.click(ref(s, "^continue")["ref"]); s = await b.snapshot()
        assert "Check your answers" in s.title or any("Check your answers" in h for h in s.headings), s.headings
        submit = ref(s, "^submit booking")
        try:
            g.check_click(submit, s); raise AssertionError("submit should need approval")
        except SafetyBlock as e:
            print("blocked without approval:", str(e)[:70])
        g.submit_approved = True
        g.check_click(submit, s)
        await b.click(submit["ref"]); s = await b.snapshot()
        assert "Booking complete" in s.headings, s.headings
        print("submitted after approval:", re.search(r"CT-\d+", s.text).group(0))
        print("ALL MOCK FORM CHECKS PASSED")
    finally:
        await b.close(); server.should_exit = True; await st

asyncio.run(main())
