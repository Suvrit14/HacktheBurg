"""P0 spike, part 2: walk the real flows by hand (no LLM) to confirm they work under Playwright."""
import asyncio
from myopilot.browser import Browser

def find(s, pred):
    return next((e for e in s.elements if pred(e)), None)

async def main():
    b = Browser(headless=True)
    await b.start()
    # 1. NHS find a pharmacy by postcode
    await b.goto("https://www.nhs.uk/service-search/pharmacy/find-a-pharmacy")
    s = await b.snapshot()
    print(s.render()[:2500]); print("-----")
    box = find(s, lambda e: e["tag"] == "input" and e["type"] in ("text", "search", ""))
    await b.type_text(box["ref"], "NW1 2DB", press_enter=True)
    s = await b.snapshot()
    print("PHARMACY RESULTS:", s.url, s.headings[:5], len(s.elements), "elements")
    print(s.render(max_elements=40)[:3000]); print("=====")

    # 2. NHS find a GP -> practice page -> registration link
    await b.goto("https://www.nhs.uk/service-search/find-a-gp")
    s = await b.snapshot()
    box = find(s, lambda e: e["tag"] == "input" and e["type"] in ("text", "search", ""))
    await b.type_text(box["ref"], "NW1 2DB", press_enter=True)
    s = await b.snapshot()
    print("GP RESULTS:", s.url, s.headings[:5])
    practice = find(s, lambda e: e["tag"] == "a" and e.get("href", "").startswith("/services/gp-surgery/"))
    print("first practice:", practice)
    if practice:
        await b.click(practice["ref"])
        s = await b.snapshot()
        print("PRACTICE:", s.url, s.headings[:6])
        reg = [e for e in s.elements if e["tag"] == "a" and "regist" in (e["name"] + e.get("href", "")).lower()]
        for e in reg: print("  reg link:", e["name"], "->", e.get("href"))
    print("=====")

    # 3. GOV.UK register to vote start page -> first form page
    await b.goto("https://www.gov.uk/register-to-vote")
    s = await b.snapshot()
    start = find(s, lambda e: "start now" in e["name"].lower())
    print("start:", start)
    if start:
        await b.click(start["ref"])
        s = await b.snapshot()
        print("VOTE FORM:", s.url, s.headings[:5])
        print(s.render(max_elements=40)[:2500])
    await b.close()

asyncio.run(main())
