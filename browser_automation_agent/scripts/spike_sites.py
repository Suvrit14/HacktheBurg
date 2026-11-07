"""P0 spike: load each candidate live site with Playwright and report what the agent would see."""
import asyncio, sys
from myopilot.browser import Browser

SITES = [
    "https://www.nhs.uk/service-search/pharmacy/find-a-pharmacy",
    "https://www.nhs.uk/service-search/find-a-gp",
    "https://gp-registration.nhs.uk/",
    "https://www.gov.uk/find-local-council",
    "https://www.edinburgh.gov.uk/bins-recycling",
    "https://www.gov.uk/register-to-vote",
    "https://www.nationalrail.co.uk/",
    "https://www.bbc.co.uk/weather",
]

async def main(verbose: str | None):
    b = Browser(headless=True)
    await b.start()
    for url in SITES:
        try:
            await b.goto(url)
            s = await b.snapshot()
            r = s.render()
            print(f"OK  {url}\n    final={s.url}\n    title={s.title!r} elements={len(s.elements)} chars={len(r)} (~{len(r)//4} tokens)")
            print(f"    headings={s.headings[:4]}")
            if verbose and verbose in url:
                print(r[:6000])
        except Exception as e:
            print(f"ERR {url}: {e}")
    await b.close()

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
