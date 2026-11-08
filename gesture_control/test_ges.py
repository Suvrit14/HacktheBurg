"""Run one gesture on a Chrome window, one command at a time.

    uv run gesture_control/test_ges.py --open https://en.wikipedia.org
    uv run gesture_control/test_ges.py --scroll-down
    uv run gesture_control/test_ges.py --scroll-up
    uv run gesture_control/test_ges.py --tab
    uv run gesture_control/test_ges.py --click
    uv run gesture_control/test_ges.py --back
    uv run gesture_control/test_ges.py --close

The first run starts Chrome and leaves it open. Every later run connects to
that same window (through Chrome's debugging port), runs one gesture, and
exits, so the page, its history and the highlight stay between runs.
"""

import argparse
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from gestures import GESTURES, apply_gesture

PORT = 9333
CDP_URL = f"http://127.0.0.1:{PORT}"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE_DIR = Path(tempfile.gettempdir()) / "myopilot-gesture-chrome"
DEFAULT_URL = "http://localhost:3000"


def chrome_running() -> bool:
    try:
        urllib.request.urlopen(f"{CDP_URL}/json/version", timeout=0.5)
        return True
    except OSError:
        return False


def start_chrome(url: str) -> None:
    subprocess.Popen(
        [
            CHROME,
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # keep Chrome open after this script exits
    )
    for _ in range(50):
        if chrome_running():
            return
        time.sleep(0.2)
    sys.exit("Chrome did not start")


def visible_page(browser):
    context = browser.contexts[0]
    pages = [p for p in context.pages if p.url.startswith(("http", "file", "about"))]
    for page in pages:
        if page.evaluate("document.visibilityState") == "visible":
            return page
    return pages[-1] if pages else context.new_page()


def parse_args():
    parser = argparse.ArgumentParser(description="Run one gesture on the Chrome window.")
    group = parser.add_mutually_exclusive_group()
    for name in GESTURES:
        flag = name.replace("_", "-")
        options = [f"--{flag}"] + ([f"--{name}"] if flag != name else [])
        group.add_argument(*options, dest="gesture", action="store_const", const=name,
                           help=f"run the {name} gesture")
    parser.add_argument("--open", metavar="URL", help="open this website first")
    parser.add_argument("--close", action="store_true", help="close the Chrome window")
    return parser.parse_args(), parser


def main():
    args, parser = parse_args()

    if args.close:
        if not chrome_running():
            print("Chrome is not running")
            return
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            browser.new_browser_cdp_session().send("Browser.close")
        print("Chrome closed")
        return

    if args.gesture is None and args.open is None:
        parser.print_help()
        return

    started = False
    if not chrome_running():
        start_chrome(args.open or DEFAULT_URL)
        started = True

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        page = visible_page(browser)
        if args.open and not started:
            page.goto(args.open)
        page.wait_for_load_state("domcontentloaded")

        if args.gesture:
            result = apply_gesture(page, args.gesture)
            print(f"{args.gesture}: {result}")
        print(f"page: {page.url}")


if __name__ == "__main__":
    main()
