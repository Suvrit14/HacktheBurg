"""Playwright wrapper the agent acts through.

Every action targets an element by the numeric ref assigned by snapshot.js,
so the LLM never has to write CSS selectors.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from playwright.async_api import TimeoutError as PWTimeout

SNAPSHOT_JS = (Path(__file__).parent / "snapshot.js").read_text()

HIGHLIGHT_CSS = """
[data-mp-active] { outline: 4px solid #ff6a00 !important; outline-offset: 2px !important;
  box-shadow: 0 0 0 8px rgba(255,106,0,.25) !important; transition: outline .1s; }
#mp-banner { position: fixed; z-index: 2147483647; left: 50%; bottom: 16px; transform: translateX(-50%);
  background: #111; color: #fff; font: 600 15px/1.3 system-ui, sans-serif; padding: 10px 16px;
  border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,.35); max-width: 80vw; pointer-events: none; }
"""


@dataclass
class Snapshot:
    url: str
    title: str
    headings: list[str]
    error_summary: str
    text: str
    elements: list[dict[str, Any]]
    scroll: dict[str, int]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def element(self, ref: int) -> dict[str, Any] | None:
        return next((e for e in self.elements if e["ref"] == ref), None)

    def render(self, max_elements: int | None = None) -> str:
        """Compact text form sent to the LLM."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.headings:
            lines.append("HEADINGS: " + " | ".join(self.headings))
        if self.error_summary:
            lines.append(f"PAGE ERRORS: {self.error_summary}")
        lines.append(f"SCROLL: {self.scroll['y']}/{max(self.scroll['max'], 0)}px")
        lines.append("ELEMENTS:")
        els = self.elements if max_elements is None else self.elements[:max_elements]
        for e in els:
            lines.append("  " + _render_element(e))
        if max_elements is not None and len(self.elements) > max_elements:
            lines.append(f"  ... {len(self.elements) - max_elements} more elements (scroll to reveal)")
        lines.append("PAGE TEXT:")
        lines.append(self.text)
        return "\n".join(lines)


def _render_element(e: dict[str, Any]) -> str:
    kind = e["role"] or e["tag"]
    if e["tag"] == "input" and e["type"]:
        kind = f"input[{e['type']}]"
    s = f"[{e['ref']}] {kind} \"{e['name']}\""
    if e.get("question"):
        s += f" (question: {e['question']})"
    if "href" in e and e["href"]:
        s += f" -> {e['href'][:80]}"
    if "value" in e and e["value"]:
        s += f" value=\"{str(e['value'])[:60]}\""
    if e.get("checked"):
        s += " CHECKED"
    if e.get("options"):
        s += " options=" + "|".join(e["options"][:25])
    if e.get("required"):
        s += " required"
    if e.get("disabled"):
        s += " disabled"
    if e.get("autocomplete"):
        s += f" autocomplete={e['autocomplete']}"
    if e.get("hint"):
        s += f" hint=\"{e['hint']}\""
    if e.get("error"):
        s += f" ERROR=\"{e['error']}\""
    if not e.get("in_view"):
        s += " (off-screen)"
    return s


class BrowserError(Exception):
    """Raised for action failures the agent should see and recover from."""


class Browser:
    def __init__(self, headless: bool = False, slow_mo: int = 0, screenshot_dir: Path | None = None,
                 channel: str | None = None):
        self.headless = headless
        self.channel = channel or None  # "chrome" = the installed Google Chrome; None = bundled Chromium
        self.slow_mo = slow_mo
        self.screenshot_dir = screenshot_dir
        self._pw: Playwright | None = None
        self._browser: PWBrowser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self.last_snapshot: Snapshot | None = None
        self._shot_n = 0

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless, slow_mo=self.slow_mo,
                                                       channel=self.channel)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 860}, locale="en-GB", timezone_id="Europe/London"
        )
        self.page = await self._context.new_page()
        await self._context.add_init_script(
            "document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');"
            f"s.textContent={HIGHLIGHT_CSS!r};document.head.appendChild(s);}});"
        )

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ---------- observation ----------
    async def snapshot(self, max_elements: int = 250, max_text: int = 3500) -> Snapshot:
        assert self.page
        await self._settle()
        raw = await self.page.evaluate(SNAPSHOT_JS, [max_elements, max_text])
        snap = Snapshot(raw=raw, **raw)
        self.last_snapshot = snap
        return snap

    async def screenshot(self, label: str = "") -> str | None:
        if not (self.page and self.screenshot_dir):
            return None
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._shot_n += 1
        path = self.screenshot_dir / f"{self._shot_n:03d}{'-' + label if label else ''}.png"
        await self.page.screenshot(path=str(path))
        return str(path)

    # ---------- actions ----------
    async def goto(self, url: str) -> None:
        assert self.page
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout as exc:
            raise BrowserError(f"Timed out loading {url}") from exc

    async def click(self, ref: int) -> None:
        loc = await self._locate(ref)
        await self._highlight(loc)
        try:
            await loc.click(timeout=8000)
        except PWTimeout:
            # Visually-hidden radios/checkboxes: click their label instead.
            try:
                await loc.check(force=True, timeout=4000)
            except Exception as exc:  # noqa: BLE001
                raise BrowserError(f"Could not click element [{ref}]: {exc}") from exc

    async def type_text(self, ref: int, text: str, press_enter: bool = False) -> None:
        loc = await self._locate(ref)
        await self._highlight(loc)
        try:
            await loc.fill(text, timeout=8000)
            if press_enter:
                await loc.press("Enter")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"Could not type into [{ref}]: {exc}") from exc

    async def select_option(self, ref: int, option: str) -> None:
        loc = await self._locate(ref)
        await self._highlight(loc)
        try:
            await loc.select_option(label=option, timeout=8000)
        except Exception:
            try:
                await loc.select_option(value=option, timeout=4000)
            except Exception as exc:  # noqa: BLE001
                raise BrowserError(f"Option '{option}' not found in [{ref}]: {exc}") from exc

    async def set_checked(self, ref: int, checked: bool) -> None:
        loc = await self._locate(ref)
        await self._highlight(loc)
        try:
            await loc.set_checked(checked, timeout=8000, force=True)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"Could not set [{ref}] checked={checked}: {exc}") from exc

    async def scroll(self, direction: str) -> None:
        assert self.page
        dy = {"down": 600, "up": -600}.get(direction)
        if dy is None:
            raise BrowserError("direction must be 'up' or 'down'")
        await self.page.mouse.wheel(0, dy)
        await asyncio.sleep(0.3)

    async def go_back(self) -> None:
        assert self.page
        await self.page.go_back(wait_until="domcontentloaded")

    async def banner(self, text: str | None) -> None:
        """Show (or clear) a caption on the page so viewers see what the agent is doing."""
        if not self.page:
            return
        try:
            await self.page.evaluate(
                """(t) => { let b = document.getElementById('mp-banner');
                   if (!t) { if (b) b.remove(); return; }
                   if (!b) { b = document.createElement('div'); b.id = 'mp-banner'; document.body.appendChild(b); }
                   b.textContent = t; }""",
                text,
            )
        except Exception:  # noqa: BLE001 - cosmetic only
            pass

    # ---------- helpers ----------
    async def _locate(self, ref: int):
        assert self.page
        loc = self.page.locator(f'[data-mp-ref="{int(ref)}"]')
        if await loc.count() == 0:
            raise BrowserError(f"Element [{ref}] no longer exists; take a fresh look at the page")
        return loc.first

    async def _highlight(self, loc) -> None:
        try:
            await loc.scroll_into_view_if_needed(timeout=3000)
            await loc.evaluate(
                "el => { document.querySelectorAll('[data-mp-active]').forEach(n => n.removeAttribute('data-mp-active'));"
                " el.setAttribute('data-mp-active', '1'); }"
            )
        except Exception:  # noqa: BLE001 - cosmetic only
            pass

    async def _settle(self) -> None:
        assert self.page
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            await self.page.wait_for_load_state("networkidle", timeout=3000)
        except PWTimeout:
            pass
