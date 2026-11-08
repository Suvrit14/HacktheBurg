"""Gesture server: commands over a WebSocket -> Playwright gestures on the GOV.UK demo form.

    uv run gesture_control/server.py                 starts the site (if needed), opens Chrome on it,
                                                     and listens on ws://localhost:8800
    uv run gesture_control/server.py --url http://localhost:3000/walking
    uv run gesture_control/server.py --port 8801 --headless

Who sends commands:
    the camera pipeline   uv run python -m camera --ws ws://localhost:8800
    a keyboard client     uv run gesture_control/ws_client.py

Live dashboard: http://localhost:8800 (opened automatically). The camera also sends its frames, every
decoded EMG window and the recording events; the server passes them on to every dashboard, together
with the commands, the browser results and these log lines.

One Chrome window and one page stay open the whole time, so every command acts on the
same page straight away (no reconnecting per command like test_ges.py).

Messages (JSON, one per WebSocket message):
  client -> server
    {"type": "hello", "client": "camera"}
    {"type": "command", "command": "tab", "source": "camera", "id": 7,
     "camera_gesture": "tab", "confidence": 0.998, "match": true}      (extra fields are only logged)
    A plain string like "tab" also works.
  server -> every client
    {"type": "hello", "commands": [...], "state": "ready", "url": "...", "title": "..."}
    {"type": "status", "state": "busy" | "ready", "command": "tab"}
    {"type": "result", "id": 7, "command": "tab", "ok": true, "result": "highlighted <a> \\"Start now\\"",
     "ms": 42, "url": "...", "title": "..."}
    {"type": "result", ..., "ok": false, "skipped": "busy"}          (a command arrived during another one)
    {"type": "error", "error": "..."}

Commands are run one at a time. A command that arrives while another is running is skipped
(not queued), so a person never gets a surprise action seconds later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from gestures import GESTURES, apply_gesture

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "govuk_form_site"
DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"
DEFAULT_URL = "https:/"

# ------------------------------------------------------------------ logging

_COLOURS = {"SERVER": "90", "SITE": "35", "CLIENT": "34", "CAMERA": "36", "BROWSER": "32", "SKIP": "33", "ERROR": "31"}
_USE_COLOUR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


_log_listeners: list = []  # functions(stamp, tag, message): the dashboard gets every log line too


def log(tag: str, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    label = f"{tag:<7}"
    if _USE_COLOUR:
        label = f"\033[{_COLOURS.get(tag, '0')}m{label}\033[0m"
    print(f"{stamp}  {label} {message}", flush=True)
    for listener in _log_listeners:
        listener(stamp, tag, message)


# ------------------------------------------------------------------ the website

def site_is_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except OSError:
        return False


def start_site(url: str) -> subprocess.Popen | None:
    """Start the GOV.UK demo site with npm if it is not already running. Returns the process we started."""
    if site_is_up(url):
        log("SITE", f"already running at {url}")
        return None
    if not (SITE_DIR / "node_modules").exists():
        log("SITE", "installing the site's packages (npm install) ...")
        subprocess.run(["npm", "install"], cwd=SITE_DIR, check=True, stdout=subprocess.DEVNULL)
    log("SITE", f"starting the GOV.UK demo site (npm start in {SITE_DIR.name}/) ...")
    proc = subprocess.Popen(["npm", "start"], cwd=SITE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)

    def pipe_output():
        for line in proc.stdout:
            line = line.rstrip()
            if line and not line.startswith(">"):
                log("SITE", line)

    threading.Thread(target=pipe_output, daemon=True).start()
    for _ in range(100):
        if site_is_up(url):
            return proc
        if proc.poll() is not None:
            sys.exit("the site did not start (see the SITE lines above)")
        time.sleep(0.2)
    sys.exit(f"the site did not answer at {url}")


def stop_site(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)  # npm and the node server it started
    except ProcessLookupError:
        pass
    log("SITE", "stopped")


# ------------------------------------------------------------------ the browser

class Browser:
    """One Chrome window driven by Playwright. Playwright's sync API must stay on one thread,
    so every call goes through a single worker thread."""

    def __init__(self, url: str, headless: bool) -> None:
        self.url = url
        self.headless = headless
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        self._pw = self._browser = self.page = None

    # --- run on the worker thread
    def _start(self) -> tuple[str, str]:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(channel="chrome", headless=self.headless,
                                                 args=["--start-maximized"])
        self._open_page()
        return self.page.url, self.page.title()

    def _open_page(self) -> None:
        self.page = self._browser.new_page(no_viewport=True)  # page fills the window
        self.page.goto(self.url)

    def _run(self, command: str) -> dict:
        if self.page is None or self.page.is_closed():  # the window was closed by hand: open a new one
            self._open_page()
        started = time.perf_counter()
        result = apply_gesture(self.page, command)
        return {"result": result, "ms": round((time.perf_counter() - started) * 1000),
                "url": self.page.url, "title": self.page.title()}

    def _where(self) -> tuple[str, str]:
        if self.page is None or self.page.is_closed():
            return "", ""
        return self.page.url, self.page.title()

    def _close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    # --- called from asyncio
    async def start(self) -> tuple[str, str]:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._start)

    async def run(self, command: str) -> dict:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._run, command)

    async def where(self) -> tuple[str, str]:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._where)

    def close(self) -> None:
        try:
            self.pool.submit(self._close).result(timeout=10)
        except Exception:  # noqa: BLE001 - shutting down anyway
            pass
        self.pool.shutdown(wait=False)


# ------------------------------------------------------------------ the WebSocket server

def short_path(url: str) -> str:
    """http://localhost:3000/name?change=true -> /name?change=true"""
    parts = urlparse(url)
    return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")


class GestureServer:
    def __init__(self, browser: Browser) -> None:
        self.browser = browser
        self.clients: dict = {}  # connection -> client name
        self.dashboards: set = set()
        self.busy = False
        self.count = 0
        self._tasks: set = set()
        self.meta: dict | None = None            # the camera's model info, for dashboards that connect later
        self.history: deque = deque(maxlen=300)  # recent log lines, for dashboards that connect later
        self.loop: asyncio.AbstractEventLoop | None = None
        _log_listeners.append(self._on_log)

    # --- dashboards
    def _on_log(self, stamp: str, tag: str, message: str) -> None:
        entry = {"type": "log", "time": stamp, "tag": tag, "message": message}
        self.history.append(entry)
        if self.loop is not None and self.dashboards:  # log() can be called from other threads
            self.loop.call_soon_threadsafe(self._spawn, self.to_dashboards(json.dumps(entry)))

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def to_dashboards(self, data: str | bytes) -> None:
        for ws in list(self.dashboards):
            try:
                await ws.send(data)
            except ConnectionClosed:
                self.dashboards.discard(ws)

    async def on_data(self, msg: dict, raw: str) -> None:
        # Live data from the camera (not commands): pass it on to the dashboards
        kind = msg.get("type")
        if kind == "meta":
            self.meta = msg
            log("CAMERA", f"decoder: {msg.get('model')} (NinaPro subject {msg.get('subject')}, "
                          f"vote {msg.get('config', {}).get('vote_window')} windows, "
                          f"min confidence {msg.get('config', {}).get('min_confidence')})")
        elif kind == "recording" and msg.get("kind") == "start":
            log("CAMERA", f"camera saw {msg.get('camera_gesture')} -> playing recorded EMG "
                          f"(repetition {msg.get('repetition')}), expecting {msg.get('expected_command')}")
        elif kind == "recording" and msg.get("kind") == "end":
            mark = "" if msg.get("match") else "   MISMATCH"
            log("CAMERA", f"recording finished: decoded {msg.get('decoded_commands')}{mark}")
        await self.to_dashboards(raw)

    async def broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except ConnectionClosed:
                pass

    async def handler(self, ws) -> None:
        name = f"client@{ws.remote_address[1]}" if ws.remote_address else "client"
        self.clients[ws] = name
        url, title = await self.browser.where()
        await ws.send(json.dumps({"type": "hello", "commands": list(GESTURES),
                                  "state": "busy" if self.busy else "ready", "url": url, "title": title}))
        log("CLIENT", f"{name} connected ({len(self.clients)} connected)")
        try:
            async for raw in ws:
                if isinstance(raw, bytes):  # a camera frame: straight to the dashboards
                    await self.to_dashboards(raw)
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    msg = None
                if isinstance(msg, dict) and msg.get("type") in ("window", "camera", "recording", "meta"):
                    await self.on_data(msg, raw)  # live data: in order, no log line per window
                    continue
                # Handle commands in their own task, so the next message is read straight away:
                # a command that arrives while another is running is then skipped, not queued.
                self._spawn(self.on_message(ws, raw))
        except ConnectionClosed:
            pass
        finally:
            name = self.clients.pop(ws, name)
            self.dashboards.discard(ws)
            log("CLIENT", f"{name} disconnected ({len(self.clients)} connected)")

    async def on_message(self, ws, raw) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            msg = {"type": "command", "command": str(raw).strip()}
        if isinstance(msg, str):
            msg = {"type": "command", "command": msg}

        if msg.get("type") == "hello":
            old = self.clients.get(ws)
            self.clients[ws] = msg.get("client") or old
            log("CLIENT", f"{old} is the {self.clients[ws]}")
            if self.clients[ws] == "dashboard":
                self.dashboards.add(ws)
                for entry in list(self.history):
                    await ws.send(json.dumps(entry))
                if self.meta is not None:
                    await ws.send(json.dumps(self.meta))
            return
        if msg.get("type") != "command":
            await ws.send(json.dumps({"type": "error", "error": f"unknown message type {msg.get('type')!r}"}))
            return
        await self.run_command(ws, msg)

    async def run_command(self, ws, msg: dict) -> None:
        command = str(msg.get("command", "")).strip().lower().replace("-", "_")
        source = msg.get("source") or self.clients.get(ws, "client")
        reply = {"type": "result", "id": msg.get("id"), "command": command}

        details = []
        if msg.get("camera_gesture"):
            details.append(f"camera saw {msg['camera_gesture']}")
        if msg.get("confidence") is not None:
            details.append(f"EMG confidence {float(msg['confidence']):.3f}")
        if msg.get("match") is False:
            details.append(f"MISMATCH, expected {msg.get('expected_command')}")
        tag = "CAMERA" if source == "camera" else "CLIENT"
        log(tag, f"-> {command or '?'}" + (f"   ({', '.join(details)})" if details else "") + f"   [from {source}]")

        if command not in GESTURES:
            log("ERROR", f"unknown command {command!r}; known: {', '.join(GESTURES)}")
            await ws.send(json.dumps({**reply, "ok": False, "error": f"unknown command {command!r}"}))
            return
        if self.busy:
            log("SKIP", f"{command} ignored: still doing the previous command")
            await ws.send(json.dumps({**reply, "ok": False, "skipped": "busy"}))
            return

        self.busy = True
        self.count += 1
        await self.broadcast({"type": "status", "state": "busy", "command": command})
        try:
            done = await self.browser.run(command)
            log("BROWSER", f"#{self.count} {command}: {done['result']}   "
                           f"({done['ms']} ms, page {short_path(done['url'])})")
            await self.broadcast({**reply, "ok": True, **done})
        except Exception as error:  # noqa: BLE001 - report it and keep serving
            log("ERROR", f"{command} failed: {error}")
            await self.broadcast({**reply, "ok": False, "error": str(error)})
        finally:
            self.busy = False
            await self.broadcast({"type": "status", "state": "ready", "command": command})


def serve_dashboard(connection, request):
    # Plain HTTP requests get the dashboard page; WebSocket upgrades go on to the handler
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    path = request.path.split("?", 1)[0]
    if path in ("/", "/dashboard"):
        body = DASHBOARD_FILE.read_bytes()  # read every time, so edits show on reload
        headers = Headers([("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body))),
                           ("Cache-Control", "no-store")])
        return Response(200, "OK", headers, body)
    return Response(404, "Not Found", Headers([("Content-Length", "9")]), b"not found")


async def main_async(args) -> None:
    site = None if args.no_site else start_site(args.url)
    browser = Browser(args.url, args.headless)
    try:
        log("BROWSER", f"opening Chrome on {args.url} ...")
        url, title = await browser.start()
        log("BROWSER", f"ready on {short_path(url)}  \"{title}\"")
        server = GestureServer(browser)
        server.loop = asyncio.get_running_loop()
        async with serve(server.handler, "localhost", args.port, process_request=serve_dashboard,
                         max_size=4 * 1024 * 1024):
            log("SERVER", f"listening on ws://localhost:{args.port}  (commands: {', '.join(GESTURES)})")
            dashboard_url = f"http://localhost:{args.port}"
            log("SERVER", f"dashboard: {dashboard_url}")
            if not (args.no_dashboard or args.headless):
                webbrowser.open(dashboard_url)
            log("SERVER", f"camera:   uv run python -m camera --ws ws://localhost:{args.port}")
            log("SERVER", f"keyboard: uv run gesture_control/ws_client.py --url ws://localhost:{args.port}")
            log("SERVER", "Ctrl+C to stop")
            stop = asyncio.Event()  # Ctrl+C or a kill signal: shut down cleanly (browser and site too)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await stop.wait()
            log("SERVER", "stopping ...")
    finally:
        browser.close()
        log("BROWSER", "closed")
        stop_site(site)


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket -> Playwright gestures on the GOV.UK demo form")
    parser.add_argument("--port", type=int, default=8800, help="WebSocket port (default 8800)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"page to open (default {DEFAULT_URL})")
    parser.add_argument("--no-site", action="store_true", help="do not start the GOV.UK site automatically")
    parser.add_argument("--headless", action="store_true", help="hide the Chrome window (for tests)")
    parser.add_argument("--no-dashboard", action="store_true", help="do not open the dashboard page automatically")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    except OSError as error:
        sys.exit(f"could not start: {error} (is another server already using port {args.port}?)")


if __name__ == "__main__":
    main()
