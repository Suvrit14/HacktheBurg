"""Control panel server: shows the switch-access menu and live agent log, and
receives the four commands (keyboard now, EMG later) plus dictated text."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..intents import Intent, IntentBus
from ..mock_site import router as mock_router

INDEX = Path(__file__).with_name("index.html")


def create_app(session: Any, bus: IntentBus) -> FastAPI:
    app = FastAPI(title="MyoPilot panel")
    app.include_router(mock_router)  # local demo form at /mock/transport

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX.read_text()

    @app.post("/intent/{name}")
    async def post_intent(name: str) -> dict[str, str]:
        """HTTP hook for other gesture sources (e.g. the classifier process)."""
        bus.publish(Intent(name.upper()), source="http")
        return {"ok": name.upper()}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()

        async def send(msg: dict[str, Any]) -> None:
            await sock.send_text(json.dumps(msg, default=str))

        session.listeners.append(send)
        await send({"kind": "view", **session.view})
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if "intent" in msg:
                    bus.publish(Intent(msg["intent"]), source=msg.get("source", "keyboard"))
                elif "text" in msg:
                    session.text_q.put_nowait(str(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            if send in session.listeners:
                session.listeners.remove(send)

    return app
