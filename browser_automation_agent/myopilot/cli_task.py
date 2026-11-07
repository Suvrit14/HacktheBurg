"""Run one task from the terminal (dev entry point, no control panel).

    uv run python -m myopilot.cli_task "find a pharmacy near me and tell me its phone number" \
        --start-url https://www.nhs.uk/service-search/pharmacy/find-a-pharmacy --headed

    --fake   use the scripted stand-in model (no AWS needed) to test the pipeline.

When the agent asks something, answer in the terminal (keyboard stand-in for gestures):
choices by number, 'b' for BACK, confirm with y/n, free text by typing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .browser import Browser
from .config import settings
from .events import EventLog
from .graph import TaskAgent
from .models import ModelNotConfigured, get_chat_model
from .profile import load_profile
from .safety import SafetyGuard


async def terminal_interrupt(payload: dict[str, Any]) -> Any:
    kind, q = payload.get("kind"), payload.get("question", "")
    print(f"\n>>> MyoPilot asks: {q}")
    loop = asyncio.get_running_loop()
    read = lambda prompt: loop.run_in_executor(None, input, prompt)  # noqa: E731
    if kind == "choice":
        opts = payload.get("options", [])
        for i, o in enumerate(opts, 1):
            print(f"    {i}. {o}")
        while True:
            a = (await read("    pick number (b = BACK): ")).strip().lower()
            if a == "b":
                return "BACK"
            if a.isdigit() and 1 <= int(a) <= len(opts):
                return opts[int(a) - 1]
    if kind == "confirm":
        a = (await read("    approve? [y/n]: ")).strip().lower()
        return "APPROVED" if a.startswith("y") else "REJECTED"
    return (await read("    your answer (voice stand-in): ")).strip() or "BACK"


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("goal")
    p.add_argument("--start-url", default=None)
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--fake", action="store_true", help="scripted stand-in model, no AWS")
    p.add_argument("--model", default=None, help="override MYOPILOT_MODEL_ID")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--auto", default=None, help="comma-separated option numbers for choice asks (non-interactive tests)")
    p.add_argument("--auto-text", default="", help="canned reply for free-text (voice) asks when --auto is used")
    p.add_argument("--auto-approve", action="store_true", help="approve confirm() asks when --auto is used")
    p.add_argument("--serve-mock", action="store_true", help="serve the local demo form on :8765 during the run")
    args = p.parse_args()

    if args.max_steps:
        settings.max_steps = args.max_steps
    if args.fake:
        from .fake_model import ScriptedPharmacyModel
        llm, model_name = ScriptedPharmacyModel(), "scripted-fake"
    else:
        try:
            llm = get_chat_model(settings, args.model)
        except ModelNotConfigured as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        model_name = args.model or settings.model_id

    handler = terminal_interrupt
    if args.auto is not None:
        canned = [a.strip() for a in args.auto.split(",")]

        async def handler(payload: dict[str, Any]) -> Any:  # noqa: F811
            kind = payload.get("kind")
            if kind == "confirm":
                reply = "APPROVED" if args.auto_approve else "REJECTED"
            elif kind == "text":
                reply = args.auto_text or "BACK"
            else:  # choice: next canned number (default: first option)
                pick = canned.pop(0) if canned else "1"
                opts = payload.get("options") or ["BACK"]
                reply = opts[min(int(pick), len(opts)) - 1] if pick.isdigit() else pick
            print(f">>> auto-answer [{kind}] {payload.get('question')!r} -> {reply!r}")
            return reply

    server_task = None
    if args.serve_mock:  # serve the local demo form for this run
        import uvicorn
        from fastapi import FastAPI

        from .mock_site import router as mock_router
        mock_app = FastAPI()
        mock_app.include_router(mock_router)
        server = uvicorn.Server(uvicorn.Config(mock_app, host="127.0.0.1", port=8765, log_level="warning"))
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.8)

    log = EventLog(settings.runs_dir)
    browser = Browser(headless=not args.headed, slow_mo=150 if args.headed else 0, screenshot_dir=log.dir / "screenshots",
                      channel=settings.browser_channel)
    await browser.start()
    try:
        agent = TaskAgent(browser=browser, guard=SafetyGuard(settings.allowed_domains, settings.live_no_submit),
                          log=log, profile=load_profile(settings.profile_path), llm=llm, settings=settings,
                          model_name=model_name)
        summary = await agent.run(args.goal, handler, start_url=args.start_url)
    finally:
        await browser.close()
        log.close()
        if server_task:
            server.should_exit = True
            await server_task
    print("\n=== RESULT ===")
    print(summary["answer"])
    if summary["follow_ups"]:
        print("Next:", " | ".join(summary["follow_ups"]))
    print("\n=== RUN ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("answer", "follow_ups")}, indent=1))
    print(f"log: {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
