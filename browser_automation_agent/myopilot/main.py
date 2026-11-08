"""Start MyoPilot: headed browser + control panel + switch-access session.

    uv run python -m myopilot.main            # needs MYOPILOT_MODEL_ID + AWS_REGION in .env
    uv run python -m myopilot.main --fake     # scripted stand-in model, no AWS

Then open http://localhost:8765 (in Chrome, for voice) next to the browser window.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

from .browser import Browser
from .config import settings
from .events import EventLog
from .graph import TaskAgent
from .intents import IntentBus
from .models import ModelNotConfigured, get_chat_model
from .panel.app import create_app
from .profile import load_profile
from .safety import SafetyGuard
from .session import Session


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fake", action="store_true", help="scripted stand-in model, no AWS")
    p.add_argument("--model", default=None, help="override MYOPILOT_MODEL_ID")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--emg", action="store_true",
                   help="drive the menus with Suvrit's EMG classifier (replayed held-out recording)")
    p.add_argument("--emg-sequence", default=None,
                   help="gestures to perform, e.g. 'scroll_down,forward' (commands) or 'fist,ulnar' (gestures) "
                        "(default: the model's gestures in the order they appear in the recording)")
    p.add_argument("--emg-speed", type=float, default=1.0, help="replay speed, 1.0 = real time")
    p.add_argument("--emg-repeat", action="store_true",
                   help="forward every repeated scroll command while a gesture is held (default: one per gesture)")
    args = p.parse_args()

    emg_sequence: list[str] = []
    if args.emg:
        from .emg_source import MODEL_PATH, SUBJECT, EMGSetupError, ensure_model, parse_sequence
        try:
            emg_sequence = parse_sequence(args.emg_sequence)
            existed = MODEL_PATH.exists()
            ensure_model()  # trains models/subject<N>_set_a.joblib if it is missing
        except (ValueError, EMGSetupError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"EMG subject {SUBJECT}: model {MODEL_PATH.name}"
              f"{'' if existed else ' (trained now, repetitions 1-5)'}")

    if args.fake:
        from .fake_model import ScriptedPharmacyModel
        llm, options_llm, model_name = ScriptedPharmacyModel(), None, "scripted-fake"
    else:
        try:
            llm = get_chat_model(settings, args.model)
            options_llm = get_chat_model(settings, settings.options_model_id or args.model)
        except ModelNotConfigured as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        model_name = args.model or settings.model_id

    log = EventLog(settings.runs_dir)
    bus = IntentBus()
    bus.on_publish = lambda intent, source: log.emit("intent", intent=intent.value, source=source)
    browser = Browser(headless=args.headless, slow_mo=120, screenshot_dir=log.dir / "screenshots",
                      channel=settings.browser_channel)
    await browser.start()
    profile = load_profile(settings.profile_path)
    agent = TaskAgent(browser=browser, guard=SafetyGuard(settings.allowed_domains, settings.live_no_submit),
                      log=log, profile=profile, llm=llm, settings=settings, model_name=model_name)
    session = Session(browser=browser, bus=bus, log=log, agent=agent, options_llm=options_llm,
                      profile=profile, model_name=model_name)
    log.subscribe(session.on_event)

    server = uvicorn.Server(uvicorn.Config(create_app(session, bus), host="127.0.0.1", port=args.port, log_level="warning"))
    print(f"MyoPilot panel: http://localhost:{args.port}   (log: {log.path})")
    tasks = [server.serve(), session.run()]
    if emg_sequence:
        from .emg_source import EMGReplaySource
        tasks.append(EMGReplaySource(bus=bus, log=log, session=session, sequence=emg_sequence,
                                     speed=args.emg_speed, repeat=args.emg_repeat).run())
        print(f"EMG gestures (Suvrit's classifier, replayed recording): {' -> '.join(emg_sequence)}")
    try:
        await asyncio.gather(*tasks)
    finally:
        await browser.close()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
