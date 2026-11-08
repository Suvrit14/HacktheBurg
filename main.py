"""MyoPilot: run the whole browser automation pipeline from the project root.

    uv run main.py            # Chrome + agent + control panel at http://localhost:8765
    uv run main.py --fake     # scripted stand-in model, no AWS needed
    uv run main.py --help     # all options

Settings come from the single .env file in this folder (see .env.example).
"""
import asyncio
import sys
from pathlib import Path

# The pipeline's code lives in browser_automation_agent/myopilot
sys.path.insert(0, str(Path(__file__).resolve().parent / "browser_automation_agent"))

from myopilot.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
