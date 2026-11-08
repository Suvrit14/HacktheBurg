# MyoPilot

An AI browser agent for people with limited hand movement. Four gesture commands (scroll_down, scroll_up, forward, back), the same ones Suvrit's classifier emits, drive an agent that completes everyday tasks on **live websites** such as NHS and GOV.UK. Muscle (EMG) gestures come from the classifier (`uv run main.py --emg`); the keyboard works as a stand-in.

Built with **LangGraph** and **AWS Bedrock**, with **Playwright** driving Google Chrome.

This folder is the browser automation part of the HacktheBurgh repo. The EMG classifier lives at the repo root (`emg_data/`). The whole repo is **one uv project**: one `pyproject.toml`, one `.venv`, one `.env` and one `.gitignore`, all at the repo root. All commands below run **from the repo root**.

## Setup
```bash
uv sync
uv run playwright install chromium
cp .env.example .env   # then set MYOPILOT_MODEL_ID and AWS_REGION
```

## Run
```bash
# Full app: Chrome + agent + control panel at http://localhost:8765 (open it in Chrome for voice)
uv run main.py

# No AWS needed: scripted stand-in model (only does the pharmacy task) to test the pipeline
uv run main.py --fake

# One task from the terminal (dev)
PYTHONPATH=browser_automation_agent uv run python -m myopilot.cli_task "find a pharmacy near me and tell me its phone number" \
  --start-url https://www.nhs.uk/service-search/pharmacy/find-a-pharmacy --headed
PYTHONPATH=browser_automation_agent uv run python -m myopilot.cli_task "book community transport for a hospital appointment" \
  --start-url http://localhost:8765/mock/transport --serve-mock --headed
```
Panel keys: ↓/→ scroll_down · ↑/← scroll_up · Enter forward · Esc back · V voice (in text questions).

## How it works
- **Switch access:** the home screen lists everyday tasks. On any other page the LLM proposes 3–4 next actions. One `forward` starts a multi-step task.
- **Typing in layers:**
  1. Fill fields from the saved profile (`profiles/demo_user.json`, a fictional persona).
  2. Turn questions into choices (`ask_choice`).
  3. Use voice only for free text (`ask_text`).
- **LangGraph:** `setup → agent → act → (human interrupt) → agent … → END`. Each wait for the user is an `interrupt()`, resumed by a gesture (`myopilot/graph.py`).
- **Safety in code** (`myopilot/safety.py`):
  - The agent can only visit allowlisted sites.
  - It never types passwords, PINs or card details.
  - A submit needs user approval through `confirm()`.
  - On live sites the final submit is always blocked (demo mode).
- **Logging:** every step goes to `runs/<id>/events.jsonl` with tokens, model, latency and a screenshot. See [docs/integration.md](docs/integration.md).

## Tests
```bash
uv run pytest -q                                              # safety + provenance rules
uv run browser_automation_agent/tests/walk_mock_form.py       # demo form + browser layer + approve-then-submit
```

## Layout
| Path | What it does |
|---|---|
| `myopilot/graph.py` | LangGraph task agent and driver |
| `myopilot/session.py` | Switch-access screens and menus |
| `myopilot/browser.py`, `snapshot.js` | Playwright wrapper and token-lean page snapshot |
| `myopilot/tools.py` | Tool schemas the LLM sees |
| `myopilot/options.py` | Home tasks and per-page option proposals |
| `myopilot/intents.py` | Command bus (scroll_down / scroll_up / forward / back) |
| `myopilot/emg_source.py` | Suvrit's EMG classifier as a command source (`--emg`) |
| `myopilot/panel/` | Control panel (FastAPI + WebSocket) |
| `myopilot/mock_site.py` | Local demo form for the full submit path |
