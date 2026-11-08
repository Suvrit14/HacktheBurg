# MyoPilot integration contract

How the other parts of the team plug into the browser automation.

## 1. Gesture input (for the EMG classifier)

The automation only understands **four commands**:

| Intent   | Meaning in a menu              | Suggested gesture (to be confirmed by M0) |
|----------|--------------------------------|-------------------------------------------|
| `NEXT`   | move the highlight forward     | hand close                                |
| `PREV`   | move the highlight back        | hand open                                 |
| `SELECT` | choose / approve               | wrist movement A                          |
| `BACK`   | cancel / go back / reject      | wrist movement B                          |

"Rest" is the absence of a command. Debounce before publishing: majority vote over a few windows, a confidence threshold, and a cooldown after each command.

There are three ways to send a command. Use whichever suits the classifier process:

1. **Same Python process.** `ClassifierSource(bus, mapping).on_prediction(label, confidence)` in `myopilot/intents.py`, or call `bus.publish("NEXT", source="emg")` directly.
2. **HTTP** from any process: `POST http://localhost:8765/intent/NEXT`
3. **WebSocket** `ws://localhost:8765/ws`: send `{"intent": "NEXT", "source": "emg"}`

Commands that arrive while the agent is busy working are dropped on purpose, so a stray gesture can't trigger a surprise action.

## 2. Event log (for the dashboard)

Every run writes `runs/<run-id>/events.jsonl`, one JSON object per line. The same events are pushed live over the WebSocket as `{"kind": "event", ...}`. Screen state is pushed as `{"kind": "view", ...}`.

Common fields: `type`, `t` (seconds since start), `ts` (ISO time).

| type | Key fields |
|---|---|
| `intent` | `intent`, `source` (keyboard / emg / http) |
| `choice` | `mode`, `title`, `index`, `label` (`BACK` if cancelled) |
| `options_proposed` | `url`, `options` |
| `task_start` / `task_end` | `goal`; at the end: `answer`, `steps`, `elapsed_s`, `input_tokens`, `output_tokens`, `total_tokens`, `llm_calls`, `user_asks`, `safety_blocks`, `model` |
| `llm_call` | `model`, `purpose` (`options` or task), `latency_s`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `tool_calls` |
| `agent_step` | `step`, `tool`, `args`, `explain` (caption), `ok`, `error`, `url_before`, `url`, `page_title`, `latency_s`, `screenshot`, `note` (`from_profile` for autofilled text) |
| `ask` / `answer` | `kind` (`choice` / `text` / `confirm`), `question`, `options`, `reply` |
| `safety_block` | `tool`, `reason`, `url` |
| `result` | `answer`, `follow_ups` |

The `task_end` numbers are the Wordsmith metrics: tokens, model and time for each task.
