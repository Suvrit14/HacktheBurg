# MyoPilot integration contract

How the other parts of the team plug into the browser automation.

## 1. Gesture input (for the EMG classifier)

The automation uses **the same four commands as Suvrit's classifier** (`emg_data/commands.py`), with no translation in between:

| Command       | Gesture (NinaPro DB5 E2)        | Meaning in a menu          |
|---------------|---------------------------------|----------------------------|
| `scroll_down` | fist (6)                        | move the highlight down    |
| `scroll_up`   | abduction of all fingers (5)    | move the highlight up      |
| `forward`     | wrist ulnar deviation (16)      | choose / approve           |
| `back`        | wrist radial deviation (15)     | cancel / go back / reject  |

Rest produces no command. Smoothing (majority vote, confidence threshold 0.9, cooldown) happens in Suvrit's `CommandLogic`.

### Running it with Suvrit's classifier
```bash
uv run main.py --emg                                   # the model's gestures in the order they appear in the recording
uv run main.py --emg --emg-sequence forward,forward    # also accepts gesture names: fist/spread/radial/ulnar
```
`myopilot/emg_source.py` replays a real recorded piece (rest, then the gesture) for each gesture in the sequence. It uses the held-out repetition 6, through `emg_data.interface.replay_commands()`, in real time. Each gesture is performed only when a menu is waiting for input.

- By default a held scroll gesture forwards **one** command; Suvrit's logic repeats it every 0.3 s. Use `--emg-repeat` to forward every repeat.
- **Subject is set in one place: `EMG_SUBJECT` in the root `.env`** (NinaPro DB5 subjects 1–10). The notebook `m1.ipynb`, the model file `models/subject<N>_set_a.joblib`, the recording `data/s<N>/S<N>_E2_A1.mat` and `--emg` all follow it.
- If the model for that subject doesn't exist yet, `--emg` (and the tests) train it automatically, exactly like `m1.ipynb` T2 (repetitions 1–5, repetition 6 held out for the replay). If the recording is missing, it stops with a clear error.

### Other ways to send a command
1. **Same Python process:** `bus.publish("scroll_down", source="emg")`.
2. **HTTP** from any process: `POST http://localhost:8765/intent/scroll_down`.
3. **WebSocket** `ws://localhost:8765/ws`: send `{"intent": "forward", "source": "emg"}`.

Commands that arrive while the agent is busy working are dropped on purpose, so a stray gesture can't trigger a surprise action.

## 2. Event log (for the dashboard)

Every run writes `runs/<run-id>/events.jsonl`, one JSON object per line. The same events are pushed live over the WebSocket as `{"kind": "event", ...}`. Screen state is pushed as `{"kind": "view", ...}`.

Common fields: `type`, `t` (seconds since start), `ts` (ISO time).

| type | Key fields |
|---|---|
| `intent` | `intent` (`scroll_down` / `scroll_up` / `forward` / `back`), `source` (keyboard / emg / http) |
| `emg_command` | `command`, `gesture`, `confidence`, `replay_t`, `intended` |
| `emg_gesture_start` / `emg_gesture_end` | `step`, `intended`; at the end: `decoded`, `forwarded`, `suppressed_repeats`, `ok` |
| `choice` | `mode`, `title`, `index`, `label` (`back` if cancelled) |
| `options_proposed` | `url`, `options` |
| `task_start` / `task_end` | `goal`; at the end: `answer`, `steps`, `elapsed_s`, `input_tokens`, `output_tokens`, `total_tokens`, `llm_calls`, `user_asks`, `safety_blocks`, `model` |
| `llm_call` | `model`, `purpose` (`options` or task), `latency_s`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `tool_calls` |
| `agent_step` | `step`, `tool`, `args`, `explain` (caption), `ok`, `error`, `url_before`, `url`, `page_title`, `latency_s`, `screenshot`, `note` (`from_profile` for autofilled text) |
| `ask` / `answer` | `kind` (`choice` / `text` / `confirm`), `question`, `options`, `reply` |
| `safety_block` | `tool`, `reason`, `url` |
| `result` | `answer`, `follow_ups` |

The `task_end` numbers are the Wordsmith metrics: tokens, model and time for each task.
