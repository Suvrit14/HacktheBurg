# MyoPilot

**Hands-free web access for people with limited hand movement.**

MyoPilot turns five muscle gestures and the user's voice into full control of a web browser. A muscle-signal (EMG) classifier recognises each gesture, Playwright carries it out in a real Chrome window, and local speech-to-text fills in text fields. A live dashboard shows every step, from the raw signal to the decision.

Built at **HackTheBurgh XIII** (Edinburgh, November 2026).

---

## How it works

```
Camera hand pose  →  Recorded EMG (NinaPro)  →  LDA classifier  →  Gesture command  →  Playwright in Chrome
                                                                                          ↓
                                                           Text field focused  →  Local Whisper voice input
```

| Gesture | Hand pose | Action in the browser |
|---|---|---|
| `scroll_down` | Fist | Smooth scroll down |
| `scroll_up` | Spread fingers | Smooth scroll up |
| `tab` | V sign | Highlight the next button, option or link on screen |
| `click` | Tilt to the little finger | Press the highlighted item |
| `back` | Tilt to the thumb | Go to the previous page |

When `tab` lands on a text field, MyoPilot starts listening. The user says the answer, and it is typed in and formatted: dates, emails, phone numbers and postcodes.

## Features

- **Five-gesture navigation** with a clear, persistent highlight. Only useful targets are reached, starting from what is on screen.
- **Voice input with no cloud calls.** `faster-whisper` runs on the device, with rules for each kind of field.
- **Live dashboard** at `localhost:8800`. It shows the camera feed, the 16-channel EMG signal and heatmap, classifier votes, the decoded command, decode time and logs.
- **GOV.UK Design System demo service**, a Blue Badge–style application form built with the official GOV.UK Frontend components.

## Quick start

**Requirements:** Python 3.12+ with [uv](https://docs.astral.sh/uv/), Node.js 20+, a webcam and a microphone.

```bash
uv sync
cd govuk_form_site && npm install && cd ..
```

**Run with the camera:**

```bash
uv run main.py direct
```

This opens the demo form in Chrome and the dashboard, then starts the camera. Ctrl+C stops everything.

**Run without a camera.** Type gestures in the terminal: `t` tab, `c` click, `d` down, `u` up, `b` back.

```bash
uv run main.py direct --keyboard
```

**Run the automated checks.** These cover the full form, completed with gestures and generated speech.

```bash
uv run main.py test direct
```

## Project structure

| Folder | Contents |
|---|---|
| `camera/` | Hand-pose capture and the link to the browser |
| `emg_data/`, `models/` | EMG processing, the classifier and trained models |
| `gesture_control/` | Gesture actions, voice input, WebSocket server, dashboard |
| `govuk_form_site/` | GOV.UK-style demo service (Express + Nunjucks) |
| `browser_automation_agent/` | Experimental LLM agent mode (LangGraph + AWS Bedrock) |

## Tech stack

Python · Playwright · MediaPipe · scikit-learn · faster-whisper · WebSockets · Node.js · GOV.UK Frontend · LangGraph · AWS Bedrock

## Team

- **Suvrit**: EMG signal processing, classifier and camera pipeline
- **Karthik**: browser automation, voice input, integration and dashboard

## License

[MIT](LICENSE)
