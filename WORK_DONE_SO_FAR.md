# MyoPilot: everything done so far

HackTheBurgh XIII, Edinburgh, 7–8 November 2026. Submission deadline: Sunday 12:00.
Status as of Sunday 8 November 2026, early morning. Written for Karthik, Suvrit and anyone who picks the project up.

> This file covers the whole project. The older `browser_automation_agent/docs/work-done-so-far.md` covers only Saturday's LLM agent work in more depth. Where the two differ, this file is current. One example: the commands are now `scroll_up / scroll_down / tab / click / back`, and `forward` is gone.

---

## Contents
1. [What MyoPilot is](#1-what-myopilot-is)
2. [Hackathon tracks and where we stand](#2-hackathon-tracks-and-where-we-stand)
3. [Team, split of work and honesty rule](#3-team-split-of-work-and-honesty-rule)
4. [Timeline: what happened, in order](#4-timeline-what-happened-in-order)
5. [Repository layout today](#5-repository-layout-today)
6. [Part A: the LLM browser agent (LangGraph + AWS Bedrock)](#6-part-a-the-llm-browser-agent-langgraph--aws-bedrock)
7. [Part B: Suvrit's EMG classifier and how we connected it](#7-part-b-suvrits-emg-classifier-and-how-we-connected-it)
8. [Part C: gesture control with Playwright (no LLM)](#8-part-c-gesture-control-with-playwright-no-llm)
9. [Part D: the GOV.UK demo form site](#9-part-d-the-govuk-demo-form-site)
10. [Part E: Suvrit's camera pipeline (MediaPipe)](#10-part-e-suvrits-camera-pipeline-mediapipe)
11. [Part F: joining camera and browser over WebSockets](#11-part-f-joining-camera-and-browser-over-websockets)
12. [Part G: the live dashboard](#12-part-g-the-live-dashboard)
13. [Git history, the merge and how conflicts were resolved](#13-git-history-the-merge-and-how-conflicts-were-resolved)
14. [All test results](#14-all-test-results)
15. [How to run everything](#15-how-to-run-everything)
16. [Settings (.env)](#16-settings-env)
17. [Incidents and lessons](#17-incidents-and-lessons)
18. [Open items and next steps](#18-open-items-and-next-steps)

---

## 1. What MyoPilot is

MyoPilot lets people with motor disabilities, who can't comfortably use a mouse and keyboard, use the web with a few **hand and muscle gestures**.

The full chain we have running today:

```
hand pose in front of a webcam (MediaPipe)
  → picks a real recorded EMG muscle signal for that gesture (NinaPro DB5)
  → the EMG classifier decodes it live, window by window (LDA + voting)
  → a command: scroll_up / scroll_down / tab / click / back
  → sent over a WebSocket
  → Playwright performs it on a real web page (our GOV.UK-style form)
  → everything is shown live on a dashboard (camera, EMG, decoder, logs)
```

Next to that, there's a **LangGraph + AWS Bedrock agent** (`browser_automation_agent/`). It can complete multi-step web tasks on live sites (NHS, GOV.UK) through switch-access menus. It isn't yet connected to the camera → WebSocket chain; that's the main next step (section 18).

---

## 2. Hackathon tracks and where we stand

| Sponsor | Challenge | Our fit today | Why |
|---|---|---|---|
| **HRT** | Best use of data for predictions and decision-making | **Strongest** | Real NinaPro EMG data trains a classifier that predicts gestures. A decision layer then turns the predictions into actions: a 5-window majority vote, a confidence threshold (0.9 for set A, 0.99 for set C), a cooldown and hold-to-repeat. It's measurable: recall per gesture, wrong commands, false triggers per minute of rest, and accuracy on held-out repetitions. The dashboard's decoder panel shows the decision as it happens |
| **G-Research** | Make the invisible visible | **Good, with the dashboard** | Muscle signals are invisible. The dashboard shows the raw 16-channel EMG scrolling, a heatmap of which muscles light up for each gesture, and how the vote turns them into a command |
| **Wordsmith** | Most complicated end-to-end agentic AI browser-use workflow | **Not yet in the live chain** | The LangGraph agent is agentic, but the camera → browser chain is direct control (one gesture = one action). It becomes a strong fit once the agent fills the GOV.UK form from voice, with gestures confirming |
| JetBrains, Bank of Scotland, InstaLily, Viridien | Developer tools, cost of living, coding agents, geology | No | Not what the project does |

The user's original focus: Wordsmith (Karthik's part), plus HRT and G-Research. It is **not** legal-focused.

---

## 3. Team, split of work and honesty rule

- **Suvrit**
  - The EMG classifier: `emg_data/`, `models/`, `m1.ipynb`, `code.ipynb`.
  - The camera pipeline: `camera/`.
  - The recording library: `emg_data/library.py`.
- **Karthik**
  - The browser automation: `browser_automation_agent/`, `gesture_control/`, `govuk_form_site/`.
  - Connecting everything together: `emg_source.py`, `camera/browser_link.py`, the WebSocket server and the dashboard.

**Honesty rule for the demo:** the EMG is **recorded** NinaPro DB5 data, replayed in real time. The classification really happens live, window by window, but the signal is not live. The camera only picks *which* recording plays. The dashboard says "Recorded NinaPro EMG, Subject 1" on the EMG panel.

---

## 4. Timeline: what happened, in order

### Saturday 7 November
1. **Brainstorm.**
   - We read the plan and the sponsor list, and decided to brainstorm before planning.
   - The Wordsmith problem statement said it's judged on workflow complexity plus tokens, model and time.
   - Decision: build on **live websites**, plus one local demo form for the full submit path.
2. **Direction set.**
   - It's an accessibility product, not legal.
   - Interaction is **switch access** (a few commands move through menus).
   - Typing is **layered**: profile autofill → choices → voice for free text.
3. **Stack chosen by the user.**
   - **LangGraph + AWS Bedrock.** The user picks the model.
   - **uv** for Python.
   - Playwright driving the **installed Google Chrome**.
4. **Built the LangGraph agent** (section 6) and tested it on live NHS / GOV.UK pages.
5. **Bedrock model journey.**
   - An empty `AWS_PROFILE` caused errors.
   - Haiku 5.5 needed an inference profile and its agreement was `NOT_AVAILABLE`.
   - We switched to **Amazon Nova Pro** (`amazon.nova-pro-v1:0`, `eu-west-2`), and fixed several Nova quirks.
6. **Organised the code.** All of Karthik's code moved into `browser_automation_agent/`. Pushed as PR #3.
7. **Reviewed Suvrit's classifier** and planned the connection.
8. **Unified the repo.**
   - One uv project at the root: one `pyproject.toml`, `uv.lock`, `.venv`, `.env`, `.env.example` and `.gitignore`.
   - A root `main.py` that runs everything.
9. **Connected the classifier** (`emg_source.py`, `--emg`), with our commands renamed to **Suvrit's command names**.
10. **Tested the classifier on its own**: the saved model's input and output, and the gesture order.
11. **Explained the `m1.ipynb` errors.** One cause: running the notebook with another subject overwrote `models/subject1_set_a.joblib`.
12. **Made the subject a single setting.** `EMG_SUBJECT` in `.env` now drives the notebook, the saved model name and the `--emg` pipeline. Models auto-train when missing.
13. **Tried subjects 2, 3 and 10** (results in section 14).
14. **Removed the hard-coded demo order.** Replays now play gestures **in the order they appear in the recording**.

### Saturday night to Sunday 8 November
15. **Direct control without the LLM.** Five gestures (`scroll_up`, `scroll_down`, `back`, `click`, `tab`), each one Playwright function, in a new `gesture_control/` folder, with `test_ges.py --back`-style testing.
16. **Smooth scrolling** added.
17. **GOV.UK demo form site** (`govuk_form_site/`) built with the official GOV.UK Frontend 6.6.0 Nunjucks macros. It's a disabled parking badge application (demo).
18. **Changed the `tab` gesture** so it reaches every radio option. Browsers' Tab key skips the other options in a radio group.
19. **Fixed the "blinking" highlight.** The `:focus` style disappeared when Chrome lost window focus, so we now use our own persistent marker.
20. **Git.**
    - Commits ended up on `main` by mistake.
    - The `karthik` branch was then pushed through a PR.
    - Conflicts with Suvrit's camera/EMG update were resolved and pushed; **PR #7 merged**.
21. **Tested Suvrit's camera pipeline** (`python -m camera --selftest` passed).
22. **Joined camera and browser over WebSockets.** `gesture_control/server.py` plus `camera/browser_link.py`, with real-time logs in both terminals.
23. **Live dashboard** with four panels: camera, EMG, decoder, logs. It's served by the same server.

---

## 5. Repository layout today

```
HacktheBurgh/
├── main.py                       # runs the LLM agent app (uv run main.py)
├── pyproject.toml, uv.lock       # ONE uv project for everything
├── requirements.txt              # generated: uv export --no-hashes --no-emit-project -o requirements.txt
├── .env, .env.example            # the only env files (.env is gitignored)
├── .gitignore                    # the only ignore file (also ignores node_modules, compiled app.css, runs/, data/)
├── WORK_DONE_SO_FAR.md           # this file
│
├── emg_data/                     # Suvrit: EMG loading, features, LDA model, command logic, replay, library
├── models/                       # saved classifiers: subject1_set_a, subject1_set_c, subject2/3/10_set_a
├── data/                         # NinaPro DB5 recordings s1 … s10 (gitignored; Suvrit keeps them at the repo root)
├── camera/                       # Suvrit: webcam → MediaPipe hand pose → recorded EMG → classifier → commands
│   └── browser_link.py           # Karthik: the camera's WebSocket link to the gesture server
├── m1.ipynb, code.ipynb, *.png   # Suvrit's notebooks and figures
│
├── browser_automation_agent/     # Karthik: the LangGraph + Bedrock agent (section 6)
├── gesture_control/              # Karthik: Playwright gestures, WebSocket server, dashboard (sections 8, 11, 12)
│   ├── gestures.py               # one Playwright function per gesture
│   ├── server.py                 # WebSocket server: site + Chrome + commands + dashboard + logs
│   ├── dashboard.html            # the live dashboard page
│   ├── ws_client.py              # keyboard test client
│   ├── test_ges.py               # one-gesture-per-run tester (Chrome on port 9333)
│   └── README.md
└── govuk_form_site/              # Karthik: GOV.UK-style demo form (Node, Express, Nunjucks)
```

**Dependencies** (`pyproject.toml`):
- boto3, fastapi, ipykernel, joblib, langchain-aws, langchain-core, langgraph
- matplotlib, mediapipe, numpy, opencv-python, pandas, playwright
- python-dotenv, python-multipart, **scikit-learn==1.7.2** (pinned by Suvrit), scipy
- uvicorn[standard], **websockets>=17**
- Dev: pytest

The site also has its own Node packages: express 4, express-session, nunjucks, govuk-frontend 6.6.0, and sass (dev).

---

## 6. Part A: the LLM browser agent (LangGraph + AWS Bedrock)

Everything is in `browser_automation_agent/`. The full detail of this part (as of Saturday) is in `browser_automation_agent/docs/work-done-so-far.md`. Summary:

### 6.1 How it works
- **Switch access** (`session.py`).
  - The home screen lists everyday tasks:
    - Find a pharmacy near me (NHS)
    - Register with a GP (NHS)
    - Check train times (National Rail)
    - Today's weather (BBC)
    - Book community transport (the local demo form)
    - Something else (speak)
  - Gestures move the highlight; one selection starts an autonomous task.
- **LangGraph task agent** (`graph.py`): `setup → agent → act → (human interrupt) → agent … → END`.
  - Each step: the model sees the page as `URL / TITLE / HEADINGS / ELEMENTS [ref] … / PAGE TEXT` and calls exactly one tool.
  - The tools (`tools.py`) are `navigate`, `click`, `type_text`, `select_option`, `set_checkbox`, `scroll`, `go_back`, `ask_choice`, `ask_text`, `confirm` and `finish`.
  - `ask_choice`, `ask_text` and `confirm` pause the graph with `interrupt()`. The session shows a menu and resumes with `Command(resume=…)`.
- **Page snapshot** (`snapshot.js`): numbers the visible interactive elements with `data-mp-ref`. It describes forms (labels, fieldset questions, options, values, errors, hints) and gives the page text, kept token-lean. Only the latest 2 snapshots go to the model in full.
- **Typing in layers:**
  1. Profile autofill from `profiles/demo_user.json`. The fictional persona is Alex Morgan, The British Library, 96 Euston Road, London NW1 2DB. The date of birth is 14 / 03 / 1990.
  2. Choices: dropdowns and radios become menus.
  3. Voice for free text (the Web Speech API in the panel).
- **Control panel** at http://localhost:8765 (FastAPI + WebSocket).
  - It shows menu cards, a live log and metrics (steps, LLM calls, tokens, time, model).
  - Buttons: ▲ scroll_up, ▼ scroll_down, ⇥ tab, ● click, ↩ back.
  - Keys: ↓/→ scroll_down, ↑/← scroll_up, Tab tab, Enter click, Esc back, V voice.
  - `POST /intent/{command}` lets other processes send commands.

### 6.2 Safety, enforced in code
- **SafetyGuard** (`safety.py`):
  - A domain allowlist: nhs.uk, gov.uk, nationalrail.co.uk, bbc.co.uk, wikipedia.org, localhost. A click that leaves these sites is undone.
  - Never types into password, PIN, card, CVV, sort-code or account-number fields.
  - Any submit needs a fresh `confirm()`.
  - With `MYOPILOT_LIVE_NO_SUBMIT=true`, a live site's final submit is **always** blocked.
- **ProvenanceGuard** (`provenance.py`): values must come from the profile, the chosen task or the user's own answers. Nova Pro was seen inventing a travel date, a wheelchair answer and a hospital. On a refusal, the guard returns a ready-made `ask_choice` suggestion.
- **Other:** analytics cookies are always rejected, and only a fictional persona is used.

### 6.3 Bedrock journey and Nova fixes
1. An empty `AWS_PROFILE=` is now treated as unset.
2. Haiku 5.5 needed the inference-profile ID, and the account agreement was `NOT_AVAILABLE`, so we switched to **Nova Pro, eu-west-2**.
3. **Fixes made for Nova Pro:**
   - Snapshot argument bug: the page text was capped at 120 characters; now 3,500.
   - Vague `finish()` answers: the agent keeps whichever text carries more facts.
   - Tool calls written as XML text: parsed back into real tool calls.
   - Schema rename `_Options` → `MenuOptions`.
   - Invented values: stopped by the ProvenanceGuard.
   - Unnecessary scrolling: fixed with prompt guidance.

### 6.4 Live website findings
- **Works under Playwright:**
  - NHS Find a pharmacy and Find a GP.
  - GP registration, using the practice-specific URLs.
  - GOV.UK register-to-vote.
  - National Rail, BBC Weather and Wikipedia.
- **Blocked:** the Edinburgh council bins page and the MOT history check returned 403 or a bot wall, so we dropped them. We don't work around bot walls.
- **Ethics point:** on live multi-page forms, each "Continue" sends that page's data to the real service.

### 6.5 Results (Nova Pro, Google Chrome)
- **NHS pharmacy, 3 runs in a row:** all correct (Boots, St Pancras, phone and opening hours). 4–5 steps, 10.7–13.0 s, about 35–39k tokens.
- **Demo transport form:** 20 steps, 27.2 s, 114k tokens. 10 fields autofilled, 6 user asks, **0 invented values**, and it submitted only after approval.
- **EMG-driven run:** Suvrit's classifier selected the pharmacy task and answered "Yes, near home" with no keyboard.

### 6.6 Command names today
After the merge with Suvrit's update (section 13), the agent uses his five commands:

| Command | In the agent's menus |
|---|---|
| `scroll_down` | Highlight down |
| `scroll_up` | Highlight up |
| `tab` | Highlight to the next option (same as `scroll_down` in menus) |
| `click` | Choose / approve (was `forward`) |
| `back` | Cancel / go back |

---

## 7. Part B: Suvrit's EMG classifier and how we connected it

### 7.1 The classifier (`emg_data/`)
- **Data:** NinaPro DB5, Exercise 2, sampled at 200 Hz with 16 channels from two Myo armbands. Values are integers from −128 to 127.
- **Windows:** 50 samples (250 ms), stepping 25 samples, so a new window arrives every **125 ms**.
- **Features:** MAV, RMS and waveform length per channel, 48 numbers in total. **Classifier:** LDA with equal priors and shrinkage.
- **Gestures and commands** (`emg_data/commands.py`, `COMMAND_FOR_GESTURE`):

| Gesture id | Movement | Command |
|---|---|---|
| 6 | Fist | `scroll_down` (repeats every 0.3 s while held) |
| 5 | Spread all fingers (abduction) | `scroll_up` (repeats while held) |
| 15 | Wrist radial deviation | `back` |
| 16 | Wrist ulnar deviation | `click` (was `forward`) |
| 2 | Index + middle extended, others folded (V) | `tab` (set-C model only) |

- **Two models:**
  - **Set A** (`CLASSES = [0, 6, 5, 15, 16]`): trained on repetitions 1–5; repetition 6 is held out. Uses `TUNED_CONFIG`: vote over 5, minimum confidence 0.9, cooldown 1 s.
  - **Set C** (`CLASSES_SET_C = [0, 6, 5, 15, 16, 2]`, `models/subject1_set_c.joblib`): trained on repetitions 1–4; repetitions 5 and 6 are held out. Settings `TUNED_CONFIG_SET_C`: vote over 5, **minimum confidence 0.99**. The settings are stored inside the model file.
- **Command logic** (`CommandLogic`): a majority vote over the last 5 windows (3 must agree). Windows below the confidence threshold don't vote. Back, click and tab fire once, then need a return to rest and the 1 s cooldown. It also reports the vote state (`leading`, `agree`, `of`, `needed`, `state`), which the dashboard shows.
- **Replay:**
  - `stream.build_replay()` joins "rest then gesture" pieces from one held-out repetition.
  - `stream_windows()` yields one window at a time in real time.
  - `interface.replay_commands()` adds the decided command to each window.
- **Recording library** (`emg_data/library.py`, Suvrit's newer code): `build_replay(["fist", "tab", "radial"])` builds any mix of recordings from repetitions 5 and 6. The camera uses it.

### 7.2 Our connection to the LLM app (`browser_automation_agent/myopilot/emg_source.py`)
- **`--emg`** replays one recorded gesture at a time through `replay_commands()`, in the same uv process. Decoded commands go onto the same `IntentBus` as the keyboard.
- **Default sequence:** the model's gestures **in the order they appear in the recording**. `--emg-sequence` picks your own, for example `click,click` or `fist,ulnar`.
- **Repeats:** only the first command of a held gesture is forwarded (one fist = one `scroll_down`). `--emg-repeat` forwards them all.
- **Timing:** a gesture is performed only when a menu is waiting, so nothing is lost while the agent works.
- **Honesty:** whatever the model decodes is forwarded, even if it's wrong. Both the intended and the decoded command are logged.
- **Subject:** one setting, `EMG_SUBJECT` in `.env`. The model path follows it (`models/subject{N}_set_a.joblib`). If the model is missing, for another subject, or saved with another scikit-learn version, it's **retrained automatically**, exactly like the notebook (reps 1–5 train, rep 6 held out).
- **Data folder:** either `s1…s10` at the repo root (Suvrit's layout) or `data/s1…` (Karthik's). `emg_data/emg.py` now falls back to `data/` when the root folder has no recording.
- **Set A only, for now:** the LLM app's EMG link uses the set-A model, so `tab` comes from the panel or keyboard there. The camera pipeline uses set C, which includes `tab`.

### 7.3 Changes we made inside `emg_data/` (tell Suvrit)

| File | Change | Why |
|---|---|---|
| `stream.py` | `recording_order()` and `gestures_in_recording_order()`; `build_replay(gestures=None)` plays in recording order | The user asked to see the data exactly in its recorded sequence, with no hand-picked demo order |
| `interface.py` | `DEMO_ORDER` removed; `gestures=None` means "all the model's gestures, in recording order". All of Suvrit's newer options were kept (`replay=`, settings from the model, the held-out check) | Same reason |
| `evaluation.py` | `build_fold(gestures=None)` | Same reason |
| `emg.py` | Fallback from `.` to `data/` for the recordings | Works on both machines |
| `stream.py` | `decode_ms` added to every window | The dashboard shows decode time per window |

### 7.4 The notebook (`m1.ipynb`)
- **At the merge we took Suvrit's version** of `m1.ipynb` and `code.ipynb` (he rewrote `m1.ipynb` for the set-C model).
- **What that lost:** our notebook edits, which read `EMG_SUBJECT` from `.env` and played the replay in recording order. The pipeline code itself still follows `EMG_SUBJECT`.
- **Earlier problem:** running the notebook with another subject saved that model as `subject1_set_a.joblib`. It was restored before pushing.

---

## 8. Part C: gesture control with Playwright (no LLM)

`gesture_control/gestures.py` has one function per gesture. Each takes the Playwright page and returns a sentence saying what happened.

| Gesture | Function | What it does |
|---|---|---|
| `scroll_up` / `scroll_down` | `scroll_up(page)` / `scroll_down(page)` | **Smooth** scroll by 80% of the screen. It waits for the browser's `scrollend` event (max 1.5 s), then reports "scrolled from X px to Y px". At the top or bottom it says so straight away |
| `back` | `back(page)` | Browser back. Waits only until the navigation commits, because pages restored from Chrome's back/forward cache never fire "load". Never leaves the site for the blank page a new tab starts on: it says "no previous page" |
| `tab` | `tab(page)` | **Our own focus walk**, not the Tab key: it visits every link, button, field, `summary` and **every radio option** (see below). It skips hidden, disabled and 1-pixel elements, and smooth-scrolls the element to the centre |
| `click` | `click(page)` | Clicks the element `tab` marked. It reports the new page if the click navigated, or "use tab first" if nothing is highlighted |

`apply_gesture(page, "tab")` runs a gesture by name. It's what the WebSocket server calls.

**Why our own `tab`:** browsers move between radio options with the arrow keys. With only Tab and click, a user could only ever choose the first radio. Our walk visits each option, so every GOV.UK radio is reachable.

**The highlight (the "blink" fix):**
- The first version used the CSS `:focus` style. It vanished as soon as the terminal became the active window, because Chrome lost focus.
- `tab` now sets two markers:
  - `data-mp-target` on the element `click` will press.
  - `data-mp-highlight` on the region drawn around it. For a radio or checkbox, that's the whole option (circle plus label), because GOV.UK hides the real input.
- The style is a 4 px orange outline, a soft orange glow and a light yellow fill. It stays until the next `tab`.

**`test_ges.py`** is the first tester: one command per run (`uv run gesture_control/test_ges.py --tab`). It starts Chrome once on debugging port **9333** with its own profile, and later runs reconnect to it. It's superseded by the WebSocket server, which keeps one page open and is much faster.

---

## 9. Part D: the GOV.UK demo form site

The site lives in `govuk_form_site/` and runs at http://localhost:3000. It's a fictional **"Apply for a disabled parking badge (demo)"** service, built to test the gestures and, later, the LLM agent.

### 9.1 Built with the official GOV.UK Design System
- Node + Express + **Nunjucks with the official `govuk-frontend` 6.6.0 macros** (radios, checkboxes, date input, error summary and so on). This matches the official markup and accessibility exactly.
- **Not a GOV.UK service,** so it follows "Using GOV.UK Frontend without GOV.UK branding" (it requires v6.3.0 or later):
  - The **Generic header** plus **Service navigation**, with no crown and no GOV.UK logotype.
  - Sass font override: `"Helvetica Neue", Helvetica, Arial, sans-serif`, never GDS Transport.
  - Own brand colour `#4c2c92`, own `theme-color`, and own icons with the same file names as GOV.UK's.
  - Our own footer, because the GOV.UK footer always shows the crown. It has no Crown copyright and no Open Government Licence.
  - A **phase banner on every page:** "Demo – This is not a real government service. No information leaves this computer."

### 9.2 The service: one thing per page

| URL | Question | Component / pattern |
|---|---|---|
| `/` | Start page | Start button, inset text, warning text, details |
| `/who-for` | Who are you applying for? | Radios |
| `/benefits` | Do you get any of these benefits? | Checkboxes with an exclusive "None of these" |
| `/walking` | How far can you walk without stopping? | Radios (4 options) |
| `/name` | What is your full name? | Text input (Names pattern) |
| `/date-of-birth` | What is your date of birth? | Date input (Dates pattern) |
| `/address` | What is your address? | Addresses pattern |
| `/contact` | How should we contact you? | Radios with a conditional reveal (email or phone) |
| `/condition` | Describe how your condition affects your walking | Character count (500) |
| `/check-answers` | Check your answers, then **Accept and send** | Summary list with Change links |
| `/confirmation` | Application complete | Panel with a reference number like `PB4K7Q9X` |

The footer links to the accessibility statement (required by the 2018 accessibility regulations), cookies (one essential session cookie, so no banner is needed) and privacy. `/start-again` clears your answers, and there's a 404 page.

### 9.3 GOV.UK behaviour
- **Validation:** an error summary at the top that gets focus, `Error:` at the start of the page title, and an inline message on each field. The wording follows the guidance, for example "Enter your full name", "Date of birth must include a month", "Enter a real postcode" and "Select who you are applying for". Only the wrong parts of a date are highlighted.
- **Navigation:** every answer is saved, then the browser is sent to the next page (POST, then GET). Each page has a Back link, and the browser's back works.
- **Change links** return to check answers after saving.
- **After sending,** the answers are cleared, so the form can't be changed or sent twice.
- **Storage:** answers are kept in memory only.

### 9.4 Built for the gestures
- **Left out on purpose:** `<select>` (GOV.UK advises against it, and a gesture can't open it), file upload, autocomplete and timeouts. There's also no National Insurance number, because our safety rules say never to type government IDs.
- **Reachable with `tab` and `click`:** every link, button, radio and checkbox. Text fields only get focus from a gesture; typing comes later from voice or the LLM.

---

## 10. Part E: Suvrit's camera pipeline (MediaPipe)

This is `camera/`, Suvrit's code. Run it with `uv run python -m camera`.

- **What it does:**
  - The webcam feeds **MediaPipe Hand Landmarker** (one hand, 21 landmarks).
  - The pose classifier (`poses.py`) turns those into a pose name.
  - The pose picks a **real recorded EMG piece** for that gesture from repetitions 5 and 6, which the model never trained on.
  - The piece is replayed in real time through the set-C classifier and command logic, which produce a command.
- **Poses → commands:**

| Pose | Command |
|---|---|
| Fist (all four fingers folded) | `scroll_down` |
| Spread (fingers extended and apart) | `scroll_up` |
| V (index and middle extended) | `tab` |
| Flat hand tilted to the thumb side | `back` |
| Flat hand tilted to the little-finger side | `click` |

- **Pose settings:**
  - A finger counts as extended at 1.4× its base distance.
  - Fingers count as spread at 22° or more apart.
  - Radial means a tilt of 30° or more; ulnar means 15° or more; anything over 75° is ignored.
  - Suvrit notes these thresholds were tuned only on **synthetic hands**.
- **Stability:**
  - A pose must hold for 5 frames.
  - "Release first": after a pose fires, the hand must relax (flat, or out of view) before the next pose counts.
  - The neutral tilt is calibrated automatically from 30 frames of a flat, upright hand.
- **Keys:** `1`–`5` trigger the poses manually, `n` recalibrates and `q` quits.
- **Modes:**
  - `--selftest`: made-up hands through the whole chain, no camera.
  - `--check`: a guided accuracy check with your hand. Results go to `camera/check_results.json`.
  - `--post URL`: sends commands to the LLM app's panel.
  - `--ws URL`: ours, section 11.

---

## 11. Part F: joining camera and browser over WebSockets

```
camera (python -m camera --ws ws://localhost:8800)            gesture_control/server.py
  pose → recorded EMG → classifier → command   ──{"type":"command","command":"tab",…}──▶  runs gestures.py on ONE
  frames, EMG windows, recording events        ──(live data for the dashboard)────────▶   open Chrome page (GOV.UK site)
                                               ◀──{"type":"status","state":"busy"/"ready"}
                                               ◀──{"type":"result","result":"highlighted <a> \"Start now\"","ms":4,…}
```

### 11.1 Why WebSockets
The user's choice. The real gains:
- **One connection stays open,** with no reconnecting for each gesture.
- **Two-way messages:** the browser tells the camera busy or ready, and what it did.
- **One Chrome page stays open:** `tab` takes 2–10 ms, compared with reconnecting for every `test_ges.py` command.

Most of the delay is elsewhere. Holding a pose takes 5 frames, and a recording takes about 1–2 s before the decision.

### 11.2 The server (`gesture_control/server.py`)
- **Starting:**
  - Starts the GOV.UK site with `npm start` if it isn't already running, and runs `npm install` first if needed. Its output appears as `SITE` log lines.
  - Opens **Chrome** with Playwright on the site (maximised, page fills the window) and keeps that one page open.
- **Serving:** listens on **ws://localhost:8800**, and serves the **dashboard** on the same port (http://localhost:8800), opening it automatically.
- **Running commands:**
  - Commands run **one at a time**.
  - A command that arrives during another is **skipped, not queued**, so there are no surprise actions later. Each message is read in its own task so this rule really works, even from one connection.
  - Unknown commands get an error.
- **Other behaviour:**
  - If you close the Chrome window, the next command opens a new page.
  - Playwright's sync API always runs on one worker thread.
  - Ctrl+C or a kill signal (SIGTERM) shuts everything down cleanly: the browser closes and the site it started stops.
- **Options:** `--port`, `--url`, `--no-site`, `--headless`, `--no-dashboard`.

### 11.3 The camera side (`camera/browser_link.py`, plus `--ws` in `camera/demo.py`)
- **Connecting:** it connects in the background and reconnects every 2 s, so the camera keeps working (printing commands) while the server is down.
- **Sending:** every decoded command goes to the server.
- **Pausing:** while the browser is **busy**, the camera pipeline is **blocked**, so new poses are ignored.
- **Showing results:** the camera window gets a "browser:" line with the last result.
- **Dashboard data** (quietly, no log line each):
  - The model info on every connection.
  - Every EMG window: the raw 50×16 samples, RMS per channel, label, confidence, true label, vote, command and decode time.
  - Recording start and end.
  - About 12 camera frames per second (JPEG, 480 px wide, with the hand drawn on) plus the current pose.
- **Self-test:** `--selftest --ws …` plays **in real time** and draws the made-up hand, so the whole system can be demoed without a webcam.

### 11.4 Message format
- **Client → server:**
  - `{"type":"hello","client":"camera"|"keyboard"|"dashboard"}`
  - `{"type":"command","command":"tab","id":7,"source":"camera","camera_gesture":"tab","confidence":0.998,"match":true}`. A plain string like `"tab"` also works.
  - Camera data: `{"type":"meta"|"window"|"recording"|"camera",…}`, plus binary JPEG frames.
- **Server → clients:**
  - `{"type":"hello","commands":[…],"state":"ready","url":…,"title":…}`
  - `{"type":"status","state":"busy"|"ready","command":"tab"}`
  - `{"type":"result","id":7,"command":"tab","ok":true,"result":"highlighted <a> \"Start now\"","ms":4,"url":…,"title":…}`, or `"ok":false` with `"skipped":"busy"` or `"error":…`
  - `{"type":"log","time":…,"tag":…,"message":…}`, sent to dashboards only.

### 11.5 Real-time logs
**Server terminal:** one timestamped, colour-coded line per event.
```
02:49:18.209  CAMERA  -> scroll_down   (camera saw fist, EMG confidence 0.999)   [from camera]
02:49:18.572  BROWSER #7 scroll_down: scrolled from 907px to 1317px   (363 ms, page /)
02:51:16.330  SKIP    tab ignored: still doing the previous command
```
The tags are `SITE`, `SERVER`, `CLIENT`, `CAMERA`, `BROWSER`, `SKIP` and `ERROR`.

**Camera terminal:** each pose from start to finish.
```
camera saw fist    -> playing a recording of repetition 5
   command: scroll_down  (camera: fist, confidence 0.999)
   -> browser: sent scroll_down (#1)
   browser: busy -> camera poses paused
   <- browser: scroll_down: scrolled from 907px to 1317px  (363 ms in the browser, 364 ms round trip)
   browser: ready -> camera poses armed
```

### 11.6 Keyboard client
`uv run gesture_control/ws_client.py` is interactive: `d` scroll_down, `u` scroll_up, `t` tab, `c` click, `b` back, `q` quit. It can also send one command and exit, for example with `--tab`.

---

## 12. Part G: the live dashboard

The dashboard is `gesture_control/dashboard.html`. The server serves it at **http://localhost:8800** and opens it automatically. It's one HTML file with no external libraries, so it works offline. It's fed by the same WebSocket.

**Where it lives:** a separate page, not the GOV.UK site (which must stay a clean "real service") and not drawn on the camera window (OpenCV text overlays can't show readable charts). In a demo, put the camera window, the GOV.UK form in Chrome and the dashboard side by side.

| Panel | What it shows |
|---|---|
| **1 · Camera** | Live feed with the tracked hand drawn on (about 12 fps), the current pose, the camera status (calibrating, ready, recording playing or blocked) and fps |
| **2 · EMG** (G-Research) | The **raw 16-channel signal** of the recording, scrolling in real time over 5 s (channels 1–8 and 9–16 are the two armbands). A **16-channel activity heatmap**: the live window's RMS, plus the average per gesture (rest, fist, spread, tab (V), radial, ulnar) by the recording's true labels, so you see which muscles light up for each gesture. Tag: **"Recorded NinaPro EMG, Subject 1 · repetition N"** |
| **3 · Decoder** (HRT) | **Decoded gesture** with vote agreement, for example "4 of 5 windows agree on tab (V) (needs 3)", shown as dots, not the raw LDA probability. The **resulting command**. The **camera label next to the EMG result**, with a **MATCH / MISMATCH** flag. **Decode time per window**: latest, average and max, windows per second (8.0 in real time) and the share of the 125 ms budget, with a small chart. A timeline of camera → EMG → command → browser result |
| **4 · Logs** | The server's log lines, live and colour-coded, including lines from before the dashboard opened (the last 300) |

The header shows the connection state, browser ready or busy, the camera state and the current page.

---

## 13. Git history, the merge and how conflicts were resolved

| Commit / PR | What |
|---|---|
| `ec26066`, PR #1 | Suvrit: Phase 01, gesture classification from EMG |
| `f027970`, PR #3 | Karthik: browser automation agent using an AWS Bedrock model |
| `5382862`, PR #4 | Suvrit: EMG with the live camera (camera/, library.py, set-C model with `tab`, `forward` → `click`) |
| `a3c5dc4`, `cacf50e` | Karthik: "web nav done" (the GOV.UK site, then everything else), committed on `main` by mistake |
| `aef9db3` | Restored the original subject 1 model |
| `970c221` | Tab/click names in comments |
| `a569926` | Merge of Suvrit's update into `karthik`, with click/tab used everywhere |
| `157292a`, **PR #7** | `karthik` merged into `main` on GitHub |

**What went wrong and how it was fixed:**
- The work was committed on `main` instead of a branch, and the push of the old `karthik` branch was rejected.
- Fix: move `karthik` to the work (GitHub's `karthik` was already contained in it), reset local `main` to GitHub's, merge GitHub's `main`, resolve the conflicts, push, then open and merge the PR.

**How the 7 conflicts were resolved:**
- `emg_data/stream.py`, `evaluation.py`, `interface.py`: **kept all of Suvrit's new work** (click/tab, set C as the default, `replay=`, the model's own settings, the held-out check) **and** our recording order. His `DEMO_ORDER` was not brought back.
- `m1.ipynb`, `code.ipynb`: **took Suvrit's versions**.
- `pyproject.toml`: both package lists, with Suvrit's `scikit-learn==1.7.2` pin kept.
- `uv.lock`: regenerated with `uv lock`; `requirements.txt` re-exported.
- **Our pipeline:**
  - `forward` → `click`, and `tab` added (intents, session menus, prompt, panel buttons and keys, docs, tests, launch config).
  - `emg_source.py` uses only the gestures the set-A model knows, finds the data in either folder, and retrains models saved with another scikit-learn version.

**Working on branches from now on:**
- Before each piece of work: `git switch main`, `git fetch origin`, `git reset --hard origin/main`, then `git switch karthik` and `git merge main`.
- Afterwards: commit and push `karthik`, then open a PR.

**Not committed yet:**
- The WebSocket server, the dashboard, `browser_link.py`, the keyboard client and the gesture_control README.
- The `back` fix, the `decode_ms` field and the `demo.py` changes.
- The `websockets` dependency (pyproject, uv.lock, requirements.txt).
- This file.

---

## 14. All test results

### Automated tests: `uv run pytest -q` → **48 passed**
- `test_safety.py`: allowlist, blocked domains, secret fields, submit rules, one approval = one submit.
- `test_provenance.py`: profile, task and answer values allowed; invented values refused.
- `test_emg_link.py`:
  - Our commands equal Suvrit's.
  - Old names (`forward`, NEXT, SELECT) are rejected.
  - Sequence parsing works.
  - The default sequence follows the recording.
  - The model matches `EMG_SUBJECT`.
  - A held fist forwards one `scroll_down` (or all with repeat).
  - Each recorded gesture decodes to its command.

### Classifier experiments

| Subject | Result |
|---|---|
| 1 (set A) | Full replay of 285 windows, all 4 gestures decoded correctly at confidence 0.955–1.000. One held fist gave 7 `scroll_down` (hold-to-repeat) |
| 2 | Window recall: rest 99.9%, spread 81.8%, fist 87.0%, radial 68.4%, ulnar 74.3%. Events on reps 1–5: 20/20 detected, 3 wrong commands, **0 false triggers in 11.5 min of rest**. Rep 6: 4/4, 0 wrong |
| 10 | Replay 42.7 s in recording order spread → fist → radial → ulnar. 5/5 detected per gesture (1, 2, 1, 1 wrong commands). Rep 6 fired scroll_up ×5, scroll_down ×6, back, then click |
| 1 (set C, camera) | `python -m camera --selftest`: all 5 poses gave the right command at confidence 0.998–1.000. A replay in recording order fired tab, scroll_up, scroll_down, back, click. A library mix of fist, tab, radial gave scroll_down, tab, back |

### Gestures on the GOV.UK site (hidden Chrome)
- **Full form run:** radios, checkboxes, links and buttons used **only `tab` and `click`**. Everything passed:
  - All 4 walking options were reachable.
  - Every empty or wrong answer showed the GOV.UK error (summary focused, `Error:` title, inline message).
  - Only the month field was highlighted for a missing month.
  - Change links returned to check answers.
  - Accept and send showed the confirmation with a reference.
  - After sending, the answers were cleared.
- **Look and branding:** no crown or GOV.UK header, Helvetica throughout, the demo banner shown, and the mobile layout works.
- **Highlight:** it stays visible after the window loses focus, and covers the whole radio option.
- **Smooth scrolling:** 0 → 576 px in about 0.45–0.6 s in about 24 steps; "already at the top" is instant.

### WebSocket chain and dashboard
- **Keyboard client:** `tab` → "Start now" (2–10 ms), `click` → `/who-for` (about 0.4–0.9 s), `back` → `/`, then `back` → "no previous page".
- **Skip rule:** a second command sent mid-action was **skipped (busy)**; unknown `jump` gave an error.
- **Camera self-test into the browser:** **passed**. scroll_down and scroll_up took about 360–380 ms in the browser, the rest 3–50 ms, and the round trip added 1–13 ms.
- **Dashboard:**
  - All four panels updated live.
  - The vote showed, for example, "4 of 5 windows agree on tab (V)".
  - Camera vs EMG showed **MATCH**.
  - Decode time was **about 0.3–1 ms per window** against a 125 ms budget (under 1%), at **8.0 windows/s**.
  - The heatmap filled in each gesture's muscle pattern.
- **Shutdown:** the browser closed, the site stopped, and no ports were left in use.

---

## 15. How to run everything

From the repo root (`/Users/karthikmandigiri/HacktheBurgh`):

**Once:**
```bash
uv sync
```
```bash
cd govuk_form_site && npm install && cd ..
```

**Camera → browser automation, with the dashboard:**

Terminal 1, browser side. This starts the site, Chrome and the dashboard. `--url` is needed while the default address in server.py is broken (section 18).
```bash
uv run gesture_control/server.py --url http://localhost:3000
```

Terminal 2, camera side:
```bash
uv run python -m camera --ws ws://localhost:8800
```
Or without a webcam (real time, with a drawn hand):
```bash
uv run python -m camera --selftest --ws ws://localhost:8800
```
Or type the commands yourself:
```bash
uv run gesture_control/ws_client.py
```

**Only the GOV.UK site:**
```bash
cd govuk_form_site && npm start
```
Then open http://localhost:3000. If port 3000 is busy, use `PORT=4000 npm start`.

**Only the camera:**
```bash
uv run python -m camera --selftest
```
```bash
uv run python -m camera
```
```bash
uv run python -m camera --check
```

**The LLM agent app:**
```bash
uv run main.py
```
This is the full app with Bedrock (panel at http://localhost:8765). Two variations:
```bash
uv run main.py --fake
```
`--fake` uses no AWS, with a scripted stand-in model (pharmacy task only).
```bash
uv run main.py --emg --emg-sequence click,click
```
This drives the app with Suvrit's classifier.

**The old one-command tester:**
```bash
uv run gesture_control/test_ges.py --open http://localhost:3000
```
```bash
uv run gesture_control/test_ges.py --tab
```

**Tests:**
```bash
uv run pytest -q
```

---

## 16. Settings (.env)

| Variable | Current value | Meaning |
|---|---|---|
| `MYOPILOT_MODEL_ID` | `amazon.nova-pro-v1:0` | Bedrock model for the LLM agent (the user's choice) |
| `MYOPILOT_OPTIONS_MODEL_ID` | (empty) | Optional cheaper model for page-option menus |
| `AWS_REGION` | `eu-west-2` | Region where the model is enabled |
| `AWS_PROFILE` | (empty) | Optional named AWS profile; empty means unset |
| `MYOPILOT_LIVE_NO_SUBMIT` | `true` | Never press the final submit on live sites |
| `MYOPILOT_MAX_STEPS` | `40` | Step cap per task |
| `MYOPILOT_BROWSER` | `chrome` | Installed Google Chrome (`chromium` = Playwright's own) |
| `EMG_SUBJECT` | `10` | NinaPro subject for the LLM app's EMG link (`.env.example` says 2). The camera pipeline uses its own model, set C for subject 1 |
| `MYOPILOT_STRICT_PROVENANCE` | (default `true`) | Block values the user or profile didn't give |

---

## 17. Incidents and lessons

- **The "blinking" highlight:** `:focus` disappears when Chrome isn't the active window, so we use our own marker (section 8).
- **Port 3000 taken:** my preview copy of the GOV.UK site was still running when the user ran `npm start`. I stopped it, and the server now reuses a running site instead of starting a second one.
- **The user's test Chrome was closed by mistake:**
  - During a test, my hidden Chrome listened on the IPv6 address while the user's `test_ges.py` window had port 9333 on IPv4.
  - The test commands drove the user's window (Wikipedia), and the final `--close` closed it.
  - It was only the separate test profile, not their main Chrome. Lesson: check that a port is free before testing.
- **Background test servers ignore Ctrl+C:** the server now also stops cleanly on SIGTERM.
- **First `back` left the site:** a new tab's first history entry is `about:blank`, so `back` now refuses to go there.
- **Commands from one connection were queued instead of skipped:** fixed by reading each message in its own task.
- **Model files and scikit-learn versions:** models saved with 1.9.1 warn under 1.7.2, so the EMG link retrains them automatically.

---

## 18. Open items and next steps

**To fix now**
- `gesture_control/server.py` has `DEFAULT_URL = "https:/"`, which looks like an accidental edit. It should be `http://localhost:3000`. Until it's fixed, start the server with `--url http://localhost:3000`.
- Commit and push the WebSocket, dashboard and related work (section 13), and tell Suvrit about the changes in `camera/demo.py`, `camera/browser_link.py` and `emg_data/stream.py`.

**For Wordsmith (the main gap)**
- Connect the LangGraph agent to the GOV.UK form and the gesture server:
  - The agent fills the form from **voice** (name, date of birth, address and so on).
  - It plans the steps across the multi-page form and handles validation errors.
  - Gestures move, confirm or correct through switch-access menus.
- Report the agent's steps, tokens and time on the dashboard, since Wordsmith judges these.
- Cut tokens per task: the form task used about 114k because each step resends the whole page.

**For G-Research and HRT**
- Show Suvrit's evaluation numbers in the demo and write-up (recall, wrong commands, false triggers per minute, held-out accuracy).
- Optional: a short "how the decision is made" view (threshold → vote → cooldown) and a mismatch counter over a session.

**Other**
- **Untested with a real webcam:** the camera → browser chain with your real hand has only been checked with the self-test. Run `python -m camera --check` to see how reliable the poses are, because radial and ulnar are the hardest on a 2D camera.
- **`tab` in the LLM app:** the LLM app's EMG link still uses set A, so `tab` there comes from the panel or keyboard. Moving it to set C would add `tab` from EMG.
- **Within-subject only:** the classifier works per subject (Subject 1 for the camera), so make no cross-subject claims.
- **Live forms:** decide whether live multi-page forms (GP registration, register-to-vote) are acceptable to demo; each Continue sends data to the real service. The local GOV.UK demo form is the safe showcase.
- **Pitch evidence:** UK disability statistics and research on everyday online tasks (ONS, Ofcom).
