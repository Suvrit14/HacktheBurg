# Gesture control

Gestures drive a real browser with Playwright. The commands come from Suvrit's camera and EMG pipeline over a WebSocket, and they run on the GOV.UK demo form (`govuk_form_site/`).

```
camera/ (Suvrit)                                         gesture_control/ (Karthik)
webcam -> MediaPipe hand pose -> recorded EMG piece   --{"command": "tab"}-->   server.py  ws://localhost:8800
-> EMG classifier -> command                          <--status, result------  -> gestures.py on one open Chrome page
                                                                                -> GOV.UK demo form (localhost:3000)
```

| File | What it does |
|---|---|
| `gestures.py` | One Playwright function per gesture: `scroll_up`, `scroll_down`, `back`, `tab`, `click` |
| `server.py` | WebSocket server. Starts the site if needed, opens Chrome on it, and runs every command it receives |
| `ws_client.py` | Sends commands from the keyboard, for testing without a camera |
| `test_ges.py` | The older one-command-per-run tester (connects to Chrome on port 9333) |
| `../camera/browser_link.py` | The camera's side of the WebSocket (`python -m camera --ws ...`) |

## Run it: two terminals, from the repo root

**Terminal 1: browser side.** This starts the GOV.UK site, opens Chrome on it and listens for commands.
```bash
uv run gesture_control/server.py
```

**Terminal 2: camera side.**
```bash
uv run python -m camera --ws ws://localhost:8800
```
Show a flat, upright hand to calibrate, then make one pose at a time. Relax your hand between poses.

| Pose | Command | What happens on the page |
|---|---|---|
| Fist | `scroll_down` | Scrolls down smoothly |
| Spread fingers | `scroll_up` | Scrolls up smoothly |
| V (index and middle up) | `tab` | Orange box on the next link, button or option (every radio option too) |
| Flat hand tilted to the thumb side | `back` | Previous page |
| Flat hand tilted to the little-finger side | `click` | Presses what the orange box is on |

In the camera window, keys `1`–`5` trigger the poses manually if the camera misses one.

**Without a camera:**
```bash
uv run python -m camera --selftest --ws ws://localhost:8800
```
This sends made-up hands through the whole chain into the browser.

```bash
uv run gesture_control/ws_client.py
```
This lets you type `d`, `u`, `t`, `c` or `b` to send commands yourself.

## Reading the logs

**Terminal 1 (server):** every line has a time and a tag.
```
02:49:18.209  CAMERA  -> scroll_down   (camera saw fist, EMG confidence 0.999)   [from camera]
02:49:18.572  BROWSER #7 scroll_down: scrolled from 907px to 1317px   (363 ms, page /)
02:50:53.165  CLIENT  -> click   [from keyboard]
02:50:53.554  BROWSER #11 click: clicked <a> "Start now", now on http://localhost:3000/who-for   (387 ms, page /who-for)
02:51:16.330  SKIP    tab ignored: still doing the previous command
```
- `SITE`: the GOV.UK site
- `CAMERA` / `CLIENT`: a command arrived, and from whom
- `BROWSER`: what Playwright did, how long it took, and which page you are on now
- `SKIP`: a command was dropped because the browser was still busy
- `ERROR`: something failed

**Terminal 2 (camera):** each pose is shown from start to finish.
```
camera saw fist    -> playing a recording of repetition 5
   command: scroll_down  (camera: fist, confidence 0.999)
   -> browser: sent scroll_down (#1)
   browser: busy -> camera poses paused
   <- browser: scroll_down: scrolled from 907px to 1317px  (363 ms in the browser, 364 ms round trip)
   browser: ready -> camera poses armed
```

## How it behaves
- **One page stays open:** one Chrome window and one page stay open, so commands act immediately. If you close the window, the next command opens a new one.
- **One command at a time:** a command that arrives while another is running is **skipped, not queued**, so there are no surprise actions seconds later. While the browser is busy, the camera also ignores new poses.
- **Reconnecting:** the camera keeps working if the server isn't running yet. It prints its commands and reconnects every 2 seconds.
- **Back:** `back` never leaves the site for the blank page a new tab starts on.
- **Stopping:** Ctrl+C in terminal 1 closes the browser and stops the site it started.

Options for `server.py`:
- `--port 8801` uses another port.
- `--url http://localhost:3000/walking` opens another page.
- `--no-site` doesn't start the site.
- `--headless` hides the window.

## WebSocket messages (JSON)

Client → server:
```json
{"type": "hello", "client": "camera"}
{"type": "command", "command": "tab", "id": 7, "source": "camera", "camera_gesture": "tab", "confidence": 0.998}
```

Server → clients:
```json
{"type": "hello", "commands": ["scroll_up", "scroll_down", "back", "tab", "click"], "state": "ready", "url": "...", "title": "..."}
{"type": "status", "state": "busy", "command": "tab"}
{"type": "result", "id": 7, "command": "tab", "ok": true, "result": "highlighted <a> \"Start now\"", "ms": 4, "url": "...", "title": "..."}
{"type": "result", "id": 8, "command": "tab", "ok": false, "skipped": "busy"}
```
