"""WebSocket link from the camera pipeline to the gesture server (gesture_control/server.py).

    link = BrowserLink("ws://localhost:8800", on_state=...)
    link.send_command(event)      # a command event from CameraPipeline.on_command
    link.state                    # "disconnected", "ready" or "busy"
    link.last_reply               # what the browser did last, e.g. 'tab: highlighted <a> "Start now"'
    link.wait_ready(timeout)      # (tests) wait until the browser has finished its command
    link.send_data(message)       # live data for the dashboard (EMG windows, camera pose); dropped if not connected
    link.send_frame(jpeg_bytes)   # one camera frame for the dashboard (binary WebSocket message)

It connects in the background and reconnects every 2 seconds if the server is not there,
so the camera keeps working (commands are still printed) while the browser side is down.
"""

import json
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


class BrowserLink:
    def __init__(self, url, on_state=None, log=print, client_name="camera", on_connect=None):
        # on_state(state): called when the browser becomes "busy", "ready" or "disconnected"
        # on_connect(): called after every (re)connection, e.g. to send the dashboard what model is used
        self.url = url
        self.on_state = on_state
        self.on_connect = on_connect
        self.log = log
        self.client_name = client_name
        self.state = "disconnected"
        self.last_reply = "-"
        self._ws = None
        self._send_lock = threading.Lock()
        self._sent_at = {}
        self._next_id = 0
        self._idle = threading.Event()
        self._idle.set()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ sending
    def send_command(self, event):
        # event: a command event from CameraPipeline (command, camera_gesture, confidence, match, ...)
        if self._ws is None:
            self.log(f"   -> browser: NOT sent ({event['command']}), no connection to {self.url}")
            return False
        self._next_id += 1
        message = {
            "type": "command", "id": self._next_id, "source": self.client_name,
            "command": event["command"],
            "camera_gesture": event.get("camera_gesture"),
            "expected_command": event.get("expected_command"),
            "confidence": event.get("confidence"),
            "match": event.get("match"),
        }
        try:
            with self._send_lock:
                self._ws.send(json.dumps(message))
        except (ConnectionClosed, OSError) as error:
            self.log(f"   -> browser: NOT sent ({event['command']}): {error}")
            return False
        self._sent_at[self._next_id] = time.perf_counter()
        self._idle.clear()
        self.log(f"   -> browser: sent {event['command']} (#{self._next_id})")
        return True

    @property
    def connected(self):
        return self._ws is not None

    def send_data(self, message):
        # Dashboard data: sent quietly (no log line), and simply dropped while there is no connection
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(json.dumps(message, separators=(",", ":")))
            return True
        except (ConnectionClosed, OSError):
            return False

    def send_frame(self, jpeg_bytes):
        # One camera frame (JPEG) as a binary WebSocket message
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(jpeg_bytes)
            return True
        except (ConnectionClosed, OSError):
            return False

    def wait_ready(self, timeout=10.0):
        return self._idle.wait(timeout)

    def close(self):
        self._stop = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ receiving
    def _set_state(self, state):
        if state != self.state:
            self.state = state
            if self.on_state is not None:
                self.on_state(state)

    def _run(self):
        warned = False
        while not self._stop:
            try:
                with connect(self.url, open_timeout=2) as ws:
                    self._ws = ws
                    warned = False
                    ws.send(json.dumps({"type": "hello", "client": self.client_name}))
                    if self.on_connect is not None:
                        self.on_connect()
                    for raw in ws:
                        if isinstance(raw, str):
                            self._handle(json.loads(raw))
            except (ConnectionClosed, OSError, TimeoutError) as error:
                if self._ws is not None:
                    self.log(f"   browser link lost ({error}); reconnecting ...")
                elif not warned:
                    self.log(f"   browser link: no gesture server at {self.url} yet "
                             f"(start it with: uv run gesture_control/server.py); retrying every 2 s")
                    warned = True
            self._ws = None
            self._set_state("disconnected")
            self._idle.set()
            if not self._stop:
                time.sleep(2)

    def _handle(self, msg):
        kind = msg.get("type")
        if kind == "hello":
            self.log(f"   browser link: connected to {self.url}, page \"{msg.get('title', '')}\"")
            self._set_state(msg.get("state", "ready"))
        elif kind == "status":
            self._set_state(msg["state"])
            if msg["state"] == "ready":
                self._idle.set()
        elif kind == "result":
            sent = self._sent_at.pop(msg.get("id"), None)
            if sent is None:
                return  # a result for another client's command
            round_trip = round((time.perf_counter() - sent) * 1000)
            if msg.get("ok"):
                self.last_reply = f"{msg['command']}: {msg['result']}"
                self.log(f"   <- browser: {self.last_reply}  ({msg['ms']} ms in the browser, "
                         f"{round_trip} ms round trip)  page: {msg.get('title', '')}")
            else:
                reason = msg.get("skipped") or msg.get("error")
                self.last_reply = f"{msg['command']}: not done ({reason})"
                self.log(f"   <- browser: {self.last_reply}")
                self._idle.set()
        elif kind == "error":
            self.log(f"   <- browser error: {msg.get('error')}")
