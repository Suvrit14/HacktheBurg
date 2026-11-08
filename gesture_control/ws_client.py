"""Send gesture commands to the gesture server from the keyboard (no camera needed).

    uv run gesture_control/ws_client.py                      interactive: type a command, press Enter
    uv run gesture_control/ws_client.py --tab                send one command and exit
    uv run gesture_control/ws_client.py --url ws://localhost:8801

Interactive shortcuts:  d scroll_down   u scroll_up   t tab   c click   b back   q quit
(full names like "scroll_down" work too)
"""

import argparse
import json
import sys
import threading

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

COMMANDS = ["scroll_up", "scroll_down", "back", "tab", "click"]
SHORTCUTS = {"d": "scroll_down", "u": "scroll_up", "t": "tab", "c": "click", "b": "back"}


def show(msg):
    kind = msg.get("type")
    if kind == "hello":
        print(f"connected. page: \"{msg.get('title', '')}\"  ({msg.get('url', '')})  browser {msg.get('state')}")
    elif kind == "result":
        if msg.get("ok"):
            print(f"<- {msg['command']}: {msg['result']}   ({msg['ms']} ms)  page: \"{msg.get('title', '')}\"")
        else:
            print(f"<- {msg['command']}: not done ({msg.get('skipped') or msg.get('error')})")
    elif kind == "error":
        print(f"<- error: {msg.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Send gesture commands to the gesture server")
    parser.add_argument("--url", default="ws://localhost:8800")
    group = parser.add_mutually_exclusive_group()
    for name in COMMANDS:
        flag = name.replace("_", "-")
        options = [f"--{flag}"] + ([f"--{name}"] if flag != name else [])
        group.add_argument(*options, dest="command", action="store_const", const=name)
    args = parser.parse_args()

    try:
        ws = connect(args.url, open_timeout=3)
    except (OSError, TimeoutError) as error:
        sys.exit(f"no gesture server at {args.url} ({error}). Start it with: uv run gesture_control/server.py")

    with ws:
        ws.send(json.dumps({"type": "hello", "client": "keyboard"}))
        show(json.loads(ws.recv()))

        if args.command:  # one-shot: send, wait for its result, exit
            ws.send(json.dumps({"type": "command", "command": args.command, "id": 1}))
            for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") in ("result", "error"):
                    show(msg)
                    return

        def receive():
            try:
                for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "status":
                        show(msg)
            except ConnectionClosed:
                print("server closed the connection")

        threading.Thread(target=receive, daemon=True).start()
        print("type: d scroll_down, u scroll_up, t tab, c click, b back, q quit")
        sequence = 0
        for line in sys.stdin:
            text = line.strip().lower()
            if text in ("q", "quit", "exit"):
                break
            command = SHORTCUTS.get(text, text.replace("-", "_"))
            if not command:
                continue
            if command not in COMMANDS:
                print(f"unknown: {text!r}. Use d, u, t, c, b or a full name")
                continue
            sequence += 1
            ws.send(json.dumps({"type": "command", "command": command, "id": sequence}))


if __name__ == "__main__":
    main()
