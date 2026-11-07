"""Turn noisy per-window predictions into commands (M1 T4).

CommandLogic.update(item) takes one window dict from stream_windows and returns a command event or None.
All settings are in DEFAULT_CONFIG, so they can be tuned in one place.
"""

from collections import Counter, deque

# NinaPro gesture id -> command (Set A, see gestures.md). Rest (0) never gives a command.
COMMAND_FOR_GESTURE = {
    6: "scroll_down",      # fist
    5: "scroll_up",        # abduction of all fingers
    15: "back",            # wrist radial deviation
    16: "forward",         # wrist ulnar deviation
}

# These commands keep firing while the gesture is held. The others fire once per gesture.
HOLD_TO_REPEAT = ["scroll_down", "scroll_up"]

DEFAULT_CONFIG = {
    "vote_window": 5,          # majority vote over the last N windows (5 windows = about 0.6 s)
    "min_confidence": 0.6,     # a window with a lower confidence votes for nothing
    "cooldown_s": 1.0,         # minimum time between two back/forward commands
    "repeat_every_s": 0.3,     # scroll fires again every this many seconds while the gesture is held
}


# The settings chosen in M1 T5 (tuned on repetitions 1 to 5 of Subject 1).
# Only min_confidence differs from DEFAULT_CONFIG. The evidence for it is thin (see m1-progress.md).
TUNED_CONFIG = dict(DEFAULT_CONFIG, min_confidence=0.9)


class CommandLogic:
    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.votes = deque(maxlen=self.config["vote_window"])   # last N (vote, confidence) pairs
        self.state = None            # smoothed state: 0 = rest, a gesture id, or None = undecided
        self.current = 0             # the gesture the smoothed state last settled on
        self.armed = True            # back/forward may only fire again after a return to rest
        self.last_repeat = {}        # command -> time it last fired (for hold-to-repeat)
        self.last_once = -1e9        # time the last back/forward command fired

    def update(self, item):
        # item: one dict from stream_windows (needs t, label, confidence). Returns an event dict or None.
        t = item["t"]
        cfg = self.config

        # 1. A window that the model is not sure about votes for nothing (None)
        vote = item["label"] if item["confidence"] >= cfg["min_confidence"] else None
        self.votes.append((vote, item["confidence"]))

        # 2. Majority vote: a gesture needs more than half of the window's votes, otherwise we stay undecided
        counts = Counter(v for v, c in self.votes if v is not None)
        needed = cfg["vote_window"] // 2 + 1
        self.state = None
        if len(counts) > 0:
            winner, count = counts.most_common(1)[0]
            if count >= needed:
                self.state = winner

        # 3. Turn the smoothed state into a command
        event = None

        if self.state == 0:
            # Rest: no command, and back/forward are allowed to fire again
            self.armed = True
            self.current = 0

        elif self.state in COMMAND_FOR_GESTURE:
            gesture = self.state
            command = COMMAND_FOR_GESTURE[gesture]
            just_started = gesture != self.current     # the smoothed state just changed to this gesture
            self.current = gesture

            fire = False
            if command in HOLD_TO_REPEAT:
                # fire when the gesture starts, then again every repeat_every_s while it is held
                waited = t - self.last_repeat.get(command, -1e9)
                fire = just_started or waited >= cfg["repeat_every_s"]
                if fire:
                    self.last_repeat[command] = t
            else:
                # fire once when the gesture starts, but only after a return to rest and after the cooldown
                fire = just_started and self.armed and (t - self.last_once >= cfg["cooldown_s"])
                if fire:
                    self.last_once = t
                    self.armed = False

            if fire:
                votes_for_gesture = [c for v, c in self.votes if v == gesture]
                event = {
                    "t": t,
                    "command": command,
                    "gesture": gesture,
                    "confidence": sum(votes_for_gesture) / len(votes_for_gesture),   # average of the winning votes
                }

        # self.state is None (undecided): nothing changes, no command
        return event
