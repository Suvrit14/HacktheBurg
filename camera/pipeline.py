"""The whole camera to command pipeline in one object (no camera needed in this file).

    camera pose (fist, spread, v, radial, ulnar)
      -> the matching REAL EMG recording from the library (repetitions 5 and 6, never trained on)
      -> replayed in real time through the classifier and the smoothing
      -> command events: scroll_down, scroll_up, back, click, tab

The camera is the recogniser; the EMG is a recorded NinaPro signal that the classifier decodes live.
Do not call it live EMG.

Use:

    pipeline = CameraPipeline(on_command=print)
    pipeline.trigger("fist")           # from the camera: returns True if a recording was started
    pipeline.blocked = True            # while the agent is working: camera triggers are ignored
    pipeline.wait()                    # (tests) wait until the recording has finished
"""

import threading

from emg_data.commands import COMMAND_FOR_GESTURE
from emg_data.interface import replay_commands
from emg_data.library import build_replay, gesture_id

from .poses import POSE_TO_GESTURE


class CameraPipeline:
    def __init__(self, on_command=None, on_window=None, on_recording=None,
                 model_path="models/subject1_set_c.joblib", data_dir=".", speed=1.0,
                 lead_in_s=0.4, tail_s=0.5, forward_repeats=False):
        # on_command(event):   a command was decoded. A dict:
        #     command, gesture, t, confidence   (as in INTERFACE.md)
        #     camera_gesture  the gesture the camera saw ("fist", "spread", "tab", "radial", "ulnar")
        #     expected_command  the command that gesture should give
        #     match           True if the decoded command is the expected one
        #     is_repeat       True for the 2nd, 3rd ... command of a held scroll gesture
        #   Repeats are only passed on if forward_repeats=True (default: one gesture = one command).
        # on_window(window):   every window of the recording (for a dashboard): the window dict of INTERFACE.md,
        #                      with raw_window, probabilities, event and vote ({"agree", "of", ...})
        # on_recording(kind, info): kind is "start" or "end"; info has camera_gesture, expected_command, repetition,
        #                      and at the end decoded_commands (list), match (bool) and mismatch (bool)
        self.on_command = on_command
        self.on_window = on_window
        self.on_recording = on_recording
        self.model_path = model_path
        self.data_dir = data_dir
        self.speed = speed
        self.lead_in_s = lead_in_s
        self.tail_s = tail_s
        self.forward_repeats = forward_repeats

        self.blocked = False                 # set True while the agent is working or a menu is not waiting for input
        self.last_reject_reason = None
        self._busy = False
        self._lock = threading.Lock()
        self._thread = None

    @property
    def busy(self):
        return self._busy

    def trigger(self, pose):
        # pose: a pose name from the camera ("fist", "spread", "v", "radial", "ulnar") or a gesture name ("tab").
        # Returns True if a recording was started, False if the trigger was ignored (see last_reject_reason).
        gesture = POSE_TO_GESTURE.get(pose, pose)
        try:
            gesture_id(gesture)
        except ValueError:
            self.last_reject_reason = f"unknown pose {pose!r}"
            return False

        with self._lock:
            if self.blocked:
                self.last_reject_reason = "blocked"
                return False
            if self._busy:
                self.last_reject_reason = "a recording is still playing"
                return False
            self._busy = True
            self.last_reject_reason = None

        self._thread = threading.Thread(target=self._play, args=(gesture,), daemon=True)
        self._thread.start()
        return True

    def wait(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def _play(self, gesture):
        try:
            replay = build_replay([gesture], lead_in_s=self.lead_in_s, tail_s=self.tail_s, data_dir=self.data_dir)
            expected = COMMAND_FOR_GESTURE[gesture_id(gesture)]
            info = {"camera_gesture": gesture, "expected_command": expected, "repetition": replay["repetitions"][0]}
            if self.on_recording is not None:
                self.on_recording("start", dict(info))

            decoded = []
            for window in replay_commands(model_path=self.model_path, replay=replay, speed=self.speed, data_dir=self.data_dir):
                if self.on_window is not None:
                    self.on_window(window)
                event = window["event"]
                if event is None:
                    continue

                is_repeat = len(decoded) > 0 and decoded[-1]["command"] == event["command"]
                output = dict(event)
                output["camera_gesture"] = gesture
                output["expected_command"] = expected
                output["match"] = event["command"] == expected
                output["is_repeat"] = is_repeat
                decoded.append(output)
                if self.on_command is not None and (self.forward_repeats or not is_repeat):
                    self.on_command(output)

            commands = []
            for output in decoded:
                if output["command"] not in commands:
                    commands.append(output["command"])
            match = commands == [expected]
            if self.on_recording is not None:
                end = dict(info)
                end.update({"decoded_commands": commands, "match": match, "mismatch": not match})
                self.on_recording("end", end)
        finally:
            self._busy = False
