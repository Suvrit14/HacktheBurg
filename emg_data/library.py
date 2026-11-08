"""Per-gesture recording library and mixer (next-steps T3).

Every recording is a real piece cut out of the NinaPro file, from the held-out repetitions 5 and 6 only:
the rest just before a gesture plus the gesture itself. Real rest-only pieces from the same repetitions
are used for the quiet stretch after a gesture and, if asked, between gestures.

    get_recording("fist")                       one recording (alternates between repetition 5 and 6)
    build_replay(["fist", "tab", "radial"])     recordings joined in any order, ready for replay_commands(replay=...)
    check_joins(replay)                         how big are the jumps where two pieces were joined?

Only the five gestures of the set-C model are in the library. The other 12 movements are left out on purpose:
in M1 they triggered commands in 61 of 65 pieces, which would cause false actions on stage.
"""

import numpy as np

from .commands import COMMAND_FOR_GESTURE
from .emg import load_raw_db5

LIBRARY_REPS = (5, 6)
GESTURE_NAMES = {"fist": 6, "spread": 5, "radial": 15, "ulnar": 16, "tab": 2}


def gesture_id(gesture):
    # Accepts the NinaPro class id (6), a gesture name ("fist") or a command name ("scroll_down")
    if isinstance(gesture, (int, np.integer)) and int(gesture) in COMMAND_FOR_GESTURE:
        return int(gesture)
    if gesture in GESTURE_NAMES:
        return GESTURE_NAMES[gesture]
    for gesture_number, command in COMMAND_FOR_GESTURE.items():
        if gesture == command:
            return gesture_number
    raise ValueError(f"unknown gesture {gesture!r}; use one of {sorted(GESTURE_NAMES)} "
                     f"or a command name {sorted(COMMAND_FOR_GESTURE.values())}")


class RecordingLibrary:
    def __init__(self, subject_number=1, exercise_number=2, repetitions=LIBRARY_REPS, data_dir=".", seed=0):
        self.repetitions = tuple(repetitions)
        self.rng = np.random.RandomState(seed)
        emg, restimulus, rerepetition, self.sampling_freq = load_raw_db5(subject_number, exercise_number, data_dir)
        self.emg = emg
        self.restimulus = restimulus
        self.rerepetition = rerepetition

        # One piece per (gesture, repetition): where the gesture is, and where the rest just before it starts
        self.pieces = {}
        for gesture in GESTURE_NAMES.values():
            for repetition in self.repetitions:
                indices = np.flatnonzero((restimulus == gesture) & (rerepetition == repetition))
                if len(indices) == 0:
                    raise ValueError(f"gesture {gesture} has no samples in repetition {repetition}")
                gesture_start = indices[0]
                gesture_end = indices[-1] + 1
                if gesture_end - gesture_start != len(indices):
                    raise ValueError(f"gesture {gesture} in repetition {repetition} is not one unbroken stretch")

                # walk back over the rest just before the gesture (it carries the same repetition number)
                rest_start = gesture_start
                while rest_start > 0 and restimulus[rest_start - 1] == 0 and rerepetition[rest_start - 1] == repetition:
                    rest_start -= 1
                self.pieces[(gesture, repetition)] = {
                    "rest_start": rest_start, "gesture_start": gesture_start, "gesture_end": gesture_end}

        # Real rest-only runs from the held-out repetitions (one unbroken stretch of rest each)
        self.rest_runs = []
        is_rest = (restimulus == 0) & np.isin(rerepetition, self.repetitions)
        edges = np.flatnonzero(np.diff(is_rest.astype(int)) != 0) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(is_rest)]
        for start, end in zip(starts, ends):
            if is_rest[start] and rerepetition[start] == rerepetition[end - 1]:
                self.rest_runs.append({"start": int(start), "end": int(end), "repetition": int(rerepetition[start])})

        self._next_repetition = {}     # gesture -> index into self.repetitions, so repeated requests alternate

    # ------------------------------------------------------------------
    def _choose_repetition(self, gesture, repetition):
        if repetition is not None:
            if repetition not in self.repetitions:
                raise ValueError(f"repetition {repetition} is not in the library {self.repetitions}")
            return repetition
        index = self._next_repetition.get(gesture, 0)
        self._next_repetition[gesture] = (index + 1) % len(self.repetitions)
        return self.repetitions[index]

    def get_rest(self, seconds):
        # A real stretch of rest of at least `seconds`, cut from a random rest run of the held-out repetitions.
        # Returns (emg, source) where source = (first sample, last sample + 1) in the original recording.
        length = int(round(seconds * self.sampling_freq))
        long_enough = [run for run in self.rest_runs if run["end"] - run["start"] >= length]
        if len(long_enough) == 0:
            raise ValueError(f"no rest run of {seconds} s in the library")
        run = long_enough[self.rng.randint(len(long_enough))]
        start = run["start"] + self.rng.randint(run["end"] - run["start"] - length + 1)
        return self.emg[start:start + length], (start, start + length)

    def get_recording(self, gesture, repetition=None, lead_in_s=1.0, tail_s=1.0):
        # One real piece: `lead_in_s` seconds of the rest just before the gesture, the gesture, and `tail_s` seconds
        # of real rest after it. repetition=None alternates between the repetitions of the library.
        gesture = gesture_id(gesture)
        repetition = self._choose_repetition(gesture, repetition)
        piece = self.pieces[(gesture, repetition)]
        fs = self.sampling_freq

        start = max(piece["rest_start"], piece["gesture_start"] - int(round(lead_in_s * fs)))
        lead = self.emg[start:piece["gesture_start"]]
        body = self.emg[piece["gesture_start"]:piece["gesture_end"]]
        parts = [lead, body]
        labels = [np.zeros(len(lead), dtype=int), self.restimulus[piece["gesture_start"]:piece["gesture_end"]]]
        sources = [(int(start), int(piece["gesture_end"]))]

        if tail_s > 0:
            tail, tail_source = self.get_rest(tail_s)
            parts.append(tail)
            labels.append(np.zeros(len(tail), dtype=int))
            sources.append(tail_source)

        return {
            "gesture": gesture,
            "command": COMMAND_FOR_GESTURE[gesture],
            "repetition": repetition,
            "emg": np.concatenate(parts, axis=0),
            "labels": np.concatenate(labels),
            "sampling_freq": fs,
            "gesture_start": len(lead),
            "gesture_end": len(lead) + len(body),
            "sources": sources,
        }


_libraries = {}


def library(subject_number=1, exercise_number=2, data_dir=".", seed=0):
    # One library per settings, built once (loading the recording takes about a second)
    key = (subject_number, exercise_number, data_dir, seed)
    if key not in _libraries:
        _libraries[key] = RecordingLibrary(subject_number, exercise_number, LIBRARY_REPS, data_dir, seed)
    return _libraries[key]


def get_recording(gesture, repetition=None, lead_in_s=1.0, tail_s=1.0, data_dir="."):
    return library(data_dir=data_dir).get_recording(gesture, repetition, lead_in_s, tail_s)


def build_replay(items, lead_in_s=1.0, tail_s=1.0, extra_rest_s=0.0, data_dir="."):
    # items: gestures in the order to play, e.g. ["fist", "tab", "radial"].
    # An item can also be (gesture, repetition), e.g. ("fist", 5), or ("rest", seconds) for extra real rest.
    # Returns a dict that replay_commands(replay=...) understands (same keys as emg_data.stream.build_replay,
    # plus "repetitions": the repetitions that were used and "joins": the sample numbers where pieces meet).
    lib = library(data_dir=data_dir)
    emg_parts = []
    label_parts = []
    segments = []
    joins = []
    repetitions_used = set()
    position = 0

    for item in items:
        if isinstance(item, tuple) and item[0] == "rest":
            rest, source = lib.get_rest(item[1])
            if position > 0:
                joins.append(position)
            emg_parts.append(rest)
            label_parts.append(np.zeros(len(rest), dtype=int))
            position += len(rest)
            continue

        gesture, repetition = (item if isinstance(item, tuple) else (item, None))
        recording = lib.get_recording(gesture, repetition, lead_in_s, tail_s)
        if position > 0:
            joins.append(position)

        segments.append({
            "gesture": recording["gesture"],
            "repetition": recording["repetition"],
            "rest_start": position,
            "gesture_start": position + recording["gesture_start"],
            "gesture_end": position + recording["gesture_end"],
        })
        repetitions_used.add(recording["repetition"])
        emg_parts.append(recording["emg"])
        label_parts.append(recording["labels"])
        position += len(recording["emg"])

        if extra_rest_s > 0:
            rest, source = lib.get_rest(extra_rest_s)
            joins.append(position)
            emg_parts.append(rest)
            label_parts.append(np.zeros(len(rest), dtype=int))
            position += len(rest)

    return {
        "emg": np.concatenate(emg_parts, axis=0),
        "labels": np.concatenate(label_parts),
        "sampling_freq": lib.sampling_freq,
        "segments": segments,
        "repetitions": sorted(repetitions_used),
        "joins": joins,
    }


def check_joins(replay, window=20, data_dir="."):
    # At every join, compare the mean of the last `window` samples before it with the mean of the first `window`
    # samples after it, channel by channel. The same number is also computed for 2000 random pairs of neighbouring
    # stretches inside real rest, which shows how big a "normal" jump is.
    # Returns the jump at every join, and the 95th percentile of the normal jumps.
    lib = library(data_dir=data_dir)
    emg = replay["emg"].astype(float)

    jumps = []
    for join in replay["joins"]:
        before = emg[join - window:join].mean(axis=0)
        after = emg[join:join + window].mean(axis=0)
        jumps.append(float(np.max(np.abs(before - after))))

    normal = []
    for _ in range(2000):
        run = lib.rest_runs[lib.rng.randint(len(lib.rest_runs))]
        if run["end"] - run["start"] < 2 * window + 1:
            continue
        position = run["start"] + window + lib.rng.randint(run["end"] - run["start"] - 2 * window)
        before = lib.emg[position - window:position].astype(float).mean(axis=0)
        after = lib.emg[position:position + window].astype(float).mean(axis=0)
        normal.append(float(np.max(np.abs(before - after))))

    return jumps, float(np.percentile(normal, 95))
