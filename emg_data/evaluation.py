"""Score the commands at event level (M1 T5).

Window accuracy is not what matters for a user. These functions count commands instead:
was the right command given for a gesture, a wrong one, or one when nothing was happening?
"""

import numpy as np

from .commands import CommandLogic, COMMAND_FOR_GESTURE
from .stream import build_replay, build_rest_replay, stream_windows


def build_fold(model, replay_rep, subject_number=1, gestures=None, exercise_number=2, data_dir="."):
    # gestures=None: the model's gestures, replayed in the order they appear in the recording.
    # Everything needed to test one repetition: the model streams (1) the gesture replay and (2) the rest-only
    # replay of that repetition. The model must NOT have been trained on replay_rep.
    gesture_replay = build_replay(subject_number, exercise_number, gestures, replay_rep, data_dir)
    rest_replay = build_rest_replay(subject_number, exercise_number, replay_rep, data_dir)
    fs = gesture_replay["sampling_freq"]

    return {
        "replay_rep": replay_rep,
        "gesture_items": list(stream_windows(gesture_replay["emg"], model, labels=gesture_replay["labels"], sampling_freq=fs)),
        "rest_items": list(stream_windows(rest_replay["emg"], model, labels=rest_replay["labels"], sampling_freq=fs)),
        "segments": gesture_replay["segments"],
        "gesture_seconds": len(gesture_replay["emg"]) / fs,
        "rest_only_seconds": len(rest_replay["emg"]) / fs,
        "sampling_freq": fs,
    }


def evaluate_folds(folds, config=None):
    # Runs the command logic on every fold and pools the counts.
    # A false trigger is any command that fires when no gesture is happening: in the rest-only replays,
    # and in the rest parts of the gesture replays.
    pieces = []
    false_events = []
    rest_seconds = 0.0

    for fold in folds:
        fs = fold["sampling_freq"]

        events = run_commands(fold["gesture_items"], config)
        fold_pieces, outside, piece_seconds = score_gesture_replay(events, fold["segments"], fs)
        pieces.extend(fold_pieces)
        false_events.extend(outside)
        rest_seconds += fold["gesture_seconds"] - piece_seconds

        rest_events = run_commands(fold["rest_items"], config)
        false_events.extend(rest_events)
        rest_seconds += fold["rest_only_seconds"]

    latencies = [p["latency"] for p in pieces if p["latency"] is not None]
    rest_minutes = rest_seconds / 60

    return {
        "n_pieces": len(pieces),
        "detected": sum(1 for p in pieces if p["detected"]),
        "wrong": sum(1 for p in pieces if p["wrong"]),
        "false_triggers": len(false_events),
        "rest_minutes": rest_minutes,
        "false_per_minute": len(false_events) / rest_minutes,
        "latency_median": float(np.median(latencies)) if len(latencies) > 0 else None,
        "latency_max": float(np.max(latencies)) if len(latencies) > 0 else None,
        "pieces": pieces,
        "false_events": false_events,
    }


def run_commands(items, config=None):
    # items: the window dicts from stream_windows. Returns the list of command events.
    logic = CommandLogic(config)
    events = []
    for item in items:
        event = logic.update(item)
        if event is not None:
            events.append(event)
    return events


def score_gesture_replay(events, segments, sampling_freq, extra_s=0.25):
    # A command counts as "during" a gesture piece if it fires between the start of the gesture
    # and extra_s seconds after its end (the vote makes commands arrive a little late).
    #
    # Returns:
    #   pieces        one dict per gesture piece: gesture, detected (the right command fired),
    #                 wrong (a different command fired), latency (seconds from gesture start to the first right command)
    #   outside       the events that fired outside every piece (false triggers)
    #   piece_seconds how many seconds of the replay count as "during a gesture piece"
    #                 (the rest of the replay is rest: replay length minus piece_seconds)
    pieces = []
    piece_seconds = 0.0

    for segment in segments:
        onset = segment["gesture_start"] / sampling_freq
        end = segment["gesture_end"] / sampling_freq + extra_s
        expected = COMMAND_FOR_GESTURE[segment["gesture"]]

        right = []
        wrong = []
        for event in events:
            if onset <= event["t"] <= end:
                if event["command"] == expected:
                    right.append(event)
                else:
                    wrong.append(event)

        latency = None
        if len(right) > 0:
            latency = right[0]["t"] - onset

        pieces.append({"gesture": segment["gesture"], "detected": len(right) > 0,
                       "wrong": len(wrong) > 0, "latency": latency})
        piece_seconds += end - onset

    outside = []
    for event in events:
        inside_a_piece = False
        for segment in segments:
            if segment["gesture_start"] / sampling_freq <= event["t"] <= segment["gesture_end"] / sampling_freq + extra_s:
                inside_a_piece = True
        if not inside_a_piece:
            outside.append(event)

    return pieces, outside, piece_seconds
