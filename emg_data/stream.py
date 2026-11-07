"""Replay a recording as a stream and classify it window by window (M1 T3).

build_replay    cuts real pieces out of one DB5 file and joins them in a demo order
stream_windows  walks over a recording window by window and yields one prediction per window
"""

import time

import numpy as np

from .emg import extract_features, load_raw_db5


def build_replay(subject_number=1, exercise_number=2, gestures=(6, 5, 15, 16), replay_rep=6, data_dir="data"):
    # For each gesture: the real rest just before it, then the gesture itself, both from repetition replay_rep.
    # The pieces are joined in the order given. Only samples labelled with replay_rep are used,
    # so nothing the final model trained on (repetitions 1 to 5) ends up in the replay.
    #
    # Returns a dict:
    #   emg            (num_samples, 16)  the joined EMG
    #   labels         (num_samples,)     restimulus of every sample (0 = rest)
    #   sampling_freq  Hz
    #   segments       one dict per gesture: gesture, rest_start, gesture_start, gesture_end (sample numbers in the replay)
    emg, restimulus, rerepetition, sampling_freq = load_raw_db5(subject_number, exercise_number, data_dir)

    emg_pieces = []
    label_pieces = []
    segments = []
    position = 0     # where the next piece starts inside the replay

    for gesture in gestures:
        # Samples of this gesture in the chosen repetition
        indices = np.flatnonzero((restimulus == gesture) & (rerepetition == replay_rep))
        if len(indices) == 0:
            raise ValueError(f"gesture {gesture} has no samples in repetition {replay_rep}")
        gesture_start = indices[0]
        gesture_end = indices[-1] + 1
        if gesture_end - gesture_start != len(indices):
            raise ValueError(f"gesture {gesture} in repetition {replay_rep} is not one unbroken stretch")

        # Walk back over the rest just before the gesture (rest that carries the same repetition number)
        rest_start = gesture_start
        while rest_start > 0 and restimulus[rest_start - 1] == 0 and rerepetition[rest_start - 1] == replay_rep:
            rest_start -= 1

        # rest + gesture are one unbroken stretch of the original recording
        emg_pieces.append(emg[rest_start:gesture_end])
        label_pieces.append(restimulus[rest_start:gesture_end])

        segments.append({
            "gesture": int(gesture),
            "rest_start": position,
            "gesture_start": position + (gesture_start - rest_start),
            "gesture_end": position + (gesture_end - rest_start),
        })
        position += gesture_end - rest_start

    return {
        "emg": np.concatenate(emg_pieces, axis=0),
        "labels": np.concatenate(label_pieces, axis=0),
        "sampling_freq": sampling_freq,
        "segments": segments,
    }


def build_rest_replay(subject_number=1, exercise_number=2, replay_rep=6, data_dir="data"):
    # All the rest of one repetition, from the rest periods before every one of the 17 movements, joined together.
    # No gesture is in it, so every command that fires while it plays is a false trigger.
    # Returns the same kind of dict as build_replay (without segments).
    emg, restimulus, rerepetition, sampling_freq = load_raw_db5(subject_number, exercise_number, data_dir)

    is_rest = (restimulus == 0) & (rerepetition == replay_rep)
    return {
        "emg": emg[is_rest],
        "labels": restimulus[is_rest],
        "sampling_freq": sampling_freq,
        "segments": [],
    }


def stream_windows(emg, model, labels=None, window_length=50, step_size=25, sampling_freq=200, speed=None):
    # Walks over emg (num_samples, 16) one window at a time, like data arriving from a sensor.
    # speed=1.0 waits so that one window arrives every step_size / sampling_freq seconds (125 ms for 25 samples at 200 Hz).
    # speed=2.0 is twice as fast; speed=None does not wait at all (for testing).
    #
    # Every yielded item is a dict:
    #   t              seconds from the start of the recording to the END of the window (when the window is complete)
    #   raw_window     (window_length, 16) the EMG of this window
    #   label          the gesture the model predicts (0 = rest)
    #   confidence     the model's probability for that gesture
    #   probabilities  {gesture: probability} for every class the model knows
    #   true_label     the most common label inside the window if labels were given, otherwise None
    #   index          number of the window, starting at 0
    classes = [int(c) for c in model.classes_]
    num_windows = ((len(emg) - window_length) // step_size) + 1
    if num_windows < 1:
        raise ValueError("the recording is shorter than one window")

    start_time = time.perf_counter()

    for i in range(num_windows):
        start = i * step_size
        end = start + window_length
        window = emg[start:end, :]
        t = end / sampling_freq

        # Wait until this window would really have arrived
        if speed:
            wait = start_time + t / speed - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

        features = extract_features(window[np.newaxis, :, :])      # (1, 48)
        probabilities = model.predict_proba(features)[0]           # one probability per class
        best = int(np.argmax(probabilities))

        true_label = None
        if labels is not None:
            true_label = int(np.bincount(labels[start:end]).argmax())

        yield {
            "t": t,
            "raw_window": window,
            "label": classes[best],
            "confidence": float(probabilities[best]),
            "probabilities": {classes[k]: float(probabilities[k]) for k in range(len(classes))},
            "true_label": true_label,
            "index": i,
        }
