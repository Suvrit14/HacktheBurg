"""The interface for the agent and the dashboard (M1 T6). See INTERFACE.md.

replay_commands   replays recorded EMG window by window and yields every window together with the command (if any)
to_json_dict      turns one yielded window into plain types that json.dumps accepts
"""

from .commands import CommandLogic, TUNED_CONFIG
from .model import load_model
from .stream import build_replay, stream_windows

COMMANDS = ["scroll_down", "scroll_up", "back", "forward"]


def replay_commands(model_path="models/subject1_set_a.joblib", subject_number=1, replay_rep=6,
                    gestures=None, speed=1.0, config=None, data_dir="data", allow_training_data=False):
    # Replays real recorded pieces (rest + gesture, from one repetition) as if they came from a sensor and
    # yields one dict per window: the window dict of stream_windows plus the key "event"
    # (the command that fired on this window, or None).
    #
    # The pieces play in the order they appear in the recording. gestures=None replays every gesture
    # the saved model knows; a list picks which ones (their order in the list does not matter).
    #
    # speed=1.0 is real time (one window every 125 ms), speed=None runs as fast as possible.
    # The replay has to be a repetition the model was NOT trained on, otherwise the results would look better
    # than they are. allow_training_data=True switches that check off.
    saved = load_model(model_path)

    if subject_number != saved["subject"]:
        raise ValueError(f"this model was trained on Subject {saved['subject']}, not Subject {subject_number}")
    if replay_rep in saved["train_reps"] and not allow_training_data:
        raise ValueError(f"the model was trained on repetition {replay_rep}; replay repetition "
                         f"{saved['held_out_rep']} or set allow_training_data=True")

    if gestures is None:
        gestures = [c for c in saved["classes"] if c != 0]
    replay = build_replay(subject_number, saved["exercise"], gestures, replay_rep, data_dir)
    logic = CommandLogic(TUNED_CONFIG if config is None else config)

    for item in stream_windows(replay["emg"], saved["model"], labels=replay["labels"],
                               window_length=saved["window_length"], step_size=saved["step_size"],
                               sampling_freq=replay["sampling_freq"], speed=speed):
        item["event"] = logic.update(item)
        yield item


def to_json_dict(item, include_raw=False):
    # Plain Python types only (JSON needs string keys, so the probabilities are keyed by the gesture id as a string).
    # The raw window (50 x 16 numbers) is left out unless include_raw=True.
    out = {
        "t": item["t"],
        "index": item["index"],
        "label": item["label"],
        "confidence": item["confidence"],
        "probabilities": {str(k): v for k, v in item["probabilities"].items()},
        "true_label": item["true_label"],
        "event": item["event"],
    }
    if include_raw:
        out["raw_window"] = item["raw_window"].tolist()
    return out
