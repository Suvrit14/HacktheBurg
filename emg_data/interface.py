"""The interface for the agent and the dashboard (M1 T6). See INTERFACE.md.

replay_commands   replays recorded EMG window by window and yields every window together with the command (if any)
to_json_dict      turns one yielded window into plain types that json.dumps accepts
"""

from .commands import CommandLogic, TUNED_CONFIG
from .model import load_model
from .stream import build_replay, stream_windows

DEMO_ORDER = [6, 5, 15, 16, 2]   # fist, spread fingers, radial deviation, ulnar deviation, tab
COMMANDS = ["scroll_down", "scroll_up", "back", "click", "tab"]    # "tab" only comes from the set-C model


def replay_commands(model_path="models/subject1_set_c.joblib", subject_number=1, replay_rep=6,
                    gestures=DEMO_ORDER, speed=1.0, config=None, data_dir=".", allow_training_data=False,
                    replay=None):
    # Replays real recorded pieces as if they came from a sensor and yields one dict per window:
    # the window dict of stream_windows plus the key "event" (the command that fired on this window, or None).
    #
    # By default it replays one repetition (replay_rep) of the gestures in the order given. To replay your own
    # mix of recordings, build it with emg_data.library.build_replay([...]) and pass it as replay=...
    #
    # speed=1.0 is real time (one window every 125 ms), speed=None runs as fast as possible.
    # The replayed repetitions have to be ones the model was NOT trained on, otherwise the results would look
    # better than they are. allow_training_data=True switches that check off.
    # config=None uses the settings stored in the model file (tuned for that model), or TUNED_CONFIG if there are none.
    saved = load_model(model_path)

    if subject_number != saved["subject"]:
        raise ValueError(f"this model was trained on Subject {saved['subject']}, not Subject {subject_number}")

    if replay is None:
        repetitions = [replay_rep]
    else:
        repetitions = replay["repetitions"]
    trained_on = [r for r in repetitions if r in saved["train_reps"]]
    if len(trained_on) > 0 and not allow_training_data:
        held_out = saved.get("held_out_reps", [saved.get("held_out_rep")])
        raise ValueError(f"the model was trained on repetition(s) {trained_on}; replay one of the held-out "
                         f"repetitions {held_out} or set allow_training_data=True")

    if replay is None:
        replay = build_replay(subject_number, saved["exercise"], gestures, replay_rep, data_dir)
    if config is None:
        config = saved.get("config", TUNED_CONFIG)
    logic = CommandLogic(config)

    for item in stream_windows(replay["emg"], saved["model"], labels=replay["labels"],
                               window_length=saved["window_length"], step_size=saved["step_size"],
                               sampling_freq=replay["sampling_freq"], speed=speed):
        item["event"] = logic.update(item)
        item["vote"] = dict(logic.vote)        # {"leading", "agree", "of", "needed", "state"}: how many of the last windows agree
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
        "vote": item["vote"],
    }
    if include_raw:
        out["raw_window"] = item["raw_window"].tolist()
    return out
