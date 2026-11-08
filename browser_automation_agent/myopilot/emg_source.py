"""EMG gesture source: Suvrit's classifier (emg_data) driving the MyoPilot menus.

For each gesture in a sequence, it replays a real recorded piece (rest, then the gesture)
from the held-out repetition through Suvrit's `replay_commands()`: the LDA model decodes
the EMG window by window, his CommandLogic turns that into commands, and the commands go
onto our IntentBus exactly like keyboard presses. Nothing in emg_data is changed.

What this layer handles:
  * paths     model and data are found from the repo root, whatever the working folder
  * repeats   his scroll commands fire every 0.3 s while a fist is held; by default only the
              first command of each gesture is forwarded (one fist = one scroll_down). --emg-repeat
              forwards every command instead.
  * timing    each gesture is performed only when a menu is waiting for input, so commands
              are never dropped while the agent is busy; replay runs in real time (speed 1.0)
  * honesty   the command that is forwarded is whatever the model decoded, even if it is
              not the one that was intended; both are logged.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, settings
from .events import EventLog
from .intents import IntentBus

if str(REPO_ROOT) not in sys.path:  # emg_data lives at the repo root
    sys.path.insert(0, str(REPO_ROOT))

from emg_data.commands import COMMAND_FOR_GESTURE  # noqa: E402
from emg_data.interface import replay_commands  # noqa: E402
from emg_data.model import CLASSES  # noqa: E402
from emg_data.stream import gestures_in_recording_order  # noqa: E402

SUBJECT = settings.emg_subject  # EMG_SUBJECT in the root .env (shared with m1.ipynb)
MODEL_PATH = REPO_ROOT / "models" / f"subject{SUBJECT}_set_a.joblib"
# Recordings live in s1 ... s10 at the repo root (Suvrit's layout) or in data/s1 ... data/s10.
DATA_DIR = REPO_ROOT if (REPO_ROOT / f"s{SUBJECT}").is_dir() else REPO_ROOT / "data"
DATA_FILE = DATA_DIR / f"s{SUBJECT}" / f"S{SUBJECT}_E2_A1.mat"
TRAIN_REPS = [1, 2, 3, 4, 5]  # same as m1.ipynb T2: repetition 6 is held out for the replay
HELD_OUT_REP = 6


class EMGSetupError(RuntimeError):
    pass


def ensure_model(log: EventLog | None = None) -> Path:
    """Make sure the model for EMG_SUBJECT exists; train it (exactly like m1.ipynb T2) if not."""
    if not DATA_FILE.exists():
        have = sorted(p.name for p in DATA_DIR.glob("s*") if p.is_dir()) if DATA_DIR.exists() else []
        raise EMGSetupError(f"EMG_SUBJECT={SUBJECT}: recording {DATA_FILE.relative_to(REPO_ROOT)} not found "
                            f"(subjects found: {', '.join(have) or 'none'}).")
    if MODEL_PATH.exists():
        import sklearn
        from emg_data.model import load_model
        try:
            saved = load_model(str(MODEL_PATH))
        except Exception:  # noqa: BLE001 - a model pickled by another scikit-learn may not load at all
            saved = {}
        # Reuse it only if it is for this subject and was saved with the scikit-learn we have now.
        if saved.get("subject") == SUBJECT and saved.get("sklearn_version") == sklearn.__version__:
            return MODEL_PATH
        if log:
            log.emit("emg_model_mismatch", path=str(MODEL_PATH.relative_to(REPO_ROOT)),
                     saved_subject=saved.get("subject"), saved_sklearn=saved.get("sklearn_version"))
    from emg_data.emg import extract_features, load_db5
    from emg_data.model import GESTURES, save_model, train_model

    emg_windows, y, rep, fs = load_db5(SUBJECT, exercise_number=2, window_length=50, step_size=25, data_dir=str(DATA_DIR))
    model = train_model(extract_features(emg_windows), y, rep, train_reps=TRAIN_REPS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(MODEL_PATH), subject=SUBJECT, exercise=2, train_reps=TRAIN_REPS, held_out_rep=HELD_OUT_REP,
               window_length=50, step_size=25, sampling_freq=fs,
               features="MAV, RMS, WL for each of the 16 channels (48 numbers)", equal_priors=True, gestures=GESTURES)
    if log:
        log.emit("emg_model_trained", subject=SUBJECT, path=str(MODEL_PATH.relative_to(REPO_ROOT)))
    return MODEL_PATH

# Gestures our (set A) model can decode. Suvrit's "tab" gesture (2) needs his set-C model, so for now
# tab comes from the panel / keyboard only.
EMG_GESTURES = [g for g in COMMAND_FOR_GESTURE if g in CLASSES]
GESTURE_FOR_COMMAND: dict[str, int] = {COMMAND_FOR_GESTURE[g]: g for g in EMG_GESTURES}
GESTURE_NAMES = {"fist": 6, "spread": 5, "radial": 15, "ulnar": 16}
# Screens where the user is expected to make a gesture.
WAITING_MODES = {"home", "browse", "result", "ask", "confirm", "text"}

def recording_sequence() -> list[str]:
    """The model's gestures as commands, in the order they appear in the subject's recording."""
    order = gestures_in_recording_order(SUBJECT, 2, EMG_GESTURES, HELD_OUT_REP, str(DATA_DIR))
    return [COMMAND_FOR_GESTURE[g] for g in order]


def parse_sequence(text: str | None) -> list[str]:
    """'scroll_down,click' (commands) or 'fist,ulnar' (gestures) -> command names.
    No sequence given: the gestures in the order they appear in the recording."""
    if not text:
        return recording_sequence()
    out = []
    for raw in text.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name in GESTURE_FOR_COMMAND:
            out.append(name)
        elif name in GESTURE_NAMES:
            out.append(COMMAND_FOR_GESTURE[GESTURE_NAMES[name]])
        else:
            raise ValueError(f"unknown gesture {raw.strip()!r}; use {', '.join(GESTURE_FOR_COMMAND)} "
                             f"or {', '.join(GESTURE_NAMES)}")
    return out


class EMGReplaySource:
    def __init__(self, *, bus: IntentBus, log: EventLog, session: Any, sequence: list[str],
                 speed: float | None = 1.0, repeat: bool = False, gap_s: float = 1.5):
        self.bus = bus
        self.log = log
        self.session = session
        self.sequence = sequence
        self.speed = speed
        self.repeat = repeat
        self.gap_s = gap_s

    async def run(self) -> None:
        self.log.emit("emg_start", sequence=self.sequence, speed=self.speed, repeat=self.repeat,
                      subject=SUBJECT, model=str(MODEL_PATH.relative_to(REPO_ROOT)))
        for i, command in enumerate(self.sequence, 1):
            await self._wait_for_menu()
            await asyncio.sleep(self.gap_s)  # a pause a person would take before the next gesture
            await self.perform(command, step=i)
        self.log.emit("emg_done", gestures=len(self.sequence))

    async def perform(self, command: str, step: int = 0) -> list[dict[str, Any]]:
        """Replay one recorded gesture through the classifier and forward what it decodes."""
        gesture = GESTURE_FOR_COMMAND[command]
        loop = asyncio.get_running_loop()
        self.log.emit("emg_gesture_start", step=step, intended=command, gesture=gesture)

        def worker() -> list[dict[str, Any]]:
            fired: list[dict[str, Any]] = []
            for item in replay_commands(model_path=str(MODEL_PATH), subject_number=SUBJECT, data_dir=str(DATA_DIR),
                                        gestures=[gesture], speed=self.speed):
                ev = item["event"]
                if not ev:
                    continue
                fired.append(ev)
                if len(fired) == 1 or self.repeat:
                    loop.call_soon_threadsafe(self._forward, ev, command)
            return fired

        fired = await asyncio.to_thread(worker)
        forwarded = len(fired) if self.repeat else min(len(fired), 1)
        self.log.emit("emg_gesture_end", step=step, intended=command,
                      decoded=[e["command"] for e in fired], forwarded=forwarded,
                      suppressed_repeats=len(fired) - forwarded,
                      ok=bool(fired) and fired[0]["command"] == command)
        return fired

    def _forward(self, ev: dict[str, Any], intended: str) -> None:
        self.log.emit("emg_command", command=ev["command"], gesture=ev["gesture"],
                      confidence=round(ev["confidence"], 3), replay_t=ev["t"], intended=intended)
        self.bus.publish(ev["command"], source="emg")

    async def _wait_for_menu(self) -> None:
        while self.session.view.get("mode") not in WAITING_MODES:
            await asyncio.sleep(0.2)
