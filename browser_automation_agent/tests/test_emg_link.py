"""Link between Suvrit's EMG classifier (emg_data) and our IntentBus."""
import asyncio

import pytest

from myopilot.config import REPO_ROOT
from myopilot.intents import Intent, IntentBus, to_intent


def test_our_commands_are_exactly_suvrits():
    from emg_data.commands import COMMAND_FOR_GESTURE
    assert sorted(i.value for i in Intent) == sorted(COMMAND_FOR_GESTURE.values())


@pytest.mark.parametrize("name,intent", [
    ("scroll_down", Intent.SCROLL_DOWN), ("scroll_up", Intent.SCROLL_UP), ("forward", Intent.FORWARD),
    ("back", Intent.BACK), ("FORWARD", Intent.FORWARD), (Intent.BACK, Intent.BACK),
])
def test_commands_parse(name, intent):
    assert to_intent(name) is intent


@pytest.mark.parametrize("name", ["jump", "NEXT", "SELECT"])
def test_unknown_or_old_command_rejected(name):
    with pytest.raises(ValueError):
        to_intent(name)


def test_parse_sequence_accepts_commands_and_gesture_names():
    from myopilot.emg_source import parse_sequence
    assert parse_sequence("fist, ulnar") == ["scroll_down", "forward"]
    assert parse_sequence("scroll_down,forward,back") == ["scroll_down", "forward", "back"]
    assert parse_sequence("scroll_up") == ["scroll_up"]
    with pytest.raises(ValueError):
        parse_sequence("wave")


from myopilot.config import settings  # noqa: E402

_S = settings.emg_subject
needs_data = pytest.mark.skipif(not (REPO_ROOT / "data" / f"s{_S}" / f"S{_S}_E2_A1.mat").exists(),
                                reason=f"NinaPro DB5 recording for subject {_S} not present")


@pytest.fixture(autouse=True, scope="module")
def _model_for_subject():
    """Train models/subject<N>_set_a.joblib for EMG_SUBJECT if it is missing (same as main.py --emg)."""
    if (REPO_ROOT / "data" / f"s{_S}" / f"S{_S}_E2_A1.mat").exists():
        from myopilot.emg_source import ensure_model
        ensure_model()


@needs_data
def test_default_sequence_follows_the_recording():
    """No hand-picked order: the replay and --emg default follow the recording itself."""
    import numpy as np
    from emg_data.commands import COMMAND_FOR_GESTURE
    from emg_data.emg import load_raw_db5
    from emg_data.stream import build_replay
    from myopilot.emg_source import DATA_DIR, parse_sequence

    _, restim, rerep, _ = load_raw_db5(_S, 2, str(DATA_DIR))
    first = {g: np.flatnonzero((restim == g) & (rerep == 6))[0] for g in COMMAND_FOR_GESTURE}
    expected = sorted(first, key=first.get)
    replay = build_replay(_S, 2, gestures=[16, 15, 6, 5], replay_rep=6, data_dir=str(DATA_DIR))  # list order ignored
    assert [s["gesture"] for s in replay["segments"]] == expected
    assert parse_sequence(None) == [COMMAND_FOR_GESTURE[g] for g in expected]


@needs_data
def test_model_matches_emg_subject():
    from emg_data.model import load_model
    from myopilot.emg_source import MODEL_PATH, SUBJECT
    assert MODEL_PATH.name == f"subject{SUBJECT}_set_a.joblib"
    assert load_model(str(MODEL_PATH))["subject"] == SUBJECT == _S


class _Log:
    def __init__(self):
        self.events = []

    def emit(self, type_, **data):
        self.events.append({"type": type_, **data})


@needs_data
@pytest.mark.parametrize("repeat,expected_forwarded", [(False, 1), (True, None)])
def test_held_fist_forwards_one_scroll_down_unless_repeat(repeat, expected_forwarded):
    from myopilot.emg_source import EMGReplaySource

    async def run():
        bus, log = IntentBus(), _Log()
        src = EMGReplaySource(bus=bus, log=log, session=None, sequence=[], speed=None, repeat=repeat)
        fired = await src.perform("scroll_down")
        await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run
        published = []
        while not bus._q.empty():
            published.append(bus._q.get_nowait()[0])
        return fired, published, log.events

    fired, published, events = asyncio.run(run())
    assert len(fired) > 1, "a held fist should make Suvrit's logic repeat scroll_down"
    assert all(e["command"] == "scroll_down" for e in fired)
    assert published == [Intent.SCROLL_DOWN] * (expected_forwarded or len(fired))
    end = next(e for e in events if e["type"] == "emg_gesture_end")
    assert end["ok"] is True


@needs_data
@pytest.mark.parametrize("command,intent", [("scroll_up", Intent.SCROLL_UP), ("back", Intent.BACK),
                                            ("forward", Intent.FORWARD)])
def test_each_recorded_gesture_decodes_to_its_command(command, intent):
    from myopilot.emg_source import EMGReplaySource

    async def run():
        bus = IntentBus()
        await EMGReplaySource(bus=bus, log=_Log(), session=None, sequence=[], speed=None).perform(command)
        await asyncio.sleep(0)
        return bus._q.get_nowait()[0]

    assert asyncio.run(run()) is intent
