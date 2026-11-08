"""Made-up hand landmarks for testing the pose classifier without a camera.

A simple flat hand in the 21-landmark layout of MediaPipe. It is NOT a real hand, so tests with it show that the
logic is consistent, not how well it works on a camera.
"""

import numpy as np

# Palm of size 1 (wrist to the base of the middle finger), fingers pointing up (negative y), radial side towards +x
BASES = {"index": (0.30, -0.95), "middle": (0.0, -1.0), "ring": (-0.28, -0.93), "pinky": (-0.54, -0.82)}
SPREAD_ANGLES = {"index": 20, "middle": 5, "ring": -8, "pinky": -22}     # degrees from straight up, for a spread hand
FLAT_ANGLES = {"index": 4, "middle": 0, "ring": -3, "pinky": -6}          # fingers together
FIRST_LANDMARK = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def _direction(angle_deg):
    a = np.radians(angle_deg)
    return np.array([np.sin(a), -np.cos(a)])      # 0 degrees = up, positive = towards +x


def make_hand(extended, spread=False, tilt_deg=0.0, scale=1.0, shift=(0.0, 0.0), mirror=False,
              noise=0.0, rotate_deg=0.0, rng=None):
    # extended: dict finger name -> True/False
    # tilt_deg: positive = the whole hand tilted towards the thumb (radial) side, negative = towards the little finger
    # rotate_deg: an extra rotation of the whole hand in the picture (a hand that is not held upright)
    # noise: standard deviation of the noise on every landmark, as a fraction of the palm size
    points = np.zeros((21, 2))
    angles = SPREAD_ANGLES if spread else FLAT_ANGLES

    # thumb (not used by the classifier, but present in a real hand)
    points[1] = (0.25, -0.25)
    points[2] = (0.45, -0.45)
    points[3] = (0.60, -0.62)
    points[4] = (0.70, -0.78)

    for name, first in FIRST_LANDMARK.items():
        base = np.array(BASES[name])
        direction = _direction(angles[name])
        if extended[name]:
            pip = base + 0.40 * direction
            dip = pip + 0.25 * direction
            tip = dip + 0.20 * direction
        else:                                      # folded into the palm
            pip = base + 0.30 * direction
            dip = pip - 0.05 * direction
            tip = base + 0.12 * direction
        points[first] = base
        points[first + 1] = pip
        points[first + 2] = dip
        points[first + 3] = tip

    # rotate about the wrist: tilt (towards +x for positive angles) plus any extra rotation
    phi = np.radians(tilt_deg + rotate_deg)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    points = points @ rotation.T

    if rng is not None and noise > 0:
        points = points + rng.normal(0, noise, points.shape)
    if mirror:
        points[:, 0] = -points[:, 0]
    return points * scale + np.array(shift)


ALL = {"index": True, "middle": True, "ring": True, "pinky": True}
NONE = {"index": False, "middle": False, "ring": False, "pinky": False}
V = {"index": True, "middle": True, "ring": False, "pinky": False}


def make_pose(name, **kwargs):
    # A hand showing the named pose: fist, spread, v, radial, ulnar or flat
    if name == "fist":
        return make_hand(NONE, **kwargs)
    if name == "spread":
        return make_hand(ALL, spread=True, **kwargs)
    if name == "v":
        return make_hand(V, **kwargs)
    if name == "flat":
        return make_hand(ALL, **kwargs)
    if name == "radial":
        return make_hand(ALL, tilt_deg=kwargs.pop("tilt_deg", 40.0), **kwargs)
    if name == "ulnar":
        return make_hand(ALL, tilt_deg=kwargs.pop("tilt_deg", -25.0), **kwargs)
    raise ValueError(name)
