"""Hand pose from the 21 MediaPipe hand landmarks (no camera or MediaPipe needed in this file).

PoseClassifier.classify(landmarks)  one frame -> a pose name
PoseDebouncer.update(pose)          frame by frame -> the pose, once, when it has been stable for a few frames

Poses and the command each one stands for (through the recording library):

    fist     all four fingers folded                         -> gesture "fist"    -> scroll_down
    spread   four fingers extended and apart                 -> gesture "spread"  -> scroll_up
    v        index and middle extended, ring and pinky folded -> gesture "tab"    -> tab
    radial   flat hand tilted towards the thumb side         -> gesture "radial"  -> back
    ulnar    flat hand tilted towards the little-finger side -> gesture "ulnar"   -> click

Other results: "flat" (fingers together, hand upright: the neutral pose between radial and ulnar) and "none".

Landmarks are an array of shape (21, 2): x to the right, y downwards, any scale (MediaPipe gives 0 to 1).
Only x and y are used. The numbers of the landmarks: 0 wrist; 1 to 4 thumb; 5 to 8 index; 9 to 12 middle;
13 to 16 ring; 17 to 20 little finger; each finger goes base (MCP), PIP, DIP, tip.

Radial and ulnar are the hard pair on a 2D camera. The assumption is that the forearm is held upright in the
picture and the hand is bent sideways at the wrist, so the tilt of the hand axis against the vertical is used.
The thresholds are settings; they have been checked on synthetic hands only, not on a real camera.
"""

import numpy as np

WRIST = 0
FINGERS = {            # name -> (MCP, PIP, DIP, tip)
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# The poses that trigger a recording, and the gesture name in the recording library
POSE_TO_GESTURE = {"fist": "fist", "spread": "spread", "v": "tab", "radial": "radial", "ulnar": "ulnar"}
TRIGGER_POSES = list(POSE_TO_GESTURE)

DEFAULT_SETTINGS = {
    "extended_ratio": 1.4,      # a finger counts as extended if its tip is this many times farther from the wrist than its base
    "spread_angle_deg": 22.0,   # extended fingers count as spread if index and little finger point this far apart
    "radial_min_deg": 30.0,     # a flat hand tilted at least this far towards the thumb side is radial
    "ulnar_min_deg": 15.0,      # a flat hand tilted at least this far towards the little-finger side is ulnar
    "tilt_max_deg": 75.0,       # tilted further than this is probably not a wrist movement
}


def _unit(vector):
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length


def _angle_between(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


class PoseClassifier:
    def __init__(self, settings=None, neutral_tilt_deg=0.0):
        self.settings = dict(DEFAULT_SETTINGS)
        if settings is not None:
            self.settings.update(settings)
        self.neutral_tilt_deg = neutral_tilt_deg      # the tilt of this user's neutral hand (see calibrate_neutral)

    def measure(self, landmarks):
        # Everything the decision is based on, so that it can be shown on screen and tuned.
        points = np.asarray(landmarks, dtype=float)[:, :2]
        wrist = points[WRIST]
        palm_size = np.linalg.norm(points[9] - wrist)

        extended = {}
        for name, (mcp, pip, dip, tip) in FINGERS.items():
            ratio = np.linalg.norm(points[tip] - wrist) / max(np.linalg.norm(points[mcp] - wrist), 1e-9)
            extended[name] = bool(ratio > self.settings["extended_ratio"])

        # The angle between the index finger and the little finger (base to tip)
        index_axis = points[8] - points[5]
        pinky_axis = points[20] - points[17]
        spread_angle = _angle_between(index_axis, pinky_axis)

        # The tilt of the hand axis (wrist to the base of the middle finger) against the picture's vertical.
        # Positive = leaning towards the thumb side (radial), negative = towards the little-finger side (ulnar).
        #   lean  = how far the hand axis leans to the right of the picture (negative = to the left)
        #   side  = which side the thumb is on, taken from the landmarks (little-finger base to index base):
        #           +1 if the thumb side is on the right of an upright hand, -1 if it is on the left
        # tilt = lean * side, so it works for a left or a right hand, palm or back of the hand, and a mirrored picture.
        axis = _unit(points[9] - wrist)
        radial_direction = _unit(points[5] - points[17])
        side = 1.0
        if axis[0] * radial_direction[1] - axis[1] * radial_direction[0] < 0:
            side = -1.0
        lean = float(np.degrees(np.arctan2(axis[0], -axis[1])))
        fingers_up = float(axis[1]) < 0
        tilt = lean * side - self.neutral_tilt_deg

        return {"extended": extended, "spread_angle_deg": spread_angle, "tilt_deg": tilt,
                "fingers_up": fingers_up, "palm_size": float(palm_size)}

    def classify(self, landmarks):
        # Returns (pose name, measurements). Pose name is one of fist, spread, v, radial, ulnar, flat, none.
        if landmarks is None:
            return "none", None
        m = self.measure(landmarks)
        e = m["extended"]
        count = sum(e.values())

        if count == 0:
            pose = "fist"
        elif e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
            pose = "v"
        elif count == 4:
            if m["spread_angle_deg"] >= self.settings["spread_angle_deg"]:
                pose = "spread"
            elif not m["fingers_up"]:
                pose = "none"
            elif self.settings["radial_min_deg"] <= m["tilt_deg"] <= self.settings["tilt_max_deg"]:
                pose = "radial"
            elif -self.settings["tilt_max_deg"] <= m["tilt_deg"] <= -self.settings["ulnar_min_deg"]:
                pose = "ulnar"
            elif -self.settings["ulnar_min_deg"] < m["tilt_deg"] < self.settings["radial_min_deg"]:
                pose = "flat"
            else:
                pose = "none"
        else:
            pose = "none"
        return pose, m

    def calibrate_neutral(self, landmarks):
        # Call with the user's hand held upright and flat: whatever tilt it shows becomes the zero point.
        self.neutral_tilt_deg = 0.0
        self.neutral_tilt_deg = self.measure(landmarks)["tilt_deg"]
        return self.neutral_tilt_deg


class PoseDebouncer:
    def __init__(self, stable_frames=5, release_frames=5):
        # stable_frames: how many frames in a row a pose must be seen before it counts (5 frames = about 0.17 s at 30 fps)
        # release_frames: how many frames in a row of "flat", "none" or no hand it takes to release the hand.
        # RELEASE FIRST: after a pose has fired, no new pose can fire until the hand has been released. This keeps the
        # shapes a hand passes through while relaxing (an opening fist can look like spread or ulnar) from firing.
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.candidate = None
        self.count = 0
        self.fired = False
        self.suppressed = None       # set to the pose on the frame where a stable pose was held back by release-first

    def update(self, pose):
        # Call once per frame. Returns the pose on the frame where it is confirmed, otherwise None.
        self.suppressed = None
        if pose == self.candidate:
            self.count += 1
        else:
            self.candidate = pose
            self.count = 1

        if pose in TRIGGER_POSES:
            if self.count == self.stable_frames:
                if self.fired:
                    self.suppressed = pose      # a stable pose, but the hand has not been released since the last one
                else:
                    self.fired = True
                    return pose
        elif self.count == self.release_frames:
            self.fired = False          # released: the next pose is a new onset
        return None


class NeutralCalibrator:
    # Finds the tilt of the user's own "upright and flat" hand automatically: it listens for frames in which all four
    # fingers are extended and together, and after frames_needed of them it takes the median tilt as the neutral tilt.
    def __init__(self, classifier, frames_needed=30, max_abs_tilt=60.0):
        self.classifier = classifier
        self.frames_needed = frames_needed
        self.max_abs_tilt = max_abs_tilt
        self.classifier.neutral_tilt_deg = 0.0
        self.values = []
        self.done = False

    @property
    def progress(self):
        return min(len(self.values), self.frames_needed)

    def update(self, landmarks):
        # Call once per frame while calibrating. Returns True once the neutral tilt has been set.
        if self.done:
            return True
        if landmarks is None:
            return False
        m = self.classifier.measure(landmarks)
        flat_hand = (all(m["extended"].values()) and m["spread_angle_deg"] < self.classifier.settings["spread_angle_deg"]
                     and m["fingers_up"] and abs(m["tilt_deg"]) < self.max_abs_tilt)
        if flat_hand:
            self.values.append(m["tilt_deg"])
            if len(self.values) >= self.frames_needed:
                self.classifier.neutral_tilt_deg = float(np.median(self.values))
                self.done = True
        return self.done
