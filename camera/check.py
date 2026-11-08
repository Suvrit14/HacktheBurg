"""A guided check of the pose recogniser with a real hand: how often is each pose recognised?

    python -m camera --check                 5 tries per pose
    python -m camera --check --tries 10

The window waits for your hand, calibrates your neutral hand (upright and flat), and then asks for one pose at a time:
"Show FIST, try 3 of 5". Each try is a fixed window of 4 seconds, then a relax phase. For every try it records, frame by
frame, what was recognised and the measurements (tilt, spread, number of extended fingers), and it scores two things:

  first pose     the first pose that held for 5 frames (what would trigger a recording in the real pipeline)
  held pose      the share of the frames in the last 2 seconds of the try that show the pose that was asked for

At the end it prints a table and saves everything to camera/check_results.json, so the thresholds in camera/poses.py
can be tuned from real data.

run_check() does not need a camera: it takes a function that returns the landmarks of the next frame.
"""

import json
import os
import time

import numpy as np

from .poses import TRIGGER_POSES, PoseClassifier, PoseDebouncer

CHECK_POSES = ["fist", "spread", "v", "radial", "ulnar"]
HINTS = {
    "fist": "close your hand into a fist",
    "spread": "open your hand, fingers wide apart",
    "v": "index and middle finger up, ring and little finger folded",
    "radial": "hand upright and flat, fingers together, then tilt it towards your THUMB",
    "ulnar": "hand upright and flat, fingers together, then tilt it towards your LITTLE FINGER",
}
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_results.json")


def run_check(next_landmarks, show, tries=5, poses=CHECK_POSES, stable_frames=5, try_s=4.0, relax_s=1.5,
              calibrate_s=3.0, wait_for_hand_s=30.0, clock=time.perf_counter, classifier=None):
    # next_landmarks(): reads the next frame and returns its landmarks (or None if there is no hand)
    # show(lines, landmarks): shows the prompt on the screen; returns False if the user wants to stop
    # Returns the results dict, or None if the user stopped or no hand was ever shown.
    classifier = classifier or PoseClassifier()
    classifier.neutral_tilt_deg = 0.0

    # 1. Wait until a hand has been in view for one second, then measure the neutral tilt for calibrate_s seconds
    started = clock()
    seen_since = None
    while True:
        landmarks = next_landmarks()
        now = clock()
        if landmarks is None:
            seen_since = None
        elif seen_since is None:
            seen_since = now
        if seen_since is not None and now - seen_since >= 1.0:
            break
        if now - started > wait_for_hand_s:
            print("no hand was shown, stopping")
            return None
        if not show(["Show your hand to the camera", "(upright, fingers together)"], landmarks):
            return None

    tilts = []
    calibration_started = clock()
    while clock() - calibration_started < calibrate_s:
        landmarks = next_landmarks()
        if landmarks is not None:
            tilts.append(classifier.measure(landmarks)["tilt_deg"])
        if not show(["Hold your hand UPRIGHT and FLAT", "(fingers together, palm or back to the camera)",
                     f"calibrating... {max(0.0, calibrate_s - (clock() - calibration_started)):.0f} s"], landmarks):
            return None
    if len(tilts) < 10:
        print("the hand was lost during the calibration, stopping")
        return None
    classifier.neutral_tilt_deg = float(np.median(tilts))

    results = {"neutral_tilt_deg": classifier.neutral_tilt_deg, "settings": dict(classifier.settings),
               "stable_frames": stable_frames, "try_s": try_s, "tries": []}

    # 2. Every pose, several tries each. One debouncer for the whole check, so a pose that is still held from the
    #    previous try is not counted again: the hand has to be released first.
    debouncer = PoseDebouncer(stable_frames=stable_frames)
    for pose in poses:
        for number in range(1, tries + 1):
            frames = {"t": [], "pose": [], "tilt_deg": [], "spread_angle_deg": [], "n_extended": []}
            first_pose = None
            first_time = None
            started = clock()
            while clock() - started < try_s:
                landmarks = next_landmarks()
                name, m = classifier.classify(landmarks)
                elapsed = clock() - started
                if m is not None:
                    frames["t"].append(round(elapsed, 3))
                    frames["pose"].append(name)
                    frames["tilt_deg"].append(round(m["tilt_deg"], 1))
                    frames["spread_angle_deg"].append(round(m["spread_angle_deg"], 1))
                    frames["n_extended"].append(sum(m["extended"].values()))
                onset = debouncer.update(name)
                if onset is not None and first_pose is None:
                    first_pose, first_time = onset, elapsed
                if not show([f"Show {pose.upper()}   (try {number} of {tries})", HINTS[pose],
                             f"now: {name}      {max(0.0, try_s - elapsed):.0f} s"], landmarks):
                    return None

            results["tries"].append({"target": pose, "try": number, "first_pose": first_pose or "none",
                                     "first_pose_time": first_time, "frames": frames})

            # relax: a flat hand (or no hand) until the debouncer has seen it released
            started = clock()
            released_frames = 0
            while clock() - started < relax_s or (released_frames < stable_frames and clock() - started < 5.0):
                landmarks = next_landmarks()
                name, m = classifier.classify(landmarks)
                debouncer.update(name)
                released_frames = released_frames + 1 if name not in TRIGGER_POSES else 0
                if not show([f"{'RIGHT' if first_pose == pose else 'got: ' + str(first_pose or 'nothing')}  -  relax your hand",
                             "(flat hand, or take it out of view)"], landmarks):
                    return None
    return results


def held_share(try_result, pose, try_s, last_seconds=2.0):
    # Of the frames (with a hand) in the last `last_seconds` of the try, how many show `pose`? None if there are none.
    frames = try_result["frames"]
    chosen = [p for p, t in zip(frames["pose"], frames["t"]) if t >= try_s - last_seconds]
    if not chosen:
        return None
    return sum(1 for p in chosen if p == pose) / len(chosen)


def summarize(results):
    lines = []
    lines.append(f"neutral tilt of your hand: {results['neutral_tilt_deg']:+.1f} degrees")
    lines.append("pose     first pose right  held pose (last 2 s)  what the first pose was instead       median tilt   median spread")
    for pose in CHECK_POSES:
        tries = [t for t in results["tries"] if t["target"] == pose]
        if not tries:
            continue
        right = sum(1 for t in tries if t["first_pose"] == pose)
        others = {}
        for t in tries:
            if t["first_pose"] != pose:
                others[t["first_pose"]] = others.get(t["first_pose"], 0) + 1
        shares = [s for s in (held_share(t, pose, results['try_s'], 2.0) for t in tries) if s is not None]
        tilts = []
        spreads = []
        for t in tries:
            end = results["try_s"]
            for tilt, spread, time_ in zip(t["frames"]["tilt_deg"], t["frames"]["spread_angle_deg"], t["frames"]["t"]):
                if time_ >= end - 2.0:
                    tilts.append(tilt)
                    spreads.append(spread)
        lines.append(f"{pose:7s}  {right:2d} of {len(tries):<2d}         "
                     f"{(np.mean(shares) * 100 if shares else float('nan')):6.0f}% of frames    {str(others) if others else '-':35s} "
                     f"{(np.median(tilts) if tilts else float('nan')):+8.1f}     {(np.median(spreads) if spreads else float('nan')):8.1f}")
    return "\n".join(lines)


def save(results, path=RESULTS_PATH):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return path


def check_with_camera(args):
    import cv2
    from .capture import HandTracker, draw_overlay

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"could not open camera {args.camera}. Try --camera 1, and close other programs that use the camera.")
        return 1
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    tracker = HandTracker()
    state = {"frame": None, "frames": 0, "started": time.perf_counter()}

    def next_landmarks():
        ok, frame = capture.read()
        if not ok:
            state["frame"] = None
            return None
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        state["frame"] = frame
        state["frames"] += 1
        landmarks, handedness = tracker.process(frame)
        return landmarks

    def show(lines, landmarks):
        if state["frame"] is None:
            return True
        cv2.imshow("MyoPilot pose check", draw_overlay(state["frame"].copy(), landmarks, lines))
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    print("Pose check. Keep one hand in view. Press q in the window to stop.")
    try:
        results = run_check(next_landmarks, show, tries=args.tries, stable_frames=args.stable_frames)
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
    fps = state["frames"] / max(time.perf_counter() - state["started"], 1e-9)

    if results is None:
        print("stopped")
        return 1
    results["fps"] = fps
    print()
    print(f"camera and tracking ran at {fps:.0f} frames per second")
    print(summarize(results))
    print("saved to", save(results))
    return 0
