"""Run the camera pipeline: webcam -> hand pose -> recorded EMG -> classifier -> commands.

    python -m camera.demo                       camera window; commands are printed
    python -m camera.demo --post http://localhost:8765     also send every command to the control panel (POST /intent/<command>)
    python -m camera.demo --selftest            no camera: feed made-up hands through the whole chain and check the commands
    python -m camera.demo --check               guided check with your hand: how often is each pose recognised?

Keys in the camera window:  1 fist   2 spread   3 V (tab)   4 radial   5 ulnar   (manual triggers, the fallback)
                            n  take the hand shown now as "upright and flat" (calibrates radial / ulnar)
                            q  quit

The camera is the recogniser; the EMG is a recorded NinaPro signal that the classifier decodes. Not live EMG.
"""

import argparse
import sys
import time
import urllib.request

from .pipeline import CameraPipeline
from .poses import POSE_TO_GESTURE, NeutralCalibrator, PoseClassifier, PoseDebouncer

MANUAL_KEYS = {"1": "fist", "2": "spread", "3": "v", "4": "radial", "5": "ulnar"}


def post_intent(base_url, command):
    # The control panel takes commands from other processes with POST /intent/<command>
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/intent/{command}", data=b"", method="POST")
        urllib.request.urlopen(request, timeout=1.0).read()
        return True
    except Exception as error:
        print(f"   (could not send {command} to {base_url}: {error})")
        return False


def make_pipeline(args, state):
    def on_command(event):
        mark = "" if event["match"] else "   <-- MISMATCH, camera saw " + event["camera_gesture"] + ", expected " + event["expected_command"]
        print(f"   command: {event['command']:12s} (camera: {event['camera_gesture']}, confidence {event['confidence']:.3f}){mark}")
        state["last_command"] = event["command"] + ("" if event["match"] else " (mismatch)")
        if args.post:
            post_intent(args.post, event["command"])

    def on_recording(kind, info):
        if kind == "start":
            print(f"camera saw {info['camera_gesture']:7s} -> playing a recording of repetition {info['repetition']}")
        else:
            print(f"   recording finished: decoded {info['decoded_commands']}")

    return CameraPipeline(on_command=on_command, on_recording=on_recording, speed=args.speed,
                          lead_in_s=args.lead_in, model_path=args.model, data_dir=args.data_dir)


def selftest(args):
    # No camera: made-up hands through the pose classifier, the debouncer and the pipeline
    from .synthetic import make_pose
    state = {}
    pipeline = make_pipeline(args, state)
    classifier = PoseClassifier()
    debouncer = PoseDebouncer()
    commands = []
    inner = pipeline.on_command
    pipeline.on_command = lambda event: (commands.append(event["command"]), inner(event))

    expected = {"fist": "scroll_down", "spread": "scroll_up", "v": "tab", "radial": "back", "ulnar": "click"}
    for pose in expected:
        for frame in [make_pose("flat")] * 8 + [make_pose(pose)] * 10:
            name, _ = classifier.classify(frame)
            onset = debouncer.update(name)
            if onset is not None:
                pipeline.trigger(onset)
                pipeline.wait()
    ok = commands == list(expected.values())
    print("selftest:", "PASSED" if ok else f"FAILED, got {commands}")
    return 0 if ok else 1


def run_camera(args):
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
    classifier = PoseClassifier()
    debouncer = PoseDebouncer(stable_frames=args.stable_frames)
    state = {"last_command": "-"}
    pipeline = make_pipeline(args, state)
    calibrator = None if args.no_auto_calibrate else NeutralCalibrator(classifier)
    last_landmarks = None
    frames = 0
    started = time.perf_counter()
    print("camera running. Keys: 1 fist, 2 spread, 3 V, 4 radial, 5 ulnar (manual), n = calibrate, q = quit")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("no frame from the camera")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)          # like a mirror; the classifier does not care

            landmarks, handedness = tracker.process(frame)
            last_landmarks = landmarks if landmarks is not None else last_landmarks

            if calibrator is not None and not calibrator.done:
                # start-up: wait for the user's flat hand, nothing can trigger yet
                if calibrator.update(landmarks):
                    print(f"calibrated: your neutral tilt is {classifier.neutral_tilt_deg:+.1f} degrees. Ready.")
                lines = ["Show a FLAT hand, fingers together, upright", f"calibrating {calibrator.progress} / {calibrator.frames_needed}"]
                cv2.imshow("MyoPilot camera", draw_overlay(frame, landmarks, lines))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            pose, m = classifier.classify(landmarks)
            onset = debouncer.update(pose)
            if debouncer.suppressed is not None:
                print(f"saw {debouncer.suppressed} but it was ignored: relax your hand (flat, or out of view) between poses")
            if onset is not None:
                if m is not None:
                    print(f"saw {onset}: tilt {m['tilt_deg']:+.0f} deg, spread {m['spread_angle_deg']:.0f} deg, "
                          f"fingers extended {sum(m['extended'].values())} of 4, neutral tilt {classifier.neutral_tilt_deg:+.0f} deg")
                if not pipeline.trigger(onset):
                    print(f"camera saw {onset} but it was ignored: {pipeline.last_reject_reason}")

            frames += 1
            lines = [f"pose: {pose}" + ("" if m is None else f"   tilt {m['tilt_deg']:+.0f} deg, spread {m['spread_angle_deg']:.0f} deg"),
                     "status: " + ("recording playing" if pipeline.busy else ("blocked" if pipeline.blocked else "ready"))
                     + ("   |   RELAX your hand to arm the next pose" if debouncer.fired else "   |   armed"),
                     f"last command: {state['last_command']}",
                     f"{frames / (time.perf_counter() - started):.0f} fps    1-5 manual, n calibrate, q quit"]
            cv2.imshow("MyoPilot camera", draw_overlay(frame, landmarks, lines))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                if last_landmarks is not None:
                    offset = classifier.calibrate_neutral(landmarks if landmarks is not None else last_landmarks)
                    print(f"neutral tilt set to {offset:+.1f} degrees")
            if chr(key) in MANUAL_KEYS:
                pipeline.trigger(MANUAL_KEYS[chr(key)])
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()
        pipeline.wait(5)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camera hand pose -> recorded EMG -> classifier -> commands")
    parser.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    parser.add_argument("--width", type=int, default=0, help="ask the camera for this width (0 = camera default)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-mirror", action="store_true", help="do not flip the picture like a mirror")
    parser.add_argument("--post", default="", help="control panel address, e.g. http://localhost:8765: send every command to POST /intent/<command>")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = recordings play in real time")
    parser.add_argument("--lead-in", type=float, default=0.4, help="seconds of rest played before a gesture")
    parser.add_argument("--stable-frames", type=int, default=5, help="frames a pose must be held before it counts")
    parser.add_argument("--model", default="models/subject1_set_c.joblib")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--no-auto-calibrate", action="store_true", help="do not wait for a flat hand at the start (use n to calibrate)")
    parser.add_argument("--selftest", action="store_true", help="no camera: check the whole chain with made-up hands")
    parser.add_argument("--check", action="store_true", help="guided check of the pose recogniser with your hand (counts the successes)")
    parser.add_argument("--tries", type=int, default=5, help="tries per pose in --check")
    args = parser.parse_args(argv)

    if args.check:
        from .check import check_with_camera
        return check_with_camera(args)
    if args.selftest:
        args.speed = None if args.speed == 1.0 else args.speed        # as fast as possible unless a speed is given
        return selftest(args)
    return run_camera(args)


if __name__ == "__main__":
    sys.exit(main())
