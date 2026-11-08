"""Webcam frames and hand landmarks (needs OpenCV and MediaPipe: see CAMERA.md).

HandTracker.process(frame) -> the 21 hand landmarks of one hand in PIXEL coordinates, or None
draw_overlay(frame, ...)   draws the hand and the recognised pose on the frame
"""

import os
import urllib.request

import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hand_landmarker.task")

# which landmarks to join with a line when drawing
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def ensure_model(path=MODEL_PATH):
    # The hand landmark model (7.8 MB) is downloaded once from Google's official MediaPipe model storage.
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading the MediaPipe hand landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, min_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        # Returns (landmarks, handedness): landmarks is an array (21, 2) in pixels (x right, y down), handedness is
        # "Left" or "Right" as MediaPipe sees it (not used for the pose). (None, None) if no hand is found.
        import cv2
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        if len(result.hand_landmarks) == 0:
            return None, None

        # MediaPipe gives x and y as fractions of the width and the height. They are turned into pixels, because
        # an angle measured on fractions would be wrong in a picture that is wider than it is tall.
        points = np.array([[p.x * width, p.y * height] for p in result.hand_landmarks[0]])
        handedness = result.handedness[0][0].category_name
        return points, handedness

    def close(self):
        self._detector.close()


def draw_overlay(frame, landmarks, lines, colour=(0, 200, 255)):
    # lines: the text to show in the top left corner, one string per line
    import cv2
    if landmarks is not None:
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(int(v) for v in landmarks[a]), tuple(int(v) for v in landmarks[b]), (0, 255, 0), 2)
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    y = 28
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        y += 28
    return frame
