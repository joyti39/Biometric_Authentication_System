"""
liveness/blink_check.py - Layer 1 of liveness detection: Blink Detection

Why blink detection catches photo spoofing:
  A printed photo or static image on a screen NEVER blinks.
  A real, live person blinks involuntarily every few seconds.
  So if we watch someone's eyes for a few seconds and see zero blinks,
  it's very likely we're looking at a photo, not a real face.

How it works (MediaPipe Tasks API, mediapipe >= 0.10 / 1.0.0):
  The old `mp.solutions.face_mesh` API is gone in newer mediapipe
  versions. The replacement is the "Tasks" API, which uses a
  downloadable .task model file (FaceLandmarker).

  Conveniently, FaceLandmarker can directly output "blendshapes" --
  pre-computed expression scores like "eyeBlinkLeft" and
  "eyeBlinkRight", each a number from 0 (eye fully open) to
  1 (eye fully closed). We don't need to manually compute Eye
  Aspect Ratio (EAR) from landmark points anymore -- the model
  gives us the blink signal directly.

  A blink = a brief moment where the blink score rises above a
  threshold, then falls back down.

This file can be run standalone to test blink detection before
wiring it into verify.py.
"""

import cv2
import numpy as np
import time
import sys
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Allow importing camera.py from the parent biometric_auth folder
# when this script is run directly from inside liveness/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import open_camera

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task"
)

BLINK_SCORE_THRESHOLD = 0.5   # blendshape score above this = eye considered "closing/closed"
CONSEC_FRAMES = 1              # how many consecutive closed-eye detections count as one blink
                                # (kept at 1 since we detect at ~10-15fps over a slow network
                                # stream -- a real blink may only register in a single frame;
                                # the cooldown below is what actually prevents over-counting)


def get_blink_score(blendshapes):
    """
    Given the list of blendshape categories for one face, extract the
    average of the left/right eye blink scores.

    Returns None if blink categories aren't found (shouldn't normally happen).
    """
    left = None
    right = None
    for category in blendshapes:
        if category.category_name == "eyeBlinkLeft":
            left = category.score
        elif category.category_name == "eyeBlinkRight":
            right = category.score
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def test_blink_detection():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        print("Download it first with:")
        print(f"  curl -L -o {MODEL_PATH} "
              f"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              f"face_landmarker/float16/latest/face_landmarker.task")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    cam = open_camera()
    if cam is None:
        print("Could not open camera. Make sure ffmpeg is running on Windows.")
        return

    blink_count = 0
    consec_closed = 0
    last_blink_time = 0
    BLINK_COOLDOWN_SECONDS = 0.4  # minimum time between two counted blinks,
                                  # prevents one real blink's noisy score from
                                  # being counted multiple times
    start_time = time.time()

    print("Look at the camera and blink naturally. Press 'q' to quit.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        blink_score = None
        if result.face_blendshapes:
            blink_score = get_blink_score(result.face_blendshapes[0])

            now_t = time.time()
            if blink_score is not None and blink_score > BLINK_SCORE_THRESHOLD:
                consec_closed += 1
            else:
                if (consec_closed >= CONSEC_FRAMES
                        and (now_t - last_blink_time) > BLINK_COOLDOWN_SECONDS):
                    blink_count += 1
                    last_blink_time = now_t
                consec_closed = 0

        elapsed = time.time() - start_time

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if blink_score is not None:
            cv2.putText(frame, f"Blink score: {blink_score:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Blink Detection Test - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nTotal blinks detected: {blink_count}")


if __name__ == "__main__":
    test_blink_detection()
