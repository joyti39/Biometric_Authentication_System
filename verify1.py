"""
verify.py - Face Verification + Liveness (Blink) Detection

"""

import cv2
import face_recognition
import sqlite3
import numpy as np
import os
import csv
import time
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from camera import open_camera, is_good_frame


# -- Load enrolled faces from database --------------------------------

def load_known_faces(conn):
    """
    Load all face embeddings from the database into memory.

    Returns two parallel lists:
      known_names      : ["Joyti", "Alice", ...]
      known_embeddings : [array(128,), array(128,), ...]
    """
    rows = conn.execute("SELECT name, embedding FROM faces").fetchall()

    known_names = []
    known_embeddings = []

    for name, blob in rows:
        embedding = np.frombuffer(blob, dtype=np.float64)
        known_names.append(name)
        known_embeddings.append(embedding)

    print(f"Loaded {len(known_names)} enrolled face(s): {set(known_names)}")
    return known_names, known_embeddings


# -- Logging ------------------------------------------------------------

def log_attempt(name, result, distance=None):
    """
    Append one authentication attempt to the CSV log file.
    Creates the file and header if it doesn't exist.
    """
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(config.LOG_PATH)

    with open(config.LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "result", "distance"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            result,
            f"{distance:.4f}" if distance is not None else ""
        ])


# -- Core face matching logic --------------------------------------------

def find_match(embedding, known_names, known_embeddings):
    """
    Compare a live face embedding against all stored embeddings.
    Returns: (matched_name or None, best_distance)
    """
    if not known_embeddings:
        return None, None

    matches = face_recognition.compare_faces(
        known_embeddings, embedding,
        tolerance=config.FACE_MATCH_TOLERANCE
    )
    distances = face_recognition.face_distance(known_embeddings, embedding)

    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]

    if matches[best_idx]:
        return known_names[best_idx], best_distance
    else:
        return None, best_distance


# -- Blink / liveness helpers ---------------------------------------------

# Single source of truth for the detection downscale factor used for
# face_recognition. Only change this ONE constant -- never add a second,
# separate scale-back multiplier anywhere else in the code.
DETECTION_SCALE = 0.4

# Blendshape score above this = eye considered "closing/closed".
# (Same value used and tuned in liveness/blink_check.py)
BLINK_SCORE_THRESHOLD = 0.5
BLINK_COOLDOWN_SECONDS = 0.4  # minimum time between two counted blinks


def get_blink_score(blendshapes):
    """
    Given the list of blendshape categories for one face, extract the
    average of the left/right eye blink scores (0 = open, 1 = closed).
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


def create_face_landmarker():
    if not os.path.exists(config.FACE_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"Face landmarker model not found at {config.FACE_LANDMARKER_MODEL_PATH}.\n"
            f"Download it with:\n"
            f"  curl -L -o {config.FACE_LANDMARKER_MODEL_PATH} "
            f"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            f"face_landmarker/float16/latest/face_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=config.FACE_LANDMARKER_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# -- Display helpers -------------------------------------------------------

def draw_scanning(frame, locations, matched_name=None, distance=None):
    display = frame.copy()
    for (top, right, bottom, left) in locations:
        color = (0, 200, 255)  # orange while just scanning
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        label = f"Scanning... ({matched_name})" if matched_name else "Scanning..."
        cv2.putText(display, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return display


def draw_liveness(frame, blinks_needed, blinks_so_far, seconds_left):
    display = frame.copy()
    cv2.putText(display, "Please blink naturally...", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.putText(display, f"Blinks: {blinks_so_far}/{blinks_needed}   Time left: {seconds_left:.1f}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return display


def draw_final(frame, locations, name, granted, distance=None, reason=""):
    color = (0, 255, 0) if granted else (0, 0, 255)
    display = frame.copy()
    for (top, right, bottom, left) in locations:
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
    label = f"Welcome, {name}!" if granted else "Access Denied"
    cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if reason:
        cv2.putText(display, reason, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display


# -- Main verification flow ------------------------------------------------

def verify():
    conn = sqlite3.connect(config.DB_PATH)
    known_names, known_embeddings = load_known_faces(conn)
    conn.close()

    if not known_names:
        print("No faces enrolled. Run enroll.py first.")
        return

    cam = open_camera()
    if cam is None:
        print("Cannot open camera. Make sure ffmpeg is running on Windows.")
        return

    detector = create_face_landmarker()

    print("\nVerification running (face match + blink liveness). Press 'q' to quit.")
    print(f"Match tolerance: {config.FACE_MATCH_TOLERANCE} (lower = stricter)")
    print(f"Blinks required: {config.BLINK_MIN_COUNT} within {config.BLINK_TIMEOUT_SECONDS}s\n")

    # State machine
    STATE_SCANNING = "SCANNING"
    STATE_LIVENESS = "LIVENESS_CHECK"
    STATE_RESULT = "RESULT"
    state = STATE_SCANNING

    last_face_check = 0
    last_locations = []
    last_display = None

    # Liveness-window variables (reset each time we enter LIVENESS_CHECK)
    liveness_person = None
    liveness_distance = None
    liveness_deadline = 0
    liveness_blink_count = 0
    liveness_consec_closed = 0
    liveness_last_blink_time = 0

    # Result-display variables
    result_until = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        now = time.time()
        show = frame

        # ---- SCANNING: look for a known face once per second ----
        if state == STATE_SCANNING:
            if now - last_face_check >= 1.0 and is_good_frame(frame):
                last_face_check = now

                small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
                locations = [
                    (int(t / DETECTION_SCALE), int(r / DETECTION_SCALE),
                     int(b / DETECTION_SCALE), int(l / DETECTION_SCALE))
                    for t, r, b, l in locations
                ]
                last_locations = locations

                if locations:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_full, locations)
                    if encodings:
                        matched_name, distance = find_match(
                            encodings[0], known_names, known_embeddings
                        )
                        if matched_name:
                            # Found a match -- move into the liveness check
                            state = STATE_LIVENESS
                            liveness_person = matched_name
                            liveness_distance = distance
                            liveness_deadline = now + config.BLINK_TIMEOUT_SECONDS
                            liveness_blink_count = 0
                            liveness_consec_closed = 0
                            liveness_last_blink_time = 0

            show = draw_scanning(frame, last_locations)

        # ---- LIVENESS_CHECK: watch for real blinks within the time window ----
        elif state == STATE_LIVENESS:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(now * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_blendshapes:
                blink_score = get_blink_score(result.face_blendshapes[0])
                if blink_score is not None and blink_score > BLINK_SCORE_THRESHOLD:
                    liveness_consec_closed += 1
                else:
                    if (liveness_consec_closed >= 1
                            and (now - liveness_last_blink_time) > BLINK_COOLDOWN_SECONDS):
                        liveness_blink_count += 1
                        liveness_last_blink_time = now
                    liveness_consec_closed = 0

            seconds_left = max(0.0, liveness_deadline - now)
            show = draw_liveness(frame, config.BLINK_MIN_COUNT, liveness_blink_count, seconds_left)

            if liveness_blink_count >= config.BLINK_MIN_COUNT:
                # Liveness passed -- GRANTED
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"GRANTED - {liveness_person} (dist={liveness_distance:.3f}, "
                      f"blinks={liveness_blink_count})")
                log_attempt(liveness_person, config.AUTH_SUCCESS, liveness_distance)
                last_display = draw_final(frame, last_locations, liveness_person,
                                           True, liveness_distance)
                state = STATE_RESULT
                result_until = now + 2.0

            elif now >= liveness_deadline:
                # Timed out without enough blinks -- likely a spoof (photo)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"DENIED  - {liveness_person} matched face but failed liveness "
                      f"(blinks={liveness_blink_count}) -- possible spoof")
                log_attempt(liveness_person, config.AUTH_FAIL_SPOOF, liveness_distance)
                last_display = draw_final(frame, last_locations, liveness_person, False,
                                           liveness_distance,
                                           reason="No blink detected -- possible photo/spoof")
                state = STATE_RESULT
                result_until = now + 2.0

        # ---- RESULT: briefly show the outcome, then go back to scanning ----
        elif state == STATE_RESULT:
            show = last_display if last_display is not None else frame
            if now >= result_until:
                state = STATE_SCANNING
                last_locations = []
                last_display = None

        cv2.imshow("Biometric Auth - press 'q' to quit", show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("\nVerification stopped.")


if __name__ == "__main__":
    verify()
