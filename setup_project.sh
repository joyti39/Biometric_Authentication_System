#!/bin/bash
# Run once from ~/biometric_auth/ to create all project files

echo "Setting up biometric_auth project..."
mkdir -p db logs models liveness

# ── config.py ──────────────────────────────────────────────────────
cat > config.py << 'PYEOF'
"""
config.py - Central settings for the Biometric Auth project.
Camera source: Windows host streams via ffmpeg TCP → Kali reads it.
"""

import os

# -------------------------------------------------------------------
# Camera source
# Windows streams webcam over TCP using ffmpeg.
# Kali reads that stream instead of using USB passthrough (which glitches).
# Format: tcp://<VMnet8 Windows IP>:<port>
# -------------------------------------------------------------------
CAMERA_SOURCE = "tcp://192.168.194.1:8080"

# Fallback: if stream not available, try local USB camera
CAMERA_FALLBACK = 0

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FRAME_FPS    = 30

# -------------------------------------------------------------------
# Face recognition
# -------------------------------------------------------------------
FACE_MATCH_TOLERANCE = 0.5   # lower = stricter match
ENROLLMENT_SAMPLES   = 5     # photos taken per person during enrollment

# -------------------------------------------------------------------
# Liveness detection thresholds
# -------------------------------------------------------------------
BLINK_EAR_THRESHOLD   = 0.25  # Eye Aspect Ratio below this = eye closed
BLINK_MIN_COUNT       = 2     # must blink at least this many times
BLINK_TIMEOUT_SECONDS = 5     # seconds given to blink
LBP_SPOOF_THRESHOLD   = 0.6   # texture score below this = likely fake

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "db",     "faces.db")
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "auth_log.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spoof_cnn.h5")

# -------------------------------------------------------------------
# Auth result codes
# -------------------------------------------------------------------
AUTH_SUCCESS       = "SUCCESS"
AUTH_FAIL_NO_FACE  = "FAIL_NO_FACE"
AUTH_FAIL_NO_MATCH = "FAIL_NO_MATCH"
AUTH_FAIL_SPOOF    = "FAIL_SPOOF"
AUTH_FAIL_TIMEOUT  = "FAIL_TIMEOUT"
PYEOF

# ── camera.py ──────────────────────────────────────────────────────
cat > camera.py << 'PYEOF'
"""
camera.py - Shared camera utility for the entire project.

How it works:
  Windows host runs:
    ffmpeg -f dshow -rtbufsize 100M -i video="Integrated Webcam"
           -vf scale=640:480 -vcodec mjpeg -f mpjpeg -q 5
           tcp://0.0.0.0:8080?listen

  Kali connects to that TCP stream and reads clean frames.
  No USB passthrough = no glitches.
"""

import cv2
import numpy as np
import config


def open_camera():
    """
    Connect to the Windows TCP camera stream.
    Falls back to local USB camera if stream is unavailable.
    """
    print(f"Connecting to camera stream: {config.CAMERA_SOURCE}")
    cam = cv2.VideoCapture(config.CAMERA_SOURCE)

    if cam.isOpened():
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"Stream connected! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return cam
        cam.release()

    # Fallback to local USB camera
    print("Stream unavailable, trying local USB camera...")
    cam = cv2.VideoCapture(config.CAMERA_FALLBACK)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS,          config.FRAME_FPS)
        print("Local USB camera connected.")
        return cam

    print("ERROR: No camera source available.")
    return None


def is_good_frame(frame):
    """
    Filter out corrupt or empty frames.

    Checks:
      1. Frame exists and has pixels
      2. Not a solid color (std deviation too low = green/black screen)
      3. No large torn bands (VMware glitch artifact)
    """
    if frame is None or frame.size == 0:
        return False

    # Must have enough visual variation
    if np.std(frame) < 15:
        return False

    # No large bands of identical rows (torn frame artifact)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_diffs = np.abs(np.diff(gray.astype(np.int16), axis=0))
    zero_rows = np.sum(row_diffs.sum(axis=1) == 0)
    if zero_rows > gray.shape[0] * 0.20:
        return False

    return True


def read_good_frame(cam, max_attempts=10):
    """
    Read frames until a good one is found, or give up after max_attempts.
    Returns: (frame, success_bool)
    """
    for _ in range(max_attempts):
        ret, frame = cam.read()
        if ret and is_good_frame(frame):
            return frame, True
    return None, False
PYEOF

# ── test_webcam_final.py ────────────────────────────────────────────
cat > test_webcam_final.py << 'PYEOF'
"""
Step 1 (Final) - Webcam Test using Windows TCP stream.
Shows live feed with glitch stats. Press 'q' to quit.
"""

import cv2
import config
from camera import open_camera, is_good_frame

cam = open_camera()
if cam is None:
    print("No camera available. Make sure ffmpeg is running on Windows.")
    exit()

print("Press 'q' to quit.")

total  = 0
bad    = 0
last_good = None

while True:
    ret, frame = cam.read()
    if not ret:
        print("Lost stream connection.")
        break

    total += 1

    if is_good_frame(frame):
        last_good = frame.copy()
    else:
        bad += 1
        if last_good is not None:
            frame = last_good.copy()
        else:
            continue

    pct = (bad / total * 100) if total > 0 else 0
    cv2.putText(frame, f"Frames: {total}  Glitched: {bad} ({pct:.1f}%)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Webcam Test - press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
pct = (bad / total * 100) if total > 0 else 0
print(f"Done. Total: {total} frames | Glitched: {bad} ({pct:.1f}%)")
PYEOF

echo ""
echo "All files created:"
ls -la *.py
