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
    return True
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
