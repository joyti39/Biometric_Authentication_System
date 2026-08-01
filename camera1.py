"""
camera.py - Shared camera utility for the entire project.

How it works:
  Windows host runs ffmpeg, streaming the webcam (via OBS Virtual
  Camera for sharp focus) over UDP broadcast on port 8080.
  Kali connects to that stream and reads frames, retrying a few
  times in case the stream isn't fully up yet at the moment we try
  to connect.
"""
import cv2
import numpy as np
import time
import config


def open_camera():
    """
    Connect to the Windows UDP camera stream, retrying a few times.
    Falls back to a local USB camera if the stream is unavailable.
    """
    for attempt in range(1, 6):
        print(f"Connecting to camera stream (attempt {attempt}/5): {config.CAMERA_SOURCE}")
        cam = cv2.VideoCapture(config.CAMERA_SOURCE)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"Stream connected! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return cam
        cam.release()
        time.sleep(1.5)

    print("Stream unavailable after 5 attempts, trying local USB camera...")
    cam = cv2.VideoCapture(config.CAMERA_FALLBACK)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, config.FRAME_FPS)
        print("Local USB camera connected.")
        return cam
    print("ERROR: No camera source available.")
    return None


def is_good_frame(frame):
    """
    Filter out corrupt or empty frames.
    """
    if frame is None or frame.size == 0:
        return False
    if np.std(frame) < 15:
        return False
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
