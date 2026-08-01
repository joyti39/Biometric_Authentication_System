"""
liveness/texture_check.py - Layer 2 of liveness detection: LBP Texture Analysis

Why texture analysis catches photo/screen spoofing:
  Real skin has fine, irregular texture (pores, subtle wrinkles,
  natural light scattering). A printed photo has ink-dot/paper-fiber
  texture; a screen replay has a pixel-grid texture. These are
  measurably different from real skin under Local Binary Pattern (LBP)
  analysis.

How LBP works (short version):
  For each pixel, look at its 8 neighbors. For each neighbor, write
  a 1 if it's brighter than the center pixel, 0 if darker. Reading
  those 8 bits gives an 8-bit "code" for that pixel (0-255).
  Do this for every pixel in the face region, then build a histogram
  of how often each code (0-255) appears.

  Real skin's histogram is "spread out" (many different codes appear,
  because skin texture is naturally irregular) -- this means HIGH
  ENTROPY. A printed photo or screen tends to have a more repetitive,
  uniform texture -- LOWER ENTROPY.

  We use Shannon entropy of the LBP histogram as our "texture score".

This file is a CALIBRATION / OBSERVATION tool: it shows the live
texture score on screen so you can compare real-face vs printed-photo
scores yourself and decide on a good threshold, before wiring this
into verify.py as a hard pass/fail gate.
"""

import cv2
import numpy as np
import face_recognition
from skimage.feature import local_binary_pattern
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import open_camera, is_good_frame

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
DETECTION_SCALE = 0.4


def compute_texture_entropy(face_gray):
    """
    Compute the Shannon entropy of the LBP histogram for a face crop.

    Higher entropy = more irregular texture (more likely REAL skin).
    Lower entropy  = more uniform/repetitive texture (more likely a
                      printed photo or screen).
    """
    lbp = local_binary_pattern(face_gray, LBP_POINTS, LBP_RADIUS, method="uniform")

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # Shannon entropy: -sum(p * log2(p)), skipping zero-probability bins
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def test_texture_check():
    cam = open_camera()
    if cam is None:
        print("Could not open camera. Make sure ffmpeg is running on Windows.")
        return

    print("Hold your REAL face up first, note the entropy score.")
    print("Then hold up a PRINTED PHOTO or PHONE SCREEN photo of yourself, compare.")
    print("Press 'q' to quit.\n")

    last_check = 0
    last_entropy = None
    last_locations = []

    import time
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        now = time.time()

        if now - last_check >= 0.5 and is_good_frame(frame):
            last_check = now

            small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=1)
            locations = [
                (int(t / DETECTION_SCALE), int(r / DETECTION_SCALE),
                 int(b / DETECTION_SCALE), int(l / DETECTION_SCALE))
                for t, r, b, l in locations
            ]
            last_locations = locations

            if locations:
                top, right, bottom, left = locations[0]
                # Clamp to frame bounds just in case
                top, left = max(0, top), max(0, left)
                bottom = min(frame.shape[0], bottom)
                right = min(frame.shape[1], right)

                face_crop = frame[top:bottom, left:right]
                if face_crop.size > 0:
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    last_entropy = compute_texture_entropy(face_gray)

        display = frame.copy()
        for (top, right, bottom, left) in last_locations:
            cv2.rectangle(display, (left, top), (right, bottom), (255, 200, 0), 2)

        if last_entropy is not None:
            cv2.putText(display, f"Texture entropy: {last_entropy:.3f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(display, "Higher = more likely REAL skin",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Texture Check Calibration - press q to quit", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_texture_check()
