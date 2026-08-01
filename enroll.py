"""
enroll.py - Face Enrollment Module

What it does:
  1. Opens the camera stream
  2. Detects a face in the frame
  3. Takes ENROLLMENT_SAMPLES number of photos
  4. Computes a 128-dimension face embedding for each photo
  5. Averages them into one final embedding
  6. Saves name + embedding to SQLite database

"""

import cv2
import face_recognition
import sqlite3
import numpy as np
import os
import sys
import time
import config
from camera import open_camera, read_good_frame


# ── Database setup ──────────────────────────────────────────────────

def init_db():
    """
    Create the database and faces table if they don't exist yet.
    Called once at startup — safe to call multiple times.

    Table structure:
      id            : auto-incrementing unique ID
      name          : person's name (e.g. "Joyti")
      embedding     : 128 float numbers stored as raw bytes (numpy tobytes)
      enrolled_at   : timestamp of enrollment
    """
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            enrolled_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def save_face(conn, name, embedding):
    """
    Save a face embedding to the database.

    embedding is a numpy array of 128 floats.
    We store it as raw bytes (BLOB) — fast to read back later.
    Like saving a fingerprint as a binary file instead of a number list.
    """
    conn.execute(
        "INSERT INTO faces (name, embedding) VALUES (?, ?)",
        (name, embedding.tobytes())
    )
    conn.commit()
    print(f"Saved '{name}' to database.")


def list_enrolled(conn):
    """Print all enrolled people in the database."""
    rows = conn.execute("SELECT id, name, enrolled_at FROM faces").fetchall()
    if not rows:
        print("No faces enrolled yet.")
    else:
        print(f"\n{'ID':<5} {'Name':<20} {'Enrolled At'}")
        print("-" * 45)
        for row in rows:
            print(f"{row[0]:<5} {row[1]:<20} {row[2]}")
    print()


# ── Face detection helpers ──────────────────────────────────────────

def detect_face(frame):
    """
    Detect faces in a frame and return their locations.

    face_recognition uses HOG (Histogram of Oriented Gradients) model by default.
    It returns a list of (top, right, bottom, left) tuples — one per face found.

    We resize to half size before detection for speed — detection on 320x240
    is 4x faster than on 640x480, and accuracy is still fine for enrollment.
    """
    # Shrink frame for faster detection
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # face_recognition needs RGB, OpenCV gives BGR — swap channels
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Get face locations
    locations = face_recognition.face_locations(rgb_small)

    # Scale locations back up to original frame size
    locations = [(t*2, r*2, b*2, l*2) for t, r, b, l in locations]

    return locations


def draw_face_box(frame, locations, label=""):
    """
    Draw a green rectangle around detected faces.
    Shows the label (name or status) above the box.
    """
    display = frame.copy()
    for (top, right, bottom, left) in locations:
        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        if label:
            cv2.putText(display, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return display


# ── Main enrollment flow ────────────────────────────────────────────

def enroll_person(name):
    """
    Full enrollment flow for one person.

    Steps:
      1. Open camera
      2. Show live feed — wait until exactly 1 face is visible
      3. Capture ENROLLMENT_SAMPLES good frames with a face
      4. Compute embedding for each frame
      5. Average all embeddings → final embedding
      6. Save to database
    """
    conn = init_db()

    # Check if this name already exists
    existing = conn.execute(
        "SELECT COUNT(*) FROM faces WHERE name=?", (name,)
    ).fetchone()[0]

    if existing > 0:
        print(f"'{name}' is already enrolled. Re-enrolling will add a new entry.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Enrollment cancelled.")
            conn.close()
            return

    print(f"\nStarting enrollment for: {name}")
    print("Look at the camera. Stay still.")
    print("Press 'q' to cancel.\n")

    cam = open_camera()
    if cam is None:
        print("Cannot open camera. Make sure ffmpeg is running on Windows.")
        conn.close()
        return

    embeddings_collected = []
    target = config.ENROLLMENT_SAMPLES

    while len(embeddings_collected) < target:
        frame, ok = read_good_frame(cam)
        if not ok:
            continue

        locations = detect_face(frame)

        # We need exactly 1 face — 0 means no one there, 2+ means too crowded
        if len(locations) == 0:
            msg = "No face detected — move closer"
        elif len(locations) > 1:
            msg = "Multiple faces — only 1 person please"
        else:
            count = len(embeddings_collected) + 1
            msg = f"Capturing {count}/{target}..."

            # Compute 128-d embedding for this frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embeddings = face_recognition.face_encodings(rgb, locations)

            if embeddings:
                embeddings_collected.append(embeddings[0])
                time.sleep(0.3)  # small pause between captures

        # Draw face box with status message
        display = draw_face_box(frame, locations, msg)

        # Progress bar at bottom of screen
        progress = int((len(embeddings_collected) / target) * frame.shape[1])
        cv2.rectangle(display,
                      (0, frame.shape[0]-10),
                      (progress, frame.shape[0]),
                      (0, 255, 0), -1)

        cv2.imshow(f"Enrolling: {name} — press 'q' to cancel", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled.")
            cam.release()
            cv2.destroyAllWindows()
            conn.close()
            return

    cam.release()
    cv2.destroyAllWindows()

    # Average all collected embeddings into one final embedding
    # This smooths out small variations between frames
    final_embedding = np.mean(embeddings_collected, axis=0)

    save_face(conn, name, final_embedding)
    print(f"\nEnrollment complete for '{name}'!")
    print(f"Collected {target} samples, averaged into 1 embedding (128 values).")

    list_enrolled(conn)
    conn.close()


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 enroll.py <name>")
        print("Example: python3 enroll.py Joyti")
        sys.exit(1)

    name = sys.argv[1].strip()
    enroll_person(name)
