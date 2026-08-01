#!/bin/bash
cd ~/biometric_auth

# ── db.py ──────────────────────────────────────────────────────────
cat > db.py << 'PYEOF'
"""
db.py - Database helper functions.
Used by both enroll.py and app.py.
"""
import sqlite3
import numpy as np
import os
import config

def init_db():
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
    conn.close()

def save_face(name, embedding):
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        "INSERT INTO faces (name, embedding) VALUES (?, ?)",
        (name, embedding.tobytes())
    )
    conn.commit()
    conn.close()

def load_all_faces():
    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
    conn.close()
    names, embeddings = [], []
    for name, blob in rows:
        names.append(name)
        embeddings.append(np.frombuffer(blob, dtype=np.float64))
    return names, embeddings
PYEOF

# ── config.py (add missing FACE_LANDMARKER_MODEL_PATH) ─────────────
cat > config.py << 'PYEOF'
import os

# Camera
CAMERA_SOURCE   = "udp://192.168.194.255:8080"
CAMERA_FALLBACK = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
FRAME_FPS       = 20

# Face recognition
FACE_MATCH_TOLERANCE = 0.5
ENROLLMENT_SAMPLES   = 5

# Liveness
BLINK_EAR_THRESHOLD   = 0.25
BLINK_MIN_COUNT       = 2
BLINK_TIMEOUT_SECONDS = 5
LBP_SPOOF_THRESHOLD   = 0.6

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "db",     "faces.db")
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "auth_log.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spoof_cnn.h5")
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

# Auth result codes
AUTH_SUCCESS       = "SUCCESS"
AUTH_FAIL_NO_FACE  = "FAIL_NO_FACE"
AUTH_FAIL_NO_MATCH = "FAIL_NO_MATCH"
AUTH_FAIL_SPOOF    = "FAIL_SPOOF"
AUTH_FAIL_TIMEOUT  = "FAIL_TIMEOUT"
PYEOF

# ── Download face landmarker model ──────────────────────────────────
mkdir -p models
if [ ! -f models/face_landmarker.task ]; then
    echo "Downloading face landmarker model..."
    curl -L -o models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    echo "Model downloaded!"
else
    echo "Model already exists."
fi

echo ""
echo "Done! All files ready:"
ls -la *.py models/
