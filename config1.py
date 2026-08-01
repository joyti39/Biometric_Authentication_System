import os

# Camera
CAMERA_SOURCE   = "udp://@:8080"
CAMERA_FALLBACK = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
FRAME_FPS       = 20

# Face recognition
FACE_MATCH_TOLERANCE = 0.5
ENROLLMENT_SAMPLES   = 5

# Liveness
BLINK_EAR_THRESHOLD   = 0.25
BLINK_MIN_COUNT       = 1
BLINK_TIMEOUT_SECONDS = 8
LBP_SPOOF_THRESHOLD   = 0.6

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "db",     "faces.db")
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "auth_log.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spoof_cnn.h5")
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

# Auth result codes
AUTH_SUCCESS       = "SUCCESS"
AUTH_FAIL_SCREEN = "DENIED_SCREEN_SPOOF"
AUTH_FAIL_NO_FACE  = "FAIL_NO_FACE"
AUTH_FAIL_NO_MATCH = "FAIL_NO_MATCH"
AUTH_FAIL_SPOOF    = "FAIL_SPOOF"
AUTH_FAIL_TIMEOUT  = "FAIL_TIMEOUT"
 
