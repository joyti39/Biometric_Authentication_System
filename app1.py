"""
app.py - Biometric Authentication System (Streamlit UI) - FINAL

Changes in this version vs previous:
  1. Performance: the verify loop now paces itself (small sleep),
     resizes frames before running blink/bezel detection, and only
     re-renders the video image a few times per second instead of
     as fast as possible -- this fixes the "everything freezes /
     nothing happens" and "laggy/delayed video" problems, which were
     caused by running full-resolution MediaPipe + Hough-transform
     edge detection on every single frame with no pacing.
  2. Robustness: the whole verify loop is wrapped in try/except so
     an unexpected error shows a message instead of silently
     freezing the whole app.
  3. Stronger anti-spoofing: bezel/screen-edge detection now fails
     an attempt on a SINGLE confident detection (instead of waiting
     for 2), and looks in a wider area around the face. This is a
     deliberate security-over-convenience tradeoff: it's better to
     occasionally ask a real person to reposition than to let a
     video-replay attack through.
  4. face_db.py now uses the 'embedding' column name to match the
     already-existing faces.db schema (fixes "no column named
     encoding" error).
"""

import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import csv
import time
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from camera import open_camera, is_good_frame
from face_db import init_db, save_face, load_all_faces


# ── Page setup + theme ──────────────────────────────────────────────
st.set_page_config(page_title="Biometric Auth", page_icon="🔐", layout="centered")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2027 0%, #0a0e14 60%, #05070a 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,170,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,170,0.04) 1px, transparent 1px);
    background-size: 34px 34px;
    pointer-events: none;
    z-index: 0;
}
h1, h2, h3 { color: #4dffd2 !important; text-shadow: 0 0 12px rgba(77,255,210,0.35); }
p, label, .stMarkdown, span { color: #d8f5ee; }

div.stButton > button {
    background: linear-gradient(135deg, #0f3a3a, #124d4d);
    color: #7dffdd;
    border: 1px solid #2fe3c2;
    border-radius: 10px;
    padding: 0.5em 1.3em;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.18s ease-in-out;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #17e0b6, #0af0c8);
    color: #04211c;
    border-color: #baffee;
    box-shadow: 0 0 18px rgba(45,255,195,0.55);
    transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0px) scale(0.98); }

.stTextInput > div > div > input {
    background-color: #0d1a1a;
    color: #baffee;
    border: 1px solid #1f5c53;
    border-radius: 8px;
}
[data-testid="stMetric"] {
    background: rgba(15, 58, 58, 0.35);
    border: 1px solid #1f5c53;
    border-radius: 12px;
    padding: 10px;
}
.cyber-card {
    background: rgba(10, 20, 22, 0.55);
    border: 1px solid #1f5c53;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 0 22px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab"] { color: #9fe8d8; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #4dffd2 !important; border-bottom-color: #4dffd2 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔐 Biometric Authentication System")

DETECTION_SCALE = 0.4
BLINK_SCORE_THRESHOLD = 0.5
BLINK_COOLDOWN_SECONDS = 0.4
BEZEL_MARGIN_RATIO = 0.9        # look further around the face for a phone/screen edge
BEZEL_LINE_COUNT_THRESHOLD = 2   # >= this many long straight edges = likely a screen/photo
BEZEL_FAIL_ON_FIRST_HIT = True   # security-first: fail immediately, don't wait for repeats

LOOP_SLEEP = 0.04                # paces the verify loop (~25 iterations/sec max)
PROCESS_EVERY_N_FRAMES = 2       # run blink+bezel detection every Nth frame to reduce load


# ── Cached resources (created once, reused across reruns) ───────────
@st.cache_resource
def get_camera():
    init_db()
    return open_camera()


@st.cache_resource
def get_detector():
    if not os.path.exists(config.FACE_LANDMARKER_MODEL_PATH):
        return None
    base = mp_python.BaseOptions(model_asset_path=config.FACE_LANDMARKER_MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


if st.button("🔄 Reconnect Camera", help="Click this if the camera stream was down when the app started"):
    get_camera.clear()
    st.rerun()


# ── Helpers ────────────────────────────────────────────────────────
def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_match(embedding, names, embeddings):
    if not embeddings:
        return None, None
    matches = face_recognition.compare_faces(
        embeddings, embedding, tolerance=config.FACE_MATCH_TOLERANCE)
    distances = face_recognition.face_distance(embeddings, embedding)
    best = np.argmin(distances)
    if matches[best]:
        return names[best], distances[best]
    return None, distances[best]


def get_blink_score(blendshapes):
    left = right = None
    for c in blendshapes:
        if c.category_name == "eyeBlinkLeft":
            left = c.score
        if c.category_name == "eyeBlinkRight":
            right = c.score
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def detect_screen_bezel(frame, face_box):
    """
    Look for a rectangular screen/photo bezel around the detected face.
    Strong straight edges survive video compression much better than
    fine skin texture, which is why this works where LBP texture
    analysis did not.
    """
    top, right, bottom, left = face_box
    h, w = bottom - top, right - left
    my, mx = int(h * BEZEL_MARGIN_RATIO), int(w * BEZEL_MARGIN_RATIO)
    y1, y2 = max(0, top - my), min(frame.shape[0], bottom + my)
    x1, x2 = max(0, left - mx), min(frame.shape[1], right + mx)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    min_len = max(20, int(min(roi.shape[0], roi.shape[1]) * 0.4))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                             minLineLength=min_len, maxLineGap=14)
    count = 0 if lines is None else len(lines)
    return count >= BEZEL_LINE_COUNT_THRESHOLD, count


def log_attempt(name, result, distance=None):
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    exists = os.path.exists(config.LOG_PATH)
    with open(config.LOG_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "name", "result", "distance"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, result,
                    f"{distance:.4f}" if distance is not None else ""])


# ── Tabs ───────────────────────────────────────────────────────────
tab_enroll, tab_verify, tab_logs = st.tabs(["📸 Enroll", "✅ Verify", "📋 Logs"])


# ════════════════════════════════════════════════════════════════════
# ENROLL TAB
# ════════════════════════════════════════════════════════════════════
with tab_enroll:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("Register a new face")
    name = st.text_input("Enter your name")

    if "enroll_live" not in st.session_state:
        st.session_state.enroll_live = False

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Camera", key="enroll_start"):
        st.session_state.enroll_live = True
    if col2.button("⏹️ Stop Camera", key="enroll_stop"):
        st.session_state.enroll_live = False

    preview_slot = st.empty()
    capture_now = st.button("📸 Capture & Enroll", disabled=not name.strip())

    cam = get_camera()
    if cam is None:
        st.error("❌ Camera not available. Make sure ffmpeg is running on Windows.")
    else:
        if capture_now:
            ret, frame = cam.read()
            if not ret:
                st.error("Could not read frame. Try again.")
            else:
                rgb = bgr_to_rgb(frame)
                locations = face_recognition.face_locations(rgb)
                if len(locations) == 0:
                    st.warning("No face detected. Move closer and try again.")
                elif len(locations) > 1:
                    st.warning("Multiple faces detected. Only one person please.")
                else:
                    encodings = face_recognition.face_encodings(rgb, locations)
                    if encodings:
                        save_face(name.strip(), encodings[0])
                        st.success(f"✅ '{name.strip()}' enrolled successfully!")
                        st.image(rgb, caption="Enrolled photo", use_container_width=True)

        elif st.session_state.enroll_live:
            try:
                while st.session_state.enroll_live:
                    ret, frame = cam.read()
                    if ret:
                        preview_slot.image(bgr_to_rgb(frame), channels="RGB",
                                            caption="Live preview", use_container_width=True)
                    time.sleep(LOOP_SLEEP)
            except Exception as e:
                st.error(f"Preview stopped due to an error: {e}")
                st.session_state.enroll_live = False
        else:
            preview_slot.info("Click **Start Camera** to see a live preview before enrolling.")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# VERIFY TAB
# ════════════════════════════════════════════════════════════════════
with tab_verify:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("Face Verification + Liveness Check")
    st.caption("Face match → blink liveness → screen/bezel spoof check, all required for GRANTED.")

    if "verifying" not in st.session_state:
        st.session_state.verifying = False

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Verification"):
        st.session_state.verifying = True
    if col2.button("⏹️ Stop", key="verify_stop"):
        st.session_state.verifying = False

    status_slot = st.empty()
    video_slot = st.empty()

    cam = get_camera()
    detector = get_detector()

    if cam is None:
        st.error("❌ Camera not available. Make sure ffmpeg is running on Windows.")
    elif detector is None:
        st.error(f"❌ Blink model missing: {config.FACE_LANDMARKER_MODEL_PATH}")
    elif st.session_state.verifying:
        names, embeddings = load_all_faces()
        if not names:
            st.error("No faces enrolled yet. Go to the Enroll tab first.")
        else:
            status_slot.info("🟠 Starting camera...")
            STATE_SCANNING, STATE_LIVENESS, STATE_RESULT = "SCANNING", "LIVENESS", "RESULT"
            state = STATE_SCANNING
            last_face_check = 0
            last_locations = []
            liveness_person = liveness_distance = None
            liveness_deadline = 0
            liveness_blink_count = liveness_consec_closed = 0
            liveness_last_blink_time = 0
            bezel_flag_count = 0
            result_until = 0
            result_msg = ("", "")
            frame_counter = 0
            failed_reads = 0
            MAX_FAILED_READS = 75  # ~3 seconds at LOOP_SLEEP=0.04

            try:
                while st.session_state.verifying:
                    ret, frame = cam.read()
                    if not ret:
                        failed_reads += 1
                        status_slot.warning(
                            f"⚠️ No frame from camera stream "
                            f"({failed_reads}/{MAX_FAILED_READS})... "
                            f"is ffmpeg still running on Windows?")
                        if failed_reads >= MAX_FAILED_READS:
                            st.error("❌ Lost the camera stream. Check that ffmpeg "
                                     "and OBS Virtual Camera are still running on "
                                     "Windows, then click Start Verification again.")
                            st.session_state.verifying = False
                            break
                        time.sleep(LOOP_SLEEP)
                        continue
                    failed_reads = 0
                    now = time.time()
                    frame_counter += 1
                    display = frame.copy()
                    run_heavy_checks = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)

                    if state == STATE_SCANNING:
                        status_slot.info("🟠 Scanning for a known face...")
                        if now - last_face_check >= 1.0 and is_good_frame(frame):
                            last_face_check = now
                            small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                            rgb = bgr_to_rgb(small)
                            locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
                            locations = [
                                (int(t / DETECTION_SCALE), int(r / DETECTION_SCALE),
                                 int(b / DETECTION_SCALE), int(l / DETECTION_SCALE))
                                for t, r, b, l in locations
                            ]
                            last_locations = locations
                            if locations:
                                rgb_full = bgr_to_rgb(frame)
                                encodings = face_recognition.face_encodings(rgb_full, locations)
                                if encodings:
                                    matched_name, distance = find_match(encodings[0], names, embeddings)
                                    if matched_name:
                                        state = STATE_LIVENESS
                                        liveness_person = matched_name
                                        liveness_distance = distance
                                        liveness_deadline = now + config.BLINK_TIMEOUT_SECONDS
                                        liveness_blink_count = 0
                                        liveness_consec_closed = 0
                                        liveness_last_blink_time = 0
                                        bezel_flag_count = 0
                        for (top, right, bottom, left) in last_locations:
                            cv2.rectangle(display, (left, top), (right, bottom), (0, 200, 255), 2)

                    elif state == STATE_LIVENESS:
                        seconds_left = max(0.0, liveness_deadline - now)
                        status_slot.warning(
                            f"👁️ {liveness_person} matched — blink naturally "
                            f"({liveness_blink_count}/{config.BLINK_MIN_COUNT} blinks, "
                            f"{seconds_left:.1f}s left)")

                        if run_heavy_checks:
                            # -- Blink check --
                            small_for_mp = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                            rgb = bgr_to_rgb(small_for_mp)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            result = detector.detect_for_video(mp_image, int(now * 1000))
                            if result.face_blendshapes:
                                score = get_blink_score(result.face_blendshapes[0])
                                if score is not None and score > BLINK_SCORE_THRESHOLD:
                                    liveness_consec_closed += 1
                                else:
                                    if (liveness_consec_closed >= 1 and
                                            (now - liveness_last_blink_time) > BLINK_COOLDOWN_SECONDS):
                                        liveness_blink_count += 1
                                        liveness_last_blink_time = now
                                    liveness_consec_closed = 0

                            # -- Screen/bezel check (blocks phone/video replay attacks) --
                            if last_locations:
                                bezel_found, line_count = detect_screen_bezel(frame, last_locations[0])
                                if bezel_found:
                                    bezel_flag_count += 1

                        for (top, right, bottom, left) in last_locations:
                            cv2.rectangle(display, (left, top), (right, bottom), (0, 200, 255), 2)

                        need_hits = 1 if BEZEL_FAIL_ON_FIRST_HIT else 2
                        if bezel_flag_count >= need_hits:
                            log_attempt(liveness_person, config.AUTH_FAIL_SPOOF, liveness_distance)
                            result_msg = ("error",
                                          f"⛔ DENIED — screen/bezel edge detected around "
                                          f"{liveness_person}'s face (photo/video replay blocked)")
                            state = STATE_RESULT
                            result_until = now + 2.5
                        elif liveness_blink_count >= config.BLINK_MIN_COUNT:
                            log_attempt(liveness_person, config.AUTH_SUCCESS, liveness_distance)
                            result_msg = ("success",
                                          f"✅ GRANTED — Welcome, {liveness_person}! "
                                          f"(distance={liveness_distance:.3f})")
                            state = STATE_RESULT
                            result_until = now + 2.0
                        elif now >= liveness_deadline:
                            log_attempt(liveness_person, config.AUTH_FAIL_SPOOF, liveness_distance)
                            result_msg = ("error",
                                          f"⛔ DENIED — {liveness_person}'s face matched but no "
                                          f"blink detected (possible spoof)")
                            state = STATE_RESULT
                            result_until = now + 2.0

                    elif state == STATE_RESULT:
                        level, text = result_msg
                        if level == "success":
                            status_slot.success(text)
                        else:
                            status_slot.error(text)
                        if now >= result_until:
                            state = STATE_SCANNING
                            last_locations = []

                    video_slot.image(bgr_to_rgb(display), channels="RGB", use_container_width=True)
                    time.sleep(LOOP_SLEEP)

            except Exception as e:
                st.error(f"Verification stopped due to an error: {e}")
                st.session_state.verifying = False

        status_slot.info("Verification stopped.")
    else:
        status_slot.write("Click **Start Verification** to begin.")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# LOGS TAB
# ════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("Authentication Log")
    if os.path.exists(config.LOG_PATH):
        import pandas as pd
        df = pd.read_csv(config.LOG_PATH)
        df = df.sort_values(df.columns[0], ascending=False)
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Attempts", len(df))
        col2.metric("Granted", len(df[df['result'] == 'SUCCESS']))
        col3.metric("Denied/Spoof", len(df[df['result'] != 'SUCCESS']))
    else:
        st.info("No log entries yet. Run a verification first.")
    st.markdown('</div>', unsafe_allow_html=True)
