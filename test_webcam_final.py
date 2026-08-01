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
