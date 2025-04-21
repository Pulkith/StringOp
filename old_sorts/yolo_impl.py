import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ———————————————— CONFIG ————————————————
MODEL_NAME     = "yolov8s.pt"
CONF_THRESH    = 0.5     # raised to reduce low‑conf false positives
MAX_AGE        = 30
N_INIT         = 5       # require a few frames before confirming a new ID
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

BRIGHTNESS_THRESH    = 50    # skip frames this dark (0–255)
ROI_BRIGHTNESS_THRESH= 30    # skip boxes whose region is too dark
LARGE_BOX_RATIO      = 0.8   # discard boxes >80% of frame area
# ——————————————————————————————————————————————

# Load YOLO
model = YOLO(MODEL_NAME)

# DeepSort
tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_area = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 0️⃣ Pre-filter: skip very dark/uniform frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < BRIGHTNESS_THRESH or gray.std() < 10:
        # no reliable data here
        cv2.imshow("People Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Initialize frame_area once
    if frame_area is None:
        h, w = frame.shape[:2]
        frame_area = h * w

    # 1️⃣ Run YOLO detection
    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    # 2️⃣ Collect filtered detections
    detections = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls.item()) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            box_area = (x2 - x1) * (y2 - y1)

            # 1) discard absurdly large boxes
            if box_area > LARGE_BOX_RATIO * frame_area:
                continue

            # 2) discard too‐dark ROIs
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0 or roi.mean() < ROI_BRIGHTNESS_THRESH:
                continue

            detections.append(((x1, y1, x2, y2), float(conf.item()), "person"))

    # 3️⃣ Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # 4️⃣ Draw & log confirmed tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"ID: {tid}, Box: ({l}, {t}, {r}, {b})")

    # 5️⃣ Display
    cv2.imshow("People Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()