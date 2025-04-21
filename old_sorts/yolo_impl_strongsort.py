import cv2
import torch
import numpy as np
from ultralytics import YOLO
from strongsort import StrongSORT

import os
import urllib.request

# ———————————————— CONFIG ————————————————
MODEL_NAME            = "yolov8s.pt"
CONF_THRESH           = 0.5
MAX_AGE               = 30
NN_BUDGET             = 100
MAX_IOU_DIST          = 0.7
DEVICE                = "mps" if torch.backends.mps.is_available() else "cpu"
BRIGHTNESS_THRESH     = 50
ROI_BRIGHTNESS_THRESH = 30
LARGE_BOX_RATIO       = 0.8
REID_WEIGHTS          = "osnet_x0_25_market1501.pt"  # convert to .pt if needed
# ——————————————————————————————————————————————

def download_osnet(weights_path="osnet_x0_25_market1501.pt"):
    url = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v0.0.1/osnet_x0_25_market1501.pt"
    if not os.path.exists(weights_path):
        print(f"[INFO] Downloading {weights_path}...")
        urllib.request.urlretrieve(url, weights_path)
        print("[INFO] Download complete.")
    else:
        print(f"[INFO] {weights_path} already exists.")

# Call this before initializing StrongSORT
download_osnet()

# Load YOLO
model = YOLO(MODEL_NAME)

# Load StrongSORT
tracker = StrongSORT(
    model_weights=REID_WEIGHTS,
    device=DEVICE,
    fp16=False,
    max_age=MAX_AGE,
    max_iou_distance=MAX_IOU_DIST,
    nn_budget=NN_BUDGET,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_area = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < BRIGHTNESS_THRESH or gray.std() < 10:
        cv2.imshow("People Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if frame_area is None:
        h, w = frame.shape[:2]
        frame_area = h * w

    # YOLOv8 inference
    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    # Collect detections
    dets = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls.item()) != 0:
                continue
            x1, y1, x2, y2 = box.int().tolist()
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > LARGE_BOX_RATIO * frame_area:
                continue
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0 or roi.mean() < ROI_BRIGHTNESS_THRESH:
                continue
            dets.append([x1, y1, x2, y2, float(conf.item()), int(cls.item())])

    dets = np.asarray(dets, dtype=float) if dets else np.empty((0, 6), dtype=float)

    # StrongSORT tracking
    outputs = tracker.update(dets, frame)

    # Draw results
    if outputs is not None:
        for output in outputs:
            x1, y1, x2, y2, track_id, class_id, conf = output.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("People Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()