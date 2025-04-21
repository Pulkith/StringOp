import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ocsort.ocsort import OCSort  # Make sure this import works or adjust based on your folder structure

# ———————————————— CONFIG ————————————————
MODEL_NAME  = "yolov8s.pt"
CONF_THRESH = 0.5
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
# ——————————————————————————————————————————————

# Load YOLO
model = YOLO(MODEL_NAME)

# Initialize OCSort tracker
tracker = OCSort(
    det_thresh=CONF_THRESH,       # Set detection threshold
    iou_threshold=0.5,            # IOU threshold for association
    min_hits=3,                   # Minimum hits to confirm track
    max_age=150,                  # Max age of tracks
    use_byte=False,               # Use ByteTrack association
    delta_t=1                     # Time step
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    # Gather detections in [x1, y1, x2, y2, conf] format
    detections = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls.item()) != 0:  # Only track person class
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            conf = conf.item()
            detections.append([x1, y1, x2, y2, conf])

    det_array = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)

    # Update OCSort
    track_outputs = tracker.update(det_array, frame)

    # Draw tracked boxes
    for track in track_outputs:
        x1, y1, x2, y2, track_id = track[:5].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"ID: {track_id}, Box: ({x1}, {y1}, {x2}, {y2})")

    cv2.imshow("OCSort Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()