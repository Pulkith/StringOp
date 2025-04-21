import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ———————————————— CONFIG ————————————————
MODEL_NAME            = "yolov8s.pt"
CONF_THRESH           = 0.5
DEVICE                = "mps" if torch.backends.mps.is_available() else "cpu"
BRIGHTNESS_THRESH     = 50
ROI_BRIGHTNESS_THRESH = 30
LARGE_BOX_RATIO       = 0.8
# ——————————————————————————————————————————————

# Load YOLO
model = YOLO(MODEL_NAME)
model.to(DEVICE)

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
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if frame_area is None:
        h, w = frame.shape[:2]
        frame_area = h * w

    # YOLOv8 inference
    results = model.predict(frame, conf=CONF_THRESH, stream=True)

    # Process detections
    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf
        classes = r.boxes.cls
        names = model.names  # Mapping of class index to name

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > LARGE_BOX_RATIO * frame_area:
                continue
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0 or roi.mean() < ROI_BRIGHTNESS_THRESH:
                continue

            class_id = int(cls.item())
            confidence = float(conf.item())
            class_name = names[class_id]

            print(f"Detected: {class_name} ({confidence:.2f})")

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()