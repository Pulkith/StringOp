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
BROWN_THRESHOLD       = 0.50  # 35%
# ——————————————————————————————————————————————

def is_wearing_brown(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    # Approx HSV range for various browns (tweakable)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(mask > 0) / (roi_bgr.shape[0] * roi_bgr.shape[1])
    
    return brown_ratio > BROWN_THRESHOLD

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
        cv2.imshow("Brown Clothing Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if frame_area is None:
        h, w = frame.shape[:2]
        frame_area = h * w

    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf
        classes = r.boxes.cls
        names = model.names

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > LARGE_BOX_RATIO * frame_area:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0 or cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean() < ROI_BRIGHTNESS_THRESH:
                continue

            if not is_wearing_brown(roi):
                continue

            confidence = float(conf.item())
            class_name = names[int(cls.item())]

            print(f"Detected: {class_name} ({confidence:.2f}) wearing brown")

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (42, 42, 165), 2)  # Brown-ish color
            label = f"{class_name} {confidence:.2f} - Brown"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 42, 165), 2)

    cv2.imshow("Brown Clothing Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()