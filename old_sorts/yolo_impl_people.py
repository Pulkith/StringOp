import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (Nano = lightweight, or use 'yolov8s.pt')
model = YOLO("yolov8n.pt")  # Ensure this is downloaded or available locally

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference (returns generator of Results)
    results = model.predict(frame, conf=0.3, stream=True)

    for r in results:
        annotated = frame.copy()
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:  # Class ID 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display
        cv2.imshow('YOLOv8 - People Only', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()