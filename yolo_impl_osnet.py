import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ———————————————— CONFIG ————————————————
MODEL_NAME          = "yolov8s.pt"
CONF_THRESH         = 0.5
# Allow tracks to survive ~5 seconds of being out of frame (at 30 FPS)
MAX_AGE             = 150        
# Require a few frames before confirming new IDs
N_INIT              = 5          
# How similar embeddings must be to match (lower = stricter)
MAX_COSINE_DISTANCE = 0.2        
# Keep up to 100 embeddings in the gallery
NN_BUDGET           = 100        
# Use OSNet re‑id model for stronger appearance features
EMBEDDER_MODEL_NAME = "osnet_x0_25"  
DEVICE              = "mps" if torch.backends.mps.is_available() else "cpu"
# ——————————————————————————————————————————————

# Load YOLOv8
model = YOLO(MODEL_NAME)

# Initialize DeepSort with appearance re‑id
tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    max_cosine_distance=MAX_COSINE_DISTANCE,
    nn_budget=NN_BUDGET,
    embedder_model_name=EMBEDDER_MODEL_NAME,
    half=True,
    embedder_gpu=torch.cuda.is_available()
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect people
    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    # 2️⃣ Prepare DeepSort inputs
    detections = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls.item()) != 0:
                continue
            x1, y1, x2, y2 = map(int, box)
            detections.append(((x1, y1, x2, y2), float(conf.item()), "person"))

    # 3️⃣ Update tracks (with re‑id)
    tracks = tracker.update_tracks(detections, frame=frame)

    # 4️⃣ Draw & log
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        print(f"ID: {tid}, Box: ({l}, {t}, {r}, {b})")

    cv2.imshow("Robust People Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()