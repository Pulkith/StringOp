import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ———————————————— CONFIG ————————————————
MODEL_NAME          = "yolov8s.pt"
CONF_THRESH         = 0.5
MAX_AGE             = 150        
N_INIT              = 5          
MAX_COSINE_DISTANCE = 0.2        
NN_BUDGET           = 100        
EMBEDDER_MODEL_NAME = "osnet_x0_25"  
DEVICE              = "mps" if torch.backends.mps.is_available() else "cpu"
# ——————————————————————————————————————————————

model = YOLO(MODEL_NAME)

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

    results = model.predict(frame, conf=CONF_THRESH, classes=[0], stream=True)

    detections = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls.item()) != 0:
                continue
            x1, y1, x2, y2 = map(int, box)
            detections.append(((x1, y1, x2, y2), float(conf.item()), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

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