import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiPersonBackpackTracker:
    def __init__(
        self,
        model_name: str = "yolov8m-seg.pt",
        conf_thresh: float = 0.6,
        proximity_thresh: float = 50,
        device: str = None,
        # DeepSort parameters
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        embedder_model_name: str = "osnet_x0_25",
    ):
        # choose device
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")
        # load YOLOv8 segmentation model
        self.model = YOLO(model_name).to(self.device)
        self.conf_thresh = conf_thresh
        print(f"Loaded YOLO model '{model_name}'")

        # threshold in pixels for person–backpack proximity
        self.proximity_thresh = proximity_thresh

        # initialize DeepSort tracker
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder_model_name=embedder_model_name,
            half=(self.device == "cuda"),
            embedder_gpu=(self.device == "cuda"),
        )
        print("Initialized DeepSort tracker")

    @staticmethod
    def _rect_distance(boxA, boxB):
        """
        Compute minimal distance between the borders of two axis-aligned rectangles.
        box = [x1, y1, x2, y2]
        """
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        dx = max(bx1 - ax2, ax1 - bx2, 0)
        dy = max(by1 - ay2, ay1 - by2, 0)
        return np.hypot(dx, dy)

    def process_frame(self, frame: np.ndarray):
        """
        1) Run segmentation on people (class 0) and backpacks (class 24)
        2) Collect person & backpack contours
        3) Keep only persons whose nearest backpack is within proximity_thresh
        4) Track those persons with DeepSort
        Returns list of dicts: [{"track_id", "bbox", "contour"}].
        """
        H, W = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            classes=[0, 24],
            verbose=False
        )
        if not results or results[0].masks is None:
            return []

        r = results[0]
        person_dets = []
        backpack_boxes = []

        # split detections into persons and backpacks
        for i in range(len(r.boxes)):
            cls = int(r.boxes.cls[i].item())
            contour = r.masks.xy[i].astype(np.int32)  # polygon Nx2
            if contour.shape[0] < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            conf = float(r.boxes.conf[i].item())

            if cls == 24:  # backpack
                backpack_boxes.append([x1, y1, x2, y2])
            elif cls == 0:  # person
                person_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "contour": contour
                })

        # filter persons by proximity to any backpack
        nearby_persons = []
        for pd in person_dets:
            dist = min(
                (self._rect_distance(pd["bbox"], bb) for bb in backpack_boxes),
                default=float("inf")
            )
            if dist <= self.proximity_thresh:
                nearby_persons.append(pd)

        # prepare DeepSort detections: only persons near a backpack
        ds_detections = [
            (det["bbox"], det["conf"], 0)
            for det in nearby_persons
        ]

        # update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        output = []
        # associate each track back to its detection contour via IoU
        for tr in tracks:
            if not tr.is_confirmed() or tr.is_deleted():
                continue
            l, t, r_, b = map(int, tr.to_ltrb())
            best_match = None
            best_iou = 0.0
            for det in nearby_persons:
                dx1, dy1, dx2, dy2 = det["bbox"]
                # compute IoU
                xx1, yy1 = max(l, dx1), max(t, dy1)
                xx2, yy2 = min(r_, dx2), min(b, dy2)
                inter_w, inter_h = max(0, xx2 - xx1), max(0, yy2 - yy1)
                inter = inter_w * inter_h
                areaA = (r_ - l) * (b - t)
                areaB = (dx2 - dx1) * (dy2 - dy1)
                union = areaA + areaB - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_match = det

            contour = best_match["contour"] if best_match and best_iou > 0.05 else None
            output.append({
                "track_id": tr.track_id,
                "bbox": (l, t, r_, b),
                "contour": contour
            })

        return output

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: list[dict]):
        """
        Draw bounding boxes, track IDs, and segmentation contours.
        """
        for tr in tracks:
            l, t, r_, b = tr["bbox"]
            pid = tr["track_id"]
            cv2.rectangle(frame, (l, t), (r_, b), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {pid}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            if tr["contour"] is not None:
                cv2.drawContours(frame, [tr["contour"]], -1, (0, 255, 0), 2)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {source}")
        print(f"Video source opened: {source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = self.process_frame(frame)
            vis = self.draw_tracks(frame, tracks)
            cv2.imshow("People+Backpack Tracker", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MultiPersonBackpackTracker(
        model_name="yolov8m-seg.pt",
        conf_thresh=0.5,
        proximity_thresh=150,
        max_age=60,
        n_init=3,
        max_cosine_distance=0.4,
        nn_budget=100,
        embedder_model_name="osnet_x0_25"
    )
    tracker.run(source=0)