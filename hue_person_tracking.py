import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiPersonColorTracker:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf_thresh: float = 0.65,
        device: str = None,
        uniform_thresh: float = 0.40,  # 40% mask for matching
        K: int = 2,                     # clusters for initial lock
        hue_delta: int = 15,            # ± hue window
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        embedder_model_name: str = "osnet_x0_25"
    ):
        # Device + YOLO
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(model_name).to(self.device)
        self.conf_thresh = conf_thresh

        # DeepSort (for initial track stability)
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder_model_name=embedder_model_name,
            half=True,
            embedder_gpu=torch.cuda.is_available()
        )

        # Color‑locking parameters
        self.uniform_thresh = uniform_thresh
        self.K = K
        self.hue_delta = hue_delta

        # Store locked persons: perm_id → { "hue": int }
        self.known = {}
        self.next_id = 1

    def _find_dominant_hue(self, roi: np.ndarray) -> tuple[int, float]:
        """K‑means on HSV hue to find largest cluster and its coverage."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H = hsv[...,0].reshape(-1,1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(H, self.K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=self.K)
        idx = np.argmax(counts)
        hue_center = int(centers[idx][0])  # 0–179
        ratio = counts[idx] / H.shape[0]
        return hue_center, ratio

    def _mask_ratio(self, roi: np.ndarray, hue_center: int) -> float:
        """Fraction of pixels within ± hue_delta of hue_center."""
        hsv_h = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[...,0]
        diff = cv2.absdiff(hsv_h, np.full_like(hsv_h, hue_center))
        mask = (diff <= self.hue_delta).astype(np.uint8)
        return mask.sum() / mask.size

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detect & assign each person a permanent ID based on locked hue.
        Returns [{"track_id": int, "bbox": (l,t,r,b)} ...].
        """
        # 1) Run YOLO detection for people
        results = self.model.predict(frame, conf=self.conf_thresh, classes=[0], stream=False)
        dets = []
        for r in results:
            for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                dets.append(((x1, y1, x2, y2), float(conf.item()), "person"))

        # 2) Update DeepSort to get stable boxes
        tracks = self.tracker.update_tracks(dets, frame=frame)

        output = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r_, b = map(int, tr.to_ltrb())
            roi = frame[t:b, l:r_]
            if roi.size == 0:
                continue

            # 3a) Try match to existing locked persons
            best_pid, best_ratio = None, 0.0
            for pid, info in self.known.items():
                ratio = self._mask_ratio(roi, info["hue"])
                if ratio > best_ratio:
                    best_ratio, best_pid = ratio, pid

            if best_pid is not None and best_ratio >= self.uniform_thresh:
                # matched an existing person
                output.append({"track_id": best_pid, "bbox": (l, t, r_, b)})
                continue

            # 3b) Not matched: see if ROI is uniform enough to *lock* as new person
            hue, ratio = self._find_dominant_hue(roi)
            if ratio >= self.uniform_thresh:
                pid = self.next_id
                self.next_id += 1
                self.known[pid] = {"hue": hue}
                output.append({"track_id": pid, "bbox": (l, t, r_, b)})

            # else: neither matched nor lockable → skip drawing

        return output

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: list[dict]):
        for tr in tracks:
            l, t, r_, b = tr["bbox"]
            pid = tr["track_id"]
            cv2.rectangle(frame, (l, t), (r_, b), (0,255,0), 2)
            cv2.putText(frame, f"ID {pid}", (l, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = self.process_frame(frame)
            annotated = self.draw_tracks(frame, tracks)
            cv2.imshow("MultiPersonColorTracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MultiPersonColorTracker(
        conf_thresh=0.6,
        uniform_thresh=0.50,
        K=3,
        hue_delta=10
    )
    tracker.run(0)