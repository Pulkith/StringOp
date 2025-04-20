import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf_thresh: float = 0.66,
        device: str = None,
        color_filter: bool = False,
        brown_threshold: float = 0.50,
        max_age: int = 500,
        n_init: int = 5,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 200,
        embedder_model_name: str = "osnet_x0_25"
    ):
        # device setup
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # load YOLO
        self.model = YOLO(model_name).to(self.device)
        self.conf_thresh = conf_thresh

        # color filter params
        self.color_filter = color_filter
        self.brown_threshold = brown_threshold
        self.lower_brown = np.array([10, 50, 20])
        self.upper_brown = np.array([30, 255, 200])

        # tracker setup
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder_model_name=embedder_model_name,
            half=True,
            embedder_gpu=torch.cuda.is_available()
        )

    @staticmethod
    def _is_wearing_brown(
        roi_bgr: np.ndarray,
        lower_brown: np.ndarray,
        upper_brown: np.ndarray,
        threshold: float
    ) -> bool:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        ratio = np.count_nonzero(mask) / (roi_bgr.shape[0] * roi_bgr.shape[1])
        return ratio > threshold

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detects & tracks people in a frame.
        Returns a list of dicts: {"track_id": int, "bbox": (l, t, r, b)}
        """
        # run detection
        results = self.model.predict(frame, conf=self.conf_thresh, classes=[0], stream=False)
        dets = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                if int(cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # Ensure coordinates are properly ordered
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                
                # Get original frame dimensions
                orig_h, orig_w = frame.shape[:2]
                # Get YOLO processing dimensions (from the log: 480x640)
                yolo_h, yolo_w = 480, 640

                # Scale factors
                scale_x = orig_w / yolo_w
                scale_y = orig_h / yolo_h

                # Scale the coordinates back to original resolution
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # Now use the properly ordered coordinates
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if self.color_filter and not self._is_wearing_brown(
                    roi, self.lower_brown, self.upper_brown, self.brown_threshold
                ):
                    continue
                dets.append(((x1, y1, x2, y2), float(conf.item()), "person"))

        # update tracker
        tracks = self.tracker.update_tracks(dets, frame=frame)
        output = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            output.append({"track_id": tid, "bbox": (l, t, r, b)})
        return output

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        for tr in tracks:
            l, t, r, b = tr["bbox"]
            tid = tr["track_id"]
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )
        return frame

    def run_webcam(self, source: int = 0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = self.process_frame(frame)
            annotated = self.draw_tracks(frame, tracks)
            cv2.imshow("Person Tracker", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = PersonTracker(color_filter=False)
    tracker.run_webcam()