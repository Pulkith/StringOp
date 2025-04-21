# File: person_tracker.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle # Added for saving features
import argparse # Added for command-line arguments
import sys # Added for flushing print output

class PersonTracker:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf_thresh: float = 0.65,
        device: str = None,
        color_filter: bool = False,
        brown_threshold: float = 0.50,
        max_age: int = 500,
        n_init: int = 6,
        max_cosine_distance: float = 0.5,
        nn_budget: int = 200, # Keep this reasonable for memory
        embedder_model_name: str = "osnet_x0_25"
    ):
        # device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") # Prioritize CUDA if available
        print(f"Using device: {self.device}")
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
            nn_budget=nn_budget, # Make sure this is >= feature_threshold if you want unique features per track
            embedder_model_name=embedder_model_name,
            half=True if self.device == "cuda" else False, # Use half precision only on GPU
            embedder_gpu=True if self.device == "cuda" else False
        )
        # Ensure the DeepSort's internal tracker metric is accessible
        if not hasattr(self.tracker, 'tracker') or not hasattr(self.tracker.tracker, 'metric') or not hasattr(self.tracker.tracker.metric, 'samples'):
             raise AttributeError("Could not access tracker.tracker.metric.samples. DeepSort structure might have changed.")


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
        # Using stream=False is less efficient for videos but simpler here
        results = self.model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False, device=self.device) 
        dets = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                # Class 0 is 'person' in standard COCO models
                if int(cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                # Basic check for valid bounding box size
                if x1 >= x2 or y1 >= y2:
                    continue

                # Optional color filter
                if self.color_filter:
                     roi = frame[y1:y2, x1:x2]
                     if roi.size == 0:
                         continue # Skip if ROI is empty
                     if not self._is_wearing_brown(
                         roi, self.lower_brown, self.upper_brown, self.brown_threshold
                     ):
                         continue

                # Format for DeepSort: [[l, t, w, h], conf, class_name]
                # Class name is optional but good practice if used elsewhere
                w = x2 - x1
                h = y2 - y1
                dets.append(([x1, y1, w, h], float(conf.item()), "person"))

        # update tracker - this is where features are added internally
        tracks = self.tracker.update_tracks(dets, frame=frame)
        output = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            ltrb = tr.to_ltrb()
            if ltrb is None: # Handle case where track might not have bbox temporarily
                continue
            l, t, r, b = map(int, ltrb)
            output.append({"track_id": tid, "bbox": (l, t, r, b)})
        return output

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (0, 0, 255) # Changed text color for visibility
    ) -> np.ndarray:
        for tr in tracks:
            l, t, r, b = tr["bbox"]
            tid = tr["track_id"]
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (l, t - 10 if t > 10 else t + 10), # Adjust text position if box is near top
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, # Slightly larger font
                text_color,
                2
            )
        return frame

    def run_webcam(self, source: int = 0, output_filename: str = "track_features.pkl", feature_threshold: int = 50):
        """
        Runs the tracker on a webcam feed and saves features when threshold is met.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        print(f"\nCollecting features. Target: {feature_threshold}")
        print(f"Saving to: {output_filename}")
        print("Press 'q' to stop early and save current features.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video stream or cannot read frame.")
                break

            frame_count += 1
            tracks = self.process_frame(frame) # This internally calls tracker.update_tracks which updates the metric
            annotated = self.draw_tracks(frame.copy(), tracks) # Draw on a copy

            # --- Feature Counting and Saving Logic ---
            # Access the features stored in the NearestNeighborDistanceMetric
            all_features_dict = self.tracker.tracker.metric.samples
            total_feature_count = sum(len(f_list) for f_list in all_features_dict.values())

            # Display info on frame
            info_text = f"Frame: {frame_count} | Features: {total_feature_count}/{feature_threshold}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Person Tracker", annotated)

            # Print progress to console (overwrite line)
            print(f"\rCollected: {total_feature_count}/{feature_threshold} features...", end="")
            sys.stdout.flush() # Ensure it prints immediately

            # Check if threshold is reached
            if total_feature_count >= feature_threshold:
                print(f"\nReached feature threshold ({total_feature_count}). Saving features...")
                try:
                    with open(output_filename, "wb") as f:
                        pickle.dump(all_features_dict, f)
                    print(f"Successfully saved features to '{output_filename}'")
                except Exception as e:
                    print(f"\nError saving features: {e}")
                break # Exit the loop

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n'q' pressed. Saving current features...")
                try:
                    # Get the latest features before saving
                    final_features_dict = self.tracker.tracker.metric.samples
                    with open(output_filename, "wb") as f:
                         pickle.dump(final_features_dict, f)
                    print(f"Successfully saved {sum(len(fl) for fl in final_features_dict.values())} features to '{output_filename}'")
                except Exception as e:
                     print(f"\nError saving features on quit: {e}")
                break # Exit the loop

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Tracking stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Person Tracker and save features.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source path or webcam ID (e.g., 0, 1, /path/to/video.mp4)."
    )
    parser.add_argument(
        "--output",
        default="track_features_1.pkl",
        help="Output filename for pickled features (e.g., features_run1.pkl)."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Number of total features (embeddings) to collect before saving and exiting."
    )
    parser.add_argument(
        "--color-filter",
        action='store_true', # Sets to True if flag is present
        help="Enable the brown color filter."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a YAML configuration file (optional)."
    )

    args = parser.parse_args()

    # Convert source to int if it's a digit, otherwise keep as string (path)
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    print("Initializing PersonTracker...")
    # --- Initialize Tracker ---
    # Add any other necessary parameters from args if needed
    tracker = PersonTracker(color_filter=args.color_filter)

    print("Starting webcam processing...")
    tracker.run_webcam(source=video_source, output_filename=args.output, feature_threshold=args.threshold)

    print("Script finished.")