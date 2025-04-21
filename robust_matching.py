import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import time
from typing import Dict, Tuple, List, Optional

class YoloDeepSortTracker:
    def __init__(
        self,
        yolo_model_name: str = "yolov8s.pt",      # YOLO model (can be 'yolov8s-seg.pt' for mask-based boxes)
        conf_thresh: float = 0.5,                # YOLO detection confidence threshold
        # DeepSort Parameters
        max_age: int = 30,                       # Max frames to keep unmatched track
        n_init: int = 3,                         # Min hits to confirm track
        max_cosine_distance: float = 0.4,        # Max cosine distance for appearance matching (lower means stricter)
        nn_budget: Optional[int] = 100,          # Size of appearance feature gallery (None for unlimited)
        embedder_model_name: str = "osnet_x0_25",# Appearance embedder model name
        device: Optional[str] = None,
        use_segmentation: bool = False,          # Set True if yolo_model_name ends with '-seg.pt' and you want to use mask bounding boxes
        mask_padding: int = 5                     # Padding around mask bounding box if use_segmentation=True
    ):
        # --- Device Setup ---
        self.device = device or self._get_default_device()
        print(f"Using device: {self.device}")
        self.use_half = self.device == "cuda"

        # --- Model Loading ---
        print(f"Loading YOLO model: {yolo_model_name}...")
        self.yolo_model = YOLO(yolo_model_name).to(self.device)
        self.yolo_model.model.eval()
        if self.use_half:
            self.yolo_model.model.half() # Use half-precision on CUDA
        print("YOLO model loaded.")

        # --- DeepSort Tracker Initialization ---
        print("Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None, # Let DeepSort handle classes if needed, but we filter persons
            # embedder=embedder_model_name, # Specify the embedder model
            half=self.use_half,
            bgr=True, # Set to True if input frames are BGR (standard for OpenCV)
            embedder_gpu=self.device != "cpu", # Use GPU for embedder if available
            embedder_model_name=embedder_model_name, # Redundant if 'embedder' is set
            embedder_wts=None, # Use default weights
            polygon=False, # We are using bounding boxes
            today=None # Not needed
        )
        print(f"DeepSort initialized with embedder: {embedder_model_name}")

        # --- Parameters ---
        self.conf_thresh = conf_thresh
        self.use_segmentation = use_segmentation and yolo_model_name.endswith("-seg.pt")
        self.mask_padding = mask_padding
        if self.use_segmentation:
            print("Segmentation mask bounding boxes will be used for detection.")

        # --- Visualization ---
        self.track_colors: Dict[str, Tuple[int, int, int]] = {} # DeepSORT uses string IDs

        self.frame_count: int = 0

    def _get_default_device(self) -> str:
        """Selects default device."""
        if torch.cuda.is_available():
            return "cuda"
        # elif torch.backends.mps.is_available(): # MPS support in libraries can vary
        #     return "mps"
        else:
            return "cpu"

    def _get_color(self, track_id: str) -> Tuple[int, int, int]:
        """Generates or retrieves a unique *base* color for a track ID."""
        if track_id not in self.track_colors:
            random.seed(int(track_id)) # Seed with track ID for consistency
            # Generate a bright base color (BGR)
            self.track_colors[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        return self.track_colors[track_id]

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """
        Processes a single frame: Detects persons, updates DeepSORT tracker.
        Returns a list of tuples: [(x1, y1, x2, y2, track_id_str), ...] for confirmed tracks.
        """
        self.frame_count += 1
        H_IMG, W_IMG = frame.shape[:2]

        # --- 1. YOLO Detection ---
        # Input frame should be BGR (OpenCV default), YOLO handles conversion if needed
        results = self.yolo_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False) # class 0 is 'person' in COCO

        detections_for_deepsort = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Extract boxes, confidences, and potentially masks
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes (x1, y1, x2, y2)
            confs = results[0].boxes.conf.cpu().numpy() # Confidence scores
            masks = results[0].masks if self.use_segmentation and results[0].masks is not None else None

            for i in range(len(boxes)):
                conf = confs[i]
                x1, y1, x2, y2 = map(int, boxes[i])

                # Use mask contour for tighter bounding box if segmentation is enabled
                if self.use_segmentation and masks is not None and i < len(masks.xy):
                    contour = masks.xy[i].astype(np.int32) # Polygon points
                    if contour.size >= 6: # Need at least 3 points
                        # Calculate bbox from contour
                        mx, my, mw, mh = cv2.boundingRect(contour)
                        # Apply padding, clamping to image bounds
                        x1 = max(0, mx - self.mask_padding)
                        y1 = max(0, my - self.mask_padding)
                        x2 = min(W_IMG, mx + mw + self.mask_padding)
                        y2 = min(H_IMG, my + mh + self.mask_padding)

                # Ensure valid box dimensions
                if x1 >= x2 or y1 >= y2:
                    print(f"Warning: Invalid box dimensions skipped: {(x1, y1, x2, y2)}")
                    continue

                # Prepare detection for DeepSORT: ([x1, y1, x2, y2], confidence, class_id)
                # Class ID 0 represents 'person'
                detections_for_deepsort.append(([x1, y1, x2, y2], conf, 0))

        # --- 2. DeepSort Update ---
        # Provide the raw frame for appearance feature extraction
        try:
            tracks = self.tracker.update_tracks(detections_for_deepsort, frame=frame)
        except Exception as e:
            print(f"Error during DeepSORT update: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed error
            tracks = [] # Continue with empty tracks if update fails

        # --- 3. Process Confirmed Tracks ---
        output_tracks = []
        for tr in tracks:
            # Check if track is confirmed and not deleted
            if not tr.is_confirmed() or tr.time_since_update > 1: # Allow 1 frame gap
                 continue

            track_id = tr.track_id # This is a string ID assigned by DeepSORT
            ltrb = tr.to_ltrb()   # Get bounding box in (left, top, right, bottom) format
            x1, y1, x2, y2 = map(int, ltrb)

            # Clamp box to image dimensions (DeepSORT prediction might go out of bounds)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W_IMG, x2)
            y2 = min(H_IMG, y2)

            # Ensure valid box dimensions after clamping
            if x1 < x2 and y1 < y2:
                output_tracks.append((x1, y1, x2, y2, track_id))
                # Ensure base color exists for visualization later
                self._get_color(track_id)

        return output_tracks

    def draw_tracks(self, frame: np.ndarray, tracks: List[Tuple[int, int, int, int, str]]) -> np.ndarray:
        """Draws bounding boxes and track IDs on the frame with time-varying hue."""
        hue_shift_speed = 2 # Controls how fast the hue changes (higher value = slower change)
        for (x1, y1, x2, y2, track_id) in tracks:
            base_color_bgr = self.track_colors.get(track_id, (0, 0, 255)) # Default to red if color missing

            # --- Calculate dynamic color ---
            # Convert base BGR to HSV
            base_color_hsv = cv2.cvtColor(np.uint8([[base_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            # Modify Hue based on frame count (modulo 180 for HSV hue range)
            hue = (int(base_color_hsv[0]) + (self.frame_count // hue_shift_speed)) % 180
            # Keep Saturation and Value from base color
            saturation = base_color_hsv[1]
            value = base_color_hsv[2]
            # Create new HSV color
            dynamic_hsv = np.uint8([[[hue, saturation, value]]])
            # Convert back to BGR
            dynamic_bgr = cv2.cvtColor(dynamic_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, dynamic_bgr)) # Convert numpy array elements to int tuple
            # --- End Calculate dynamic color ---

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            # Calculate text size for background rectangle
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Draw background rect for text
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w + 5 , y1 - 5), color, -1) # -1 fills rect
            # Put white text on the background rect
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) # White text

        return frame

# --- Main Execution ---
def main():
    # --- Configuration ---
    VIDEO_SOURCE = 0 # Webcam index or "path/to/your/video.mp4"
    YOLO_MODEL = "yolov8n-seg.pt" # Use segmentation for potentially tighter boxes initially
    USE_SEG_BOX = True           # Set True to use mask bounding boxes
    CONFIDENCE_THRESHOLD = 0.5   # Detection confidence
    # DeepSort params - adjust based on environment/performance needs
    MAX_AGE = 50
    N_INIT = 3
    MAX_COS_DIST = 0.4
    NN_BUDGET = None # Unlimited gallery size (can consume more memory)

    # --- Initialize Tracker ---
    tracker = YoloDeepSortTracker(
        yolo_model_name=YOLO_MODEL,
        conf_thresh=CONFIDENCE_THRESHOLD,
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_cosine_distance=MAX_COS_DIST,
        nn_budget=NN_BUDGET,
        embedder_model_name="osnet_x0_25", # Common lightweight embedder
        use_segmentation=USE_SEG_BOX,
        mask_padding=10
    )

    # --- Video Source ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
    print(f"Input video: {frame_width}x{frame_height} @ {fps} FPS (reported)")

    # Optional: Video Writer for saving output
    # out = cv2.VideoWriter('output_deepsort.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break

        frame_num += 1
        start_time = time.time()

        # --- Process Frame ---
        try:
            confirmed_tracks = tracker.process_frame(frame)
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            # import traceback; traceback.print_exc() # Uncomment for full error details
            confirmed_tracks = [] # Continue if processing fails for a frame

        end_time = time.time()
        processing_time = end_time - start_time
        processing_fps = 1.0 / processing_time if processing_time > 0 else float('inf')

        # --- Visualize ---
        annotated_frame = tracker.draw_tracks(frame, confirmed_tracks)

        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {processing_fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Display Track Count
        cv2.putText(annotated_frame, f"Tracks: {len(confirmed_tracks)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        cv2.imshow("YOLOv8 + DeepSORT Tracking", annotated_frame)

        # Optional: Write frame to output video
        # out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    # if out: out.release()
    cv2.destroyAllWindows()
    print("Released resources.")

if __name__ == "__main__":
    main()