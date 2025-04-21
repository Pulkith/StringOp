import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random  # For distinct colors

class MultiPersonColorTracker:
    def __init__(
        self,
        model_name: str = "yolov8m-seg.pt",  # Use medium segmentation model
        conf_thresh: float = 0.60,        # Confidence threshold for YOLO detection
        device: str = None,
        uniform_thresh: float = 0.40,      # Color lock threshold
        K: int = 3,                         # K-means clusters for color
        hue_delta: int = 2,                # Hue window for color matching/locking
        mask_padding: int = 10,             # Pixels to pad mask bounding box
        # DeepSort Parameters
        max_age: int = 30,                  # Max frames to keep unmatched track
        n_init: int = 3,                    # Min hits to confirm track
        max_cosine_distance: float = 0.4,   # Max cosine distance for appearance matching
        nn_budget: int = 100,               # Size of appearance feature gallery
        embedder_model_name: str = "osnet_x0_25" # Appearance embedder
    ):
        # Device + YOLO Segmentation Model
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        self.model = YOLO(model_name).to(self.device)
        self.conf_thresh = conf_thresh
        print(f"Loaded model {model_name}")

        # DeepSort Tracker
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None, # Let DeepSort handle classes if needed, but we filter persons
            embedder_model_name=embedder_model_name,
            half=True if self.device == "cuda" else False, # Use half precision on GPU
            embedder_gpu=True if self.device == "cuda" else False,
            # embedder_wts=None, # Use default weights
            # polygon=False, # We are using bounding boxes
            # today=None # Not needed
        )
        print("Initialized DeepSort tracker")

        # Color‑locking parameters
        self.uniform_thresh = uniform_thresh
        self.K = K
        self.hue_delta = hue_delta
        self.mask_padding = mask_padding

        # Store locked persons: perm_id → { "hue": int, "color": (B,G,R) }
        self.known = {}
        self.next_id = 1
        self.track_colors = {} # Store colors for drawing assigned IDs consistently

    def _get_random_color(self):
        """Generates a random BGR color."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _find_dominant_hue(self, roi: np.ndarray, mask: np.ndarray = None) -> tuple[int, float]:
        """
        K‑means on HSV hue of masked ROI to find largest cluster and its coverage.
        If mask is provided, only considers pixels within the mask.
        """
        if roi.size == 0: return -1, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if mask is not None and mask.shape[:2] == roi.shape[:2] and np.any(mask):
            # Apply mask: Only consider pixels where mask is non-zero
            pixels_h = hsv[mask > 0][:, 0] # Extract Hue values from masked area
        else:
            # Use all pixels if no valid mask provided
            pixels_h = hsv[..., 0].reshape(-1)

        if pixels_h.size == 0: return -1, 0.0 # No pixels to analyze

        # Reshape for K-means
        pixels_h = pixels_h.reshape(-1, 1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        best_labels, centers = None, None
        try:
            # Ensure K is not greater than the number of samples
            k_actual = min(self.K, pixels_h.shape[0])
            if k_actual < 1: return -1, 0.0
            compactness, labels, centers = cv2.kmeans(pixels_h, k_actual, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten()
        except cv2.error as e:
            print(f"KMeans Error: {e}. ROI shape: {roi.shape}, Mask shape: {mask.shape if mask is not None else 'None'}, Pixel count: {pixels_h.shape[0]}")
            return -1, 0.0 # Return invalid hue if KMeans fails


        counts = np.bincount(labels, minlength=k_actual)
        idx = np.argmax(counts)
        hue_center = int(centers[idx][0])  # 0–179
        ratio = counts[idx] / pixels_h.shape[0]
        return hue_center, ratio

    def _mask_ratio(self, roi: np.ndarray, hue_center: int, mask: np.ndarray = None) -> float:
        """
        Fraction of *masked* pixels within ± hue_delta of hue_center.
        If mask is provided, only considers pixels within the mask.
        """
        if roi.size == 0: return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if mask is not None and mask.shape[:2] == roi.shape[:2] and np.any(mask):
            # Apply mask
            hsv_h = hsv[..., 0][mask > 0]
            total_pixels = np.sum(mask > 0)
        else:
            # Use all pixels
            hsv_h = hsv[..., 0].flatten()
            total_pixels = hsv_h.size

        if total_pixels == 0: return 0.0

        # Calculate absolute difference (handling wrap-around for hue)
        diff = np.abs(hsv_h - hue_center)
        diff = np.minimum(diff, 180 - diff) # Hue wraps around at 180

        within_delta = (diff <= self.hue_delta)
        match_count = np.sum(within_delta)

        return match_count / total_pixels

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detect persons using YOLOv8-seg, derive tight bounding boxes from masks,
        track using DeepSORT, and assign permanent IDs based on locked hue from mask area.
        Returns list of tracked objects: [{"track_id": int, "bbox": (l,t,r,b), "contour": ndarray, "color": (b,g,r)}]
        """
        H_IMG, W_IMG = frame.shape[:2]

        # 1) Run YOLOv8 segmentation for persons (class 0)
        results = self.model.predict(frame, conf=self.conf_thresh, classes=[0], stream=False, verbose=False)

        detections_for_deepsort = []
        contours_in_frame = [] # Store contours corresponding to detections

        if results and results[0].masks is not None: # Check if masks exist
            r = results[0]

            for i in range(len(r.boxes)): # Iterate through each detected instance
                contour = r.masks.xy[i] # This is a numpy array of [N, 2] points (polygon)
                if contour.size < 6: continue # Need at least 3 points for a contour

                # Calculate bounding box tightly around the mask contour
                x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Apply padding to the mask bounding box, clamping to image bounds
                pad_x1 = max(x1 - self.mask_padding, 0)
                pad_y1 = max(y1 - self.mask_padding, 0)
                pad_x2 = min(x2 + self.mask_padding, W_IMG)
                pad_y2 = min(y2 + self.mask_padding, H_IMG)

                # Confidence score
                conf = float(r.boxes.conf[i].item())

                # Prepare detection for DeepSORT: (bbox_xyxy, confidence, class_id/name)
                # DeepSORT expects [left, top, width, height] format, NOT xyxy
                # bbox_ltwh = [pad_x1, pad_y1, pad_x2 - pad_x1, pad_y2 - pad_y1]
                # However, the docs for deep_sort_realtime say update_tracks expects
                # detections = [([x1,y1,x2,y2], conf, cls_id), ...]
                # Let's stick to the documented xyxy format for DeepSORT input here.
                bbox_xyxy = [pad_x1, pad_y1, pad_x2, pad_y2]

                detections_for_deepsort.append((bbox_xyxy, conf, 0)) # Class 0 for person
                contours_in_frame.append(contour.astype(np.int32)) # Store contour


        # 2) Update DeepSort tracker
        # The 'frame' argument is used by DeepSort for appearance feature extraction from the bbox area
        tracks = self.tracker.update_tracks(detections_for_deepsort, frame=frame)

        # 3) Process confirmed tracks: Match to known IDs or assign new ones based on color
        output = []
        track_id_to_contour_map = {} # Map DeepSORT track ID to original contour index

        # Simple spatial matching: Map track ID to contour based on IoU or center distance
        # This is needed because DeepSORT might miss frames or tracks can be lost/re-initialized
        unmatched_contour_indices = list(range(len(contours_in_frame)))
        if tracks and contours_in_frame:
            for tr in tracks:
                if not tr.is_confirmed() or tr.is_deleted(): continue

                trk_l, trk_t, trk_r, trk_b = map(int, tr.to_ltrb())
                trk_box = np.array([trk_l, trk_t, trk_r, trk_b])
                trk_center_x = (trk_l + trk_r) / 2
                trk_center_y = (trk_t + trk_b) / 2

                best_match_idx = -1
                min_dist = float('inf')
                # Find the *closest* detection contour center to this track center
                for idx in unmatched_contour_indices:
                    contour = contours_in_frame[idx]
                    x, y, w, h = cv2.boundingRect(contour)
                    det_center_x = x + w / 2
                    det_center_y = y + h / 2
                    dist = np.sqrt((trk_center_x - det_center_x)**2 + (trk_center_y - det_center_y)**2)

                    # Basic check: distance threshold (e.g., within 50 pixels)
                    if dist < min_dist and dist < 1000:
                         min_dist = dist
                         best_match_idx = idx

                if best_match_idx != -1:
                     track_id_to_contour_map[tr.track_id] = best_match_idx
                     unmatched_contour_indices.remove(best_match_idx) # Don't match this contour again


        # Process tracks for color locking and output generation
        for tr in tracks:
            if not tr.is_confirmed() or tr.is_deleted():
                continue

            track_id = tr.track_id # This is DeepSORT's temporary track ID
            l, t, r_, b = map(int, tr.to_ltrb()) # Use DeepSORT's tracked box

            # Get the associated contour mask if found
            contour_mask = None
            full_mask_binary = np.zeros((H_IMG, W_IMG), dtype=np.uint8) # Full frame mask
            contour_points = None
            if track_id in track_id_to_contour_map:
                contour_idx = track_id_to_contour_map[track_id]
                if 0 <= contour_idx < len(contours_in_frame):
                     contour_points = contours_in_frame[contour_idx]
                     # Create a binary mask for the ROI based on the contour
                     cv2.drawContours(full_mask_binary, [contour_points], -1, 255, thickness=cv2.FILLED)
                     # Crop the binary mask to the bounding box region [t:b, l:r_]
                     contour_mask = full_mask_binary[t:b, l:r_]

            # Extract ROI using DeepSORT's potentially smoothed/predicted bbox
            roi = frame[t:b, l:r_]
            if roi.size == 0: continue

            # Perform color analysis using the ROI and the corresponding contour mask (if available)
            # This focuses color analysis on the actual person pixels, not background in the bbox
            best_pid, best_ratio = None, 0.0
            for pid, info in self.known.items():
                # Match against known hues using the mask
                ratio = self._mask_ratio(roi, info["hue"], mask=contour_mask)
                if ratio > best_ratio:
                    best_ratio, best_pid = ratio, pid

            final_pid = None
            if best_pid is not None and best_ratio >= self.uniform_thresh:
                final_pid = best_pid # Matched existing person
            else:
                # Find dominant hue within the mask to potentially lock a new person
                hue, ratio = self._find_dominant_hue(roi, mask=contour_mask)
                if hue != -1 and ratio >= self.uniform_thresh:
                    # Check if this hue is too close to an existing known hue? (Optional refinement)
                    is_new_hue = True
                    for known_info in self.known.values():
                        hue_diff = abs(hue - known_info["hue"])
                        hue_diff = min(hue_diff, 180 - hue_diff) # Wrap around
                        if hue_diff <= self.hue_delta * 1.5: # Allow some buffer
                            is_new_hue = False
                            break

                    if is_new_hue:
                        final_pid = self.next_id # Assign new ID
                        self.next_id += 1
                        new_color = self._get_random_color()
                        self.known[final_pid] = {"hue": hue, "color": new_color}
                        self.track_colors[final_pid] = new_color


            # Append to output if matched or newly locked
            if final_pid is not None:
                 # Get the color for this permanent ID
                 track_color = self.known[final_pid].get("color", (0, 255, 0))
                 if final_pid not in self.track_colors: # Ensure color exists if re-acquired
                     self.track_colors[final_pid] = track_color

                 track_info = {
                     "track_id": final_pid,         # Our permanent ID
                     "bbox": (l, t, r_, b),         # DeepSORT's box for drawing
                     "contour": contour_points,     # The contour points (can be None)
                     "color": self.track_colors[final_pid] # Consistent color for this ID
                 }
                 output.append(track_info)

        return output

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: list[dict], draw_mask: bool = True, draw_box: bool = True, box_padding: int = 10): # Added box_padding for drawing
        """
        Draws bounding boxes derived *directly* from associated contours (with padding)
        and semi-transparent masks. Uses DeepSORT only for ID assignment.
        """
        overlay = frame.copy()
        alpha = 0.4 # Mask transparency
        H_IMG, W_IMG = frame.shape[:2] # Get frame dimensions for clamping padding

        for tr in tracks:
            pid = tr["track_id"]
            color = tr["color"]
            contour = tr["contour"] # Get the associated contour for this track ID

            # --- Calculate Box from Contour for Drawing ---
            final_box_coords = None
            if contour is not None and contour.size > 0:
                # Calculate bounding box directly from the current contour
                x, y, w, h = cv2.boundingRect(contour)
                # Apply padding *for drawing*, clamping to image bounds
                draw_l = max(x - box_padding, 0)
                draw_t = max(y - box_padding, 0)
                draw_r = min(x + w + box_padding, W_IMG)
                draw_b = min(y + h + box_padding, H_IMG)
                final_box_coords = (draw_l, draw_t, draw_r, draw_b)

                # --- Draw the Mask ---
                if draw_mask:
                    cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)
            else:
                # Fallback: If contour is missing for some reason, use DeepSORT's box
                # This might still happen if association fails briefly
                 if draw_box: # Only use fallback if drawing boxes is enabled
                    final_box_coords = tr["bbox"] # This is the potentially offset box from DeepSORT


            # --- Draw the Bounding Box ---
            if draw_box and final_box_coords is not None:
                l, t, r_, b = final_box_coords
                cv2.rectangle(frame, (l, t), (r_, b), color, 2)
                cv2.putText(frame, f"ID {pid}", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Blend overlay with masks onto the original frame
        # Ensure blending happens even if no masks were drawn in this frame to maintain consistency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {source}")
        print(f"Opened video source: {source}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_count += 1
            print(f"\nProcessing frame {frame_count}...")

            # Process frame to get tracked objects with boxes, contours, IDs
            try:
                 tracked_objects = self.process_frame(frame)
                 print(f"Found {len(tracked_objects)} tracked objects in frame {frame_count}")
            except Exception as e:
                 print(f"Error processing frame {frame_count}: {e}")
                 # Optionally add traceback: import traceback; traceback.print_exc()
                 continue # Skip frame on error


            # Draw the results
            try:
                annotated_frame = self.draw_tracks(frame, tracked_objects, draw_mask=True, draw_box=True)
            except Exception as e:
                 print(f"Error drawing tracks on frame {frame_count}: {e}")
                 annotated_frame = frame # Show original frame if drawing fails

            cv2.imshow("MultiPersonColorTracker - Segmentation", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Received 'q' key press. Exiting.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Released video capture and destroyed windows.")

if __name__ == "__main__":
    print("Starting tracker...")
    # Adjust parameters as needed for drone environment
    tracker = MultiPersonColorTracker(
        model_name="yolov8s-seg.pt", # Medium model, good balance
        conf_thresh=0.5,          # Lowered slightly, might need tuning
        uniform_thresh=0.40,       # Adjust based on clothing variety
        mask_padding=20,            # Smaller padding
        max_age=200,                # Increase if tracks get lost easily
        n_init=4,                  # Require more hits to confirm
        max_cosine_distance=0.4   # Slightly more lenient appearance matching
    )
    # Use camera index 0 or path to video file
    tracker.run(source=0)
    # tracker.run(source="path/to/your/video.mp4")