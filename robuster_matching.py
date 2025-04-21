import cv2
import torch
import numpy as np
from ultralytics import YOLO
# Ensure you have deep_sort_realtime installed: pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import time
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import itertools # For generating permanent IDs
import math # For Hue circular distance

# Define a type hint for the permanent tracker data
# Now stores dominant hue info instead of histogram
PermanentTrackData = Dict[str, Any] # {"face_track_id": str, "dominant_hue_info": Optional[Tuple[float, float, float]], "last_seen_frame": int, "bbox": tuple}
DominantHueInfo = Optional[Tuple[float, float, float]] # (avg_hue, avg_saturation, percentage_size)

# Define COCO Keypoint indices (for Ultralytics YOLOv8 Pose models)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

HEAD_KP_INDICES = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
TORSO_KP_INDICES = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]

class YoloPosePermanentTracker:
    def __init__(
        self,
        yolo_model_name: str = "yolov8n-pose.pt", # *** Requires a POSE model ***
        conf_thresh: float = 0.5,                 # YOLO detection confidence threshold
        # --- Face DeepSort Parameters ---
        face_max_iou_distance: float = 0.7,
        face_max_age: int = 75,
        face_n_init: int = 5,
        face_max_cosine_distance: float = 0.6,
        face_nn_budget: Optional[int] = 100,
        face_embedder_model_name: str = "osnet_x0_25",
        # --- Torso Dominant Hue Group Parameters ---
        hue_tolerance: int = 10,                  # Allowable Hue variation within a group (+/- degrees from center)
        saturation_threshold: int = 50,           # Min saturation for a pixel to be considered 'colored'
        value_threshold: int = 50,                # Min value (brightness) for a pixel to be considered
        min_group_size_perc: float = 0.15,        # Largest hue group must cover at least this % of valid torso pixels
        min_absolute_group_pixels: int = 50,      # Largest hue group must have at least this many pixels
        hue_match_tolerance: int = 15,            # Allowable difference in avg. Hue between current and stored
        saturation_match_tolerance: int = 40,     # Allowable difference in avg. Saturation
        # --- General Tracking ---
        max_permanent_track_age: int = 600,
        device: Optional[str] = None,
        # --- Debugging ---
        enable_color_score_print: bool = True,
        enable_debug_draw: bool = True,
    ):
        # --- Device Setup ---
        self.device = device or self._get_default_device()
        print(f"Using device: {self.device}")
        self.use_half = self.device == "cuda"

        # --- Model Loading ---
        print(f"Loading YOLO Pose model: {yolo_model_name}...")
        if "-pose" not in yolo_model_name:
             print(f"Warning: Provided model '{yolo_model_name}' might not be a pose estimation model.")
        self.yolo_model = YOLO(yolo_model_name).to(self.device)
        self.yolo_model.model.eval()
        if self.use_half:
            self.yolo_model.model.half()
        print("YOLO Pose model loaded.")

        # --- Face DeepSort Tracker Initialization ---
        print("Initializing Face DeepSort tracker...")
        self.face_tracker = DeepSort(
            max_iou_distance=face_max_iou_distance, max_age=face_max_age, n_init=face_n_init,
            max_cosine_distance=face_max_cosine_distance, nn_budget=face_nn_budget,
            override_track_class=None, embedder_model_name=face_embedder_model_name,
            half=self.use_half, bgr=True, embedder_gpu=self.device != "cpu", polygon=False,
        )
        print(f"Face DeepSort initialized with embedder: {face_embedder_model_name}")

        # --- Parameters ---
        self.conf_thresh = conf_thresh
        # Color parameters
        self.hue_tolerance = hue_tolerance
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        self.min_group_size_perc = min_group_size_perc
        self.min_absolute_group_pixels = min_absolute_group_pixels
        self.hue_match_tolerance = hue_match_tolerance
        self.saturation_match_tolerance = saturation_match_tolerance
        # Other parameters
        self.max_permanent_track_age = max_permanent_track_age
        self.enable_color_score_print = enable_color_score_print
        self.enable_debug_draw = enable_debug_draw

        # --- Permanent Tracking Data ---
        self.permanent_tracker_data: Dict[str, PermanentTrackData] = {}
        self._permanent_id_counter = itertools.count(1)
        self.next_permanent_id = lambda: f"P{next(self._permanent_id_counter)}"

        # --- Visualization ---
        self.permanent_track_colors: Dict[str, Tuple[int, int, int]] = {}

        # --- Frame Specific Data (for debugging draw) ---
        self.last_person_info : Dict[int, Dict] = {}
        self.last_assignments : Dict[int, str] = {}

        self.frame_count: int = 0
        print("--- Tracker Initialized with Dominant Hue Group Color Matching ---")
        print(f"Hue Tolerance (Group): +/- {self.hue_tolerance}")
        print(f"Min Saturation (Group): {self.saturation_threshold}")
        print(f"Min Value (Group): {self.value_threshold}")
        print(f"Min Group Size (%): {self.min_group_size_perc*100:.1f}%")
        print(f"Min Abs Group Size (px): {self.min_absolute_group_pixels}")
        print(f"Hue Tolerance (Match): +/- {self.hue_match_tolerance}")
        print(f"Saturation Tolerance (Match): +/- {self.saturation_match_tolerance}")
        print("----------------------------------------------------------------")


    def _get_default_device(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        else: return "cpu"

    def _get_color(self, permanent_id: str) -> Tuple[int, int, int]:
        if permanent_id not in self.permanent_track_colors:
            random.seed(hash(permanent_id))
            self.permanent_track_colors[permanent_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        return self.permanent_track_colors[permanent_id]

    def _get_bbox_from_kpts(self, kpts_xy: np.ndarray, indices: List[int], frame_shape: Tuple[int, int], padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
        H_IMG, W_IMG = frame_shape
        valid_kpts = kpts_xy[indices]
        valid_kpts = valid_kpts[np.all(valid_kpts > 0, axis=1)] # Filter out invalid (0,0) points

        if valid_kpts.shape[0] < 2: return None

        min_x, min_y = np.min(valid_kpts, axis=0)
        max_x, max_y = np.max(valid_kpts, axis=0)

        x1 = max(0, int(min_x - padding))
        y1 = max(0, int(min_y - padding))
        x2 = min(W_IMG, int(max_x + padding))
        y2 = min(H_IMG, int(max_y + padding))

        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 5 or (y2 - y1) < 5: return None
        return (x1, y1, x2, y2)

    def _get_face_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, HEAD_KP_INDICES, frame_shape, padding=15)

    def _get_torso_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, TORSO_KP_INDICES, frame_shape, padding=10)

    def _circular_hue_distance(self, h1: float, h2: float) -> float:
        """Calculates the shortest distance between two hues on the HSV circle (0-180)."""
        diff = abs(h1 - h2)
        return min(diff, 180.0 - diff)

    def _get_dominant_hue_group(self, torso_patch: np.ndarray) -> DominantHueInfo:
        """
        Finds the largest connected group of pixels within a similar hue range
        in the torso patch, subject to saturation/value thresholds.
        """
        if torso_patch is None or torso_patch.shape[0] < 10 or torso_patch.shape[1] < 5:
            return None

        try:
            hsv_patch = cv2.cvtColor(torso_patch, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_patch)

            # 1. Mask for valid pixels (above S and V thresholds)
            valid_pixel_mask = cv2.bitwise_and(
                (s > self.saturation_threshold).astype(np.uint8),
                (v > self.value_threshold).astype(np.uint8)
            ) * 255
            total_valid_pixels = cv2.countNonZero(valid_pixel_mask)

            if total_valid_pixels < self.min_absolute_group_pixels: # Not enough valid pixels
                # print("Debug: Not enough valid pixels in torso patch.")
                return None

            # 2. Find dominant Hue among valid pixels
            # Using 18 bins for Hue (each covers 10 degrees)
            hist = cv2.calcHist([hsv_patch], [0], valid_pixel_mask, [18], [0, 180])
            dominant_hue_bin = np.argmax(hist)
            dominant_hue_center = (dominant_hue_bin + 0.5) * 10 # Center of the dominant bin

            # 3. Define Hue range around the dominant hue
            lower_hue = (dominant_hue_center - self.hue_tolerance + 180) % 180
            upper_hue = (dominant_hue_center + self.hue_tolerance) % 180

            # 4. Create mask for pixels within the dominant Hue range
            if lower_hue <= upper_hue:
                hue_mask = cv2.inRange(h, lower_hue, upper_hue)
            else: # Handle wrap-around (e.g., range 170 to 10)
                mask1 = cv2.inRange(h, lower_hue, 179)
                mask2 = cv2.inRange(h, 0, upper_hue)
                hue_mask = cv2.bitwise_or(mask1, mask2)

            # 5. Combine valid pixel mask and hue mask
            final_mask = cv2.bitwise_and(valid_pixel_mask, hue_mask)

            # 6. Find largest connected component in the final mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

            if num_labels <= 1: # Only background label
                # print(f"Debug: No connected components found for dominant hue {dominant_hue_center:.1f}")
                return None

            # Find label with largest area (excluding background label 0)
            areas = stats[1:, cv2.CC_STAT_AREA] # Get areas, skip background
            max_label = np.argmax(areas) + 1    # Add 1 back to get correct label index
            group_area = stats[max_label, cv2.CC_STAT_AREA]

            # 7. Check size thresholds
            group_percentage = group_area / total_valid_pixels
            # print(f"Debug: Dominant Hue {dominant_hue_center:.1f}, Largest Group Area: {group_area}, Perc: {group_percentage:.2f}")

            if group_percentage >= self.min_group_size_perc and group_area >= self.min_absolute_group_pixels:
                # 8. Calculate average Hue and Saturation of the largest component
                component_mask = (labels == max_label).astype(np.uint8) * 255
                mean_hsv = cv2.mean(hsv_patch, mask=component_mask)
                avg_hue = mean_hsv[0]
                avg_saturation = mean_hsv[1]
                # print(f"Debug: Found Dominant Group -> Avg H: {avg_hue:.1f}, Avg S: {avg_saturation:.1f}, Perc: {group_percentage:.2f}")
                return (avg_hue, avg_saturation, group_percentage)
            else:
                # print("Debug: Largest group too small.")
                return None

        except Exception as e:
            print(f"Warning: Error calculating dominant hue group: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed error
            return None

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        self.frame_count += 1
        H_IMG, W_IMG = frame.shape[:2]
        self.last_person_info = {}
        self.last_assignments = {}

        # --- 1. YOLO Pose Detection ---
        results = self.yolo_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        face_detections_for_deepsort = []
        person_info: Dict[int, Dict] = {}

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            keypoints_data = results[0].keypoints
            if keypoints_data is None: return []
            keypoints_xy = keypoints_data.xy.cpu().numpy()

            for i in range(len(boxes)):
                person_bbox = tuple(map(int, boxes[i]))
                conf = confs[i]; kpts = keypoints_xy[i]
                face_bbox = self._get_face_bbox_from_kpts(kpts, (H_IMG, W_IMG))
                torso_bbox = self._get_torso_bbox_from_kpts(kpts, (H_IMG, W_IMG))
                face_idx_ds = -1; dominant_hue_info = None

                if face_bbox:
                    fx1, fy1, fx2, fy2 = face_bbox
                    face_detections_for_deepsort.append(([fx1, fy1, fx2, fy2], conf, 0))
                    face_idx_ds = len(face_detections_for_deepsort) - 1

                if torso_bbox:
                    tx1, ty1, tx2, ty2 = torso_bbox
                    torso_patch = frame[ty1:ty2, tx1:tx2]
                    # --- Calculate Dominant Hue Group ---
                    dominant_hue_info = self._get_dominant_hue_group(torso_patch)
                    # --- End Calculate Dominant Hue Group ---

                person_info[i] = {
                    "person_bbox": person_bbox, "face_bbox": face_bbox, "torso_bbox": torso_bbox,
                    "face_idx_ds": face_idx_ds, "dominant_hue_info": dominant_hue_info, # Store new info
                    "current_face_track_id": None, "kpts": kpts,
                }
            self.last_person_info = person_info

        # --- 2. Update Face DeepSort Tracker ---
        face_tracks = []
        if face_detections_for_deepsort:
            try: face_tracks = self.face_tracker.update_tracks(face_detections_for_deepsort, frame=frame)
            except Exception as e: print(f"Error during Face DeepSORT update: {e}")

        active_face_track_ids = set()
        for track in face_tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            face_track_id = track.track_id; active_face_track_ids.add(face_track_id)
            original_det_idx = getattr(track, 'original_det_idx', None)
            if original_det_idx is not None:
                for p_idx, info in person_info.items():
                    if info["face_idx_ds"] == original_det_idx: info["current_face_track_id"] = face_track_id; break

        # --- 3. Gather Matching Evidence ---
        person_candidates = defaultdict(lambda: {"color": [], "face": None, "info": None})
        active_permanent_ids = {pid: data for pid, data in self.permanent_tracker_data.items() if self.frame_count - data['last_seen_frame'] <= self.max_permanent_track_age}

        if self.enable_color_score_print and person_info and active_permanent_ids: print(f"--- Frame {self.frame_count} Dominant Hue Comparisons ---")

        for person_idx, info in person_info.items():
            person_candidates[person_idx]["info"] = info
            current_hue_info : DominantHueInfo = info["dominant_hue_info"]

            print(info)

            # --- 3a. Collect Color Matches (Dominant Hue Group) ---
            if current_hue_info is not None:
                if self.enable_color_score_print and active_permanent_ids: print(f" Person {person_idx} (Hue:{current_hue_info[0]:.1f}, Sat:{current_hue_info[1]:.1f}, Perc:{current_hue_info[2]:.2f}):")
                current_h, current_s, current_perc = current_hue_info

                for perm_id, track_data in active_permanent_ids.items():
                    stored_hue_info : DominantHueInfo = track_data.get("dominant_hue_info")
                    match = False
                    reason = "No stored info"
                    if stored_hue_info is not None:
                        stored_h, stored_s, stored_perc = stored_hue_info
                        reason = ""
                        # Check size percentage
                        if current_perc >= self.min_group_size_perc and stored_perc >= self.min_group_size_perc:
                            # Check Hue distance (circular)
                            hue_dist = self._circular_hue_distance(current_h, stored_h)
                            if hue_dist <= self.hue_match_tolerance:
                                # Check Saturation distance
                                sat_dist = abs(current_s - stored_s)
                                if sat_dist <= self.saturation_match_tolerance:
                                    match = True # All checks passed
                                    reason = "Match"
                                else: reason = f"Sat Fail ({sat_dist:.1f} > {self.saturation_match_tolerance})"
                            else: reason = f"Hue Fail ({hue_dist:.1f} > {self.hue_match_tolerance})"
                        else: reason = f"Size Fail (C:{current_perc:.2f} S:{stored_perc:.2f} < {self.min_group_size_perc})"

                    if self.enable_color_score_print: print(f"  vs PermID {perm_id}: -> {reason}")

                    if match:
                         # Store as match (score 1.0 for binary match)
                         person_candidates[person_idx]["color"].append((perm_id, 1.0))

                # Keep only best match if multiple (unlikely with binary match)
                person_candidates[person_idx]["color"].sort(key=lambda x: x[1], reverse=True)


            # --- 3b. Identify Face Links ---
            current_face_id = info.get("current_face_track_id")
            if current_face_id and current_face_id in active_face_track_ids:
                linked_perm_ids = [pid for pid, data in active_permanent_ids.items() if data.get("face_track_id") == current_face_id]
                if len(linked_perm_ids) == 1: person_candidates[person_idx]["face"] = linked_perm_ids[0]
                elif len(linked_perm_ids) > 1:
                    print(f"Warning: Face {current_face_id} links to MULTIPLE PermIDs: {linked_perm_ids}. Skip P{person_idx}.")
                    person_candidates[person_idx]["face"] = "SKIP"


        # --- 4. Resolve Assignments (Prioritized Matching) ---
        assignments: Dict[int, str] = {}
        assigned_person_indices = set(); assigned_perm_ids = set()
        persons_to_process = list(person_candidates.keys())

        for person_idx in persons_to_process: # Face matches first
            face_match_result = person_candidates[person_idx]["face"]
            if face_match_result == "SKIP": assigned_person_indices.add(person_idx)
            elif face_match_result is not None:
                perm_id = face_match_result
                if perm_id not in assigned_perm_ids:
                     assignments[person_idx] = perm_id; assigned_person_indices.add(person_idx); assigned_perm_ids.add(perm_id)
                else: assigned_person_indices.add(person_idx) # Skip conflicting face

        potential_color_assignments = [] # Then color matches
        for person_idx in persons_to_process:
            if person_idx not in assigned_person_indices:
                # Since color match is binary (score=1.0), just take the first one if exists
                if person_candidates[person_idx]["color"]:
                    perm_id, score = person_candidates[person_idx]["color"][0]
                    if perm_id not in assigned_perm_ids:
                        potential_color_assignments.append((person_idx, perm_id, score))
        # Sort just in case scoring is added later
        potential_color_assignments.sort(key=lambda x: x[2], reverse=True)

        for person_idx, perm_id, score in potential_color_assignments:
            if person_idx not in assigned_person_indices and perm_id not in assigned_perm_ids:
                face_link = person_candidates[person_idx]["face"]
                if face_link is None or face_link == perm_id or face_link == "SKIP":
                     assignments[person_idx] = perm_id; assigned_person_indices.add(person_idx); assigned_perm_ids.add(perm_id)

        self.last_assignments = assignments

        # --- 4c. Update Data for Assigned Tracks ---
        for person_idx, perm_id in assignments.items():
             if perm_id in self.permanent_tracker_data:
                 info = person_candidates[person_idx]["info"]
                 # Update dominant hue info
                 if info["dominant_hue_info"] is not None:
                     self.permanent_tracker_data[perm_id]["dominant_hue_info"] = info["dominant_hue_info"]
                 self.permanent_tracker_data[perm_id]["last_seen_frame"] = self.frame_count
                 self.permanent_tracker_data[perm_id]["bbox"] = info["person_bbox"]

        # --- 5. Assign New Permanent IDs ---
        for person_idx in persons_to_process:
             if person_idx not in assigned_person_indices:
                 info = person_candidates[person_idx]["info"]
                 current_face_id = info.get("current_face_track_id")
                 dominant_hue_info = info.get("dominant_hue_info")

                 has_active_face = current_face_id and current_face_id in active_face_track_ids
                 is_new_face_track = False
                 if has_active_face: is_new_face_track = not any(data.get("face_track_id") == current_face_id for data in self.permanent_tracker_data.values())

                 has_valid_color = dominant_hue_info is not None
                 is_new_color = True # Assume new unless color match occurred with an already assigned ID
                 if person_candidates[person_idx]["color"]:
                     # Check if the matched perm_id was already assigned to someone else
                     matched_perm_id, _ = person_candidates[person_idx]["color"][0]
                     if matched_perm_id in assigned_perm_ids:
                         is_new_color = False

                 if is_new_face_track and has_valid_color and is_new_color:
                     new_perm_id = self.next_permanent_id()
                     self.permanent_tracker_data[new_perm_id] = {
                         "face_track_id": current_face_id,
                         "dominant_hue_info": dominant_hue_info, # Store new info
                         "last_seen_frame": self.frame_count,
                         "bbox": info["person_bbox"]
                     }
                     assignments[person_idx] = new_perm_id
                     print(f"Assigned NEW PermID: {new_perm_id} to P{person_idx} (Face: {current_face_id})")
                     self.last_assignments = assignments

        # --- 6. Prepare Output ---
        output_tracks = []
        for person_idx, permanent_id in assignments.items():
            if person_idx in person_info:
                x1, y1, x2, y2 = person_info[person_idx]["person_bbox"]
                output_tracks.append((x1, y1, x2, y2, permanent_id))
                self._get_color(permanent_id) # Ensure vis color exists
            else: print(f"Error: P_idx {person_idx} not found in P_info during output.")

        return output_tracks

    def draw_tracks(self, frame: np.ndarray, tracks: List[Tuple[int, int, int, int, str]]) -> np.ndarray:
        DRAW_DEBUG_BOXES = self.enable_debug_draw
        person_info = self.last_person_info
        assignments = self.last_assignments

        for (x1, y1, x2, y2, permanent_id) in tracks:
            color = self._get_color(permanent_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{permanent_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w + 5 , y1 - 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if DRAW_DEBUG_BOXES:
                 p_idx = -1
                 for idx, assigned_id in assignments.items():
                      if assigned_id == permanent_id and idx in person_info:
                           px1, py1, px2, py2 = person_info[idx]["person_bbox"]
                           if abs(x1-px1)<2 and abs(y1-py1)<2 and abs(x2-px2)<2 and abs(y2-py2)<2: p_idx = idx; break
                 if p_idx != -1:
                     info = person_info[p_idx]
                     if info.get("face_bbox"):
                         fx1, fy1, fx2, fy2 = info["face_bbox"]
                         cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 1)
                         cv2.putText(frame, "F", (fx1, fy1+10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
                     if info.get("torso_bbox"):
                         tx1, ty1, tx2, ty2 = info["torso_bbox"]
                         cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 1)
                         # Add dominant hue info text if available
                         hue_info = info.get("dominant_hue_info")
                         torso_label = "T"
                         if hue_info: torso_label += f" H:{hue_info[0]:.0f} S:{hue_info[1]:.0f}"
                         cv2.putText(frame, torso_label, (tx1, ty1+10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1)

        return frame

# --- Main Execution ---
def main():
    VIDEO_SOURCE = 0
    YOLO_MODEL = "yolov8n-pose.pt"
    CONFIDENCE_THRESHOLD = 0.55
    # Face DeepSort params
    FACE_MAX_IOU_DIST = 0.7; FACE_MAX_AGE = 75; FACE_N_INIT = 5; FACE_MAX_COS_DIST = 0.65; FACE_NN_BUDGET = None
    # --- Torso Dominant Hue Group Parameters --- Tuning needed! ---
    HUE_TOLERANCE_GROUP = 150         # How much Hue variation within the dominant group (+/-)
    SATURATION_THRESHOLD_GROUP = 0  # Min saturation of pixels considered for grouping
    VALUE_THRESHOLD_GROUP = 0       # Min value/brightness of pixels considered
    MIN_GROUP_PERC = 0.10            # Group must be at least 10% of valid torso pixels
    MIN_ABS_PIXELS = 40              # Group must have at least 40 pixels absolute
    HUE_MATCH_TOLERANCE = 20         # How close avg. Hues must be to match IDs (+/-)
    SAT_MATCH_TOLERANCE = 50         # How close avg. Sats must be to match IDs (+/-)
    # --- General ---
    MAX_PERMANENT_AGE = 900
    # --- Debugging ---
    ENABLE_COLOR_SCORE_PRINT = True # Print detailed color match info
    ENABLE_DEBUG_DRAW = True      # Draw face/torso boxes and hue info

    tracker = YoloPosePermanentTracker(
        yolo_model_name=YOLO_MODEL, conf_thresh=CONFIDENCE_THRESHOLD,
        face_max_iou_distance=FACE_MAX_IOU_DIST, face_max_age=FACE_MAX_AGE, face_n_init=FACE_N_INIT,
        face_max_cosine_distance=FACE_MAX_COS_DIST, face_nn_budget=FACE_NN_BUDGET,
        face_embedder_model_name="osnet_x0_25",
        # Pass new color params
        hue_tolerance=HUE_TOLERANCE_GROUP,
        saturation_threshold=SATURATION_THRESHOLD_GROUP,
        value_threshold=VALUE_THRESHOLD_GROUP,
        min_group_size_perc=MIN_GROUP_PERC,
        min_absolute_group_pixels=MIN_ABS_PIXELS,
        hue_match_tolerance=HUE_MATCH_TOLERANCE,
        saturation_match_tolerance=SAT_MATCH_TOLERANCE,
        # Other params
        max_permanent_track_age=MAX_PERMANENT_AGE,
        enable_color_score_print=ENABLE_COLOR_SCORE_PRINT,
        enable_debug_draw=ENABLE_DEBUG_DRAW
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {VIDEO_SOURCE}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 30
    print(f"Input: {frame_width}x{frame_height} @ {fps} FPS")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1; start_time = time.time()
        try: tracked_persons = tracker.process_frame(frame.copy())
        except Exception as e:
            print(f"--- FATAL Error processing frame {frame_num}: {e} ---")
            import traceback; traceback.print_exc(); break
        end_time = time.time(); processing_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else float('inf')
        annotated_frame = tracker.draw_tracks(frame, tracked_persons)
        cv2.putText(annotated_frame, f"FPS: {processing_fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Tracks: {len(tracked_persons)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Perm IDs: {len(tracker.permanent_tracker_data)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("YOLO Pose + Permanent Tracker (Dominant Hue)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); print("Released resources.")

if __name__ == "__main__":
    main()