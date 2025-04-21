# Imports required
import cv2
import torch
import numpy as np # <-- Added NumPy
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import time
from typing import Dict, Tuple, List, Optional, Any, Union
from collections import defaultdict
import itertools
import re
import traceback # Added for detailed error printing in new function

# --- Color Category Definitions (HSV Ranges) ---
# These ranges might need tuning based on your specific lighting and camera.
# Order matters: Check Grayscale/Black/White first.
COLOR_CATEGORIES = {
    # Achromatic (Low Saturation or Extreme Value)
    "GRAY": {"s_max": 60, "v_min": 50, "v_max": 200}, # Adjusted Gray range
    "BLACK": {"v_max": 49},
    "WHITE": {"s_max": 60, "v_min": 201}, # White needs low saturation too
    # Chromatic (Check Hue only if not Achromatic)
    "RED": {"h_ranges": [(0, 7), (170, 179)]},
    "ORANGE_BROWN": {"h_ranges": [(7, 25)]}, # Combined Orange and Brown
    "YELLOW": {"h_ranges": [(26, 35)]},
    "GREEN": {"h_ranges": [(36, 80)]},
    "CYAN": {"h_ranges": [(81, 100)]},
    "BLUE": {"h_ranges": [(101, 130)]},
    "PURPLE_PINK": {"h_ranges": [(131, 169)]},
}

# --- NEW: Define Color Names and Indices ---
# List of chromatic color names for easier iteration
CHROMATIC_COLOR_NAMES = ["RED", "ORANGE_BROWN", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE_PINK"]
# --- IMPORTANT: Order MUST match CAT_INDICES below ---
ALL_COLOR_NAMES = ["GRAY", "BLACK", "WHITE", "RED", "ORANGE_BROWN", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE_PINK"]
NUM_COLOR_CATEGORIES = len(ALL_COLOR_NAMES)

# --- IMPORTANT: Order MUST match ALL_COLOR_NAMES above ---
CAT_IDX_GRAY = 0
CAT_IDX_BLACK = 1
CAT_IDX_WHITE = 2
CAT_IDX_RED = 3
CAT_IDX_ORANGE_BROWN = 4
CAT_IDX_YELLOW = 5
CAT_IDX_GREEN = 6
CAT_IDX_CYAN = 7
CAT_IDX_BLUE = 8
CAT_IDX_PURPLE_PINK = 9

# Map indices back to names (for converting histogram result)
IDX_TO_NAME_MAP = {i: name for i, name in enumerate(ALL_COLOR_NAMES)}
# ---------------------------------------------

# --- Type Hint for Tracker Data ---
PermanentTrackData = Dict[str, Any] # {"face_track_id": str, "locked_color_dist": Optional[Dict[str, float]], "vis_color": Tuple[int,int,int], "last_seen_frame": int, "bbox": tuple}

# --- Keypoint Indices ---
KP_NOSE = 0; KP_LEFT_EYE = 1; KP_RIGHT_EYE = 2; KP_LEFT_EAR = 3; KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5; KP_RIGHT_SHOULDER = 6; KP_LEFT_ELBOW = 7; KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9; KP_RIGHT_WRIST = 10; KP_LEFT_HIP = 11; KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13; KP_RIGHT_KNEE = 14; KP_LEFT_ANKLE = 15; KP_RIGHT_ANKLE = 16
HEAD_KP_INDICES = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
TORSO_KP_INDICES = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]

def get_id_num(perm_id: str) -> int:
    match = re.search(r'\d+$', perm_id)
    return int(match.group()) if match else -1

class YoloPoseTrackerColorCategoriesEfficient: # Renamed class slightly
    def __init__(
        self,
        yolo_model_name: str = "yolov8n-pose.pt", # Changed default back to N
        conf_thresh: float = 0.5,
        # Face DeepSort Parameters
        face_max_iou_distance: float = 0.7, face_max_age: int = 75, face_n_init: int = 5,
        face_max_cosine_distance: float = 0.6, face_nn_budget: Optional[int] = 100,
        face_embedder_model_name: str = "osnet_x0_25",
        # --- Color Category Parameters ---
        color_similarity_threshold: float = 0.4, # Min distribution intersection for strong match. NEEDS TUNING!
        color_distinctness_threshold: float = 0.7, # Max distribution intersection to be distinct for new ID. NEEDS TUNING! (Higher means harder to be distinct)
        # --- General Tracking ---
        max_permanent_track_age: int = 600,
        aspect_ratio_threshold: float = 0.8, # Min ratio of shorter to longer side for bbox adjustment
        device: Optional[str] = None,
        # --- Debugging ---
        enable_color_dist_print: bool = True, # New flag
        enable_debug_draw: bool = True,
    ):
        self.device = device or self._get_default_device()
        print(f"Using device: {self.device}")
        self.use_half = self.device == "cuda"

        print(f"Loading YOLO Pose model: {yolo_model_name}..."); self._load_yolo(yolo_model_name)
        print("Initializing Face DeepSort tracker..."); self._init_deepsort(face_max_iou_distance, face_max_age, face_n_init, face_max_cosine_distance, face_nn_budget, face_embedder_model_name)

        self.conf_thresh = conf_thresh
        self.color_sim_thresh = color_similarity_threshold
        self.color_dist_thresh = color_distinctness_threshold
        self.max_permanent_track_age = max_permanent_track_age
        self.aspect_ratio_thresh = aspect_ratio_threshold
        self.enable_color_dist_print = enable_color_dist_print
        self.enable_debug_draw = enable_debug_draw

        self.permanent_tracks: Dict[str, PermanentTrackData] = {}
        self._permanent_id_counter = itertools.count(1)
        self.next_permanent_id = lambda: f"P{next(self._permanent_id_counter)}"
        self.face_id_to_perm_id_map: Dict[str, str] = {}

        self.last_person_info : Dict[int, Dict] = {}
        self.last_assignments : Dict[int, str] = {}
        self.frame_count: int = 0

        print("--- Tracker Initialized with EFFICIENT Color Category Matching & Aspect Ratio ---")
        print(f"Assignment Priority: 1) Face Match -> 2) Color Match -> 3) New ID") # <-- Added info
        print(f"Color Similarity Threshold (Intersection): {self.color_sim_thresh:.2f}")
        print(f"Color Distinctness Threshold (Max Intersection): {self.color_dist_thresh:.2f}")
        print(f"Bounding Box Aspect Ratio Threshold: {self.aspect_ratio_thresh:.2f}")
        print("--------------------------------------------------------------------------------")


    def _load_yolo(self, yolo_model_name):
        """Loads the YOLO model."""
        if "-pose" not in yolo_model_name: print(f"Warning: '{yolo_model_name}' not pose model.")
        self.yolo_model = YOLO(yolo_model_name).to(self.device); self.yolo_model.model.eval()
        if self.use_half: self.yolo_model.model.half(); print("YOLO loaded (half).")
        else: print("YOLO loaded (full).")

    def _init_deepsort(self, max_iou_distance, max_age, n_init, max_cosine_distance, nn_budget, embedder_model_name):
        """Initializes the DeepSort tracker."""
        self.face_tracker = DeepSort(
            max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init,
            max_cosine_distance=max_cosine_distance, nn_budget=nn_budget,
            override_track_class=None, embedder_model_name=embedder_model_name,
            half=self.use_half, bgr=True, embedder_gpu=self.device != "cpu", polygon=False,
        )
        print(f"Face DeepSort initialized: {embedder_model_name}")

    def _get_default_device(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        else: return "cpu"

    def _get_random_bgr_color(self):
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    # --- BBOX FUNCTION WITH ASPECT RATIO ADJUSTMENT ---
    def _get_bbox_from_kpts(self, kpts_xy: np.ndarray, indices: List[int], frame_shape: Tuple[int, int], padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
        H_IMG, W_IMG = frame_shape
        valid_kpts = kpts_xy[indices]; valid_kpts = valid_kpts[np.all(valid_kpts > 0, axis=1)]
        if valid_kpts.shape[0] < 2: return None

        min_x, min_y = np.min(valid_kpts, axis=0); max_x, max_y = np.max(valid_kpts, axis=0)
        x1 = max(0, int(min_x - padding)); y1 = max(0, int(min_y - padding))
        x2 = min(W_IMG, int(max_x + padding)); y2 = min(H_IMG, int(max_y + padding))

        # Initial validation
        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 5 or (y2 - y1) < 5: return None

        # --- Aspect Ratio Adjustment ---
        w = float(x2 - x1)
        h = float(y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        w_new, h_new = w, h
        if w > h:
            target_h = self.aspect_ratio_thresh * w
            if h < target_h:
                h_new = target_h
        elif h > w:
            target_w = self.aspect_ratio_thresh * h
            if w < target_w:
                w_new = target_w
        # else w == h, no change needed

        # Recalculate corners if size changed
        if w_new != w or h_new != h:
            x1 = int(round(cx - w_new / 2.0))
            y1 = int(round(cy - h_new / 2.0))
            x2 = int(round(cx + w_new / 2.0))
            y2 = int(round(cy + h_new / 2.0))

            # Clip to image boundaries
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W_IMG, x2); y2 = min(H_IMG, y2)

        # Final validation after potential adjustment
        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 5 or (y2 - y1) < 5: return None

        return (x1, y1, x2, y2)
    # --------------------------------

    def _get_face_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, HEAD_KP_INDICES, frame_shape, padding=15)

    def _get_torso_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, TORSO_KP_INDICES, frame_shape, padding=10)


    # --- *** NEW EFFICIENT Color Category Distribution Function *** ---
    def _get_color_category_distribution_efficient(self, patch: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculates the distribution of color categories in a patch efficiently using NumPy masking and cv2.calcHist.
        """
        if patch is None or patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
            return None
        try:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

            # Initialize map (defaulting to GRAY might be reasonable)
            # Using uint8 is fine as we have fewer than 256 categories
            category_map = np.full(hsv.shape[:2], CAT_IDX_GRAY, dtype=np.uint8)

            # --- Create masks for each category (Achromatic first) ---
            # Black
            black_mask = v < COLOR_CATEGORIES["BLACK"]["v_max"]
            category_map[black_mask] = CAT_IDX_BLACK

            # White (Exclude black)
            white_mask = (v >= COLOR_CATEGORIES["WHITE"]["v_min"]) & \
                         (s < COLOR_CATEGORIES["WHITE"]["s_max"]) & \
                         (~black_mask)
            category_map[white_mask] = CAT_IDX_WHITE

            # Gray (Exclude black and white) - Re-assign GRAY index explicitly
            gray_mask = (s < COLOR_CATEGORIES["GRAY"]["s_max"]) & \
                        (v >= COLOR_CATEGORIES["GRAY"]["v_min"]) & \
                        (v <= COLOR_CATEGORIES["GRAY"]["v_max"]) & \
                        (~black_mask) & (~white_mask)
            category_map[gray_mask] = CAT_IDX_GRAY

            # --- Combined Achromatic mask for efficient exclusion below ---
            achromatic_mask = black_mask | white_mask | gray_mask

            # --- Create masks for Chromatic colors (Exclude all achromatic) ---
            # Red
            red_h_mask = ((h >= COLOR_CATEGORIES["RED"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["RED"]["h_ranges"][0][1])) | \
                         ((h >= COLOR_CATEGORIES["RED"]["h_ranges"][1][0]) & (h <= COLOR_CATEGORIES["RED"]["h_ranges"][1][1]))
            category_map[red_h_mask & ~achromatic_mask] = CAT_IDX_RED

            # Orange/Brown
            orange_h_mask = (h >= COLOR_CATEGORIES["ORANGE_BROWN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["ORANGE_BROWN"]["h_ranges"][0][1])
            category_map[orange_h_mask & ~achromatic_mask] = CAT_IDX_ORANGE_BROWN

            # Yellow
            yellow_h_mask = (h >= COLOR_CATEGORIES["YELLOW"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["YELLOW"]["h_ranges"][0][1])
            category_map[yellow_h_mask & ~achromatic_mask] = CAT_IDX_YELLOW

            # Green
            green_h_mask = (h >= COLOR_CATEGORIES["GREEN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["GREEN"]["h_ranges"][0][1])
            category_map[green_h_mask & ~achromatic_mask] = CAT_IDX_GREEN

            # Cyan
            cyan_h_mask = (h >= COLOR_CATEGORIES["CYAN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["CYAN"]["h_ranges"][0][1])
            category_map[cyan_h_mask & ~achromatic_mask] = CAT_IDX_CYAN

            # Blue
            blue_h_mask = (h >= COLOR_CATEGORIES["BLUE"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["BLUE"]["h_ranges"][0][1])
            category_map[blue_h_mask & ~achromatic_mask] = CAT_IDX_BLUE

            # Purple/Pink
            purple_h_mask = (h >= COLOR_CATEGORIES["PURPLE_PINK"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["PURPLE_PINK"]["h_ranges"][0][1])
            category_map[purple_h_mask & ~achromatic_mask] = CAT_IDX_PURPLE_PINK

            # --- Calculate 1D Histogram of Category Indices ---
            # The histogram will have bins 0 through NUM_COLOR_CATEGORIES-1
            hist = cv2.calcHist([category_map], [0], None, [NUM_COLOR_CATEGORIES], [0, NUM_COLOR_CATEGORIES])

            # --- Normalize histogram to get distribution (sum = 1.0) ---
            total_pixels = category_map.size
            if total_pixels == 0: return None # Avoid division by zero
            hist_normalized = hist / total_pixels

            # --- Convert histogram to dictionary format ---
            # Include only categories with non-zero percentages
            distribution = {IDX_TO_NAME_MAP[i]: float(hist_normalized[i]) for i in range(NUM_COLOR_CATEGORIES) if hist_normalized[i] > 0}

            return distribution

        except Exception as e:
            print(f"Error calculating efficient color distribution: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return None
    # ---------------------------------------------------------------


    def _compare_color_distributions(self, dist1: Optional[Dict[str, float]], dist2: Optional[Dict[str, float]]) -> float:
        """Compares two color distributions using histogram intersection."""
        if dist1 is None or dist2 is None: return 0.0

        intersection = 0.0
        # Consider all possible categories defined
        for category_name in ALL_COLOR_NAMES:
            intersection += min(dist1.get(category_name, 0.0), dist2.get(category_name, 0.0))

        # Intersection score is already between 0 and 1 (higher is more similar)
        return intersection
    # -----------------------------------


    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, Tuple[int, int, int]]]:
        self.frame_count += 1; H_IMG, W_IMG = frame.shape[:2]
        self.last_person_info = {}; self.last_assignments = {}

        # --- 1. YOLO Pose Detection & Feature Extraction ---
        # (No changes needed here)
        results = self.yolo_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        face_detections_for_deepsort = []; person_info: Dict[int, Dict] = {}

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy(); confs = results[0].boxes.conf.cpu().numpy()
            keypoints_data = results[0].keypoints;
            if keypoints_data is None: return []
            keypoints_xy = keypoints_data.xy.cpu().numpy()

            for i in range(len(boxes)):
                person_bbox = tuple(map(int, boxes[i])); kpts = keypoints_xy[i]
                face_bbox = self._get_face_bbox_from_kpts(kpts, (H_IMG, W_IMG))
                torso_bbox = self._get_torso_bbox_from_kpts(kpts, (H_IMG, W_IMG))

                face_idx_ds = -1; current_color_dist = None; torso_patch = None

                if face_bbox:
                    fx1, fy1, fx2, fy2 = face_bbox
                    face_detections_for_deepsort.append(([fx1, fy1, fx2, fy2], confs[i], 0))
                    face_idx_ds = len(face_detections_for_deepsort) - 1

                if torso_bbox:
                    tx1, ty1, tx2, ty2 = torso_bbox
                    if ty2 > ty1 and tx2 > tx1:
                        torso_patch = frame[ty1:ty2, tx1:tx2]
                        current_color_dist = self._get_color_category_distribution_efficient(torso_patch)
                    else: torso_patch = None

                person_info[i] = {
                    "person_bbox": person_bbox, "face_bbox": face_bbox, "torso_bbox": torso_bbox,
                    "face_idx_ds": face_idx_ds,
                    "current_color_dist": current_color_dist,
                    "torso_patch": torso_patch,
                    "current_face_track_id": None, "kpts": kpts,
                }
            self.last_person_info = person_info

        # --- 2. Update Face DeepSort Tracker ---
        # (No changes needed here)
        face_tracks = []; active_face_track_ids = set(); current_frame_face_id_map = {}
        if face_detections_for_deepsort:
            try: face_tracks = self.face_tracker.update_tracks(face_detections_for_deepsort, frame=frame)
            except Exception as e: print(f"Error DeepSORT update: {e}")
        for track in face_tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            face_track_id = track.track_id; active_face_track_ids.add(face_track_id)
            original_det_idx = getattr(track, 'original_det_idx', None)
            if original_det_idx is not None:
                for p_idx, info in person_info.items():
                    if info["face_idx_ds"] == original_det_idx:
                        info["current_face_track_id"] = face_track_id
                        current_frame_face_id_map[face_track_id] = p_idx; break

        # --- 3. Gather Matching Evidence ---
        # (No changes needed here, but the interpretation of 'face_perm_id' in step 4 changes)
        person_candidates = defaultdict(lambda: {"color": [], "face_perm_id": None, "info": None})
        active_permanent_ids = {pid: data for pid, data in self.permanent_tracks.items() if self.frame_count - data['last_seen_frame'] <= self.max_permanent_track_age}

        # Update face_id_to_perm_id_map based on active permanent tracks
        current_face_mapping = {}
        for perm_id, data in active_permanent_ids.items():
             linked_face_id = data.get("face_track_id")
             if linked_face_id:
                  # Simple assignment, potential conflicts noted but not fully resolved here
                  if linked_face_id in current_face_mapping and current_face_mapping[linked_face_id] != perm_id:
                      print(f"Warning: Face ID {linked_face_id} linked to multiple Permanent IDs ({current_face_mapping[linked_face_id]} and {perm_id}). Using the latter.")
                  current_face_mapping[linked_face_id] = perm_id
        self.face_id_to_perm_id_map = current_face_mapping # This map now reflects persistent links

        if self.enable_color_dist_print and person_info and active_permanent_ids: print(f"--- Frame {self.frame_count} Color Distribution Comparisons ---")

        for person_idx, info in person_info.items():
            person_candidates[person_idx]["info"] = info
            current_distribution = info.get("current_color_dist")

            # 3a. Collect Color Matches
            if current_distribution is not None:
                 if self.enable_color_dist_print:
                      dist_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(current_distribution.items()) if v > 0.01])
                      print(f" Person {person_idx} (FaceID: {info.get('current_face_track_id')}) Dist: [{dist_str}]:")

                 for perm_id, track_data in active_permanent_ids.items():
                     stored_distribution = track_data.get("locked_color_dist")
                     match_similarity = 0.0
                     reason = "No stored distribution"

                     if stored_distribution is not None:
                         match_similarity = self._compare_color_distributions(current_distribution, stored_distribution)
                         reason = f"Dist Intersection={match_similarity:.3f}"
                         if match_similarity >= self.color_sim_thresh:
                             reason += f" >= Thresh({self.color_sim_thresh:.2f}) -> POTENTIAL MATCH"
                             person_candidates[person_idx]["color"].append((perm_id, match_similarity)) # Store potential match
                         else:
                             reason += f" < Thresh -> NO MATCH"

                     if self.enable_color_dist_print: print(f"  vs PermID {perm_id} (LinkedFace: {track_data.get('face_track_id')}): -> {reason}")

                 person_candidates[person_idx]["color"].sort(key=lambda x: x[1], reverse=True) # Sort potential color matches

            # 3b. Identify Potential Face Link (Used in Step 4a)
            current_face_id = info.get("current_face_track_id")
            if current_face_id and current_face_id in self.face_id_to_perm_id_map:
                # Store the perm_id that *should* be linked via face persistence
                person_candidates[person_idx]["face_perm_id"] = self.face_id_to_perm_id_map[current_face_id]


        # --- 4. Resolve Assignments (REVISED LOGIC) ---
        assignments: Dict[int, str] = {}
        assigned_person_indices = set()
        assigned_perm_ids = set()
        persons_to_process = list(person_candidates.keys())

        # --- 4a. PRIORITY: Assign based on Direct Face Link Persistence ---
        if self.enable_debug_draw: print("--- Assignment Step 4a (PRIORITY: Face Link Persistence) ---")
        for person_idx in persons_to_process:
             # Check if this person's current face_id is linked to a persistent perm_id
             linked_perm_id = person_candidates[person_idx].get("face_perm_id") # From Step 3b

             if linked_perm_id and linked_perm_id in active_permanent_ids:
                 # Check if this permanent ID hasn't already been assigned in this frame
                 if linked_perm_id not in assigned_perm_ids:
                     assignments[person_idx] = linked_perm_id
                     assigned_person_indices.add(person_idx)
                     assigned_perm_ids.add(linked_perm_id)
                     if self.enable_debug_draw: print(f"  P{person_idx} assigned {linked_perm_id} via persistent face link (FaceID: {person_candidates[person_idx]['info']['current_face_track_id']}).")
                 else:
                     # This person's face links to a perm_id already taken by someone else via face link.
                     # Mark person as processed to prevent fallback, but don't assign. Needs conflict resolution?
                     assigned_person_indices.add(person_idx)
                     if self.enable_debug_draw: print(f"  P{person_idx} face links to {linked_perm_id} (FaceID: {person_candidates[person_idx]['info']['current_face_track_id']}), but {linked_perm_id} already assigned. Person skipped.")
             # else: No face link or linked perm_id is inactive. Proceed to fallback.


        # --- 4b. FALLBACK: Assign based on Color Matches (for unassigned persons/perm_ids) ---
        if self.enable_debug_draw: print("--- Assignment Step 4b (FALLBACK: Color Dist Score) ---")
        color_candidates = []
        # Iterate through persons *not* assigned by face link
        for person_idx in persons_to_process:
            if person_idx not in assigned_person_indices:
                potential_color_matches = person_candidates[person_idx]["color"] # Sorted list from Step 3a
                # Find the best *available* color match for this person
                best_available_match = None
                for perm_id, score in potential_color_matches:
                    if perm_id in active_permanent_ids and perm_id not in assigned_perm_ids:
                        # Found the highest-scoring match to an available perm_id
                        best_available_match = {"person_idx": person_idx, "perm_id": perm_id, "score": score}
                        break # Only consider the top valid match per person
                if best_available_match:
                    color_candidates.append(best_available_match)

        # Sort all potential color assignments by score (highest first)
        color_candidates.sort(key=lambda x: x["score"], reverse=True)
        if self.enable_debug_draw and color_candidates: print(f"  Color Candidates (Dist Sim): {[{'P':c['person_idx'], 'ID':c['perm_id'], 'S':round(c['score'],3)} for c in color_candidates]}")

        # Assign based on the sorted color candidates, ensuring no conflicts
        for candidate in color_candidates:
            person_idx = candidate["person_idx"]; perm_id = candidate["perm_id"]
            # Double-check: ensure neither person nor perm_id has been assigned in the meantime
            if person_idx not in assigned_person_indices and perm_id not in assigned_perm_ids:
                assignments[person_idx] = perm_id
                assigned_person_indices.add(person_idx)
                assigned_perm_ids.add(perm_id)
                if self.enable_debug_draw: print(f"  P{person_idx} assigned {perm_id} via color fallback (Dist Sim: {candidate['score']:.3f}).")

        self.last_assignments = assignments # Store final assignments for this frame

        # --- 4c. Update Data for Assigned Tracks ---
        # (This logic remains the same, updates based on the final 'assignments' map)
        for person_idx, perm_id in assignments.items():
              if perm_id in self.permanent_tracks: # Should always be true here
                  info = person_candidates[person_idx]["info"]
                  self.permanent_tracks[perm_id]["last_seen_frame"] = self.frame_count
                  self.permanent_tracks[perm_id]["bbox"] = info["person_bbox"] # Update bbox
                  current_face_id = info.get("current_face_track_id")

                  # Update face link if necessary (e.g., face re-acquired or assigned via color)
                  if current_face_id:
                       # If the perm track wasn't linked or linked to a different face, update
                       if self.permanent_tracks[perm_id].get("face_track_id") != current_face_id:
                            old_face_id = self.permanent_tracks[perm_id].get("face_track_id")
                            # Clean up old link in the map if it existed and pointed here
                            if old_face_id and old_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[old_face_id] == perm_id:
                                del self.face_id_to_perm_id_map[old_face_id]
                            # Set the new link
                            self.permanent_tracks[perm_id]["face_track_id"] = current_face_id
                            # Update the map for future frames
                            if current_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[current_face_id] != perm_id:
                                print(f"Warning: Face ID {current_face_id} was linked to {self.face_id_to_perm_id_map[current_face_id]}, overwriting with {perm_id}.")
                            self.face_id_to_perm_id_map[current_face_id] = perm_id
                  else:
                       # Person assigned (likely via color), but no face detected this frame.
                       # Do NOT clear the existing face_track_id link in permanent_tracks,
                       # as the face might reappear later. The self.face_id_to_perm_id_map
                       # was already updated at the start of step 3 based on active tracks.
                       pass

        # --- 5. Assign New Permanent IDs ---
        # (Logic mostly the same, but now only applies to people unassigned after BOTH face and color steps)
        if self.enable_debug_draw: print("--- Assignment Step 5 (New IDs - Color Categories) ---")
        for person_idx in persons_to_process:
             # Only consider persons not assigned in Step 4a or 4b
             if person_idx not in assigned_person_indices:
                 info = person_candidates[person_idx]["info"]
                 current_face_id = info.get("current_face_track_id")
                 current_distribution = info.get("current_color_dist")

                 can_create_new_id = True; reason = ""

                 # Check 1: Does this person's face ID *already* map to an existing permanent ID?
                 # This prevents creating a duplicate permanent ID if the face link exists but
                 # the target perm_id was assigned to someone else in step 4a.
                 if current_face_id and current_face_id in self.face_id_to_perm_id_map:
                     existing_perm_id = self.face_id_to_perm_id_map[current_face_id]
                     # Check if that existing perm_id is still active
                     if existing_perm_id in active_permanent_ids:
                         can_create_new_id = False; reason = f"Face ID {current_face_id} already linked to active PermID {existing_perm_id}"
                         # We don't assign here, as the logic prevents re-assigning `existing_perm_id` if it was already taken.
                         # This scenario might indicate a tracking conflict or merge opportunity later.
                     # else: face links to an inactive perm_id, safe to create new or re-assign face later

                 # Check 2: Valid Color Distribution? (Only if still possible to create new ID)
                 if can_create_new_id:
                     if current_distribution is None or not current_distribution: # Check if empty dict
                         can_create_new_id = False; reason = "No valid color distribution"

                 # Check 3: Color Distinctness (Only if still possible)
                 if can_create_new_id:
                      is_distinct_dist = True
                      max_intersection = 0.0
                      most_similar_perm_id = None
                      # Compare against ALL currently active permanent tracks
                      for perm_id, track_data in active_permanent_ids.items():
                           locked_dist = track_data.get("locked_color_dist")
                           if locked_dist is not None:
                               intersection = self._compare_color_distributions(current_distribution, locked_dist)
                               if intersection > max_intersection:
                                   max_intersection = intersection
                                   most_similar_perm_id = perm_id
                               if intersection > self.color_dist_thresh: # Is it TOO similar?
                                   is_distinct_dist = False; break
                      if not is_distinct_dist:
                           can_create_new_id = False
                           reason = f"Color dist too similar to existing PermID {most_similar_perm_id} (max intersection: {max_intersection:.3f} > thresh: {self.color_dist_thresh:.3f})"

                 # Create New ID if all checks passed
                 if can_create_new_id:
                     new_perm_id = self.next_permanent_id()
                     new_vis_color = self._get_random_bgr_color()
                     self.permanent_tracks[new_perm_id] = {
                         "face_track_id": current_face_id, # Link face if available
                         "locked_color_dist": current_distribution, # Lock the color distribution
                         "vis_color": new_vis_color,
                         "last_seen_frame": self.frame_count,
                         "bbox": info["person_bbox"]
                     }
                     # Update the face map immediately if a face was present
                     if current_face_id:
                         if current_face_id in self.face_id_to_perm_id_map: print(f"Warning: Overwriting face map for {current_face_id} with NEW ID {new_perm_id}")
                         self.face_id_to_perm_id_map[current_face_id] = new_perm_id

                     assignments[person_idx] = new_perm_id # Add to this frame's assignments
                     assigned_person_indices.add(person_idx) # Mark as assigned
                     assigned_perm_ids.add(new_perm_id) # Mark new ID as used this frame
                     self.last_assignments = assignments # Update last assignments
                     dom_cat = max(current_distribution, key=current_distribution.get) if current_distribution else "N/A"
                     print(f"Assigned NEW PermID: {new_perm_id} to P{person_idx} (Face: {current_face_id}, Dom Col: {dom_cat})")
                 elif reason and self.enable_debug_draw:
                     # Only print reason if a new ID wasn't created and we have a reason
                     print(f"  Did not assign new ID to P{person_idx}: {reason}")


        # --- 6. Prune Old Permanent Tracks & Update Face Map ---
        # (No changes needed here)
        ids_to_remove = [pid for pid, data in self.permanent_tracks.items() if self.frame_count - data['last_seen_frame'] > self.max_permanent_track_age]
        if ids_to_remove:
            if self.enable_debug_draw: print(f"--- Pruning {len(ids_to_remove)} old PermIDs: {ids_to_remove} ---")
            for pid in ids_to_remove:
                 removed_data = self.permanent_tracks.pop(pid, None)
                 if removed_data:
                     removed_face_id = removed_data.get("face_track_id")
                     # Clean up face map if the removed track had the primary link
                     if removed_face_id and removed_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[removed_face_id] == pid:
                          del self.face_id_to_perm_id_map[removed_face_id]
                          if self.enable_debug_draw: print(f"   Removed face link for {removed_face_id} from pruned PermID {pid}")


        # --- 7. Prepare Output ---
        # (No changes needed here)
        output_tracks = []
        # Use the final 'assignments' map which includes assignments from all steps (4a, 4b, 5)
        for person_idx, permanent_id in assignments.items():
              # Ensure the permanent track still exists (it should, unless pruned mid-frame - unlikely)
              # and the person_info is available
              if permanent_id in self.permanent_tracks and person_idx in person_info:
                  x1, y1, x2, y2 = person_info[person_idx]["person_bbox"]
                  vis_color = self.permanent_tracks[permanent_id].get("vis_color", (0, 0, 255))
                  output_tracks.append((x1, y1, x2, y2, permanent_id, vis_color))

        return output_tracks


    # --- draw_tracks (Show dominant color category) ---
    # (No changes needed here, it uses the final assignment result)
    def draw_tracks(self, frame: np.ndarray, tracks: List[Tuple[int, int, int, int, str, Tuple[int,int,int]]]) -> np.ndarray:
        DRAW_DEBUG_BOXES = self.enable_debug_draw
        person_info = self.last_person_info
        assignments = self.last_assignments # Use the stored assignments from process_frame

        for (x1, y1, x2, y2, permanent_id, vis_color) in tracks:
            p_idx = -1; linked_face_id = "N/A"; current_face_id_this_frame = "N/A"
            perm_track_data = self.permanent_tracks.get(permanent_id, {})
            locked_dist = perm_track_data.get("locked_color_dist")
            persistent_face_link = perm_track_data.get("face_track_id", "N/A") # Face ID linked in permanent store

            dom_cat = "N/A"
            if locked_dist: # Check if dictionary is not None and not empty
                dom_cat = max(locked_dist, key=locked_dist.get, default="N/A")

            # Find the original person index (p_idx) and their face ID *in this frame*
            for idx, assigned_id in assignments.items():
                 if assigned_id == permanent_id and idx in person_info:
                    p_idx = idx
                    current_face_id_this_frame = person_info[idx].get("current_face_track_id", "N/A")
                    break

            label = f"ID:{permanent_id}"
            # Display the face ID *persistently linked* to this perm ID
            if persistent_face_link != "N/A": label += f" F:{persistent_face_link}"
            # Display the locked dominant color category
            if dom_cat != "N/A": label += f" C:{dom_cat}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), vis_color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10
            cv2.rectangle(frame, (x1, label_y - h - 5), (x1 + w + 5 , label_y + 5), vis_color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if DRAW_DEBUG_BOXES and p_idx != -1:
                 info = person_info[p_idx]
                 # Draw face box (if exists) and show current face ID for this frame
                 if info.get("face_bbox"):
                     fx1, fy1, fx2, fy2 = info["face_bbox"]; cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 1)
                     if current_face_id_this_frame != "N/A":
                         cv2.putText(frame, f"CurF:{current_face_id_this_frame}", (fx1, fy1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 1)

                 # Draw torso box (if exists) and show current/locked dominant color
                 if info.get("torso_bbox"):
                     tx1, ty1, tx2, ty2 = info["torso_bbox"]; cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 1)
                     current_dist = info.get("current_color_dist")
                     curr_dom_cat = "?"
                     if current_dist: # Check if not None and not empty
                         curr_dom_cat = max(current_dist, key=current_dist.get, default="?")

                     torso_label = f"Cur:{curr_dom_cat}"
                     if dom_cat != "N/A": torso_label += f" Lock:{dom_cat}" # Use dom_cat calculated earlier
                     cv2.putText(frame, torso_label, (tx1, ty1 + 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
        return frame


# --- Main Execution ---
# (No changes needed in main function)
def main():
    # --- Configuration ---
    VIDEO_SOURCE = 0
    YOLO_MODEL = "yolov8n-pose.pt" # <-- CHANGED BACK TO NANO for better performance
    CONFIDENCE_THRESHOLD = 0.50
    # Face
    FACE_MAX_IOU_DIST=0.6; FACE_MAX_AGE=250; FACE_N_INIT=20; FACE_MAX_COS_DIST=0.2; FACE_NN_BUDGET=1000
    # Color Category Params - NEED TUNING
    COLOR_SIM_THRESH = 0.6   # Min intersection for match (0.0-1.0)
    COLOR_DIST_THRESH = 0.6 # Max intersection for distinctness (0.0-1.0). Higher = harder to be distinct.
    # General
    MAX_PERMANENT_AGE = 600
    ASPECT_RATIO_THRESH = 0.8 # Enforce bbox aspect ratio
    # Debug
    ENABLE_COLOR_DIST_PRINT = False # Keep false unless debugging thresholds
    ENABLE_DEBUG_DRAW = True

    tracker = YoloPoseTrackerColorCategoriesEfficient( # Use new class name
        yolo_model_name=YOLO_MODEL, conf_thresh=CONFIDENCE_THRESHOLD,
        face_max_iou_distance=FACE_MAX_IOU_DIST, face_max_age=FACE_MAX_AGE, face_n_init=FACE_N_INIT,
        face_max_cosine_distance=FACE_MAX_COS_DIST, face_nn_budget=FACE_NN_BUDGET,
        color_similarity_threshold=COLOR_SIM_THRESH,
        color_distinctness_threshold=COLOR_DIST_THRESH,
        max_permanent_track_age=MAX_PERMANENT_AGE,
        aspect_ratio_threshold = ASPECT_RATIO_THRESH,
        enable_color_dist_print=ENABLE_COLOR_DIST_PRINT,
        enable_debug_draw=ENABLE_DEBUG_DRAW,
    )

    # (Video Input Setup - Same)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {VIDEO_SOURCE}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS); fps = int(fps_cap) if fps_cap and fps_cap > 0 else 30
    print(f"Input: {frame_width}x{frame_height} @ {fps} FPS")

    # (Processing Loop - Same)
    frame_num = 0
    while True:
        ret, frame = cap.read(); start_time = time.time()
        if not ret: break
        frame_num += 1
        try:
            # Make sure frame is writable if DeepSORT needs it (copy often ensures this)
            frame_copy = frame.copy()
            tracked_persons = tracker.process_frame(frame_copy)
            proc_time = time.time() - start_time; processing_fps = 1.0 / proc_time if proc_time > 0 else float('inf')

            # Draw tracks on the original frame
            annotated_frame = tracker.draw_tracks(frame, tracked_persons)

            # Display Info
            cv2.putText(annotated_frame, f"FPS: {processing_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Tracks: {len(tracked_persons)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Perm IDs: {len(tracker.permanent_tracks)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO Pose + Permanent Tracker (Face Priority)", annotated_frame) # Updated window title slightly
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        except Exception as e:
            print(f"\n--- FATAL ERROR frame {frame_num} ---")
            traceback.print_exc() # Print full traceback
            print("---")
            break

    # (Cleanup - Same)
    cap.release(); cv2.destroyAllWindows(); print("Resources released.")


if __name__ == "__main__":
    main()