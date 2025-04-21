# -*- coding: utf-8 -*-
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
import sys # For sys.exit

# --- Ollama Integration Imports ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("------------------------------------------------------")
    print("WARNING: 'ollama' library not found.")
    print("LLM color filtering will be disabled.")
    print("Install it with: pip install ollama")
    print("Ensure the Ollama service is running and you have pulled a model (e.g., 'ollama pull gemma:2b')")
    print("------------------------------------------------------")
    OLLAMA_AVAILABLE = True
# -----------------------------

# --- Color Category Definitions (HSV Ranges) ---
# (Same as before)
COLOR_CATEGORIES = {
    # Achromatic (Low Saturation or Extreme Value)
    "GRAY":        {"s_max": 50,  "v_min": 40,  "v_max": 200},  # tighter saturation, slightly darker
    "BLACK":       {"v_max": 30},                             # deeper blacks
    "WHITE":       {"s_max": 120,  "v_min": 150},               # very bright, low saturation

    # Chromatic (only if not caught by above)
    "RED":         {"h_ranges": [(0, 10),   (170, 180)]},     # broader red wrap
    "ORANGE_BROWN":{"h_ranges": [(11, 30)]},                  # shifts orange→brown boundary
    "YELLOW":      {"h_ranges": [(31, 40)]},                  # tighter yellow
    "GREEN":       {"h_ranges": [(41, 90)]},                  # extends into richer greens
    "CYAN":        {"h_ranges": [(91, 120)]},                 # narrowed cyan band
    "BLUE":        {"h_ranges": [(121, 150)]},                # shifted for MacBook blues
    "PURPLE_PINK": {"h_ranges": [(151, 169)]},                # covers purples and pinks
}

# --- Define Color Names and Indices ---
# (Same as before)
CHROMATIC_COLOR_NAMES = ["RED", "ORANGE_BROWN", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE_PINK"]
ALL_COLOR_NAMES = ["GRAY", "BLACK", "WHITE", "RED", "ORANGE_BROWN", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE_PINK"]
NUM_COLOR_CATEGORIES = len(ALL_COLOR_NAMES)
CAT_IDX_GRAY = 0; CAT_IDX_BLACK = 1; CAT_IDX_WHITE = 2; CAT_IDX_RED = 3
CAT_IDX_ORANGE_BROWN = 4; CAT_IDX_YELLOW = 5; CAT_IDX_GREEN = 6; CAT_IDX_CYAN = 7
CAT_IDX_BLUE = 8; CAT_IDX_PURPLE_PINK = 9
IDX_TO_NAME_MAP = {i: name for i, name in enumerate(ALL_COLOR_NAMES)}

# --- Type Hint for Tracker Data ---
# (Same as before)
PermanentTrackData = Dict[str, Any] # {"face_track_id": str, "locked_color_dist": Optional[Dict[str, float]], "vis_color": Tuple[int,int,int], "last_seen_frame": int, "bbox": tuple}

# --- Keypoint Indices ---
# (Same as before)
KP_NOSE = 0; KP_LEFT_EYE = 1; KP_RIGHT_EYE = 2; KP_LEFT_EAR = 3; KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5; KP_RIGHT_SHOULDER = 6; KP_LEFT_ELBOW = 7; KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9; KP_RIGHT_WRIST = 10; KP_LEFT_HIP = 11; KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13; KP_RIGHT_KNEE = 14; KP_LEFT_ANKLE = 15; KP_RIGHT_ANKLE = 16
HEAD_KP_INDICES = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
TORSO_KP_INDICES = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]

def get_id_num(perm_id: str) -> int:
    # (Same as before)
    match = re.search(r'\d+$', perm_id)
    return int(match.group()) if match else -1

class YoloPoseTrackerColorCategoriesEfficient:
    # --- __init__ (Removed LLM device, simplified) ---
    def __init__(
        self,
        yolo_model_name: str = "yolov8n-pose.pt",
        conf_thresh: float = 0.5,
        # Face DeepSort Parameters
        face_max_iou_distance: float = 0.7, face_max_age: int = 75, face_n_init: int = 5,
        face_max_cosine_distance: float = 0.6, face_nn_budget: Optional[int] = 100,
        face_embedder_model_name: str = "osnet_x0_25",
        # Color Category Parameters
        color_similarity_threshold: float = 0.6,
        color_distinctness_threshold: float = 0.7,
        # General Tracking
        max_permanent_track_age: int = 600,
        aspect_ratio_threshold: float = 0.8,
        # --- REMOVED LLM device parameter ---
        # Debugging
        enable_color_dist_print: bool = True,
        enable_debug_draw: bool = True,
    ):
        self.device = self._get_default_device() # YOLO/DeepSort device
        print(f"Using device for YOLO/DeepSort: {self.device}")
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
        print(f"Assignment Priority: 1) Face Match -> 2) Color Match -> 3) New ID")
        print(f"Color Similarity Threshold (Intersection): {self.color_sim_thresh:.2f}")
        print(f"Color Distinctness Threshold (Max Intersection): {self.color_dist_thresh:.2f}")
        print(f"Bounding Box Aspect Ratio Threshold: {self.aspect_ratio_thresh:.2f}")
        print("--------------------------------------------------------------------------------")

    # --- Helper methods (_load_yolo, _init_deepsort, _get_default_device, etc.) ---
    # (Remain the same as before)
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
        # elif torch.backends.mps.is_available(): return "mps" # Optional: for MacOS Metal
        else: return "cpu"

    def _get_random_bgr_color(self):
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def _get_bbox_from_kpts(self, kpts_xy: np.ndarray, indices: List[int], frame_shape: Tuple[int, int], padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
        H_IMG, W_IMG = frame_shape
        valid_kpts = kpts_xy[indices]; valid_kpts = valid_kpts[np.all(valid_kpts > 0, axis=1)]
        if valid_kpts.shape[0] < 2: return None

        min_x, min_y = np.min(valid_kpts, axis=0); max_x, max_y = np.max(valid_kpts, axis=0)
        x1 = max(0, int(min_x - padding)); y1 = max(0, int(min_y - padding))
        x2 = min(W_IMG, int(max_x + padding)); y2 = min(H_IMG, int(max_y + padding))

        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 5 or (y2 - y1) < 5: return None # Initial validation

        w = float(x2 - x1); h = float(y2 - y1); cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        w_new, h_new = w, h
        if w > h:
            target_h = self.aspect_ratio_thresh * w
            if h < target_h: h_new = target_h
        elif h > w:
            target_w = self.aspect_ratio_thresh * h
            if w < target_w: w_new = target_w

        if w_new != w or h_new != h:
            x1 = int(round(cx - w_new / 2.0)); y1 = int(round(cy - h_new / 2.0))
            x2 = int(round(cx + w_new / 2.0)); y2 = int(round(cy + h_new / 2.0))
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W_IMG, x2); y2 = min(H_IMG, y2)

        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 5 or (y2 - y1) < 5: return None # Final validation
        return (x1, y1, x2, y2)

    def _get_face_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, HEAD_KP_INDICES, frame_shape, padding=15)

    def _get_torso_bbox_from_kpts(self, kpts_xy: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        return self._get_bbox_from_kpts(kpts_xy, TORSO_KP_INDICES, frame_shape, padding=10)

    def _get_color_category_distribution_efficient(self, patch: np.ndarray) -> Optional[Dict[str, float]]:
        # (Same as before)
        if patch is None or patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1: return None
        try:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            category_map = np.full(hsv.shape[:2], CAT_IDX_GRAY, dtype=np.uint8)
            black_mask = v < COLOR_CATEGORIES["BLACK"]["v_max"]; category_map[black_mask] = CAT_IDX_BLACK
            white_mask = (v >= COLOR_CATEGORIES["WHITE"]["v_min"]) & (s < COLOR_CATEGORIES["WHITE"]["s_max"]) & (~black_mask); category_map[white_mask] = CAT_IDX_WHITE
            gray_mask = (s < COLOR_CATEGORIES["GRAY"]["s_max"]) & (v >= COLOR_CATEGORIES["GRAY"]["v_min"]) & (v <= COLOR_CATEGORIES["GRAY"]["v_max"]) & (~black_mask) & (~white_mask); category_map[gray_mask] = CAT_IDX_GRAY
            achromatic_mask = black_mask | white_mask | gray_mask
            red_h_mask = ((h >= COLOR_CATEGORIES["RED"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["RED"]["h_ranges"][0][1])) | ((h >= COLOR_CATEGORIES["RED"]["h_ranges"][1][0]) & (h <= COLOR_CATEGORIES["RED"]["h_ranges"][1][1])); category_map[red_h_mask & ~achromatic_mask] = CAT_IDX_RED
            orange_h_mask = (h >= COLOR_CATEGORIES["ORANGE_BROWN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["ORANGE_BROWN"]["h_ranges"][0][1]); category_map[orange_h_mask & ~achromatic_mask] = CAT_IDX_ORANGE_BROWN
            yellow_h_mask = (h >= COLOR_CATEGORIES["YELLOW"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["YELLOW"]["h_ranges"][0][1]); category_map[yellow_h_mask & ~achromatic_mask] = CAT_IDX_YELLOW
            green_h_mask = (h >= COLOR_CATEGORIES["GREEN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["GREEN"]["h_ranges"][0][1]); category_map[green_h_mask & ~achromatic_mask] = CAT_IDX_GREEN
            cyan_h_mask = (h >= COLOR_CATEGORIES["CYAN"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["CYAN"]["h_ranges"][0][1]); category_map[cyan_h_mask & ~achromatic_mask] = CAT_IDX_CYAN
            blue_h_mask = (h >= COLOR_CATEGORIES["BLUE"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["BLUE"]["h_ranges"][0][1]); category_map[blue_h_mask & ~achromatic_mask] = CAT_IDX_BLUE
            purple_h_mask = (h >= COLOR_CATEGORIES["PURPLE_PINK"]["h_ranges"][0][0]) & (h <= COLOR_CATEGORIES["PURPLE_PINK"]["h_ranges"][0][1]); category_map[purple_h_mask & ~achromatic_mask] = CAT_IDX_PURPLE_PINK
            hist = cv2.calcHist([category_map], [0], None, [NUM_COLOR_CATEGORIES], [0, NUM_COLOR_CATEGORIES])
            total_pixels = category_map.size
            if total_pixels == 0: return None
            hist_normalized = hist / total_pixels
            distribution = {IDX_TO_NAME_MAP[i]: float(hist_normalized[i]) for i in range(NUM_COLOR_CATEGORIES) if hist_normalized[i] > 0}
            return distribution
        except Exception as e:
            print(f"Error calculating efficient color distribution: {e}"); traceback.print_exc(); return None

    def _compare_color_distributions(self, dist1: Optional[Dict[str, float]], dist2: Optional[Dict[str, float]]) -> float:
        # (Same as before)
        if dist1 is None or dist2 is None: return 0.0
        intersection = 0.0
        for category_name in ALL_COLOR_NAMES: intersection += min(dist1.get(category_name, 0.0), dist2.get(category_name, 0.0))
        return intersection

    # --- process_frame (Core tracking logic) ---
    # (Remains the same - LLM interaction happens outside this)
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, Tuple[int, int, int]]]:
        self.frame_count += 1; H_IMG, W_IMG = frame.shape[:2]
        self.last_person_info = {}; self.last_assignments = {}

        # 1. YOLO Detection
        results = self.yolo_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        face_detections_for_deepsort = []; person_info: Dict[int, Dict] = {}
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy(); confs = results[0].boxes.conf.cpu().numpy()
            keypoints_data = results[0].keypoints;
            if keypoints_data is None: return []
            keypoints_xy = keypoints_data.xy.cpu().numpy()
            for i in range(len(boxes)):
                if i >= len(keypoints_xy): continue
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

                        # ----- INSERT: print HSV summary for this person -----
                        hsv_patch = cv2.cvtColor(torso_patch, cv2.COLOR_BGR2HSV)
                        h_mean = hsv_patch[..., 0].mean()
                        s_mean = hsv_patch[..., 1].mean()
                        v_mean = hsv_patch[..., 2].mean()
                        print(f"[Frame {self.frame_count}] Person {i}: "
                            f"H_avg={h_mean:.1f}, S_avg={s_mean:.1f}, V_avg={v_mean:.1f}")
                        # ------------------------------------------------------

                        current_color_dist = self._get_color_category_distribution_efficient(torso_patch)
                    else: torso_patch = None
                person_info[i] = {"person_bbox": person_bbox, "face_bbox": face_bbox, "torso_bbox": torso_bbox, "face_idx_ds": face_idx_ds, "current_color_dist": current_color_dist, "torso_patch": torso_patch, "current_face_track_id": None, "kpts": kpts}
            self.last_person_info = person_info

        # 2. DeepSort Update
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

        # 3. Gather Matching Evidence
        person_candidates = defaultdict(lambda: {"color": [], "face_perm_id": None, "info": None})
        active_permanent_ids = {pid: data for pid, data in self.permanent_tracks.items() if self.frame_count - data['last_seen_frame'] <= self.max_permanent_track_age}
        current_face_mapping = {}
        for perm_id, data in active_permanent_ids.items():
             linked_face_id = data.get("face_track_id")
             if linked_face_id:
                  if linked_face_id in current_face_mapping and current_face_mapping[linked_face_id] != perm_id: pass # Ignore warning print for brevity
                  current_face_mapping[linked_face_id] = perm_id
        self.face_id_to_perm_id_map = current_face_mapping
        if self.enable_color_dist_print and person_info and active_permanent_ids: print(f"--- Frame {self.frame_count} Color Distribution Comparisons ---")
        for person_idx, info in person_info.items():
            person_candidates[person_idx]["info"] = info
            current_distribution = info.get("current_color_dist")
            if current_distribution is not None:
                 if self.enable_color_dist_print:
                      dist_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(current_distribution.items()) if v > 0.01])
                      print(f" Person {person_idx} (FaceID: {info.get('current_face_track_id')}) Dist: [{dist_str}]:")
                 for perm_id, track_data in active_permanent_ids.items():
                     stored_distribution = track_data.get("locked_color_dist")
                     match_similarity = 0.0; reason = "No stored distribution"
                     if stored_distribution is not None:
                         match_similarity = self._compare_color_distributions(current_distribution, stored_distribution)
                         reason = f"Dist Intersection={match_similarity:.3f}"
                         if match_similarity >= self.color_sim_thresh:
                             reason += f" >= Thresh({self.color_sim_thresh:.2f}) -> POTENTIAL MATCH"; person_candidates[person_idx]["color"].append((perm_id, match_similarity))
                         else: reason += f" < Thresh -> NO MATCH"
                     if self.enable_color_dist_print: print(f"  vs PermID {perm_id} (LinkedFace: {track_data.get('face_track_id')}): -> {reason}")
                 person_candidates[person_idx]["color"].sort(key=lambda x: x[1], reverse=True)
            current_face_id = info.get("current_face_track_id")
            if current_face_id and current_face_id in self.face_id_to_perm_id_map:
                person_candidates[person_idx]["face_perm_id"] = self.face_id_to_perm_id_map[current_face_id]

        # 4. Resolve Assignments
        assignments: Dict[int, str] = {}
        assigned_person_indices = set(); assigned_perm_ids = set()
        persons_to_process = list(person_candidates.keys())
        # 4a. Face Link Priority
        if self.enable_debug_draw: print("--- Assignment Step 4a (PRIORITY: Face Link Persistence) ---")
        for person_idx in persons_to_process:
             linked_perm_id = person_candidates[person_idx].get("face_perm_id")
             if linked_perm_id and linked_perm_id in active_permanent_ids:
                 if linked_perm_id not in assigned_perm_ids:
                     assignments[person_idx] = linked_perm_id; assigned_person_indices.add(person_idx); assigned_perm_ids.add(linked_perm_id)
                     if self.enable_debug_draw: print(f"  P{person_idx} assigned {linked_perm_id} via persistent face link (FaceID: {person_candidates[person_idx]['info']['current_face_track_id']}).")
                 else:
                     assigned_person_indices.add(person_idx)
                     if self.enable_debug_draw: print(f"  P{person_idx} face links to {linked_perm_id} (FaceID: {person_candidates[person_idx]['info']['current_face_track_id']}), but {linked_perm_id} already assigned. Person skipped.")
        # 4b. Color Match Fallback
        if self.enable_debug_draw: print("--- Assignment Step 4b (FALLBACK: Color Dist Score) ---")
        color_candidates = []
        for person_idx in persons_to_process:
            if person_idx not in assigned_person_indices:
                potential_color_matches = person_candidates[person_idx]["color"]
                best_available_match = None
                for perm_id, score in potential_color_matches:
                    if perm_id in active_permanent_ids and perm_id not in assigned_perm_ids:
                        best_available_match = {"person_idx": person_idx, "perm_id": perm_id, "score": score}; break
                if best_available_match: color_candidates.append(best_available_match)
        color_candidates.sort(key=lambda x: x["score"], reverse=True)
        if self.enable_debug_draw and color_candidates: print(f"  Color Candidates (Dist Sim): {[{'P':c['person_idx'], 'ID':c['perm_id'], 'S':round(c['score'],3)} for c in color_candidates]}")
        for candidate in color_candidates:
            person_idx = candidate["person_idx"]; perm_id = candidate["perm_id"]
            if person_idx not in assigned_person_indices and perm_id not in assigned_perm_ids:
                assignments[person_idx] = perm_id; assigned_person_indices.add(person_idx); assigned_perm_ids.add(perm_id)
                if self.enable_debug_draw: print(f"  P{person_idx} assigned {perm_id} via color fallback (Dist Sim: {candidate['score']:.3f}).")
        self.last_assignments = assignments
        # 4c. Update Assigned Tracks Data
        for person_idx, perm_id in assignments.items():
              if perm_id in self.permanent_tracks:
                  info = person_candidates[person_idx]["info"]
                  self.permanent_tracks[perm_id]["last_seen_frame"] = self.frame_count
                  self.permanent_tracks[perm_id]["bbox"] = info["person_bbox"]
                  current_face_id = info.get("current_face_track_id")
                  if current_face_id:
                       if self.permanent_tracks[perm_id].get("face_track_id") != current_face_id:
                            old_face_id = self.permanent_tracks[perm_id].get("face_track_id")
                            if old_face_id and old_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[old_face_id] == perm_id:
                                del self.face_id_to_perm_id_map[old_face_id]
                            self.permanent_tracks[perm_id]["face_track_id"] = current_face_id
                            if current_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[current_face_id] != perm_id: pass # Ignore warning print
                            self.face_id_to_perm_id_map[current_face_id] = perm_id
                  else: pass # No face this frame, keep existing link

        # 5. Assign New IDs
        if self.enable_debug_draw: print("--- Assignment Step 5 (New IDs - Color Categories) ---")
        for person_idx in persons_to_process:
             if person_idx not in assigned_person_indices:
                 info = person_candidates[person_idx]["info"]
                 current_face_id = info.get("current_face_track_id")
                 current_distribution = info.get("current_color_dist")
                 can_create_new_id = True; reason = ""
                 if current_face_id and current_face_id in self.face_id_to_perm_id_map:
                     existing_perm_id = self.face_id_to_perm_id_map[current_face_id]
                     if existing_perm_id in active_permanent_ids and existing_perm_id not in assigned_perm_ids:
                         can_create_new_id = False; reason = f"Face ID {current_face_id} already linked to active PermID {existing_perm_id}"
                     elif existing_perm_id in assigned_perm_ids:
                         can_create_new_id = False; reason = f"Face ID {current_face_id} linked to PermID {existing_perm_id}, but it was assigned elsewhere"
                 if can_create_new_id:
                     if current_distribution is None or not current_distribution:
                         can_create_new_id = False; reason = "No valid color distribution"
                 if can_create_new_id:
                      is_distinct_dist = True; max_intersection = 0.0; most_similar_perm_id = None
                      perm_ids_to_compare = {pid: data for pid, data in active_permanent_ids.items() if pid not in assigned_perm_ids}
                      for perm_id, track_data in perm_ids_to_compare.items():
                           locked_dist = track_data.get("locked_color_dist")
                           if locked_dist is not None:
                               intersection = self._compare_color_distributions(current_distribution, locked_dist)
                               if intersection > max_intersection: max_intersection = intersection; most_similar_perm_id = perm_id
                               if intersection > self.color_dist_thresh: is_distinct_dist = False; break
                      if not is_distinct_dist:
                           can_create_new_id = False; reason = f"Color dist too similar to existing UNASSIGNED PermID {most_similar_perm_id} (max int: {max_intersection:.3f} > thresh: {self.color_dist_thresh:.3f})"
                 if can_create_new_id:
                     new_perm_id = self.next_permanent_id(); new_vis_color = self._get_random_bgr_color()
                     self.permanent_tracks[new_perm_id] = {"face_track_id": current_face_id, "locked_color_dist": current_distribution, "vis_color": new_vis_color, "last_seen_frame": self.frame_count, "bbox": info["person_bbox"]}
                     if current_face_id:
                         if current_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[current_face_id] != new_perm_id: pass # Ignore warning print
                         self.face_id_to_perm_id_map[current_face_id] = new_perm_id
                     assignments[person_idx] = new_perm_id; assigned_person_indices.add(person_idx); assigned_perm_ids.add(new_perm_id)
                     self.last_assignments = assignments
                     dom_cat = max(current_distribution, key=current_distribution.get) if current_distribution else "N/A"
                     print(f"Assigned NEW PermID: {new_perm_id} to P{person_idx} (Face: {current_face_id}, Dom Col: {dom_cat})")
                 elif reason and self.enable_debug_draw:
                     print(f"  Did not assign new ID to P{person_idx}: {reason}")

        # 6. Prune Old Tracks
        ids_to_remove = [pid for pid, data in self.permanent_tracks.items() if self.frame_count - data['last_seen_frame'] > self.max_permanent_track_age]
        if ids_to_remove:
            if self.enable_debug_draw: print(f"--- Pruning {len(ids_to_remove)} old PermIDs: {ids_to_remove} ---")
            for pid in ids_to_remove:
                 removed_data = self.permanent_tracks.pop(pid, None)
                 if removed_data:
                     removed_face_id = removed_data.get("face_track_id")
                     if removed_face_id and removed_face_id in self.face_id_to_perm_id_map and self.face_id_to_perm_id_map[removed_face_id] == pid:
                          del self.face_id_to_perm_id_map[removed_face_id]
                          if self.enable_debug_draw: print(f"   Removed face link for {removed_face_id} from pruned PermID {pid}")

        # 7. Prepare Output
        output_tracks = []
        for person_idx, permanent_id in assignments.items():
              if permanent_id in self.permanent_tracks and person_idx in person_info:
                  x1, y1, x2, y2 = person_info[person_idx]["person_bbox"]
                  vis_color = self.permanent_tracks[permanent_id].get("vis_color", (0, 0, 255))
                  output_tracks.append((x1, y1, x2, y2, permanent_id, vis_color))
        return output_tracks

    # --- draw_tracks (Applies filter based on target_color_name) ---
    # (Same as before - filtering logic is independent of LLM backend)
    def draw_tracks(self, frame: np.ndarray, tracks: List[Tuple[int, int, int, int, str, Tuple[int,int,int]]], target_color_name: Optional[str] = None) -> np.ndarray:
        DRAW_DEBUG_BOXES = self.enable_debug_draw
        person_info = self.last_person_info
        assignments = self.last_assignments
        filtering_active = target_color_name is not None
        if filtering_active:
            filter_color_bgr = (150, 255, 150); cv2.putText(frame, f"FILTERING: {target_color_name} Jacket", (frame.shape[1] - 500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, filter_color_bgr, 2)

        drawn_count = 0
        for (x1, y1, x2, y2, permanent_id, vis_color) in tracks:
            perm_track_data = self.permanent_tracks.get(permanent_id, {})
            locked_dist = perm_track_data.get("locked_color_dist")
            dom_cat = max(locked_dist, key=locked_dist.get, default="N/A") if locked_dist else "N/A"

            if filtering_active and dom_cat != target_color_name: continue
            drawn_count += 1

            p_idx = -1; current_face_id_this_frame = "N/A"
            persistent_face_link = perm_track_data.get("face_track_id", "N/A")
            for idx, assigned_id in assignments.items():
                 if assigned_id == permanent_id and idx in person_info:
                    p_idx = idx; current_face_id_this_frame = person_info[idx].get("current_face_track_id", "N/A"); break

            label = f"ID:{permanent_id}"
            if persistent_face_link != "N/A": label += f" F:{persistent_face_link}"
            if dom_cat != "N/A": label += f" C:{dom_cat}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), vis_color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, h + 15)
            cv2.rectangle(frame, (x1, label_y - h - 5), (x1 + w + 5 , label_y + 5), vis_color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if DRAW_DEBUG_BOXES and p_idx != -1 and p_idx in person_info:
                 info = person_info[p_idx]
                 if info.get("face_bbox"):
                     fx1, fy1, fx2, fy2 = info["face_bbox"]; cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 1)
                     if current_face_id_this_frame != "N/A": cv2.putText(frame, f"CurF:{current_face_id_this_frame}", (fx1, fy1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 1)
                 if info.get("torso_bbox"):
                     tx1, ty1, tx2, ty2 = info["torso_bbox"]; cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 1)
                     current_dist = info.get("current_color_dist")
                     curr_dom_cat = max(current_dist, key=current_dist.get, default="?") if current_dist else "?"
                     torso_label = f"Cur:{curr_dom_cat}"
                     if dom_cat != "N/A": torso_label += f" Lock:{dom_cat}"
                     cv2.putText(frame, torso_label, (tx1, ty1 + 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)

        drawn_count_text = f"Tracks Drawn: {drawn_count}" + (f" / {len(tracks)}" if filtering_active else "")
        cv2.putText(frame, drawn_count_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

# --- Ollama Helper Function ---
def get_color_from_prompt_ollama(
    user_prompt: str,
    color_categories_list: List[str],
    ollama_model_name: str,
    # ollama_host: Optional[str] = None, # Optional: if not default localhost
) -> Optional[str]:
    """
    Uses a running Ollama instance to determine the best matching color category.
    """
    if not OLLAMA_AVAILABLE:
        print("Ollama functionality disabled because 'ollama' library is missing.")
        return None

    # Prepare prompt for Ollama chat
    system_prompt = (
        "You are a helpful assistant. Given the user's prompt describing an object or person, "
        "your task is to identify the single best matching color from the provided list of color categories. "
        "Consider common color associations (e.g., 'khaki' often fits 'ORANGE_BROWN', 'navy' fits 'BLUE'). "
        "Output *only* the EXACT name of the chosen color category from the list, with no extra text, explanation, or punctuation."
    )
    color_list_str = ", ".join(color_categories_list)
    full_user_prompt = f"Available color categories: [{color_list_str}]\n\nUser prompt: '{user_prompt}'\n\nChosen color category:"

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': full_user_prompt},
    ]

    try:
        print(f"Querying Ollama model '{ollama_model_name}'...")
        # client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client() # If specifying host
        response = ollama.chat(
            model=ollama_model_name,
            messages=messages,
            options={'temperature': 0.1} # Lower temperature for more deterministic output
        )
        extracted_color = response['message']['content'].strip()
        print(f"Ollama Raw Output: '{extracted_color}'")

        # --- Validate against known color categories ---
        cleaned_color = extracted_color.upper().replace(" ", "_").strip(" .,:;'\"[]") # Clean up typical LLM additions, including potential brackets
        if cleaned_color in color_categories_list:
            print(f"Ollama Validated Match: {cleaned_color}")
            return cleaned_color
        else:
            # Try partial matching as a fallback
            for valid_color in color_categories_list:
                # Be a bit more careful with partial matching
                # Check if the cleaned output *contains* the valid color AND is relatively short
                # to avoid matching parts of longer unrelated words.
                if valid_color in cleaned_color and len(cleaned_color) < len(valid_color) + 10:
                     print(f"Ollama Partial Match Found: {valid_color} in '{cleaned_color}'")
                     return valid_color
            print(f"Warning: Ollama output '{extracted_color}' (cleaned: '{cleaned_color}') does not match any category in {color_categories_list}. No color filter applied.")
            return None

    except Exception as e:
        print(f"\n--- ERROR Communicating with Ollama ---")
        print(f"Model Attempted: {ollama_model_name}")
        print(f"Error: {e}")
        print("Troubleshooting:")
        print(" 1. Is the Ollama service running in the background?")
        print(f" 2. Have you pulled the model? (e.g., run 'ollama pull {ollama_model_name}' in the terminal)")
        print(" 3. Is the OLLAMA_MODEL_NAME in the script correct?")
        # traceback.print_exc() # Can be noisy, enable if needed
        print("---------------------------------------\n")
        return None

# --- Main Execution ---
def main():
    # --- Configuration ---
    VIDEO_SOURCE = 0 # Use 0 for webcam, or provide video file path
    YOLO_MODEL = "yolov8n-pose.pt" # Nano model
    CONFIDENCE_THRESHOLD = 0.50
    # Face DeepSort Params
    FACE_MAX_IOU_DIST=0.6; FACE_MAX_AGE=250; FACE_N_INIT=20; FACE_MAX_COS_DIST=0.2; FACE_NN_BUDGET=1000
    # Color Category Params
    COLOR_SIM_THRESH = 0.5
    COLOR_DIST_THRESH = 0.7
    # General Tracking
    MAX_PERMANENT_AGE = 600
    ASPECT_RATIO_THRESH = 0.8
    # Debugging
    ENABLE_COLOR_DIST_PRINT = False
    ENABLE_DEBUG_DRAW = True

    # --- Ollama Configuration ---
    ENABLE_OLLAMA_FILTER = True # Enable only if library is installed
    # --- IMPORTANT: Set this to a model you have pulled with Ollama ---
    OLLAMA_MODEL_NAME = "gemma3:4b"  # Examples: "llama3:8b", "mistral:latest", "phi3:mini"
    # OLLAMA_HOST = "http://192.168.1.100:11434" # Optional: If Ollama runs on a different machine
    DEFAULT_USER_PROMPT = "look for a person in a brown hoodie"
    # -------------------------

    # --- Initialize Tracker ---
    tracker = YoloPoseTrackerColorCategoriesEfficient(
        yolo_model_name=YOLO_MODEL, conf_thresh=CONFIDENCE_THRESHOLD,
        face_max_iou_distance=FACE_MAX_IOU_DIST, face_max_age=FACE_MAX_AGE, face_n_init=FACE_N_INIT,
        face_max_cosine_distance=FACE_MAX_COS_DIST, face_nn_budget=FACE_NN_BUDGET,
        color_similarity_threshold=COLOR_SIM_THRESH,
        color_distinctness_threshold=COLOR_DIST_THRESH,
        max_permanent_track_age=MAX_PERMANENT_AGE,
        aspect_ratio_threshold=ASPECT_RATIO_THRESH,
        enable_color_dist_print=ENABLE_COLOR_DIST_PRINT,
        enable_debug_draw=ENABLE_DEBUG_DRAW,
        # No LLM device needed here anymore
    )

    # --- Ollama Loading & Prompting ---
    target_color_name: Optional[str] = None
    ollama_ready = False

    if ENABLE_OLLAMA_FILTER:
        print("\n--- Ollama Color Filter Setup ---")
        # Check Ollama service status and model availability
        try:
            print("Checking Ollama service connection...")
            ollama.list() # Simple command to check connection
            print(f"Ollama service connected. Ensure model '{OLLAMA_MODEL_NAME}' is pulled.")
            ollama_ready = True # Assume model exists for now, error handled later
        except Exception as e:
            print(f"\n--- ERROR Connecting to Ollama Service ---")
            print(f"Error: {e}")
            print("Please ensure the Ollama service is running before starting the script.")
            print("Ollama color filtering will be disabled.")
            print("---------------------------------------\n")
            ENABLE_OLLAMA_FILTER = False

        if ollama_ready:
            user_prompt = input(f"Enter color description (or press Enter for default: '{DEFAULT_USER_PROMPT}'): ")
            if not user_prompt:
                user_prompt = DEFAULT_USER_PROMPT
            print(f"Using prompt: '{user_prompt}'")

            target_color_name = get_color_from_prompt_ollama(
                user_prompt, ALL_COLOR_NAMES, OLLAMA_MODEL_NAME #, OLLAMA_HOST # Pass host if configured
            )
            if target_color_name:
                print(f"Ollama selected target color: {target_color_name}")
            else:
                print("Ollama could not determine a valid target color. Filtering disabled.")
                # Optionally disable filter if LLM fails first time
                # ENABLE_OLLAMA_FILTER = False
    else:
        print("\nOllama Color Filtering is disabled ('ollama' library not found or connection failed).")

    # --- Video Input Setup ---
    print("\n--- Starting Video Processing ---")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"FATAL ERROR: Cannot open video source: {VIDEO_SOURCE}")
        sys.exit(1) # Exit if video source fails

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS); fps = int(fps_cap) if fps_cap and fps_cap > 0 else 30
    print(f"Input source opened: {frame_width}x{frame_height} @ {fps} FPS (reported)")

    # --- Processing Loop ---
    frame_num = 0
    start_overall_time = time.time()
    while True:
        ret, frame = cap.read(); loop_start_time = time.time()
        if not ret:
            print("\nEnd of video source reached.")
            break
        frame_num += 1

        try:
            frame_copy = frame.copy()
            tracked_persons = tracker.process_frame(frame_copy)
            proc_time = time.time() - loop_start_time
            processing_fps = 1.0 / proc_time if proc_time > 0 else float('inf')

            # Draw tracks, applying Ollama filter if active and successful
            annotated_frame = tracker.draw_tracks(
                frame,
                tracked_persons,
                target_color_name if ENABLE_OLLAMA_FILTER and target_color_name else None
            )

            # Display Info
            cv2.putText(annotated_frame, f"FPS: {processing_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Perm IDs: {len(tracker.permanent_tracks)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO Pose + Permanent Tracker (Ollama Color Filter)", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, exiting.")
                break

        except Exception as e:
            print(f"\n--- FATAL ERROR during processing frame {frame_num} ---")
            traceback.print_exc()
            print("--- Attempting to continue ---")
            # break # Uncomment to stop on error

    # --- Cleanup ---
    end_overall_time = time.time()
    total_time = end_overall_time - start_overall_time
    avg_fps = frame_num / total_time if total_time > 0 else 0
    print("\n--- Processing Finished ---")
    print(f"Processed {frame_num} frames in {total_time:.2f} seconds.")
    print(f"Average FPS: {avg_fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()
    # Explicitly clear GPU memory used by YOLO/DeepSort if needed
    if tracker.device == 'cuda':
        del tracker.yolo_model
        if hasattr(tracker, 'face_tracker') and hasattr(tracker.face_tracker, 'embedder'):
             del tracker.face_tracker.embedder
        # No transformers model to delete
        torch.cuda.empty_cache()
        print("CUDA cache cleared (YOLO/DeepSort).")
    print("Resources released.")


if __name__ == "__main__":
    main()