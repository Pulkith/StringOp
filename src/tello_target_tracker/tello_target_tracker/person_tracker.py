import cv2
import torch
import numpy as np
from ultralytics import YOLO
import random
from typing import Dict, List, Tuple

class PersonSegmentationTracker:
    def __init__(
        self,
        seg_model_name: str = "yolov8s-seg.pt",
        conf_thresh: float = 0.8,
        uniform_thresh: float = 0.4,
        K: int = 3,
        mask_padding: int = 10,
        iou_thresh: float = 0.3,
        device: str = None
    ):
        # Device selection logic
        self.device = device or (
            "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # Load and prepare segmentation model
        self.seg_model = YOLO(seg_model_name).to(self.device)
        # Eval mode for inference
        self.seg_model.model.eval()
        # Use half-precision on CUDA for better performance
        if self.device == "cuda":
            self.seg_model.model.half()

        # Parameters
        self.conf_thresh = conf_thresh
        self.uniform_thresh = uniform_thresh
        self.K = K
        self.mask_padding = mask_padding
        self.iou_thresh = iou_thresh

        # Tracking state
        self.tracks: Dict[int, Tuple[int,int,int,int]] = {}
        self.track_colors: Dict[int, Tuple[int,int,int]] = {}
        self.next_id = 1

    def _get_color(self) -> Tuple[int,int,int]:
        """Generate a random color for visualization"""
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def _iou(self, A: Tuple[int,int,int,int], B: Tuple[int,int,int,int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        xA = max(A[0], B[0]); yA = max(A[1], B[1])
        xB = min(A[2], B[2]); yB = min(A[3], B[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        areaA = (A[2]-A[0]) * (A[3]-A[1])
        areaB = (B[2]-B[0]) * (B[3]-B[1])
        denom = areaA + areaB - inter
        return inter/denom if denom>0 else 0.0

    def _dominant_hue(self, roi: np.ndarray, mask: np.ndarray=None) -> Tuple[int,float]:
        """Find the dominant hue in the region of interest"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = hsv[...,0]
        if mask is not None and mask.shape==h.shape and mask.any():
            vals = h[mask>0].reshape(-1,1).astype(np.float32)
        else:
            vals = h.flatten().reshape(-1,1).astype(np.float32)
        if vals.size==0:
            return -1, 0.0
        k = min(self.K, vals.shape[0])
        __, labels, centers = cv2.kmeans(
            vals, k, None,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),
            3, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten(), minlength=k)
        idx = np.argmax(counts)
        return int(centers[idx][0]), counts[idx]/vals.shape[0]

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> List[dict]:
        """
        Process a frame to detect and track people using segmentation.
        720*960 image size is expected.
        Returns a list of dicts: {"track_id": int, "bbox": (l, t, r, b)}
        """
        H, W = frame.shape[:2]

        # Resize frame for segmentation model
        frame = frame[:,:,::-1]

        # Run segmentation
        res = self.seg_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        
        # Process segmentation results
        matches = {}
        if res and res[0].masks is not None:
            masks = res[0].masks.xy
            
            # Build detections from segmentation masks
            dets = []
            for poly in masks:
                cnt = np.array(poly, np.int32)
                if cnt.shape[0] < 3: 
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Add padding to the bounding box
                x1 = max(x-self.mask_padding, 0)
                y1 = max(y-self.mask_padding, 0)
                x2 = min(x+w+self.mask_padding, W)
                y2 = min(y+h+self.mask_padding, H)
                dets.append({'cnt': cnt, 'bbox': (x1, y1, x2, y2)})
            
            # Track by IoU
            new_tracks = {}
            det_map = {}
            used = set()
            for det in dets:
                best, biou = None, self.iou_thresh
                for tid, old in self.tracks.items():
                    if tid in used: 
                        continue
                    iou = self._iou(old, det['bbox'])
                    if iou > biou: 
                        best, biou = tid, iou
                if best is None:
                    best = self.next_id
                    self.next_id += 1
                    self.track_colors[best] = self._get_color()
                new_tracks[best] = det['bbox']
                det_map[best] = det
                used.add(best)
            self.tracks = new_tracks
            
            # Filter by hue uniformity
            for tid, det in det_map.items():
                x1, y1, x2, y2 = det['bbox']
                # Build mask
                mask = np.zeros((H, W), np.uint8)
                cv2.drawContours(mask, [det['cnt']], -1, 255, -1)
                roi = frame[y1:y2, x1:x2]
                crop_mask = mask[y1:y2, x1:x2]
                hue, ratio = self._dominant_hue(roi, mask=crop_mask)
                if hue >= 0 and ratio >= self.uniform_thresh:
                    matches[tid] = det['bbox']
        
        # Convert matches to the format expected by the state machine
        output = []
        for tid, (l, t, r, b) in matches.items():
            output.append({"track_id": tid, "bbox": (l, t, r, b)})
        
        return output
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[dict],
        box_color: Tuple[int, int, int] = None,
        text_color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """Draw tracking results on the frame"""
        result = frame.copy()
        for tr in tracks:
            tid = tr["track_id"]
            l, t, r, b = tr["bbox"]
            
            # Use track-specific color if no color is provided
            color = box_color or self.track_colors.get(tid, (0, 255, 0))
            txt_color = text_color or color
            
            # Draw bounding box
            cv2.rectangle(result, (l, t), (r, b), color, 2)
            # Draw ID
            cv2.putText(
                result,
                f"ID {tid}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                txt_color,
                2
            )
        return result