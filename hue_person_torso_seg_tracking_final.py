import cv2
import torch
import numpy as np
from ultralytics import YOLO
import random
from typing import Dict, Tuple

class PersonPoseHueProcessor:
    def __init__(
        self,
        seg_model_name: str = "yolov8s-seg.pt",
        pose_model_name: str = "yolov8s-pose.pt",
        conf_thresh: float = 0.6,
        uniform_thresh: float = 0.4,
        K: int = 3,
        hue_delta: int = 5,
        mask_padding: int = 10,
        iou_thresh: float = 0.3,
        device: str = None
    ):
        # Device selection
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # Load and prepare models
        self.seg_model = YOLO(seg_model_name).to(self.device)
        self.pose_model = YOLO(pose_model_name).to(self.device)
        # Eval mode & mixed precision on CUDA
        self.seg_model.model.eval()
        self.pose_model.model.eval()
        if self.device.startswith('cuda'):
            self.seg_model.model.half()
            self.pose_model.model.half()

        # Parameters
        self.conf_thresh = conf_thresh
        self.uniform_thresh = uniform_thresh
        self.K = K
        self.hue_delta = hue_delta
        self.mask_padding = mask_padding
        self.iou_thresh = iou_thresh

        # Tracking state
        self.tracks: Dict[int, Tuple[int,int,int,int]] = {}
        self.track_colors: Dict[int, Tuple[int,int,int]] = {}
        self.next_id = 1

        # COCO skeleton pairs
        self.skel_pairs = np.array([
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (11,12),(5,11),(6,12),
            (11,13),(13,15),(12,14),(14,16)
        ])

    def _get_color(self) -> Tuple[int,int,int]:
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def _iou(self, A: Tuple[int,int,int,int], B: Tuple[int,int,int,int]) -> float:
        xA = max(A[0], B[0]); yA = max(A[1], B[1])
        xB = min(A[2], B[2]); yB = min(A[3], B[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        areaA = (A[2]-A[0]) * (A[3]-A[1])
        areaB = (B[2]-B[0]) * (B[3]-B[1])
        denom = areaA + areaB - inter
        return inter/denom if denom>0 else 0.0

    def _dominant_hue(self, roi: np.ndarray, mask: np.ndarray=None) -> Tuple[int,float]:
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
    def process(self, frame: np.ndarray) -> Dict[int, Tuple[int,int,int,int]]:
        H, W = frame.shape[:2]
        # 1) segmentation
        res = self.seg_model.predict(frame, conf=self.conf_thresh, classes=[0], verbose=False)
        if not res or res[0].masks is None:
            return {}
        masks = res[0].masks.xy

        # 2) build detections
        dets = []
        for poly in masks:
            cnt = np.array(poly, np.int32)
            if cnt.shape[0]<3: continue
            x,y,w,h = cv2.boundingRect(cnt)
            x1 = max(x-self.mask_padding,0); y1 = max(y-self.mask_padding,0)
            x2 = min(x+w+self.mask_padding, W); y2 = min(y+h+self.mask_padding, H)
            dets.append({'cnt':cnt, 'bbox':(x1,y1,x2,y2)})

        # 3) track by IoU
        new_tracks = {}; det_map = {}; used = set()
        for det in dets:
            best, biou = None, self.iou_thresh
            for tid, old in self.tracks.items():
                if tid in used: continue
                iou = self._iou(old, det['bbox'])
                if iou>biou: best, biou = tid, iou
            if best is None:
                best = self.next_id; self.next_id+=1
                self.track_colors[best]=self._get_color()
            new_tracks[best]=det['bbox']; det_map[best]=det; used.add(best)
        self.tracks = new_tracks

        # 4) filter by hue uniformity
        matches = {}
        for tid, det in det_map.items():
            x1,y1,x2,y2 = det['bbox']
            # build mask
            mask = np.zeros((H,W), np.uint8)
            cv2.drawContours(mask, [det['cnt']], -1, 255, -1)
            roi = frame[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            hue, ratio = self._dominant_hue(roi, mask=crop_mask)
            if hue>=0 and ratio>=self.uniform_thresh:
                matches[tid] = det['bbox']
        print(matches)
        return matches


def main():
    p = PersonPoseHueProcessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        if not ret: break
        matches = p.process(frame)
        # visualize
        for tid, (x1,y1,x2,y2) in matches.items():
            color = p.track_colors.get(tid, (0,255,0))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Matches", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
