import cv2
import torch
import numpy as np
from ultralytics import YOLO
import random

class PersonPoseHueProcessor:
    def __init__(
        self,
        seg_model_name: str = "yolov8s-seg.pt",
        pose_model_name: str = "yolov8s-pose.pt",
        conf_thresh: float = 0.6,
        uniform_thresh: float = 0.4,
        K: int = 3,
        hue_delta: int = 4,
        mask_padding: int = 10,
        iou_thresh: float = 0.3,
        device: str = None
    ):
        # Device selection
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # Load segmentation and pose models
        self.seg_model = YOLO(seg_model_name).to(self.device)
        self.pose_model = YOLO(pose_model_name).to(self.device)

        self.conf_thresh = conf_thresh
        self.uniform_thresh = uniform_thresh
        self.K = K
        self.hue_delta = hue_delta
        self.mask_padding = mask_padding
        self.iou_thresh = iou_thresh

        # Tracking state: track_id -> last bbox (x1,y1,x2,y2)
        self.tracks = {}
        self.track_colors = {}
        self.next_id = 1

        # COCO skeleton pairs for drawing
        self.skel_pairs = [
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (11,12),(5,11),(6,12),
            (11,13),(13,15),(12,14),(14,16)
        ]

    def _get_random_color(self):
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0

    def _find_dominant_hue(self, roi, mask=None):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        vals = None
        if mask is not None and mask.shape == roi.shape[:2] and mask.any():
            vals = hsv[...,0][mask>0].reshape(-1,1).astype(np.float32)
        else:
            vals = hsv[...,0].reshape(-1,1).astype(np.float32)
        if vals is None or vals.size == 0:
            return -1, 0.0
        k = min(self.K, vals.shape[0])
        _, labels, centers = cv2.kmeans(
            vals, k, None,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),
            3, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k)
        idx = np.argmax(counts)
        return int(centers[idx][0]), counts[idx]/vals.shape[0]

    def process_frame(self, frame):
        H, W = frame.shape[:2]
        overlay = frame.copy()
        alpha = 0.7

        # 1) YOLO segmentation
        res = self.seg_model.predict(
            frame, conf=self.conf_thresh, classes=[0], verbose=False
        )
        if not res or not hasattr(res[0], 'masks') or res[0].masks is None:
            return frame
        r = res[0]

        # 2) collect current detections
        detections = []
        for poly in r.masks.xy:
            cnt = np.array(poly, dtype=np.int32)
            if cnt.shape[0] < 3:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            x1 = max(x-self.mask_padding,0)
            y1 = max(y-self.mask_padding,0)
            x2 = min(x+w+self.mask_padding,W)
            y2 = min(y+h+self.mask_padding,H)
            detections.append({'contour': cnt, 'bbox': (x1,y1,x2,y2)})

        # 3) match detections to existing tracks via IoU
        new_tracks = {}
        det_to_id = {}
        used = set()
        for det in detections:
            best_id, best_iou = None, self.iou_thresh
            for tid, old_bbox in self.tracks.items():
                if tid in used:
                    continue
                iou = self._compute_iou(old_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.track_colors[best_id] = self._get_random_color()
            new_tracks[best_id] = det['bbox']
            det_to_id[best_id] = det
            used.add(best_id)
        self.tracks = new_tracks

        # 4) process each tracked person
        for tid, det in det_to_id.items():
            contour = det['contour']
            x1,y1,x2,y2 = det['bbox']
            color = self.track_colors[tid]

            # draw box and ID
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness=3)
            cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # pose estimation with safety checks
            roi = frame[y1:y2, x1:x2]
            keypoints = None
            try:
                pose_res = self.pose_model.predict(
                    roi, conf=self.conf_thresh, verbose=False
                )
                if pose_res and hasattr(pose_res[0], 'keypoints') and pose_res[0].keypoints is not None:
                    xy = pose_res[0].keypoints.xy
                    confs = pose_res[0].keypoints.conf
                    if xy is not None and confs is not None and len(xy) > 0 and len(confs) > 0:
                        kpts_arr = xy[0].cpu().numpy()
                        cfs_arr = confs[0].cpu().numpy()
                        # map back to full frame coords
                        kpts_arr[:,0] += x1
                        kpts_arr[:,1] += y1
                        keypoints = [(int(x),int(y),float(c)) for (x,y),c in zip(kpts_arr, cfs_arr)]
                        # draw skeleton
                        for a,b in self.skel_pairs:
                            if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
                                cv2.line(frame,
                                         keypoints[a][:2],
                                         keypoints[b][:2],
                                         color, 2)
                        # draw keypoints
                        for xk,yk,cf in keypoints:
                            if cf > 0.3:
                                cv2.circle(frame, (xk,yk), 3, color, -1)
            except Exception:
                keypoints = None

            # segmentation mask
            mask = np.zeros((H,W), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            # torso mask via circle or fallback
            torso_mask = None
            if keypoints:
                pts = np.array([ keypoints[5][:2], keypoints[6][:2],
                                 keypoints[11][:2], keypoints[12][:2] ])
                center = tuple(pts.mean(axis=0).astype(int))
                sh_dist = np.linalg.norm(
                    np.array(keypoints[5][:2]) - np.array(keypoints[6][:2])
                )
                radius = int(max(sh_dist/2, 0.2*(y2-y1)))
                torso_mask = np.zeros((H,W), dtype=np.uint8)
                cv2.circle(torso_mask, center, radius, 255, thickness=-1)

            # select mask for hue analysis
            if torso_mask is not None and torso_mask.any():
                use_mask = torso_mask
                roi_box = frame
            else:
                use_mask = mask[y1:y2, x1:x2]
                roi_box = frame[y1:y2, x1:x2]

            # hue analysis
            hue, ratio = self._find_dominant_hue(roi_box, mask=use_mask)
            if hue != -1 and ratio >= self.uniform_thresh:
                cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)

        # blend shading
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {source}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis = self.process_frame(frame)
            cv2.imshow("PersonPoseHue", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = PersonPoseHueProcessor()
    p.run()
