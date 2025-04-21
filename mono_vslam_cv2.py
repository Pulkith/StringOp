#!/usr/bin/env python3

import argparse
import time
import os
from pathlib import Path
import sys

import cv2
import numpy as np

# --- Constants ---
MIN_FEATURES = 100  # Minimum number of features to trigger re-detection

################################################################################
# Utilities
################################################################################

def load_intrinsics(args, frame_shape):
    """
    Load camera intrinsics (K matrix) and distortion coefficients (dist).
    Priority: --calib file > CLI args (--fx, etc.) > Fallback estimate.
    """
    h, w = frame_shape[:2]
    K, dist = None, None

    if args.calib:
        calib_file = Path(args.calib)
        if calib_file.is_file() and calib_file.suffix == '.npz':
            try:
                data = np.load(calib_file)
                K = data['K']
                dist = data['dist']
                print(f"[INFO] Loaded intrinsics and distortion from {calib_file}")
                if K.shape != (3, 3) or dist.ndim != 1:
                     raise ValueError("Invalid K or dist shape in calibration file.")
                return K.astype(np.float64), dist.astype(np.float64)
            except Exception as e:
                print(f"[ERROR] Failed to load calibration file {calib_file}: {e}")
                sys.exit(1)
        else:
            print(f"[WARN] --calib provided, but '{args.calib}' is not a valid .npz file. Trying other methods.")

    if args.fx is not None and args.fy is not None:
        fx, fy = args.fx, args.fy
        cx = args.cx if args.cx is not None else w / 2.0
        cy = args.cy if args.cy is not None else h / 2.0
        # Assume no distortion if loaded from CLI args
        dist = np.zeros(5, dtype=np.float64)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        print("[INFO] Using intrinsics from CLI arguments (assuming zero distortion).")
        return K, dist

    # --- Fallback ---
    # This is highly inaccurate and should be avoided. Calibrate your camera!
    # Estimate focal length assuming a diagonal FOV (e.g., 70 degrees for many webcams)
    # This is just a rough guess!
    diagonal_fov_deg = 70.0
    diagonal_res = np.sqrt(w**2 + h**2)
    f_estimated = (diagonal_res / 2.0) / np.tan(np.deg2rad(diagonal_fov_deg / 2.0))
    K = np.array([[f_estimated, 0, w / 2.0],
                  [0, f_estimated, h / 2.0],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64) # Assume no distortion
    print("[WARN] Using fallback intrinsics estimate based on assumed FOV.")
    print("[WARN] This is likely INACCURATE. Calibrate your camera and use --calib option!")
    print("[WARN] Press 'c' during runtime to save calibration shots.")
    return K, dist


def save_calib_image(frame, folder="calib_shots"):
    """Save frame so the user can run offline calibrateCamera later."""
    calib_dir = Path(folder)
    calib_dir.mkdir(exist_ok=True)
    fname = calib_dir / f"shot_{int(time.time()*1e3)}.png"
    # Save the *original* (distorted) frame for calibration
    # Note: If called during the run loop, 'frame' might already be undistorted.
    # For true calibration, it's better to capture raw frames separately.
    # However, for convenience, we save the currently displayed frame.
    cv2.imwrite(str(fname), frame)
    print(f"[INFO] Calibration shot saved to {fname}")


################################################################################
# Main monocular VO class
################################################################################
class MonoVO:
    def __init__(self, cap, K, dist, fps, baseline):
        self.cap = cap
        self.K = K
        self.dist = dist
        self.invK = np.linalg.inv(K)
        self.baseline = baseline # Target physical distance for scale init [m]
        self.fps_target = fps
        self.delay_ms = int(1000 / fps) if fps > 0 else 1 # Avoid division by zero, 1ms delay minimum

        # Tracking params
        self.feat_params = dict(maxCorners=500,      # Increased corners
                                qualityLevel=0.01,
                                minDistance=8,       # Slightly increased distance
                                blockSize=8)         # Slightly increased block size
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,            # Pyramid levels
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                        30, 0.01))   # Termination criteria

        # Pose state
        self.R = np.eye(3, dtype=np.float64) # Current Rotation (World -> Camera)
        self.t = np.zeros((3, 1), dtype=np.float64) # Current Translation (World -> Camera)
        self.scale = 1.0            # Default scale if initialization skipped/fails
        self.scale_initialized = False
        self.traj_len = 0.0         # Cumulative distance travelled [m]

        # Frame state
        self.prev_frame_undistorted = None
        self.prev_gray = None
        self.prev_pts = None

        # Bootstrap first frame
        self._bootstrap_first_frame()

    def _bootstrap_first_frame(self):
        """Read the first frame, undistort, find features."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read first frame from camera")

        # Undistort the first frame
        self.prev_frame_undistorted = cv2.undistort(frame, self.K, self.dist, None, self.K)
        self.prev_gray = cv2.cvtColor(self.prev_frame_undistorted, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feat_params)

        if self.prev_pts is None or len(self.prev_pts) < MIN_FEATURES:
             raise RuntimeError("Unable to find sufficient features in the first frame.")
        print(f"[INFO] Initialized with {len(self.prev_pts)} features.")

    # --------------------------------------------------------------------- #
    def _estimate_relative_pose(self, p1, p2):
        """
        Estimate relative rotation (R) and translation (t) between two frames
        using the Essential Matrix decomposition. Points p1, p2 must be undistorted.
        """
        # Ensure points are in the correct shape (N, 2)
        p1_norm = p1.reshape(-1, 2).astype(np.float32)
        p2_norm = p2.reshape(-1, 2).astype(np.float32)

        # Find Essential Matrix using RANSAC
        # We use undistorted points, so K is needed.
        # Threshold: max distance from epipolar line (pixels). Lower values are stricter.
        E, mask = cv2.findEssentialMat(p2_norm, p1_norm, self.K,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0) # Adjust threshold if needed (0.5-1.5 typical)

        if E is None or mask is None:
            print("[WARN] Essential Matrix estimation failed.")
            return None, None, None

        # Count inliers
        inlier_count = np.sum(mask)
        if inlier_count < 8: # Need at least 8 points for recoverPose
            print(f"[WARN] Not enough inliers for pose recovery ({inlier_count} found).")
            return None, None, None

        # Recover relative pose (R, t) from Essential Matrix
        # Note: t is up to scale at this point
        # Points must be the *inliers* identified by findEssentialMat mask
        _, R, t, _ = cv2.recoverPose(E, p2_norm[mask.ravel() == 1], p1_norm[mask.ravel() == 1], self.K)

        # Return relative rotation, translation, and the mask of inliers
        return R, t, mask

    # --------------------------------------------------------------------- #
    def _scale_initialisation(self):
        """
        Initialize the metric scale by moving the camera a known baseline distance.
        This version accumulates the pose change over the calibration period for robustness.
        """
        print("\n" + "="*40)
        print("[INFO] Starting Scale Initialization...")
        print(f"[INFO] ==> Move the camera smoothly sideways by exactly {self.baseline:.2f} meters.")
        print("[INFO] ==> Keep features in view. Avoid fast rotations.")
        print("="*40 + "\n")

        start_time = time.time()
        init_duration_sec = 3.0 # Duration for the user to perform the movement
        needed_frames = int(self.fps_target * init_duration_sec) if self.fps_target > 0 else 60 # Target frames
        bar_len = 300

        # Store the initial state
        first_gray = self.prev_gray.copy()
        first_pts = self.prev_pts.copy()
        if first_pts is None:
            print("[ERROR] No initial points available for scale calibration.")
            return False

        # Accumulators for total relative pose change during calibration
        R_accum = np.eye(3)
        t_accum = np.zeros((3, 1))
        actual_frames = 0

        # Use a temporary copy of prev_pts/gray for the init loop
        current_gray = self.prev_gray
        current_pts = self.prev_pts

        while actual_frames < needed_frames:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Camera feed stopped during scale initialization.")
                break

            # Undistort current frame
            frame_undistorted = cv2.undistort(frame, self.K, self.dist, None, self.K)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

            # Track features from previous frame to current frame
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(current_gray, gray, current_pts, None, **self.lk_params)

            if next_pts is None or status is None:
                print("[WARN] Optical flow failed during scale init.")
                # Try to re-detect and continue? Or just use previous state?
                # For simplicity, let's just skip this frame's contribution
                current_gray = gray # Update gray for next iteration
                continue

            # Select good points (tracked successfully)
            good_mask = (status == 1).flatten()
            good_prev = current_pts[good_mask]
            good_next = next_pts[good_mask]

            if len(good_prev) < 8:
                print("[WARN] Not enough points tracked during scale init.")
                current_gray = gray
                current_pts = cv2.goodFeaturesToTrack(gray, **self.feat_params) # Re-detect
                if current_pts is None: current_pts = good_next.reshape(-1, 1, 2) # Fallback if redetect fails
                continue

            # Estimate relative pose change for this frame step
            R_c, t_c, _ = self._estimate_relative_pose(good_prev, good_next)

            if R_c is not None and t_c is not None:
                # Accumulate the pose transformation: T_new = T_step * T_old
                # Note: We track camera pose relative to the start.
                # t_accum represents the position of the *current* frame relative to the *first* frame, expressed in the first frame's coordinates.
                # R_accum represents the orientation of the *current* frame relative to the *first* frame.
                t_accum = t_accum + (R_accum @ t_c) # Add scaled translation vector
                R_accum = R_c @ R_accum          # Compose rotations

            # Update for next iteration
            current_gray = gray
            current_pts = good_next.reshape(-1, 1, 2)

            # Re-detect features if number drops too low
            if len(current_pts) < MIN_FEATURES:
                current_pts = cv2.goodFeaturesToTrack(current_gray, **self.feat_params)
                if current_pts is None:
                    print("[WARN] Feature re-detection failed during scale init.")
                    # Use the few points we still have
                    current_pts = good_next.reshape(-1, 1, 2)


            # --- Visualization ---
            actual_frames += 1
            progress = min(1.0, actual_frames / needed_frames)
            disp_len = int(bar_len * progress)
            bar_img = np.zeros((30, bar_len + 20, 3), dtype=np.uint8) # Increased width for text
            bar_img[:, :disp_len] = (0, 255, 0)
            bar_img[:, disp_len:] = (50, 50, 50)
             # Display percentage
            cv2.putText(bar_img, f"{int(progress * 100)}%", (bar_len + 2, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Put progress bar on the undistorted frame
            h_f, w_f = frame_undistorted.shape[:2]
            bar_y_start = 10
            bar_x_start = 10
            frame_undistorted[bar_y_start:bar_y_start+30, bar_x_start:bar_x_start+bar_len+20] = bar_img

            cv2.putText(frame_undistorted, f"Move sideways {self.baseline:.2f}m...",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Monocular VO Initializing...", frame_undistorted)

            key = cv2.waitKey(1) # Use waitKey(1) for max speed during init
            if key == ord('q'):
                print("[WARN] Scale initialization aborted by user.")
                cv2.destroyWindow("Monocular VO Initializing...")
                return False
            elif key == 27: # ESC key
                print("[WARN] Scale initialization aborted by user (ESC).")
                cv2.destroyWindow("Monocular VO Initializing...")
                return False

        cv2.destroyWindow("Monocular VO Initializing...")

        # --- Calculate Scale ---
        elapsed_time = time.time() - start_time
        print(f"[INFO] Scale initialization finished in {elapsed_time:.2f} seconds.")

        # Total translation magnitude (distance between first and last frame's estimated position)
        total_raw_translation_norm = np.linalg.norm(t_accum)
        print(f"[DEBUG] Accumulated Raw Translation Vector: {t_accum.flatten()}")
        print(f"[DEBUG] Accumulated Raw Translation Norm: {total_raw_translation_norm}")

        if total_raw_translation_norm < 1e-3: # Need some significant motion
            print("[ERROR] Scale initialization failed: Not enough translation detected.")
            print("[ERROR] Ensure you moved the camera sideways significantly (~{self.baseline:.2f}m).")
            return False

        # Calculate the scale factor
        self.scale = self.baseline / total_raw_translation_norm
        self.scale_initialized = True
        print(f"[INFO] Scale factor established: {self.scale:.4f} meters per VO unit.")
        print("="*40 + "\n")

        # --- Crucial Update ---
        # Update the main prev_gray and prev_pts to the *last* state from the init loop
        self.prev_gray = current_gray
        self.prev_pts = current_pts
        # No need to update self.R and self.t here, they represent the global pose which starts at Identity/Zero

        return True

    # --------------------------------------------------------------------- #
    def run(self):
        """Main VO processing loop."""

        # Attempt scale initialization first
        if not self._scale_initialisation():
             print("[WARN] Failed to initialize scale. Using default scale=1.0.")
             print("[WARN] Distances will NOT be metric. Press 's' to retry initialization.")
             self.scale = 1.0 # Set a default scale, but mark as uninitialized
             self.scale_initialized = False


        while True:
            frame_start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # 1. Undistort Frame
            frame_undistorted = cv2.undistort(frame, self.K, self.dist, None, self.K)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

            if self.prev_pts is None or len(self.prev_pts) < MIN_FEATURES // 2:
                # If points are lost or too few, re-detect
                print("[INFO] Low feature count, re-detecting...")
                self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feat_params)
                if self.prev_pts is None or len(self.prev_pts) < 10:
                    print("[WARN] Failed to find enough features after re-detection. Skipping frame.")
                    self.prev_gray = gray # Update gray anyway
                    continue # Skip rest of loop

            # 2. Track Features (Optical Flow)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )

            if next_pts is None or status is None:
                print("[WARN] Optical flow failed completely. Re-detecting on next frame.")
                self.prev_gray = gray
                self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feat_params) # Force re-detect
                continue

            # Select good points
            good_mask = (status == 1).flatten()
            good_prev = self.prev_pts[good_mask]
            good_next = next_pts[good_mask]
            num_tracked = len(good_prev)

            if num_tracked < 8: # Need at least 8 for Essential Matrix
                print(f"[WARN] Tracking lost ({num_tracked} points tracked). Re-detecting features.")
                self.prev_gray = gray
                self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feat_params)
                continue

            # 3. Estimate Relative Pose
            R_c, t_c, inlier_mask = self._estimate_relative_pose(good_prev, good_next)

            if R_c is None or t_c is None:
                print("[WARN] Pose estimation failed. Maintaining previous pose.")
                # Keep previous state, but update points/gray for next frame
                # Maybe re-detect if pose fails consistently? For now, just update.
                self.prev_gray = gray
                # Use only the points that were considered 'good' by optical flow for the next frame
                self.prev_pts = good_next.reshape(-1, 1, 2)
                if len(self.prev_pts) < MIN_FEATURES:
                    self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feat_params) # Try re-detecting
                continue


            # Filter points used for pose estimation (inliers from RANSAC)
            # We update prev_pts for the *next* iteration using these reliable points
            final_good_next = good_next[inlier_mask.ravel() == 1]
            num_inliers = len(final_good_next)

            # 4. Update Global Pose
            # Apply the *scaled* translation (t_c is unit vector from recoverPose)
            # Rotate the relative translation into the world frame using the *current* orientation R
            scaled_t_c = self.scale * t_c
            self.t = self.t + (self.R @ scaled_t_c)
            # Update the rotation matrix (compose relative rotation R_c with current R)
            self.R = R_c @ self.R

            # Update total trajectory length (using the scaled translation)
            self.traj_len += np.linalg.norm(scaled_t_c)

            # 5. Visualization
            vis_frame = frame_undistorted.copy()

            # Draw tracked features (green circles for inliers used in pose)
            for pt in final_good_next:
                 x, y = pt.astype(int).ravel()
                 cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1) # Green for inliers

            # Draw HUD (Heads-Up Display)
            h_f, w_f = vis_frame.shape[:2]
            hud_bg_color = (0, 0, 0)
            hud_text_color = (255, 255, 255)
            hud_info_color = (255, 255, 0) # Yellow for info
            cv2.rectangle(vis_frame, (0, 0), (300, 120), hud_bg_color, -1)

            # Use world coordinates (often Y is up/down, Z is forward/backward)
            # Note: OpenCV camera coord system: +Z forward, +Y down, +X right
            # Conversion to common world coord (e.g., +Y up) might be needed depending on convention.
            # Here, we display the camera's coordinate relative to the world origin.
            pos_x, pos_y, pos_z = self.t.flatten()
            cv2.putText(vis_frame, f"X: {pos_x:+.2f} m", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_text_color, 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Y: {pos_y:+.2f} m", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_text_color, 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Z: {pos_z:+.2f} m", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_text_color, 1, cv2.LINE_AA)

            cv2.putText(vis_frame, f"Traj: {self.traj_len:.2f} m", (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_info_color, 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Scale: {self.scale:.3f}{'' if self.scale_initialized else ' (DEF!)'}", (150, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_info_color, 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Feats: {num_tracked}/{num_inliers}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_info_color, 1, cv2.LINE_AA)


            cv2.imshow("Monocular VO", vis_frame)

            # 6. Handle User Input & Timing
            processing_time_ms = (time.time() - frame_start_time) * 1000
            wait_time = max(1, self.delay_ms - int(processing_time_ms))
            key = cv2.waitKey(wait_time)

            if key == ord('q') or key == 27: # q or ESC
                print("[INFO] Quitting...")
                break
            elif key == ord('s'):
                print("[INFO] Retrying scale initialization...")
                # Reset pose and attempt scale init again
                self.R = np.eye(3)
                self.t = np.zeros((3, 1))
                self.traj_len = 0.0
                # Need to ensure prev_pts and prev_gray are valid before calling
                if self.prev_gray is not None:
                     self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feat_params)
                     if self.prev_pts is not None:
                         if not self._scale_initialisation():
                            print("[WARN] Failed to re-initialize scale. Using previous or default scale.")
                         # Continue the loop after attempting re-initialization
                     else:
                         print("[WARN] Cannot re-initialize scale: failed to find features.")
                else:
                    print("[WARN] Cannot re-initialize scale: no previous frame data.")

            elif key == ord('c'):
                 # Save the *original* frame for calibration if possible
                 # We might only have the undistorted one easily available here
                 print("[INFO] Saving current (likely undistorted) view for calibration reference.")
                 save_calib_image(frame_undistorted) # Save the frame being displayed


            # 7. Prepare for Next Iteration
            self.prev_gray = gray
            # Update features: Use the inliers from the pose estimation for the next frame
            self.prev_pts = final_good_next.reshape(-1, 1, 2)

            # Optional: Re-detect if features drop low even after filtering
            if len(self.prev_pts) < MIN_FEATURES:
                 print("[INFO] Feature count low after pose estimation, re-detecting for next frame.")
                 self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feat_params)


        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")


################################################################################
# Entry-point
################################################################################
def main():
    ap = argparse.ArgumentParser(
        description="Real-time Monocular Visual Odometry with Metric Scale",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("-cam", "--camera_id", type=int, default=0,
                    help="Video capture device index. Check your system (often 0 or 1 for built-in webcams).")
    ap.add_argument("-fps", "--fps", type=float, default=30.0,
                    help="Target processing frame-rate (0 for max possible).")
    ap.add_argument("-b", "--baseline", type=float, default=0.30, # Default 30cm
                    help="Physical distance (meters) to move the camera sideways during scale initialization.")

    # Intrinsics Arguments: Prioritize --calib file if provided
    ap.add_argument("-calib", "--calib", type=str, default=None,
                    help="Path to camera calibration .npz file (containing 'K' matrix and 'dist' coeffs). Overrides fx, fy, cx, cy.")
    ap.add_argument("--fx", type=float, default=None, help="Focal length x (pixels). Used if --calib not set.")
    ap.add_argument("--fy", type=float, default=None, help="Focal length y (pixels). Used if --calib not set.")
    ap.add_argument("--cx", type=float, default=None, help="Principal point x (pixels). Used if --calib not set.")
    ap.add_argument("--cy", type=float, default=None, help="Principal point y (pixels). Used if --calib not set.")


    args = ap.parse_args()

    # --- Input Validation ---
    if args.baseline <= 0:
        print("[ERROR] Baseline distance must be positive.")
        sys.exit(1)
    if args.calib and not Path(args.calib).is_file():
         print(f"[ERROR] Calibration file not found: {args.calib}")
         sys.exit(1)

    # --- Camera Setup ---
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera ID {args.camera_id}. Try different IDs (0, 1, ...).")
        # Add platform specific tips if possible, e.g., check permissions on macOS/Linux
        if sys.platform == "darwin": # macOS
            print("[INFO] On macOS, ensure the application has camera permissions in System Preferences > Security & Privacy > Camera.")
        sys.exit(1)

    # Set desired FPS (best effort, depends on camera capabilities)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Get actual camera properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {args.camera_id} opened successfully.")
    print(f"[INFO] Resolution: {w0}x{h0}")
    print(f"[INFO] Target FPS: {args.fps}, Actual FPS reported by camera: {actual_fps:.2f}")
    if args.fps > 0 and abs(actual_fps - args.fps) > 5:
        print("[WARN] Camera might not support the requested FPS.")


    # --- Load Intrinsics ---
    K, dist = load_intrinsics(args, (h0, w0))
    print("[INFO] Camera Matrix K:\n", K)
    print("[INFO] Distortion Coefficients dist:\n", dist)

    # --- Initialize and Run VO ---
    try:
        vo = MonoVO(cap, K, dist, fps=args.fps, baseline=args.baseline)
        vo.run()
    except Exception as e:
        print(f"[FATAL ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure camera is released even if error occurs
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application finished.")


if __name__ == "__main__":
    main()