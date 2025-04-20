#!/usr/bin/env python3
import yaml
import cv2
import time
import threading
import numpy as np
from tello_target_tracker.DJITelloPy.djitellopy import Tello

# from DJITelloPy.djitellopy.tello import Tello
import ament_index_python

class TelloCamera:
    def __init__(self, tello, camera_info_file=''):
        # Load camera calibration if available
        self.camera_info = None
        self.camera_info_file = camera_info_file
        if len(self.camera_info_file) > 0:
            try:
                with open(self.camera_info_file, 'r') as file:
                    self.camera_info = yaml.load(file, Loader=yaml.FullLoader)
            except Exception as e:
                print(f"Warning: Failed to load camera info: {str(e)}")
        
        # Store reference to the tello instance
        self.tello = tello
        
        # Camera state variables
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.thread = None
        self.last_frame_time = 0
        print("TelloCamera initialized.")
    
    def start_video_capture(self):
        """Start the video capture thread"""
        if self.running:
            print("Video capture already running")
            return
        
        try:
            print("Starting video stream...")
            self.tello.streamon()
            time.sleep(1.0)  # Give time for stream to initialize
            
            # Get the frame reader from Tello
            self.frame_reader = self.tello.get_frame_read()
            if self.frame_reader is None:
                raise Exception("Failed to get frame reader from Tello")
                
            # Start the capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            print("Video capture thread started")
        except Exception as e:
            self.running = False
            print(f"Error starting video capture: {str(e)}")
            raise
    
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        frame_count = 0
        error_count = 0
        
        while self.running:
            try:
                # Get frame from Tello
                print("getting frame")
                raw_frame = self.frame_reader.frame
                print("got frame")
                
                if raw_frame is not None:
                    # Store a copy of the frame with thread safety
                    with self.frame_lock:
                        self.frame = raw_frame.copy()
                        self.last_frame_time = time.time()
                    
                    # Count successful frames
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Captured {frame_count} frames, current shape: {self.frame.shape}")
                    
                    # Optional: Display frame
                    # cv2.imshow("Tello Camera", self.frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.stop()
                    #     break
                else:
                    error_count += 1
                    if error_count % 5 == 0:  # Don't spam logs
                        print(f"Received None frame from Tello ({error_count} errors)")
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                print(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)  # Longer sleep on error
    
    def get_frame(self):
        """Get the latest frame with thread safety"""
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def get_frame_age(self):
        """Get the age of the current frame in seconds"""
        if self.last_frame_time == 0:
            return float('inf')  # No frame yet
        return time.time() - self.last_frame_time
    
    def stop(self):
        """Stop video capture"""
        self.running = False
        
        # Wait for the thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        try:
            self.tello.streamoff()
            print("Video stream stopped")
        except Exception as e:
            print(f"Error stopping video stream: {str(e)}")
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    
    camera = TelloCamera(tello)
    camera.start_video_capture()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("Tello Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame available")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        tello.end()  # Ensure Tello connection is closed