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
    def __init__(self, tello, logger, camera_info_file=''):
        # Load camera calibration if available
        self.camera_info = None
        self.camera_info_file = camera_info_file
        if len(self.camera_info_file) > 0:
            try:
                with open(self.camera_info_file, 'r') as file:
                    self.camera_info = yaml.load(file, Loader=yaml.FullLoader)
            except Exception as e:
                logger.error(f"Failed to load camera info: {str(e)}")
        
        # Store reference to the tello instance
        self.tello = tello
        
        # Logger for proper ROS logging
        self.logger = logger
        
        # Camera state variables
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.thread = None
        self.last_frame_time = 0
        
        # Performance monitoring
        self.frame_count = 0
        self.error_count = 0
        self.fps_count = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        
        # Frame buffer to reduce impact of dropped frames
        self.frame_buffer_size = 5
        self.frame_buffer = []
        
        # wait for Tello to be ready
        self.logger.info("Waiting for Tello to be ready...")
        
        # Frame buffer for smoother video
        self.frame_buffer_size = 3  # Small buffer to help with jitter
        self.frame_buffer = []
        self.frame_timestamps = []
        
        # Performance tracking
        self.frame_count = 0
        self.error_count = 0
        self.fps_tracking_start = time.time()


        # Apply video optimizations
        self.optimize_video_stream()

        self.logger.info("TelloCamera initialized.")
    
    def start_video_capture(self):
        """Start the video capture thread"""
        if self.running:
            self.logger.warn("Video capture already running")
            return
        
        try:
            self.logger.info("Starting video stream...")
            self.tello.streamon()
            time.sleep(1.0)  # Give time for stream to initialize
            
            # Reset performance counters
            self.frame_count = 0
            self.error_count = 0
            self.fps_start_time = time.time()
            
            # Get the frame reader from Tello
            self.frame_reader = self.tello.get_frame_read()
            if self.frame_reader is None:
                raise Exception("Failed to get frame reader from Tello")
                
            # Start the capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Video capture thread started")
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting video capture: {str(e)}")
            raise
    
    def optimize_video_stream(self):
        """Apply optimizations to improve video streaming performance"""
        try:
            # Set video preferences for better streaming
            self.logger.info("Applying video streaming optimizations...")
            
            # Lower resolution if possible for your Tello model
            try:
                self.tello.set_video_resolution(self.tello.RESOLUTION_480P)
                self.logger.info("Set video resolution to 480p")
            except Exception as e:
                self.logger.warn(f"Failed to set video resolution: {str(e)}")
            
            # Lower frame rate to reduce bandwidth
            try:
                self.tello.set_video_fps(self.tello.FPS_15)
                self.logger.info("Set video FPS to 15")
            except Exception as e:
                self.logger.warn(f"Failed to set video FPS: {str(e)}")
            
            # Adjust bitrate
            try:
                self.tello.set_video_bitrate(self.tello.BITRATE_2MBPS)
                self.logger.info("Set video bitrate to 2Mbps")
            except Exception as e:
                self.logger.warn(f"Failed to set video bitrate: {str(e)}")
                
            self.logger.info("Video streaming optimizations applied")
        except Exception as e:
            self.logger.error(f"Unable to apply video optimizations: {str(e)}")

    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        consecutive_errors = 0
        last_stats_time = time.time()
        
        while self.running:
            try:
                # Get frame from Tello
                raw_frame = self.frame_reader.frame
                
                if raw_frame is not None:
                    # Process and store the frame
                    self._process_new_frame(raw_frame)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                else:
                    # Handle missing frame
                    self.error_count += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        self.logger.warn(f"Received {consecutive_errors} consecutive None frames")
                    elif self.error_count % 10 == 0:
                        self.logger.warn(f"Received None frame from Tello (total errors: {self.error_count})")
                
                # Log statistics periodically
                current_time = time.time()
                if current_time - last_stats_time > 10.0:  # Every 10 seconds
                    self._log_performance_stats()
                    last_stats_time = current_time
                
                # Sleep to avoid tight loop - adjust based on desired frame rate
                # Shorter sleep = more CPU but potentially more frames
                time.sleep(0.03)  # ~30fps theoretical max
                
            except Exception as e:
                # Log and handle errors
                self.error_count += 1
                consecutive_errors += 1
                
                if consecutive_errors < 5:
                    self.logger.warn(f"Error in capture loop: {str(e)}")
                elif consecutive_errors == 5:
                    self.logger.error(f"Multiple consecutive errors in capture loop: {str(e)}")
                    
                time.sleep(0.1)  # Longer sleep on error
    
    def _process_new_frame(self, raw_frame):
        """Process a newly received frame"""
        try:
            # Create a copy of the raw frame
            processed_frame = raw_frame.copy()
            
            # Update frame counter and calculate FPS
            self.frame_count += 1
            self.fps_count += 1
            current_time = time.time()
            
            # Calculate FPS every second
            if current_time - self.fps_start_time >= 1.0:
                self.actual_fps = self.fps_count / (current_time - self.fps_start_time)
                self.fps_count = 0
                self.fps_start_time = current_time
            
            # Add frame to buffer
            with self.frame_lock:
                # Update the current frame
                self.frame = processed_frame
                self.last_frame_time = current_time
                
                # Manage frame buffer
                self.frame_buffer.append(processed_frame)
                if len(self.frame_buffer) > self.frame_buffer_size:
                    self.frame_buffer.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        elapsed_time = time.time() - self.fps_start_time + 0.001  # Avoid division by zero
        if self.frame_count == 0:
            self.logger.warn("No frames received yet")
            return
            
        avg_fps = self.fps_count / elapsed_time
        error_rate = (self.error_count / (self.frame_count + self.error_count)) * 100 if self.frame_count + self.error_count > 0 else 0
        
        self.logger.info(
            f"Video Stats: Frames: {self.frame_count}, "
            f"Current FPS: {self.actual_fps:.1f}, "
            f"Errors: {self.error_count} ({error_rate:.1f}%)"
        )
        
        if self.frame is not None:
            with self.frame_lock:
                if self.frame is not None:
                    self.logger.info(f"Current frame: {self.frame.shape}, Age: {self.get_frame_age():.2f}s")
    
    def get_frame(self):
        """Get the latest frame with thread safety"""
        with self.frame_lock:
            if not self.frame_buffer:
                return None
                
            # Return the most recent frame from the buffer
            return self.frame_buffer[-1].copy()
    
    def get_frame_age(self):
        """Get the age of the current frame in seconds"""
        if self.last_frame_time == 0:
            return float('inf')  # No frame yet
        return time.time() - self.last_frame_time
    
    def get_stats(self):
        """Get current performance statistics"""
        return {
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "fps": self.actual_fps,
            "frame_age": self.get_frame_age(),
            "buffer_size": len(self.frame_buffer)
        }
    
    def stop(self):
        """Stop video capture"""
        self.running = False
        
        # Wait for the thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        try:
            self.tello.streamoff()
            self.logger.info("Video stream stopped")
        except Exception as e:
            self.logger.error(f"Error stopping video stream: {str(e)}")
        
        # Final stats
        self._log_performance_stats()
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    
    # Create a simple logger for standalone testing
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warn(self, msg): print(f"WARN: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    camera = TelloCamera(tello, SimpleLogger())
    camera.start_video_capture()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Add FPS display
                stats = camera.get_stats()
                cv2.putText(
                    frame, 
                    f"FPS: {stats['fps']:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow("Tello Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame available")
            time.sleep(0.03)  # ~30fps display rate
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        tello.end()  # Ensure Tello connection is closed