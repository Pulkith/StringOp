#!/usr/bin/env python3

import yaml
import cv2
import time
import threading
from djitellopy import Tello
import ament_index_python  # Optional: only used if you still want to resolve the default YAML path

class TelloCamera:
    def __init__(self, camera_info_file=''):
        # Load camera calibration
        self.camera_info = None
        self.camera_info_file = camera_info_file

        if len(self.camera_info_file) == 0:
            try:
                share_directory = ament_index_python.get_package_share_directory('tello')
                self.camera_info_file = share_directory + '/ost.yaml'
            except:
                raise FileNotFoundError("Camera info file not provided and 'ament_index_python' can't resolve default path.")

        with open(self.camera_info_file, 'r') as file:
            self.camera_info = yaml.load(file, Loader=yaml.FullLoader)

        # Initialize drone
        self.tello = Tello()
        self.tello.connect()
        print("Connected to Tello drone.")

        self.running = False
        self.frame = None
        self.bridge = None

    def start_video_capture(self):
        self.tello.streamon()
        self.running = True

        def capture_loop():
            while self.running:
                frame = self.tello.get_frame_read().frame
                self.frame = frame
                # Display the frame (optional)
                cv2.imshow("Tello Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

        thread = threading.Thread(target=capture_loop)
        thread.start()

    def stop(self):
        self.running = False
        self.tello.streamoff()
        self.tello.end()
        cv2.destroyAllWindows()
        print("Video capture stopped and Tello disconnected.")

if __name__ == "__main__":
    camera_info_file = ''  # Replace with actual path to ost.yaml if needed
    tello_cam = TelloCamera(camera_info_file)
    tello_cam.start_video_capture()
