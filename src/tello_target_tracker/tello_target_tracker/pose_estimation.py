import threading
import time
import pandas as pd

class PoseEstimation:
    def __init__(self, tello, x_init = 0):
        # All drones are aligned at the beginning with an offset in x
        self.tello = tello
        self.x = x_init
        self.y = 0
        self.z = 0
        self.yaw = 0

        self.rate = 10
        self.dt = 1 / self.rate
        self.running = True
        
        self.previous_time = time.time()

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()


    def join(self):
        print("Pose estimation data saved to pose_estimation.csv")
        self.thread.join()
        
    def loop(self):
        while self.running:
            self.previous_time = time.time()
            self.update()
            time.sleep(self.dt)

    def get_state(self):
        # Get the current state of the drone
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
        }

    def update(self):
        try:
            vx = self.tello.get_speed_x()
            vy = self.tello.get_speed_y()
            vz = self.tello.get_speed_z()

            #print(f"Speed: vx={vx}, vy={vy}, vz={vz}")
            #print(f"Acceleration: ax={ax}, ay={ay}, az={az}")
            # y is the height of the drone
            # z is the depth direction

            self.x += vx * self.dt
            self.y += vy * self.dt
            self.z = self.tello.get_height()
            #self.z += vz * self.dt
            self.yaw = self.tello.get_yaw()

            #print(f"Pose: x={self.x}, y={self.y}, z={self.z}, yaw={self.yaw}")
            
        except Exception as e:
            print(f"Error: {e}")