from src.tello_target_tracker.tello_target_tracker.DJITelloPy.djitellopy import Tello
import time

# Create Tello object
tello = Tello()

# Connect to the drone
print("Connecting to Tello...")
tello.connect()

# Enter SDK mode
print("Entering SDK mode...")
tello.send_control_command("command")

# Check battery
battery = tello.get_battery()
print(f"Battery level: {battery}%")

# Attempt takeoff
print("Attempting takeoff...")
tello.takeoff()

# Wait and get height
time.sleep(3)
height = tello.get_height()
print(f"Current height: {height} cm")

# Land
print("Landing...")
tello.land()

# End connection
tello.end()