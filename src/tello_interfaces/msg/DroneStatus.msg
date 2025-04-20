# DroneStatus.msg
# Status information for a Tello drone

int32 drone_id                # Unique identifier for the drone
string state                  # Current state of the drone's state machine
int32 battery                 # Battery percentage (0-100)
int32 target_id               # ID of target being tracked (-1 if none)
float32 height                # Current height in meters
geometry_msgs/Vector3 velocity # Current velocity