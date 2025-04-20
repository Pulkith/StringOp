# TargetInfo.msg
# Information about a detected target

int32 target_id               # Unique identifier for the target
geometry_msgs/Point relative_position  # Position relative to drone (x,y,z)
float32 confidence            # Detection confidence (0.0-1.0)