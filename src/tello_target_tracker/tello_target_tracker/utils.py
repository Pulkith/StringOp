import math
import numpy as np
from geometry_msgs.msg import Point

def calculate_distance(point1: Point, point2: Point) -> float:
    """Calculate the Euclidean distance between two points"""
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

def estimate_drone_position(initial_pos, velocities, time_delta):
    """Estimate drone position based on velocity integration"""
    pos = np.array(initial_pos)
    vel = np.array(velocities)
    
    # Simple integration
    pos += vel * time_delta
    
    return pos

def filter_position(current_pos, new_measurement, alpha=0.3):
    """Apply a simple low-pass filter to position measurements"""
    if current_pos is None:
        return new_measurement
    
    return alpha * new_measurement + (1 - alpha) * current_pos