from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy

@dataclass
class GlobalPose:
    x: float
    y: float
    z: float
    yaw: Optional[float] = None

@dataclass
class LocalPose:
    x: float
    y: float
    z: float
    yaw: Optional[float] = None

@dataclass
class ObjectAttributes:
    r: float
    g: float
    b: float
    size_x: float
    size_y: float

# Constants
f = 0.5
cx, cy = 320, 240



def score(a: ObjectAttributes, b: ObjectAttributes) -> float:
    # Calculate the score based on the distance and size difference
    size_diff = ((a.size_x - b.size_x) ** 2 + (a.size_y - b.size_y) ** 2) ** 0.5
    color_diff = ((a.color[0] - b.color[0]) ** 2 + (a.color[1] - b.color[1]) ** 2 + (a.color[2] - b.color[2]) ** 2) ** 0.5
    return 0.1*size_diff + 0.9*color_diff

def dist(a: GlobalPose, b: GlobalPose) -> float:
    # Calculate the Euclidean distance between two poses
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def update_vector_low_pass(prev_pose: GlobalPose, new_pose: GlobalPose, alpha=0.5) -> GlobalPose:
    # alpha = 1.0 for no filtering
    pose = GlobalPose(
        x=alpha * new_pose.x + (1 - alpha) * prev_pose.x,
        y=alpha * new_pose.y + (1 - alpha) * prev_pose.y,
        z=alpha * new_pose.z + (1 - alpha) * prev_pose.z,
    )

    if new_pose.yaw is not None:
        pose.yaw = alpha * new_pose.yaw + (1 - alpha) * prev_pose.yaw

    return pose


class GlobalState:
    def __init__(self):
        self.drones: Dict[str, GlobalPose] = {}
        self.objects: List[Tuple[GlobalPose, ObjectAttributes]] = []

    def update_drone_pose(self, drone_id: str, new_pose: GlobalPose) -> None:
        self.drones[drone_id] = new_pose

    def add_or_update_object(self, object_pose: GlobalPose, object_attributes: ObjectAttributes) -> None:
        is_new_object = True
        for index, existing_object in enumerate(self.objects):
            if score(object_attributes, existing_object[1]) < 0.5 and dist(object_pose, existing_object[0]) < 0.5:
                print("Updating existing object")
                existing_object_pose = update_vector_low_pass(existing_object[0], object_pose)
                print(f"Existing object pose: {existing_object_pose}")
                existing_object_attributes = existing_object[1]
                self.objects[index] = (existing_object_pose, existing_object_attributes)
                is_new_object = False
                break
    
        if is_new_object:
            self.objects.append((object_pose, object_attributes))

    def update_drone_objects(self, drone_id: str, objects: List[Tuple[LocalPose, ObjectAttributes]]) -> Dict:
        drone_pose = self.drones[drone_id]
        # Update object poses with low pass filter (ADMM like approach)
        for drone_object in objects:
            relative_object_vector = drone_object[0]
            global_object_vector = GlobalPose(
                x=drone_pose.x + relative_object_vector.x * numpy.cos(drone_pose.yaw) - relative_object_vector.y * numpy.sin(drone_pose.yaw),
                y=drone_pose.y + relative_object_vector.x * numpy.sin(drone_pose.yaw) + relative_object_vector.y * numpy.cos(drone_pose.yaw),
                z=drone_pose.z + relative_object_vector.z,
            )

            print(f"Global object vector: {global_object_vector}")
            self.add_or_update_object(global_object_vector, drone_object[1])


    def get_drone_action():
        # Placeholder for drone action logic
        pass

if __name__ == "__main__":
    # Example usage
    global_state = GlobalState()


    global_state