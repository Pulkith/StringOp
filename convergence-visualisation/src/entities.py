import numpy as np
import uuid

class Drone:
    def __init__(self, position):
        self.position = np.array(position)
        self.id = str(uuid.uuid4())
        self.target = None
        self.region_polygon = None
        self.center_point = None
        self.targets_in_region = []
        self.alive = True
        self.death_prob = 0.01

    def set_death_prob(self, prob):
        self.death_prob = prob
        return self
    


class Quadcopter(Drone):
    def __init__(self, position):
        super().__init__(position)

    def move_towards(self, target, step_size):
        direction = target - self.position
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.position += (direction / norm) * min(step_size, norm)

class FixedWing(Drone):
    def __init__(self, position, heading):
        super().__init__(position)
        self.heading = heading

class PointOfInterest:
    def __init__(self, position, weight=1.0):
        self.position = np.array(position)
        self.id = str(uuid.uuid4())
        self.weight = weight

class AreaOfInterest:
    def __init__(self, polygon_coords, weight=1.0):
        from shapely.geometry import Polygon
        self.polygon = Polygon(polygon_coords)
        self.id = str(uuid.uuid4())
        self.weight = weight