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
    def __init__(self, position, weight=1.0, moving=False, speed=1.0):
        self.position = np.array(position)
        self.id = str(uuid.uuid4())
        self.weight = weight
        self.moving = moving  # Whether the POI is moving
        self.speed = speed  # Speed of movement
        self.direction = np.random.uniform(0, 2 * np.pi)  # Random initial direction

    def move(self, bounds):
        if self.moving:
            # Update position based on direction and speed
            dx = self.speed * np.cos(self.direction)
            dy = self.speed * np.sin(self.direction)
            new_position = self.position + np.array([dx, dy])

            # Check if the new position is within bounds
            if 0 <= new_position[0] <= bounds[0] and 0 <= new_position[1] <= bounds[1]:
                self.position = new_position
            else:
                # Reverse direction if hitting a boundary
                self.direction += np.pi
                self.direction %= 2 * np.pi

    def __str__(self):
        return f"POI(id={self.id}, position={self.position.tolist()}, weight={self.weight}, moving={self.moving})"
    def __repr__(self):
        return f"POI(id={self.id}, position={self.position.tolist()}, weight={self.weight}, moving={self.moving})"

class AreaOfInterest:
    def __init__(self, polygon_coords, weight=1.0):
        from shapely.geometry import Polygon
        self.polygon = Polygon(polygon_coords)
        self.id = str(uuid.uuid4())
        self.weight = weight