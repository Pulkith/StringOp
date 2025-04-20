import os
import numpy as np
import random
from src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets
from src.visualization import draw_scene, draw_static, draw_priority_surface

def run(bounds=(160, 90), num_steps=50, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Create results directory
    results_dir = "results/area_sim"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize drones, POIs, and AOIs
    drones = [Quadcopter((random.uniform(0, bounds[0]), random.uniform(0, bounds[1]))) for _ in range(4)]
    pois = [PointOfInterest((random.uniform(0, bounds[0]), random.uniform(0, bounds[1])), weight=random.uniform(1, 10)) for _ in range(10)]
    aois = [
        AreaOfInterest(
            [(random.uniform(0, bounds[0]), random.uniform(0, bounds[1])) for _ in range(5)],
            weight=random.uniform(1, 5)
        )
        for _ in range(3)
    ]

    # Draw initial configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "initial.png"))

    # Simulate
    for step in range(num_steps):
        # Assign Voronoi targets
        points = np.array([drone.position for drone in drones if drone.alive])
        voronoi_regions = clipped_voronoi_polygons_2d(points, bounds)
        assign_voronoi_targets(drones, pois, aois, bounds, voronoi_regions)

        # Move drones
        for drone in drones:
            if drone.alive and drone.target is not None:
                drone.move_towards(drone.target, step_size=1.0)

    # Draw final configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "final.png"))

    # Draw priority surface
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"))