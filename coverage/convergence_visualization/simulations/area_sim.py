from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Voronoi
import os
import numpy as np
import random
from src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets
from src.visualization import *

def generate_random_position(bounds):
    return np.random.uniform(0, bounds[0]), np.random.uniform(0, bounds[1])

def run(bounds=(160, 90), num_steps=200, seed=1):
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Create results directory
    results_dir = "results/area_sim"
    os.makedirs(results_dir, exist_ok=True)

    # Define the values of the simulation
    num_drones = 3
    num_fw = 0
    num_pois = 10
    num_aois = np.random.randint(3, 6)     # random number of AOIs with generic convex shapes

    poi_range = (1, 10)
    aoi_range = (1, 5)

    # Initialize drones, POIs, and AOIs
    def random_point():
        return (random.uniform(0, bounds[0]), random.uniform(0, bounds[1]))
    drones = [Quadcopter(random_point()) for _ in range(num_drones)]
    pois = [PointOfInterest(random_point(), weight=random.uniform(*poi_range)) for _ in range(num_pois)]
    aois = []

    for _ in range(num_aois):
        cx, cy = generate_random_position(bounds)
        num_vertices = np.random.randint(3, 9)
        angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
        r_min, r_max = min(bounds) * 0.05, min(bounds) * 0.15
        poly_coords = []
        for ang in angles:
            r = np.random.uniform(r_min, r_max)
            x = np.clip(cx + r * np.cos(ang), 0, bounds[0])
            y = np.clip(cy + r * np.sin(ang), 0, bounds[1])
            poly_coords.append((x, y))
        aois.append(AreaOfInterest(poly_coords, weight=np.random.uniform(*aoi_range)))
        
    # Draw initial configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "initial.png"))

    # Set up the figure for animation
    animate_simulation(drones, pois, aois, bounds, results_dir, num_steps=num_steps, seed=seed)

    # Draw final configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "final.png"))

    # Draw priority surface
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"))

