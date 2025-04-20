from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Voronoi
import os
import numpy as np
import random
from convergence_visualization.src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from convergence_visualization.src.voronoi_utils import clipped_voronoi_polygons_2d
from convergence_visualization.src.planner import assign_voronoi_targets
from convergence_visualization.src.visualization import *

def run(bounds=(160, 90), num_steps=200, seed=1, drone_positions=None, poi_positions=None):
    drones = [Quadcopter(pos) for pos in drone_positions]
    pois = [PointOfInterest(position=pos, weight=1.0) for pos in poi_positions]
    aois = []

    results_dir = os.path.join(".")

    # Draw initial configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "initial.png"))

    # Set up the figure for animation
    animate_simulation(drones, pois, aois, bounds, results_dir, num_steps=num_steps, seed=seed)

    # Draw final configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "final.png"))

    # Draw priority surface
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"))
