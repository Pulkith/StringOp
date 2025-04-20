from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Voronoi
import os
import numpy as np
import random
from src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets
from src.visualization import draw_scene, draw_static, draw_priority_surface

def run(bounds=(160, 90), num_steps=50, seed=1):
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Create results directory
    results_dir = "results/basic_sim"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize drones, POIs, and AOIs
    drones = [Quadcopter((random.uniform(0, bounds[0]), random.uniform(0, bounds[1]))) for _ in range(4)]
    pois = [PointOfInterest((random.uniform(0, bounds[0]), random.uniform(0, bounds[1])), weight=random.uniform(1, 10)) for _ in range(10)]
    aois = []

    # Draw initial configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "initial.png"))

    # Set up the figure for animation
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    def update(frame):
        # Assign Voronoi targets
        points = np.array([drone.position for drone in drones if drone.alive])
        if len(points) > 0:  # Ensure there are points to create a Voronoi diagram
            vor = Voronoi(points)  # Create the Voronoi object
            voronoi_regions = clipped_voronoi_polygons_2d(vor, bounds)
            assign_voronoi_targets(drones, pois, aois, bounds, voronoi_regions)

        # Move drones
        for drone in drones:
            if drone.alive and drone.target is not None:
                drone.move_towards(drone.target, step_size=1.0)

        # Redraw the scene
        draw_scene(ax, drones, pois, aois, bounds)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=200)

    # Save the animation as a GIF
    gif_path = os.path.join(results_dir, "simulation.gif")
    anim.save(gif_path, dpi=80, writer=PillowWriter(fps=5))
    plt.close()

    # Draw final configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "final.png"))

    # Draw priority surface
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"))

