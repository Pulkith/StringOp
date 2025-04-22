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
    results_dir = "results/back2_sim"
    os.makedirs(results_dir, exist_ok=True)

    # Add a plot with background image
    background_image_path = "simulations/backgrounds/background2.jpg"
    background_image = plt.imread(os.path.join(os.getcwd(), background_image_path))
    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=(0, bounds[0], 0, bounds[1]))
    ax.grid(True)
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_title("Voronoi Animation with Background Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    
    plt.show()


if __name__ == "__main__":
    run()