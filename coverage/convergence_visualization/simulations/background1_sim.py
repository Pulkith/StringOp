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
    results_dir = "results/back1_sim"
    os.makedirs(results_dir, exist_ok=True)

    
    # Define the values of the simulation
    num_drones = 4
    num_fw = 0
    num_pois = 10
    num_aois = np.random.randint(0, 2)     # random number of AOIs with generic convex shapes

    poi_range = (1, 10)
    aoi_range = (1, 3)

    # Initialize drones, POIs, and AOIs
    def random_point():
        return (random.uniform(0, bounds[0]), random.uniform(0, bounds[1]))
    
    drones = [Quadcopter(random_point()) for _ in range(num_drones)]
    points = [  PointOfInterest((1, 75), weight=10, moving=True, speed=60/200, direction=-np.pi/6),
                PointOfInterest((54, 4), weight=4, moving=True, speed=0.6, direction=1.08083),
                PointOfInterest((50, 1), weight=6, moving=True, speed=0.2, direction=1.08083),
                PointOfInterest((75, 74), weight=10),
                PointOfInterest((85, 27), weight=9),
    ]
    
    # Add a group of lower prio targets
    center_coord = (150, 50)
    sigma = 5
    num_pois = 10

    def gaus_point(center, sigma):
        x = np.random.normal(center[0], sigma)
        y = np.random.normal(center[1], sigma)
        return (x, y)
    points += [PointOfInterest(gaus_point(center_coord, sigma), weight=2) for _ in range(int(num_pois/2-1))]

    # add some more random points to the fig
    pois = points + [PointOfInterest(random_point(), random.uniform(*poi_range)) for _ in range(int(num_pois/2-1))]
    
    aois = [AreaOfInterest([(10, 80), (38, 69), (40, 72), (36,76), (38, 90), (19, 90)], weight=2.0),
            AreaOfInterest([(42, 30), (60, 30), (60, 44), (48, 44)], weight=2.0),
    ]

    background_image_path = "simulations/backgrounds/background1.jpg"
    back_img = plt.imread(os.path.join(os.getcwd(), background_image_path))
    back_img = None

    # Draw initial configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "initial.png"), background_image=back_img)

    # Set up the figure for animation
    animate_simulation(drones, pois, aois, bounds,results_dir,  num_steps=num_steps, seed=seed,background_image=back_img)

    # Draw final configuration
    draw_static(drones, pois, aois, bounds, os.path.join(results_dir, "final.png"), background_image=back_img)

    # Draw priority surface
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"))
    draw_priority_surface(drones, pois, aois, bounds, os.path.join(results_dir, "priority_surface.png"), animate_rotation=True)


if __name__ == "__main__":
    run()