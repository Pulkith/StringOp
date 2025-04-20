import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import cm
from matplotlib.colors import Normalize
from src.entities import *
import random
import os, io
import numpy as np

from scipy.spatial import Voronoi
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets
from shapely.geometry import Point
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import cm
from matplotlib.colors import Normalize
from shapely.geometry import LineString
from src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets

# --- helper functions ---

def create_rectangle_marker(bounds):
    x = [0, 0, bounds[0], bounds[0], 0]
    y = [0, bounds[1], bounds[1], 0, 0]
    return x, y



# --- Visualization ---
def draw_scene(ax, drones, pois, aois, bounds,
               assignments=None,
               show_voronoi=False,
               use_custom_voronoi=False):
    ax.cla()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.plot(*create_rectangle_marker(bounds), color='black', linewidth=2)

    # colormap for POI priority (weight)
    norm_poi = Normalize(vmin=1, vmax=10)
    cmap_poi = cm.Reds

    # Draw clipped Voronoi polygons for alive drones only
    patches = []
    for drone in drones:
        if isinstance(drone, Quadcopter) and drone.alive and drone.region_polygon is not None:
            coords = np.array(drone.region_polygon.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            patches.append(patch)
    if patches:
        ax.add_collection(PatchCollection(
            patches,
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=1,
            alpha=0.2
        ))

    # Draw areas of interest outlines
    for area in aois:
        coords = np.array(area.polygon.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], color='magenta', linestyle='--', linewidth=1)

    # Draw movement & indicator lines
    for drone in drones:
        if isinstance(drone, Quadcopter) and drone.alive:
            # Draw line from drone to its target (if it has one)
            if drone.target is not None:
                ax.plot(
                    [drone.position[0], drone.target[0]],
                    [drone.position[1], drone.target[1]],
                    color='blue', linewidth=1.5, label='Target Line'
                )
            # Draw dashed lines from the target to each POI in the region
            if drone.target is not None and len(drone.targets_in_region) > 1:
                for poi_pos in drone.targets_in_region:
                    ax.plot(
                        [drone.target[0], poi_pos[0]],
                        [drone.target[1], poi_pos[1]],
                        linestyle='--', color='gray', linewidth=1, label='POI Link'
                    )

    # Draw drones, centers, targets, POIs
    scale = min(bounds) / 20
    for drone in drones:
        if isinstance(drone, Quadcopter):
            if drone.alive:
                # Draw alive drones
                ax.scatter(*drone.position, marker='x', color='black', s=100 * scale)
                if drone.center_point is not None:
                    ax.scatter(*drone.center_point, marker='o', color='orange', s=1, alpha=0.5)
                if drone.target is not None:
                    ax.scatter(*drone.target, marker='*', color='green', s=100)
            else:
                # Draw dead drones with skull marker
                ax.scatter(*drone.position, marker='$\u2620$', color='red', s=150 * scale)

    for poi in pois:
        color = cmap_poi(norm_poi(poi.weight))
        ax.scatter(*poi.position, color=color, s=40)

    # Legend (applies to both static and animation)
    drone_handle      = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=8, label='Quadcopter')
    poi_handle        = mlines.Line2D([], [], color='red',   marker='o', linestyle='None', markersize=6, label='POI')
    target_handle     = mlines.Line2D([], [], color='green', marker='*', linestyle='None', markersize=10, label='Target')
    # center_handle     = mlines.Line2D([], [], color='orange',marker='o', linestyle='None', markersize=8, label='Region Center', alpha=0.5)
    goal_line_handle  = mlines.Line2D([], [], color='blue',  linewidth=1.5,           label='Goal Line')
    link_line_handle  = mlines.Line2D([], [], color='gray',  linestyle='--', linewidth=1, label='POI Links')
    area_handle      = mlines.Line2D([], [], color='magenta', linestyle='--', linewidth=1, label='Area of Interest')
    skull_handle = mlines.Line2D([], [], color='none', marker='$\u2620$',linestyle='None',markersize=12,label='Dead Drone')
    ax.legend(
        handles=[
            drone_handle,
            poi_handle,
            target_handle,
            area_handle,
            # center_handle,
            goal_line_handle,
            link_line_handle,
            skull_handle
        ],
        loc='upper right'
    )

    ax.set_title("Drones Moving to POI Region Centers")


# --- Drawing Utilities ---
def draw_static(drones, pois, aois, bounds, filename, assignments=None, streamlit_display=False):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    draw_scene(ax, drones, pois, aois, bounds, assignments, show_voronoi=True, use_custom_voronoi=False)
    ax.set_title("Initial Configuration of Drones and POIs")
    plt.tight_layout()
    
    if streamlit_display:
        import streamlit as st
        st.pyplot(fig)
    else:
        plt.savefig(filename, dpi=300)
        plt.close()

def draw_priority_surface(drones, pois, aois, bounds, filename, grid_res=100, sigma=None):
    """
    Draw a 3D surface of POI priority weights as Gaussian blobs,
    overlaid on the 2D map at z=0.
    """
    if sigma is None:
        sigma = min(bounds) / 10.0

    # Create grid
    x = np.linspace(0, bounds[0], grid_res)
    y = np.linspace(0, bounds[1], grid_res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Accumulate Gaussian contributions
    for poi in pois:
        xi, yi = poi.position
        w = poi.weight
        Z += w * np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * sigma ** 2))

    # include area contributions via their centroids
    for area in aois:
        xi, yi = area.polygon.centroid.coords[0]
        w = area.weight
        Z += w * np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * sigma ** 2))

    # Create 3D figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Draw 2D map underneath at z=0
    # map border
    rect_x, rect_y = create_rectangle_marker(bounds)
    ax.plot(rect_x, rect_y, zs=0, zdir='z', color='black', linewidth=2)

    # Voronoi regions
    for drone in drones:
        if drone.region_polygon is not None:
            coords = np.array(drone.region_polygon.exterior.coords)
            ax.plot(coords[:,0], coords[:,1], zs=0, zdir='z',
                    color='blue', alpha=0.3)

    # Drones as black 'x'
    for drone in drones:
        ax.scatter(drone.position[0], drone.position[1], 0,
                   marker='x', color='black', s=50)

    # POIs colored by weight
    norm_poi = Normalize(vmin=1, vmax=10)
    cmap_poi = cm.Reds
    for poi in pois:
        color = cmap_poi(norm_poi(poi.weight))
        ax.scatter(poi.position[0], poi.position[1], 0,
                   color=color, s=30)

    # draw areas of interest outlines
    for area in aois:
        coords = np.array(area.polygon.exterior.coords)
        ax.plot(coords[:,0], coords[:,1], color='magenta', linestyle='--', linewidth=1)

    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, Z,
                           cmap='viridis', edgecolor='none', alpha=0.7)

    # Labels and colorbar
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Priority Intensity")
    ax.set_box_aspect((1,1,(0.5)))  # X=Y, Z other ration
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Weighted Intensity")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()
    plt.close()



# --- Animation ---
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_simulation(drones, pois, aois, bounds, results_dir, num_steps=200, seed=1, streamlit_display=False):
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

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

    if streamlit_display:
        # show the GIF in Streamlit
        import streamlit as st
        st.image(gif_path, caption="Simulation Animation")
        