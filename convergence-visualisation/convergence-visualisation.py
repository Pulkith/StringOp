import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial import Voronoi, QhullError
from scipy.optimize import linear_sum_assignment
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union, split
import uuid

def clipped_voronoi_polygons_2d(vor, bounds):
    from shapely.geometry import LineString

    # Create the bounding box polygon
    min_x, min_y = 0, 0
    max_x, max_y = bounds
    bbox_poly = box(min_x, min_y, max_x, max_y)

    regions = []
    for i, point in enumerate(vor.points):
        # Start with full bounding box
        region = bbox_poly
        # Clip by each bisector
        for j, other in enumerate(vor.points):
            if i == j:
                continue
            # Compute midpoint and direction of perpendicular bisector
            mid = (point + other) / 2
            direction = other - point
            # Normal vector (perpendicular to direction)
            normal = np.array([direction[1], -direction[0]])
            normal = normal / np.linalg.norm(normal)
            # Create a long line along the bisector
            L = max(bounds) * 2
            line_coords = [mid + normal * L, mid - normal * L]
            bisector = LineString(line_coords)
            # Split and keep the side containing the original point
            pieces = split(region, bisector)
            # Select the piece that contains the site
            for poly in pieces.geoms:
                if poly.contains(Point(point)) or poly.touches(Point(point)):
                    region = poly
                    break
        # Final region clipped
        regions.append(region)
    return regions

# --- Entities ---
class Drone:
    def __init__(self, position):
        self.position = np.array(position)
        self.id = str(uuid.uuid4())
        self.target = None
        self.region_polygon = None
        self.center_point = None
        self.targets_in_region = []

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
    def __init__(self, position):
        self.position = np.array(position)
        self.id = str(uuid.uuid4())

# --- Visualization ---
def draw_scene(ax, drones, pois, bounds,
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

    # Draw clipped Voronoi polygons
    patches = []
    for drone in drones:
        if isinstance(drone, Quadcopter) and drone.region_polygon is not None:
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

    # Draw movement & indicator lines
    for drone in drones:
        if isinstance(drone, Quadcopter):
            # if no target, fly to center_point (orange star)
            start = drone.target if drone.target is not None else drone.center_point
            if start is not None:
                ax.plot(
                    [drone.position[0], start[0]],
                    [drone.position[1], start[1]],
                    color='blue', linewidth=1.5
                )
            # if multiple POIs in region, dashed grey from target (green star) to each
            if drone.target is not None and len(drone.targets_in_region) > 1:
                for poi_pos in drone.targets_in_region:
                    ax.plot(
                        [drone.target[0], poi_pos[0]],
                        [drone.target[1], poi_pos[1]],
                        linestyle='--', color='gray', linewidth=1
                    )

    # Draw drones, centers, targets, POIs
    scale = min(bounds) / 20
    for drone in drones:
        if isinstance(drone, Quadcopter):
            ax.scatter(*drone.position, marker='x', color='black', s=100 * scale)
            if drone.center_point is not None:
                ax.scatter(*drone.center_point, marker='*', color='orange', s=60)
            if drone.target is not None:
                ax.scatter(*drone.target, marker='*', color='green', s=100)
        else:  # FixedWing
            triangle = create_triangle_marker(drone.position, drone.heading, size=scale)
            ax.plot(*triangle, color='black')

    for poi in pois:
        ax.scatter(*poi.position, color='red', s=40)

    # Legend (applies to both static and animation)
    drone_handle      = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=8, label='Quadcopter')
    poi_handle        = mlines.Line2D([], [], color='red',   marker='o', linestyle='None', markersize=6, label='POI')
    target_handle     = mlines.Line2D([], [], color='green', marker='*', linestyle='None', markersize=10, label='Target')
    center_handle     = mlines.Line2D([], [], color='orange',marker='*', linestyle='None', markersize=8, label='Region Center')
    goal_line_handle  = mlines.Line2D([], [], color='blue',  linewidth=1.5,           label='Goal Line')
    link_line_handle  = mlines.Line2D([], [], color='gray',  linestyle='--', linewidth=1, label='POI Links')
    ax.legend(
        handles=[
            drone_handle,
            poi_handle,
            target_handle,
            center_handle,
            goal_line_handle,
            link_line_handle
        ],
        loc='upper right'
    )

    ax.set_title("Drones Moving to POI Region Centers")

# --- Drawing Utilities ---
def draw_static(drones, pois, bounds, filename, assignments=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    draw_scene(ax, drones, pois, bounds, assignments, show_voronoi=True, use_custom_voronoi=False)
    ax.set_title("Initial Configuration of Drones and POIs")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_voronoi_polygons(ax, points, bounds):
    bbox = box(0, 0, bounds[0], bounds[1])
    vor = Voronoi(points)
    patches = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue
        poly_coords = vor.vertices[region]
        try:
            poly = Polygon(poly_coords).intersection(bbox)
            if poly.is_valid and not poly.is_empty:
                mpl_poly = MplPolygon(np.array(poly.exterior.coords), closed=True)
                patches.append(mpl_poly)
        except:
            continue
    if patches:
        patch_collection = PatchCollection(patches, facecolor='none', edgecolor='blue', linewidth=1, alpha=0.6)
        ax.add_collection(patch_collection)

def create_rectangle_marker(bounds):
    x = [0, 0, bounds[0], bounds[0], 0]
    y = [0, bounds[1], bounds[1], 0, 0]
    return x, y

def create_triangle_marker(position, heading, size=5):
    angle_rad = np.deg2rad(heading)
    base_angle1 = angle_rad + np.pi * 3/4
    base_angle2 = angle_rad - np.pi * 3/4

    front = position + size * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    left = position + size * 0.6 * np.array([np.cos(base_angle1), np.sin(base_angle1)])
    right = position + size * 0.6 * np.array([np.cos(base_angle2), np.sin(base_angle2)])

    x = [front[0], left[0], right[0], front[0]]
    y = [front[1], left[1], right[1], front[1]]
    return x, y

# --- Region Targets ---
def assign_voronoi_targets(drones, pois, bounds):
    points = np.vstack([drone.position for drone in drones])
    vor = Voronoi(points)
    bbox = box(0, 0, bounds[0], bounds[1])

    clipped_regions = clipped_voronoi_polygons_2d(vor, bounds)
    for i, drone in enumerate(drones):
        drone.center_point = None
        drone.region_polygon = None
        drone.target = None

        if i >= len(clipped_regions):
            continue

        poly = clipped_regions[i]
        drone.region_polygon = poly
        drone.center_point = np.array(poly.centroid.coords[0])

        points_in_region = [poi.position for poi in pois if poly.covers(Point(poi.position))]
        drone.targets_in_region = points_in_region
        if points_in_region:
            drone.target = np.mean(points_in_region, axis=0)
        else:
            # No POIs in region: move to center of the region polygon
            drone.target = drone.center_point

# --- Main Simulation ---
def generate_random_position(bounds):
    return np.random.uniform(0, bounds[0]), np.random.uniform(0, bounds[1])

def simulate(bounds, num_steps=50, step_size=1.0, gif_filename="voronoi_simulation.gif"):
    num_quad, num_fw, num_pois = 3, 0, 10
    drones = [Quadcopter(generate_random_position(bounds)) for _ in range(num_quad)]
    pois = [PointOfInterest(generate_random_position(bounds)) for _ in range(num_pois)]

    # assign_voronoi_targets(drones, pois, bounds)  # Initial static frame

    draw_static(drones, pois, bounds, "voronoi_initial.png")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    def update(frame):
        # Recompute regions, centers, and targets each step
        assign_voronoi_targets(drones, pois, bounds)

        # Move drones towards their current target (region center if no POIs)
        for drone in drones:
            if isinstance(drone, Quadcopter):
                if drone.target is not None:
                    drone.move_towards(drone.target, step_size)

        # Redraw the scene with updated regions and targets
        draw_scene(ax, drones, pois, bounds)

    anim = FuncAnimation(fig, update, frames=num_steps, interval=200)
    anim.save(gif_filename, dpi=80, writer=PillowWriter(fps=5))
    plt.close()

    draw_static(drones, pois, bounds, "voronoi_final.png")

# --- Entry Point ---
def main():
    bounds = (160, 90)
    simulate(bounds)

if __name__ == '__main__':
    main()
