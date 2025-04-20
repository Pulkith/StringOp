# voronoi_drones_plot.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, QhullError
from scipy.optimize import linear_sum_assignment
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


# --- Entities ---
class Drone:
    def __init__(self, position):
        self.position = np.array(position)

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

# --- Visualization ---
def draw_scene(ax, drones, pois, bounds, assignments=None, show_voronoi=False):
    ax.cla()

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    ax.plot(*create_rectangle_marker(bounds), color='black', linewidth=2)

    scale = min(bounds) / 20
    drone_positions = [drone.position for drone in drones]

    if len(drone_positions) >= 3:
        arr = np.vstack(drone_positions)
        if np.linalg.matrix_rank(arr - arr[0]) > 1:
            plot_voronoi(ax, arr, bounds)

    for drone in drones:
        if isinstance(drone, Quadcopter):
            ax.scatter(*drone.position, marker='x', color='black', s=100 * scale)
        elif isinstance(drone, FixedWing):
            triangle = create_triangle_marker(drone.position, drone.heading, size=scale)
            ax.plot(*triangle, color='black')

    if show_voronoi:
        try:
            if len(drone_positions) >= 3:
                arr = np.vstack(drone_positions)
                if np.linalg.matrix_rank(arr - arr[0]) > 1:
                    vor = Voronoi(arr)
                    voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                                    line_colors='gray', line_width=1, line_alpha=0.6, point_size=0)
        except QhullError:
            print("[WARNING] Voronoi computation failed — likely due to degenerate geometry.")

    for poi in pois:
        ax.scatter(*poi.position, color='red', s=40)

    if assignments:
        for i, j in assignments:
            p1, p2 = drones[i].position, pois[j].position
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=':', color='lightblue', linewidth=1.5)

    ax.set_title("Drones Moving to Assigned POIs")

# --- Drawing Utilities ---
def draw_static(drones, pois, bounds, filename, assignments=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    draw_scene(ax, drones, pois, bounds, assignments)
    ax.set_title("Final Positions of Drones and POIs")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_voronoi(ax, points, bounds, extension=1e3):
    try:
        vor = Voronoi(points)
        segments = []

        center = points.mean(axis=0)
        for (pointidx, simplex) in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                i, j = pointidx
                t = points[j] - points[i]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = (points[i] + points[j]) / 2
                far_point = midpoint + n * extension
                segments.append([midpoint, far_point])
            else:
                p1, p2 = vor.vertices[simplex]
                segments.append([p1, p2])

        line_collection = LineCollection(segments, colors='blue', linewidths=1, alpha=0.6)
        ax.add_collection(line_collection)
    except QhullError:
        print("[WARNING] Voronoi computation failed — likely due to degenerate geometry.")

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

# --- Matching Logic ---
def assignment_cost(drones, pois):
    cost_matrix = np.zeros((len(drones), len(pois)))
    for i, drone in enumerate(drones):
        for j, poi in enumerate(pois):
            cost_matrix[i, j] = np.linalg.norm(drone.position - poi.position)
    return cost_matrix

def compute_assignments(drones, pois):
    cost = assignment_cost(drones, pois)
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind, col_ind))

# --- Main Simulation ---
def generate_random_position(bounds):
    return np.random.uniform(0, bounds[0]), np.random.uniform(0, bounds[1])

def simulate(bounds, num_steps=50, step_size=1.0, gif_filename="voronoi_simulation.gif"):
    num_quad, num_fw, num_pois = 3, 0, 3
    drones = [Quadcopter(generate_random_position(bounds)) for _ in range(num_quad)]
    pois = [PointOfInterest(generate_random_position(bounds)) for _ in range(num_pois)]

    assignments = compute_assignments(drones, pois)

    draw_static(drones, pois, bounds, "voronoi_initial.png", assignments)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    def update(frame):
        for i, j in assignments:
            if isinstance(drones[i], Quadcopter):
                drones[i].move_towards(pois[j].position, step_size)
        draw_scene(ax, drones, pois, bounds, assignments)

    anim = FuncAnimation(fig, update, frames=num_steps, interval=200)
    anim.save(gif_filename, dpi=80, writer=PillowWriter(fps=5))
    plt.close()

    draw_static(drones, pois, bounds, "voronoi_final.png", assignments)

# --- Entry Point ---
def main():
    bounds = (160, 90)
    simulate(bounds)

if __name__ == '__main__':
    main()
