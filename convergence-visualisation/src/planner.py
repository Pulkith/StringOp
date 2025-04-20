import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

def assignment_cost(drones, pois):
    cost = np.zeros((len(drones), len(pois)))
    for i, d in enumerate(drones):
        for j, p in enumerate(pois):
            cost[i, j] = np.linalg.norm(d.position - p.position)
    return cost

def compute_assignments(drones, pois):
    cost = assignment_cost(drones, pois)
    r, c = linear_sum_assignment(cost)
    return list(zip(r, c))

def assign_voronoi_targets(drones, pois, aois, bounds, clipped_regions):
    # reset
    for d in drones:
        d.region_polygon = None
        d.center_point = None
        d.target = None
        d.targets_in_region = []
    
    # only alive drones
    alive = [d for d in drones if d.alive]
    for i, drone in enumerate(alive):
        poly = clipped_regions[i]
        drone.region_polygon = poly
        drone.center_point = np.array(poly.centroid.coords[0])
        pts, wts = [], []
        for p in pois:
            if poly.covers(Point(p.position)):
                pts.append(p.position); wts.append(p.weight)
        for a in aois:
            if poly.intersects(a.polygon):
                cent = np.array(a.polygon.centroid.coords[0])
                pts.append(cent); wts.append(a.weight)
        drone.targets_in_region = pts
        if wts:
            drone.target = np.average(pts, axis=0, weights=wts)
        else:
            drone.target = drone.center_point