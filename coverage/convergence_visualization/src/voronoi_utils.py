import numpy as np
from scipy.spatial import Voronoi, QhullError
from shapely.geometry import box, Point, LineString
from shapely.ops import split

#import points
import matplotlib.pyplot as plt

def clipped_voronoi_polygons_2d(vor, bounds):
    min_x, min_y = 0, 0
    max_x, max_y = bounds
    bbox_poly = box(min_x, min_y, max_x, max_y)
    regions = []
    for i, point in enumerate(vor.points):
        region = bbox_poly
        for j, other in enumerate(vor.points):
            if i == j: continue
            mid = (point + other) / 2
            direction = other - point
            normal = np.array([direction[1], -direction[0]])
            normal /= np.linalg.norm(normal)
            L = max(bounds) * 2
            bisector = LineString([mid + normal*L, mid - normal*L])
            pieces = split(region, bisector)
            for poly in pieces.geoms:
                if poly.contains(Point(point)) or poly.touches(Point(point)):
                    region = poly
                    break
        regions.append(region)
    return regions

def plot_voronoi_polygons(ax, points, bounds):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import Polygon
    vor = Voronoi(points)
    clipped = clipped_voronoi_polygons_2d(vor, bounds)
    patches = []
    for poly in clipped:
        coords = np.array(poly.exterior.coords)
        patches.append(MplPolygon(coords, closed=True))
    if patches:
        ax.add_collection(PatchCollection(patches,
            facecolor='lightblue', edgecolor='blue', linewidth=1, alpha=0.2))