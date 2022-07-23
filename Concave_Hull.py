from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import math
import shapely.geometry as geometry
from descartes import PolygonPatch
import pylab as plt
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
import itertools


def plot_polygon(polygon):
    fig = plt.figure(figsize=[16, 9])
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#ffdede', ec='#ffdede', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig, patch


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    @param only_outer:
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    #def add_edge(edges, edge_points, coords, i, j):
    #    """Add a line between the i-th and j-th points, if not in the list already"""
        #if (i, j) in edges or (j, i) in edges:
        #    assert (j, i) in edges, "Can't go twice over same directed edge right?"
        #    if only_outer:
        #        # if both neighboring triangles are in shape, it's not a boundary edge
        #        edges.remove((j, i))
        #    return
        #edges.add((i, j))



    coords = np.array([point for point in points])
    # print(len(coords))

    tri = Delaunay(coords)
    # print(len(tri.vertices))
    #edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area) if area > 0 else 999

        # Here's the radius filter.
        # print circum_r
        if circum_r < alpha:
            edge_points.extend([coords[[ia, ib]], coords[[ib, ic]], coords[[ic, ia]]])
            #edges.add(((ia, ib), (ib, ic), (ic, ia)))
            # withour lines below it is not uniquely an outer maring
            #c = Counter(edges)
            #edges = {key for key, value in c.items() if value == 1}

            #add_edge(edges, edge_points, coords, ia, ib)
            #add_edge(edges, edge_points, coords, ib, ic)
            #add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points

def concave_hull(coords, alpha, testing = False):  # coords is a 2D numpy array

    if len(coords) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(coords)).convex_hull

    # i removed the Qbb option from the scipy defaults.
    # it is much faster and equally precise without it.
    # unless your coords are integers.
    # see http://www.qhull.org/html/qh-optq.htm
    if type(coords).__name__ == "list":
        coords_list = coords
        coords = np.array(coords)

    tri = Delaunay(coords, qhull_options="Qc Qz Q12").vertices

    ia, ib, ic = (
        tri[:, 0],
        tri[:, 1],
        tri[:, 2],
    )  # indices of each of the triangles' points
    pa, pb, pc = (
        coords[ia],
        coords[ib],
        coords[ic],
    )  # coordinates of each of the triangles' points

    a = np.sqrt((pa[:, 0] - pb[:, 0]) ** 2 + (pa[:, 1] - pb[:, 1]) ** 2)
    b = np.sqrt((pb[:, 0] - pc[:, 0]) ** 2 + (pb[:, 1] - pc[:, 1]) ** 2)
    c = np.sqrt((pc[:, 0] - pa[:, 0]) ** 2 + (pc[:, 1] - pa[:, 1]) ** 2)

    s = (a + b + c) * 0.5  # Semi-perimeter of triangle

    area = np.sqrt(
        s * (s - a) * (s - b) * (s - c)
    )  # Area of triangle by Heron's formula

    area = np.where(area == 0, 0.0001, a)

    filter = (
            ( a * b * c / (4.0 * area) ) < alpha  # 1.0 / alpha
    )  # Radius Filter based on alpha value

    # Filter the edges
    edges = tri[filter]

    # now a main difference with the aforementioned approaches is that we don't
    # use a Set() because this eliminates duplicate edges. in the list below
    # both (i, j) and (j, i) pairs are counted. The reasoning is that boundary
    # edges appear only once while interior edges twice
    edges = [
        tuple(sorted(combo)) for e in edges for combo in itertools.combinations(e, 2)
    ]

    count = Counter(edges)  # count occurrences of each edge

    # keep only edges that appear one time (concave hull edges)
    edges = [e for e, c in count.items() if c == 1]

    # these are the coordinates of the edges that comprise the concave hull
    edges = [(coords[e[0]], coords[e[1]]) for e in edges]

    # use this only if you need to return your hull points in "order" (i think
    # its CCW)
    if not edges:
        raise Exception

    ml = MultiLineString(edges)
    poly = polygonize(ml)
    hull = unary_union(list(poly))
    # hull_vertices = hull.exterior.coords.xy

    return hull, edges # hull_vertices
