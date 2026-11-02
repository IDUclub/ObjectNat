"""
Independent brute-force visibility reference shared by the tests and the benchmarks.

It shares no logic with ObjectNat's visibility kernel, so agreement with it is a real accuracy check.
"""

import numpy as np
import shapely
from shapely import Point, Polygon


def oracle_visibility(point: Point, obstacle_geoms, radius: float, quad_segs: int = 16) -> Polygon:
    """
    Independent brute-force reference: view disc minus the union of every wall's shadow and the obstacles.

    The shadow of a wall segment is the region behind it as seen from the observer; the union of all
    wall shadows is exactly the occluded region. Each segment is split so that no piece subtends more
    than 60 degrees, which keeps its shadow quadrilateral convex and reaching beyond the disc.
    No pruning, no nearest-wall ordering: slow, but shares no logic with either implementation.
    """
    disc = point.buffer(radius, quad_segs=quad_segs)
    geoms = [g for g in obstacle_geoms if g is not None and not g.is_empty and g.intersects(disc)]
    if not geoms:
        return disc
    o = np.array([point.x, point.y])
    far = 4.0 * radius
    shadows = []
    for geom in geoms:
        boundary = geom.boundary if geom.geom_type in ("Polygon", "MultiPolygon") else geom
        for line in shapely.get_parts(boundary):
            coords = np.asarray(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                va, vb = a - o, b - o
                if np.hypot(*va) == 0 or np.hypot(*vb) == 0:
                    continue
                angle = abs(np.arctan2(va[0] * vb[1] - va[1] * vb[0], va @ vb))
                pieces = max(1, int(np.ceil(angle / (np.pi / 3))))
                for k in range(pieces):
                    p = a + (b - a) * (k / pieces)
                    q = a + (b - a) * ((k + 1) / pieces)
                    vp, vq = p - o, q - o
                    p_far = o + vp / np.hypot(*vp) * far
                    q_far = o + vq / np.hypot(*vq) * far
                    quad = Polygon([p, q, q_far, p_far])
                    if not quad.is_valid:
                        quad = quad.buffer(0)
                    if not quad.is_empty and quad.area > 0:
                        shadows.append(quad)
    hidden = shapely.union_all(shadows + geoms)
    visible = disc.difference(hidden)
    parts = [g for g in shapely.get_parts(visible) if g.geom_type == "Polygon"]
    containing = [g for g in parts if g.distance(point) < 1e-6]
    if containing:
        return max(containing, key=lambda g: g.area)
    return Polygon()
