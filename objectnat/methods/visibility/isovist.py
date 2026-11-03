"""Isovist metrics of exact visibility polygons.

A visibility polygon (isovist) is star-shaped around its observer, so every boundary edge is either an
*occluding* edge (radial: it lies on a ray from the observer and separates the visible area from a shadow),
a stretch of the *view limit* (a side of the view disc) or a visible piece of an obstacle *wall*.

The radial distance ``r(θ)`` from the observer to the boundary is a function of direction, and its moments
are integrated exactly edge by edge: along an edge at perpendicular distance ``h`` from the observer,
``r = h / cos φ``, so ``∫ r dθ``, ``∫ r² dθ`` and ``∫ r³ dθ`` have closed forms. No rays are sampled.

References: Benedikt (1979) "To take hold of space: isovists and isovist fields"; Batty (2001)
"Exploring isovist fields"; Turner et al. (2001) "From isovists to visibility graphs".
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from objectnat.methods.visibility._native import DEFAULT_QUAD_SEGS

ISOVIST_METRICS = (
    "area",
    "perimeter",
    "compactness",
    "occlusivity",
    "n_occlusions",
    "min_radial",
    "max_radial",
    "mean_radial",
    "std_radial",
    "skew_radial",
    "drift",
    "view_share",
    "open_share",
)

_REL_TOL = 1e-7  # relative tolerance for "lies on a ray" and "lies on the view limit"


def _primitive_r1(h, t):
    # ∫ r dθ along a line at distance h, with t the signed position along the line from the foot point
    return h * np.arcsinh(t / h)


def _primitive_r3(h, t):
    return 0.5 * (h * t * np.sqrt(h * h + t * t) + h**3 * np.arcsinh(t / h))


def isovist_metrics(observers: np.ndarray, polygons: np.ndarray, view_distances: np.ndarray) -> pd.DataFrame:
    """
    Isovist metrics for visibility polygons in a projected CRS.

    Args:
        observers: ``(n, 2)`` observer coordinates.
        polygons: ``n`` visibility polygons (shapely), in the same CRS; empty for blocked observers.
        view_distances: ``n`` view radii the polygons were computed with.

    Returns:
        DataFrame with one row per observer and the columns in ``ISOVIST_METRICS``
        (NaN for empty polygons).
    """
    n = len(polygons)
    polygons = np.asarray(polygons, dtype=object)
    observers = np.asarray(observers, dtype="float64").reshape(n, 2)
    view_distances = np.asarray(view_distances, dtype="float64")

    out = pd.DataFrame(np.nan, index=np.arange(n), columns=list(ISOVIST_METRICS))
    out["n_occlusions"] = 0
    valid = ~(shapely.is_empty(polygons) | shapely.is_missing(polygons))
    if not valid.any():
        out.loc[~valid, "n_occlusions"] = np.nan
        return out

    area = shapely.area(polygons)
    perimeter = shapely.length(polygons)
    out["area"] = area
    out["perimeter"] = perimeter
    with np.errstate(divide="ignore", invalid="ignore"):
        out["compactness"] = 4 * np.pi * area / perimeter**2
    out["view_share"] = area / (np.pi * view_distances**2)

    # --- per-edge geometry of every exterior ring, relative to its observer -----------------------------
    parts, part_owner = shapely.get_parts(polygons[valid], return_index=True)
    owner_ids = np.flatnonzero(valid)[part_owner]
    rings = shapely.get_exterior_ring(parts)
    coords, ring_idx = shapely.get_coordinates(rings, return_index=True)
    owner = owner_ids[ring_idx]
    rel = coords - observers[owner]

    same_ring = ring_idx[1:] == ring_idx[:-1]
    p1, p2, edge_owner = rel[:-1][same_ring], rel[1:][same_ring], owner[:-1][same_ring]
    edge = p2 - p1
    length = np.hypot(edge[:, 0], edge[:, 1])
    keep = length > 0
    p1, p2, edge, length, edge_owner = p1[keep], p2[keep], edge[keep], length[keep], edge_owner[keep]

    unit = edge / length[:, None]
    cross = p1[:, 0] * edge[:, 1] - p1[:, 1] * edge[:, 0]
    h = np.abs(cross) / length  # distance from the observer to the edge's line
    t1 = np.einsum("ij,ij->i", p1, unit)
    t2 = np.einsum("ij,ij->i", p2, unit)
    scale = np.maximum(np.hypot(p1[:, 0], p1[:, 1]), np.hypot(p2[:, 0], p2[:, 1]))
    radial = h <= _REL_TOL * np.maximum(scale, 1.0)

    # sides of the view disc: at distance R cos(π/N) with normals between the disc vertices
    n_sides = 4 * DEFAULT_QUAD_SEGS
    radius = view_distances[edge_owner]
    foot = p1 - t1[:, None] * unit
    normal_angle = np.mod(np.arctan2(foot[:, 1], foot[:, 0]), 2 * np.pi / n_sides)
    on_limit = (
        ~radial
        & (np.abs(h - radius * np.cos(np.pi / n_sides)) <= _REL_TOL * np.maximum(radius, 1.0) * 10)
        & (np.abs(normal_angle - np.pi / n_sides) <= 1e-6)
    )

    # closed-form moments of r(θ) per non-radial edge
    hs = np.where(radial, 1.0, h)
    theta = np.where(radial, 0.0, np.abs(np.arctan2(cross, np.einsum("ij,ij->i", p1, p2))))
    i1 = np.where(radial, 0.0, np.abs(_primitive_r1(hs, t2) - _primitive_r1(hs, t1)))
    i2 = np.where(radial, 0.0, np.abs(h * (t2 - t1)))
    i3 = np.where(radial, 0.0, np.abs(_primitive_r3(hs, t2) - _primitive_r3(hs, t1)))

    # nearest point of each edge to the observer
    t_near = np.clip(-t1, 0.0, length)
    nearest = p1 + t_near[:, None] * unit
    d_near = np.hypot(nearest[:, 0], nearest[:, 1])

    sums = pd.DataFrame(
        {
            "owner": edge_owner,
            "theta": theta,
            "i1": i1,
            "i2": i2,
            "i3": i3,
            "occl": np.where(radial, length, 0.0),
            "n_occl": radial.astype("int64"),
            "limit": np.where(on_limit, length, 0.0),
            "min_r": d_near,
        }
    ).groupby("owner")
    agg = sums.agg(
        {"theta": "sum", "i1": "sum", "i2": "sum", "i3": "sum", "occl": "sum", "n_occl": "sum", "limit": "sum"}
    )
    agg["min_r"] = sums["min_r"].min()
    max_r = pd.Series(np.hypot(rel[:, 0], rel[:, 1])).groupby(owner).max()

    idx = agg.index.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = agg["i1"] / agg["theta"]
        m2 = agg["i2"] / agg["theta"]
        m3 = agg["i3"] / agg["theta"]
        var = np.maximum(m2 - mean**2, 0.0)
        std = np.sqrt(var)
        skew = np.where(std > 0, (m3 - 3 * mean * var - mean**3) / std**3, 0.0)

    out.loc[idx, "occlusivity"] = agg["occl"].to_numpy()
    out.loc[idx, "n_occlusions"] = agg["n_occl"].to_numpy()
    out.loc[idx, "min_radial"] = agg["min_r"].to_numpy()
    out.loc[idx, "max_radial"] = max_r.loc[idx].to_numpy()
    out.loc[idx, "mean_radial"] = mean.to_numpy()
    out.loc[idx, "std_radial"] = std.to_numpy()
    out.loc[idx, "skew_radial"] = skew
    out.loc[idx, "open_share"] = (agg["limit"] / perimeter[idx]).to_numpy()

    centroids = shapely.get_coordinates(shapely.centroid(polygons[valid]))
    out.loc[valid, "drift"] = np.hypot(*(centroids - observers[valid]).T)
    out.loc[~valid, list(ISOVIST_METRICS)] = np.nan
    return out


def add_isovist_metrics(result: gpd.GeoDataFrame, points: gpd.GeoSeries, view_distances: np.ndarray) -> None:
    """Append the isovist metric columns to ``result`` (visibility polygons in a projected CRS)."""
    metrics = isovist_metrics(shapely.get_coordinates(points.values), result.geometry.values, view_distances)
    for column in ISOVIST_METRICS:
        result[column] = metrics[column].to_numpy()
