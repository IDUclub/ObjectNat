"""Bridge to the optional Rust core (``objectnat-core``) for visibility polygons.

Everything here degrades gracefully: when the core is not installed or is disabled through
``config.set_enable_rust_core(False)``, callers use the pure-Python implementations.
"""

import numpy as np
import shapely
from shapely import Point, Polygon

from objectnat import config

try:  # pragma: no cover - exercised only when the optional core is installed
    from objectnat_core import visibility_polygons as _visibility_polygons
except ImportError:  # pragma: no cover - exercised only when the core is missing
    _visibility_polygons = None

POLYGON_TYPE_ID = 3
LINEAR_TYPE_IDS = (1, 2)  # LineString, LinearRing
DEFAULT_QUAD_SEGS = 16  # shapely's default for Point.buffer()


def native_available() -> bool:
    """Return True when the Rust core is installed."""
    return _visibility_polygons is not None


def use_native() -> bool:
    """Return True when the Rust core is installed and enabled in the global config."""
    return native_available() and config.enable_rust_core


def _oriented_rings(polygons: np.ndarray) -> np.ndarray:
    """Rings of valid polygons oriented with the polygon interior on their left (exterior CCW, holes CW)."""
    if polygons.size == 0:
        return np.empty(0, dtype=object)
    rings, owner = shapely.get_rings(polygons, return_index=True)
    is_exterior = np.r_[True, owner[1:] != owner[:-1]]
    ccw = shapely.is_ccw(rings)
    flip = (is_exterior & ~ccw) | (~is_exterior & ccw)
    if flip.any():
        rings[flip] = shapely.reverse(rings[flip])
    return rings


def _lines_to_segments(lines: np.ndarray) -> np.ndarray:
    if lines.size == 0:
        return np.empty((0, 4), dtype="float64")
    coords, index = shapely.get_coordinates(lines, return_index=True)
    same_line = index[1:] == index[:-1]
    return np.hstack([coords[:-1][same_line], coords[1:][same_line]]).astype("float64", copy=False)


def geometries_to_segments(geometries) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert obstacle geometries to wall segments for the Rust core.

    Polygons contribute every ring (exterior and holes); lines contribute their own segments.
    Points and empty or missing geometries are ignored.

    Returns:
        ``(segments, one_sided)``: an ``(m, 4)`` float64 array ``x1, y1, x2, y2`` and an ``(m,)`` bool
        array. ``True`` marks walls of valid polygons, oriented with the interior on the left, which
        the core may skip when the observer is on their interior side (back-face culling). Walls of
        lines and of invalid polygons stay two-sided.
    """
    empty = np.empty((0, 4), dtype="float64"), np.empty(0, dtype=bool)
    geoms = np.asarray(geometries, dtype=object)
    if geoms.size == 0:
        return empty
    geoms = geoms[~shapely.is_missing(geoms)]
    geoms = geoms[~shapely.is_empty(geoms)]
    parts = shapely.get_parts(geoms)
    if parts.size == 0:
        return empty
    type_ids = shapely.get_type_id(parts)
    polygons = parts[type_ids == POLYGON_TYPE_ID]
    valid = shapely.is_valid(polygons)
    one_sided = _lines_to_segments(_oriented_rings(polygons[valid]))
    two_sided = _lines_to_segments(
        np.concatenate([shapely.get_rings(polygons[~valid]), parts[np.isin(type_ids, LINEAR_TYPE_IDS)]])
    )
    segments = np.vstack([one_sided, two_sided])
    flags = np.r_[np.ones(len(one_sided), dtype=bool), np.zeros(len(two_sided), dtype=bool)]
    return segments, flags


def native_visibility_polygons(
    points: np.ndarray,
    radii: np.ndarray,
    segments: np.ndarray,
    one_sided: np.ndarray | None = None,
    *,
    quad_segs: int = DEFAULT_QUAD_SEGS,
    n_threads: int | None = None,
) -> np.ndarray:
    """
    Compute visibility polygons for many observers with the Rust core.

    Args:
        points: ``(n, 2)`` observer coordinates in a projected CRS.
        radii: ``(n,)`` view radii in CRS units.
        segments: ``(m, 4)`` wall segments from :func:`geometries_to_segments`.
        one_sided: ``(m,)`` back-face culling flags from :func:`geometries_to_segments`.
        quad_segs: segments per quarter circle of the view disc (matches ``Point.buffer``).
        n_threads: worker threads, ``None`` for all cores.

    Returns:
        Object array of ``n`` shapely Polygons (empty Polygon when nothing is visible).
    """
    points = np.ascontiguousarray(points, dtype="float64").reshape(-1, 2)
    radii = np.ascontiguousarray(radii, dtype="float64").reshape(-1)
    segments = np.ascontiguousarray(segments, dtype="float64").reshape(-1, 4)
    if one_sided is not None:
        one_sided = np.ascontiguousarray(one_sided, dtype=bool).reshape(-1)
    coords, offsets = _visibility_polygons(points, radii, segments, int(quad_segs), n_threads, one_sided)
    has_ring = np.diff(offsets) > 0
    ring_offsets = np.concatenate([[0], offsets[1:][has_ring]]).astype("int64")
    geom_offsets = np.concatenate([[0], np.cumsum(has_ring)]).astype("int64")
    result = shapely.from_ragged_array(shapely.GeometryType.POLYGON, coords, (ring_offsets, geom_offsets))
    result = np.asarray(result, dtype=object)
    invalid = ~shapely.is_valid(result)
    if invalid.any():
        result[invalid] = [_largest_part_at(poly, Point(p)) for poly, p in zip(result[invalid], points[invalid])]
    return result


def _polygon_parts(geometry) -> list[Polygon]:
    """All non-empty polygons inside a geometry, flattening nested multi-parts and collections."""
    parts, stack = [], [geometry]
    while stack:
        geom = stack.pop()
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            parts.append(geom)
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            stack.extend(geom.geoms)
    return parts


def _largest_part_at(polygon, point: Point):
    """
    Repair a (rarely) invalid ring and keep the polygonal part around the observer.

    Near-coincident walls from different obstacles (e.g. a building and an earlier visibility
    polygon cut by the same reflection notch) can leave a micro bow-tie in the ring; the repair
    removes it without changing the area beyond floating-point noise.
    """
    parts = _polygon_parts(shapely.make_valid(polygon))
    if not parts:
        return Polygon()
    parts = _polygon_parts(shapely.union_all(parts))
    containing = [g for g in parts if g.distance(point) <= 1e-6]
    return max(containing or parts, key=lambda g: g.area)


def native_visibility_single(point_from: Point, obstacles_geometries, view_distance: float) -> Polygon:
    """Visibility polygon for one observer; drop-in for ``_visibility_accurate``."""
    segments, one_sided = geometries_to_segments(obstacles_geometries)
    poly = native_visibility_polygons(
        np.array([[point_from.x, point_from.y]]), np.array([float(view_distance)]), segments, one_sided, n_threads=1
    )[0]
    return poly
