import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely import Point, Polygon

from objectnat import config
from objectnat.methods.visibility._native import (
    geometries_to_segments,
    native_visibility_polygons,
    native_visibility_single,
)

logger = config.logger

VISIBILITY_METHODS = ("accurate", "simple")
VIEW_DISC_QUAD_SEGS = 16  # matches Point.buffer(): the view polygon the kernel clips to


def dist_to_furthest_point(point_from: Point, view_polygon: Polygon) -> float:
    """
    Compute the maximum distance from an observer point to the boundary of a visibility polygon.

    The function measures the Euclidean distance from the observer point to all
    vertices of the polygon's exterior ring and returns the maximum distance,
    rounded to one decimal place.

    Args:
        point_from (Point):
            Observer location in the same CRS as ``view_polygon``.
        view_polygon (Polygon):
            Polygon representing the visibility area or any polygonal geometry
            around the observer.

    Returns:
        float:
            Maximum distance from ``point_from`` to the vertices of
            ``view_polygon``, rounded to one decimal place.
    """
    coords = np.asarray(view_polygon.exterior.coords, dtype="float64")
    dx = coords[:, 0] - point_from.x
    dy = coords[:, 1] - point_from.y
    res = float(np.sqrt(dx * dx + dy * dy).max())
    return round(res, 1)


def _visibility_accurate(point_from: Point, obstacles: gpd.GeoDataFrame, view_distance: float) -> Polygon:
    """
    Compute the exact visibility polygon for a single observer point.

    The visible area is the view disc (the same polygon ``point_from.buffer(view_distance)`` builds)
    minus everything hidden behind obstacle walls. It is computed by the Rust extension
    ``objectnat._rust``: obstacle walls are clipped to the disc, every wall endpoint and wall crossing
    becomes a critical direction, and one ray per angular interval finds the nearest wall, which gives
    the boundary exactly.

    Args:
        point_from (Point):
            Observer location in projected coordinates.
        obstacles (gpd.GeoDataFrame):
            GeoDataFrame with obstacle geometries (buildings, walls, etc.) in
            the same CRS as ``point_from``. Expected geometry types are
            Polygon, MultiPolygon, LineString or MultiLineString.
        view_distance (float):
            Maximum viewing radius around the observer, in units of the CRS.

    Returns:
        Polygon:
            Polygon representing the visible area around ``point_from`` within
            ``view_distance``.

    Notes:
        It is assumed that coordinates are in a metric CRS (e.g., UTM). CRS
        management is done externally in ``get_visibility()``.
    """
    point_buffer = point_from.buffer(view_distance)
    idx = obstacles.sindex.query(point_buffer, predicate="intersects")
    if len(idx) == 0:
        return point_buffer
    return native_visibility_single(point_from, obstacles.geometry.values[idx], view_distance)


def _visibility_accurate_batch(
    points: gpd.GeoSeries, obstacles: gpd.GeoDataFrame, distances: np.ndarray, max_workers: int | None
) -> np.ndarray:
    """
    Visibility polygons for many observers at once (multi-threaded in Rust).

    Observers inside an obstacle get an empty polygon. ``max_workers`` limits the number of threads
    (``None`` uses all cores).
    """
    xy = shapely.get_coordinates(points.values)
    segments, one_sided = geometries_to_segments(obstacles.geometry.values)
    results = native_visibility_polygons(xy, distances, segments, one_sided, n_threads=max_workers)
    inside, _ = obstacles.sindex.query(points.values, predicate="within")
    if len(inside):
        results[np.unique(inside)] = Polygon()
    return results


def get_visibility(
    point_from: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame,
    view_distance: float | None = None,
    method: Literal["accurate", "simple"] = "accurate",
    *,
    resolution: int = 32,  # pylint: disable=unused-argument
    parallel: bool = False,  # pylint: disable=unused-argument
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute visibility polygons for one or many observer points.

    Each observer gets the exact area visible within its viewing radius: the view disc minus everything
    hidden behind obstacle walls. Points and obstacles are reprojected to a locally estimated projected
    CRS (typically UTM) for the calculation, and observers are processed in parallel by ObjectNat's Rust
    extension.

    Args:
        point_from (gpd.GeoDataFrame):
            GeoDataFrame with point geometries representing observer locations.
            Must have a valid CRS. Any additional attributes are preserved in
            the output.
        obstacles (gpd.GeoDataFrame):
            GeoDataFrame with obstacle geometries (Polygon, MultiPolygon, LineString,
            MultiLineString) in any CRS. If empty or ``None``, no occlusion is
            applied and visibility is limited only by the viewing radius.
        view_distance (float | None, optional):
            Global viewing radius for all points (in metres). Used when the
            ``"visibility_distance"`` column is not present in ``point_from``.
            If ``None`` and the column is also missing, a ``ValueError`` is raised.
        method (Literal["accurate", "simple"], optional):
            Kept for compatibility. ``"simple"`` (a ray-based approximation) is
            deprecated: the exact algorithm is now faster than it was, so both
            values return the exact result.
        resolution (int, optional):
            Deprecated, ignored. Used to set the ray count of ``method="simple"``.
        parallel (bool, optional):
            Ignored: the calculation is always multi-threaded.
        max_workers (int | None, optional):
            Maximum number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the same index and attributes as ``point_from``,
            but with the geometry column replaced by visibility polygons. The
            result is returned in the original CRS of ``point_from``. Observers
            inside an obstacle get an empty polygon.

    Raises:
        TypeError:
            If ``point_from`` is not a GeoDataFrame.
        ValueError:
            If ``method`` is not ``"accurate"`` or ``"simple"``, if
            ``point_from`` is empty, has no CRS, or if neither
            ``view_distance`` nor the ``"visibility_distance"`` column is
            provided.

    Notes:
        If a column named ``"visibility_distance"`` is present in ``point_from``,
        its values are used as per-point view distances and the ``view_distance``
        argument is ignored.
    """
    if not isinstance(point_from, gpd.GeoDataFrame):
        raise TypeError("point_from must be a GeoDataFrame with a point geometry")

    if point_from.empty:
        raise ValueError("GeoDataFrame 'point_from' is empty.")

    if method not in VISIBILITY_METHODS:
        raise ValueError(f"method must be one of: {', '.join(repr(m) for m in VISIBILITY_METHODS)}; got {method!r}")
    if method == "simple":
        warnings.warn(
            "method='simple' is deprecated and computes the exact visibility; omit the argument.",
            DeprecationWarning,
            stacklevel=2,
        )

    original_crs = point_from.crs
    local_crs = point_from.estimate_utm_crs()
    points_local = point_from.to_crs(local_crs)

    if "visibility_distance" in points_local.columns:
        distances = points_local["visibility_distance"].to_numpy(dtype="float64")
    else:
        if view_distance is None:
            raise ValueError(
                "Either provide parameter view_distance or add column 'visibility_distance' to point_from GeoDataFrame."
            )
        distances = np.full(len(points_local), float(view_distance))

    obstacles_local = None
    if obstacles is not None and len(obstacles) > 0:
        obstacles_local = obstacles.to_crs(local_crs)
        allowed_geom_types = ["MultiPolygon", "Polygon", "LineString", "MultiLineString"]
        obstacles_local = obstacles_local[obstacles_local.geom_type.isin(allowed_geom_types)]

    if obstacles_local is None or obstacles_local.empty:
        results = shapely.buffer(points_local.geometry.values, distances, quad_segs=VIEW_DISC_QUAD_SEGS)
    else:
        results = _visibility_accurate_batch(points_local.geometry, obstacles_local, distances, max_workers)

    result_gdf = points_local.copy()
    result_gdf["geometry"] = results
    result_gdf.set_crs(local_crs, inplace=True)
    return result_gdf.to_crs(original_crs)
