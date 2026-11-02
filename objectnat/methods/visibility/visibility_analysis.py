from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely import LineString, Point, Polygon
from tqdm.contrib.concurrent import process_map

from objectnat import config
from objectnat.methods.visibility._native import (
    geometries_to_segments,
    native_visibility_polygons,
    native_visibility_single,
)

logger = config.logger
enable_tqdm = config.enable_tqdm_bar

VISIBILITY_METHODS = ("accurate", "simple")


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


def _visibility_simple(
    point_from: Point, obstacles: gpd.GeoDataFrame, view_distance: float, resolution: int
) -> Polygon:
    """
    Compute a fast, approximate visibility polygon for a single observer point.

    This function provides a simplified line-of-sight estimate using radial
    rays:

    * A circular buffer with radius ``view_distance`` is created around the
      observer.
    * The buffer is discretized into multiple directions using the
      ``quad_segs`` parameter (``resolution``).
    * For each direction, a line segment is drawn from the observer to the
      buffer boundary.
    * These lines are cut by the union of obstacles, removing occluded parts.
    * The endpoints of the remaining visible segments form an approximate
      visibility contour.

    Compared to ``_visibility_accurate()``, this method is less precise,
    especially in highly complex or detailed urban scenes.

    Args:
        point_from (Point):
            Observer location in projected coordinates.
        obstacles (gpd.GeoDataFrame):
            GeoDataFrame with obstacle geometries in the same CRS as
            ``point_from``.
        view_distance (float):
            Maximum viewing radius around the observer, in units of the CRS.
        resolution (int):
            Angular resolution of the buffer. Passed as ``quad_segs`` to
            ``Point.buffer()``. Higher values produce smoother and more
            detailed visibility contours but increase computation time.

    Returns:
        Polygon:
            Approximate visibility polygon from ``point_from`` within
            ``view_distance``, clipped by ``obstacles``.
    """
    point_buffer = point_from.buffer(view_distance, quad_segs=resolution)
    sindex = obstacles.sindex
    idx = list(sindex.query(point_buffer, predicate="intersects"))
    if not idx:
        return point_buffer
    obstacles = obstacles.iloc[idx].reset_index(drop=True)

    buffer_exterior_ = list(point_buffer.exterior.coords)
    line_geometry = [LineString([point_from, ext]) for ext in buffer_exterior_]
    buffer_lines_gdf = gpd.GeoDataFrame(geometry=line_geometry)
    united_buildings = obstacles.union_all()
    if united_buildings:
        splited_lines = buffer_lines_gdf["geometry"].apply(lambda x: x.difference(united_buildings))
    else:
        splited_lines = buffer_lines_gdf["geometry"]

    splited_lines_gdf = gpd.GeoDataFrame(geometry=splited_lines).explode(index_parts=True)
    splited_lines_list = []

    for _, v in splited_lines_gdf.groupby(level=0):
        splited_lines_list.append(v.iloc[0]["geometry"].coords[-1])
    circuit = Polygon(splited_lines_list)
    if united_buildings:
        circuit = circuit.difference(united_buildings)

    return circuit


def _visibility_worker(args: tuple[Point, gpd.GeoDataFrame | None, float, str, int]) -> Polygon:
    """
    Worker function for computing visibility for a single observer point.

    This helper is designed to be used with process-based parallelization
    (e.g., ``ProcessPoolExecutor`` or ``tqdm.contrib.concurrent.process_map``).
    It unpacks the argument tuple and dispatches to either the accurate or
    simple visibility algorithm.

    Args:
        args (tuple[Point, gpd.GeoDataFrame | None, float, str, int]):
            A 5-tuple containing:

            * point_geom (Point): Observer location in local projected CRS.
            * obstacles (gpd.GeoDataFrame | None): Obstacles in the same CRS.
              If ``None`` or empty, no occlusion is applied.
            * view_distance (float): Viewing radius for this particular point.
            * method (str): Visibility algorithm to use. Must be either
              ``"accurate"`` or ``"simple"``.
            * resolution (int): Resolution parameter passed to the simple
              method (ignored for the accurate method).

    Returns:
        Polygon:
            Visibility polygon for the given observer and parameters. If there
            are no obstacles, this is a circular-like buffer. If the observer
            lies inside an obstacle, an empty polygon is returned.
    """
    point_geom, obstacles, view_distance, method, resolution = args

    if obstacles is None or len(obstacles) == 0:
        point_buffer = point_geom.buffer(
            view_distance,
            quad_segs=(32 if method == "accurate" else resolution),
        )
        return point_buffer

    if obstacles.contains(point_geom).any():
        return Polygon()

    if method == "accurate":
        return _visibility_accurate(point_geom, obstacles, view_distance)
    if method == "simple":
        return _visibility_simple(point_geom, obstacles, view_distance, resolution)
    raise ValueError(f"method must be one of: {', '.join(repr(m) for m in VISIBILITY_METHODS)}; got {method!r}")


def _visibility_accurate_batch(
    points: gpd.GeoSeries, obstacles: gpd.GeoDataFrame, distances: list[float], max_workers: int | None
) -> np.ndarray:
    """
    Batch version of ``_visibility_accurate`` for many observers at once (multi-threaded in Rust).

    Matches ``_visibility_worker`` semantics: observers inside an obstacle get an empty polygon.
    ``max_workers`` limits the number of threads (``None`` uses all cores).
    """
    xy = shapely.get_coordinates(points.values)
    segments, one_sided = geometries_to_segments(obstacles.geometry.values)
    radii = np.asarray(distances, dtype="float64")
    results = native_visibility_polygons(xy, radii, segments, one_sided, n_threads=max_workers)
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
    resolution: int = 32,
    parallel: bool = False,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute visibility polygons for one or many observer points.

    This is a high-level, batch interface to the visibility analysis:

    * Accepts a GeoDataFrame of observer points.
    * Reprojects points and obstacles to a locally estimated projected CRS
      (typically UTM) for distance-accurate calculations.
    * For each point, computes a visibility polygon using either:
        * an accurate wall-based algorithm (``method="accurate"``), or
        * a simple ray-based approximation (``method="simple"``).
    * Supports per-point visibility radii via a dedicated column or a global
      ``view_distance`` value.
    * ``method="accurate"`` always runs multi-threaded in the Rust extension;
      ``method="simple"`` can be parallelized across processes and can display
      a tqdm progress bar.

    Args:
        point_from (gpd.GeoDataFrame):
            GeoDataFrame with point geometries representing observer locations.
            Must have a valid CRS. Any additional attributes are preserved in
            the output.
        obstacles (gpd.GeoDataFrame):
            GeoDataFrame with obstacle geometries in the same CRS as
            ``point_from``. If empty or ``None``, no occlusion is applied and
            visibility is limited only by the viewing radius.
        view_distance (float | None, optional):
            Global viewing radius for all points (in units of the CRS). Used
            when the ``"visibility_distance"`` column is not present in
            ``point_from``. If ``None`` and the column is also missing, a
            ``ValueError`` is raised.
        method (Literal["accurate", "simple"], optional):
            Visibility algorithm to use:

            * ``"accurate"`` – exact visible area, based on obstacle walls,
              computed in the Rust extension.
            * ``"simple"`` – faster approximation with radial rays and
              obstacle cutting.

        resolution (int, optional):
            Resolution parameter for the simple method. Passed as ``quad_segs``
            to ``Point.buffer()``; ignored when ``method="accurate"``.
        parallel (bool, optional):
            For ``method="simple"``: if ``True``, compute visibility polygons in
            parallel using processes; if ``False``, process points sequentially.
            ``method="accurate"`` is always multi-threaded and ignores it.
        max_workers (int | None, optional):
            Maximum number of worker processes for ``method="simple"`` with
            ``parallel=True`` (``None``: the ``ProcessPoolExecutor`` default), or
            maximum number of threads for ``method="accurate"`` (``None``: all cores).

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the same index and attributes as ``point_from``,
            but with the geometry column replaced by visibility polygons. The
            result is returned in the original CRS of ``point_from``.

    Raises:
        TypeError:
            If ``point_from`` is not a GeoDataFrame.
        ValueError:
            If ``method`` is not ``"accurate"`` or ``"simple"``, if
            ``point_from`` is empty, has no CRS, or if neither
            ``view_distance`` nor the ``"visibility_distance"`` column is
            provided.

    Notes:
        * If a column named ``"visibility_distance"`` is present in
          ``point_from``, its values are used as per-point view distances and
          the ``view_distance`` argument is ignored.
        * When ``parallel=True`` with ``method="simple"`` and ``enable_tqdm`` is
          True in the global config, a progress bar is displayed using
          ``tqdm.contrib.concurrent.process_map`` during parallel execution.

    Differences between methods
    ---------------------------
    * ``method="accurate"``:
        Exact visible area from obstacle walls, computed in Rust. Precise in
        dense environments and around corners, and the faster of the two.
    * ``method="simple"``:
        Uses radial rays and line splitting in Python. An approximation whose
        detail depends on ``resolution``.
    """
    if not isinstance(point_from, gpd.GeoDataFrame):
        raise TypeError("point_from must be a GeoDataFrame with a point geometry")

    if point_from.empty:
        raise ValueError("GeoDataFrame 'point_from' is empty.")

    if method not in VISIBILITY_METHODS:
        raise ValueError(f"method must be one of: {', '.join(repr(m) for m in VISIBILITY_METHODS)}; got {method!r}")

    original_crs = point_from.crs
    local_crs = point_from.estimate_utm_crs()

    points_local = point_from.to_crs(local_crs)

    if "visibility_distance" in points_local.columns:
        distances = points_local["visibility_distance"].to_list()
    else:
        if view_distance is None:
            raise ValueError(
                "Either provide parameter view_distance or add column 'visibility_distance' to point_from GeoDataFrame."
            )
        distances = [view_distance] * len(points_local)

    if obstacles is not None and len(obstacles) > 0:
        obstacles_local = obstacles.to_crs(local_crs)
        allowed_geom_types = ["MultiPolygon", "Polygon", "LineString", "MultiLineString"]
        obstacles_local = obstacles_local[obstacles_local.geom_type.isin(allowed_geom_types)]
    else:
        obstacles_local = None

    if method == "accurate" and obstacles_local is not None and len(obstacles_local) > 0:
        results = _visibility_accurate_batch(points_local.geometry, obstacles_local, distances, max_workers)
        result_gdf = points_local.copy()
        result_gdf["geometry"] = results
        result_gdf.set_crs(local_crs, inplace=True)
        return result_gdf.to_crs(original_crs)

    tasks = [(geom, obstacles_local, dist, method, resolution) for geom, dist in zip(points_local.geometry, distances)]

    if not parallel:
        results = [_visibility_worker(t) for t in tasks]
    else:
        if enable_tqdm:
            results = process_map(
                _visibility_worker,
                tasks,
                max_workers=max_workers,
                chunksize=1,
                desc="Visibility",
            )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_visibility_worker, tasks))
        logger.info("done parallel")

    result_gdf = points_local.copy()
    result_gdf["geometry"] = results
    result_gdf.set_crs(local_crs, inplace=True)
    result_gdf = result_gdf.to_crs(original_crs)

    return result_gdf
