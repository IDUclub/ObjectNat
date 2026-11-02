"""
Frozen copy of the pure-Python ``method="accurate"`` visibility from ObjectNat 2.1.0.

ObjectNat now computes accurate visibility in its Rust extension (``objectnat._rust``). This copy is kept
only so the benchmarks can keep measuring the speed-up and the agreement against the implementation it
replaced. It is not part of the package and is not maintained.
"""

# pylint: disable=duplicate-code
import math
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from objectnat.methods.utils.geom_utils import polygons_to_multilinestring


def explode_linestring(geometry: LineString) -> list[LineString]:
    """A function to return all segments of a linestring as a list of linestrings"""
    coords_ext = geometry.coords  # Create a list of all line node coordinates
    result = [LineString(part) for part in zip(coords_ext, coords_ext[1:])]
    return result


def point_side_of_line(line: LineString, point: Point) -> int:
    """A positive indicates the left-hand side a negative indicates the right-hand side"""
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    x, y = point.coords[0]
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if cross_product > 0:
        return 1
    return -1


def get_point_from_a_thorough_b(a: Point, b: Point, dist):
    """
    Func to get Point from point a thorough point b on dist
    """
    direction = math.atan2(b.y - a.y, b.x - a.x)
    c_x = a.x + dist * math.cos(direction)
    c_y = a.y + dist * math.sin(direction)
    return Point(c_x, c_y)


def legacy_visibility_accurate(point_from: Point, obstacles: gpd.GeoDataFrame, view_distance: float) -> Polygon:
    """
    Compute a high-accuracy visibility polygon for a single observer point.

    This function implements a detailed line-of-sight algorithm based on obstacle
    boundaries:

    * A circular buffer with radius ``view_distance`` is built around the
      observer point.
    * Obstacles intersecting this buffer are selected.
    * Polygonal obstacles are converted to line boundaries and exploded into
      individual segments.
    * For each nearest wall segment, a visibility wedge (sector) is constructed
      using angular relationships between wall endpoints and the observer.
    * All wedges are combined and subtracted from the initial visibility buffer
      together with obstacles to obtain the final visible area.

    Compared to ``_visibility_simple()``, this method is more accurate in
    complex environments (e.g., dense buildings, narrow streets), but it is also
    significantly slower.

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
            ``view_distance``, after subtracting obstacles. If the result is a
            MultiPolygon, the component intersecting the observer point is
            returned.

    Notes:
        It is assumed that coordinates are in a metric CRS (e.g., UTM). CRS
        management is done externally in ``get_visibility()``.
    """
    point_buffer = point_from.buffer(view_distance)
    sindex = obstacles.sindex
    idx = list(sindex.query(point_buffer, predicate="intersects"))
    if not idx:
        return point_buffer
    obstacles = obstacles.iloc[idx].reset_index(drop=True)

    buildings_lines_in_buffer = gpd.GeoSeries(
        pd.Series(
            obstacles.geometry.apply(polygons_to_multilinestring).explode(index_parts=False).apply(explode_linestring)
        ).explode()
    )

    buildings_lines_in_buffer = buildings_lines_in_buffer.loc[buildings_lines_in_buffer.intersects(point_buffer)]

    coords = [line.coords[0] for line in buildings_lines_in_buffer.geometry] + [
        line.coords[-1] for line in buildings_lines_in_buffer.geometry
    ]

    coords = np.asarray(coords, dtype="float64")
    dx = coords[:, 0] - point_from.x
    dy = coords[:, 1] - point_from.y
    max_dist_lines = float(np.sqrt(dx * dx + dy * dy).max())

    max_dist = max(view_distance, max_dist_lines)

    polygons = []
    buildings_lines_in_buffer = gpd.GeoDataFrame(geometry=buildings_lines_in_buffer, crs=obstacles.crs).reset_index()
    while not buildings_lines_in_buffer.empty:
        gdf_sindex = buildings_lines_in_buffer.sindex
        # TODO check if 2 walls are nearest and use the widest angle between points
        _, nearest_wall_sind = gdf_sindex.nearest(point_from, return_all=False, max_distance=max_dist)
        nearest_wall = buildings_lines_in_buffer.loc[nearest_wall_sind].iloc[0]
        wall_points = [Point(coords) for coords in nearest_wall.geometry.coords]

        # Calculate angles and sort by angle
        points_with_angle = sorted(
            [(pt, math.atan2(pt.y - point_from.y, pt.x - point_from.x)) for pt in wall_points], key=lambda x: x[1]
        )
        delta_angle = 2 * math.pi + points_with_angle[0][1] - points_with_angle[-1][1]
        if round(delta_angle, 10) == round(math.pi, 10):
            wall_b_centroid = obstacles.loc[nearest_wall["index"]].centroid
            p1 = get_point_from_a_thorough_b(point_from, points_with_angle[0][0], max_dist)
            p2 = get_point_from_a_thorough_b(point_from, points_with_angle[1][0], max_dist)
            polygon = LineString([p1, p2])
            polygon = polygon.buffer(
                distance=max_dist * point_side_of_line(polygon, wall_b_centroid), single_sided=True
            )
        else:
            if delta_angle > math.pi:
                delta_angle = 2 * math.pi - delta_angle
            a = math.sqrt((max_dist**2) * (1 + (math.tan(delta_angle / 2) ** 2)))
            p1 = get_point_from_a_thorough_b(point_from, points_with_angle[0][0], a)
            p2 = get_point_from_a_thorough_b(point_from, points_with_angle[-1][0], a)
            polygon = Polygon([points_with_angle[0][0], p1, p2, points_with_angle[1][0]])

        if not polygon.is_valid or polygon.area < 1:
            buildings_lines_in_buffer.drop(nearest_wall_sind, inplace=True)
            buildings_lines_in_buffer.reset_index(drop=True, inplace=True)
            continue
        polygons.append(polygon)
        candidate_idx = list(gdf_sindex.query(polygon, predicate="intersects"))
        if candidate_idx:
            candidates = buildings_lines_in_buffer.loc[candidate_idx]
            mask_within = candidates.within(polygon)
            to_drop = candidates.index[mask_within]
        else:
            to_drop = pd.Index([])
        to_drop = to_drop.append(pd.Index(nearest_wall_sind))
        buildings_lines_in_buffer.drop(index=to_drop, inplace=True)
        buildings_lines_in_buffer.reset_index(drop=True, inplace=True)
    vis_poly = point_buffer.difference(unary_union(polygons + obstacles.geometry.to_list()))

    if isinstance(vis_poly, MultiPolygon):
        vis_poly = list(vis_poly.geoms)
        for polygon in vis_poly:
            if polygon.intersects(point_from):
                vis_poly = polygon
                break
    return vis_poly


def _worker(args):
    point_geom, obstacles, view_distance = args
    if obstacles is None or len(obstacles) == 0:
        return point_geom.buffer(view_distance, quad_segs=32)
    if obstacles.contains(point_geom).any():
        return Polygon()
    return legacy_visibility_accurate(point_geom, obstacles, view_distance)


def legacy_get_visibility(
    point_from: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame,
    view_distance: float | None = None,
    method: str = "accurate",
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    **_,
) -> gpd.GeoDataFrame:
    """``get_visibility(method="accurate")`` as it worked in ObjectNat 2.1.0."""
    assert method == "accurate", "only the accurate method was replaced"
    original_crs = point_from.crs
    local_crs = point_from.estimate_utm_crs()
    points_local = point_from.to_crs(local_crs)
    if "visibility_distance" in points_local.columns:
        distances = points_local["visibility_distance"].to_list()
    else:
        distances = [view_distance] * len(points_local)
    obstacles_local = obstacles.to_crs(local_crs) if obstacles is not None and len(obstacles) else None
    if obstacles_local is not None:
        allowed = ["MultiPolygon", "Polygon", "LineString", "MultiLineString"]
        obstacles_local = obstacles_local[obstacles_local.geom_type.isin(allowed)]
    tasks = [(geom, obstacles_local, dist) for geom, dist in zip(points_local.geometry, distances)]
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_worker, tasks, chunksize=1))
    else:
        results = [_worker(t) for t in tasks]
    result = points_local.copy()
    result["geometry"] = results
    return result.set_crs(local_crs, allow_override=True).to_crs(original_crs)
