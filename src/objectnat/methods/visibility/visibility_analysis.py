import math
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from tqdm.contrib.concurrent import process_map

from objectnat import config
from objectnat.methods.utils.geom_utils import (
    explode_linestring,
    get_point_from_a_thorough_b,
    point_side_of_line,
    polygons_to_multilinestring,
)

logger = config.logger
enable_tqdm = config.enable_tqdm_bar


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
    try:
        coords = np.asarray(view_polygon.exterior.coords, dtype="float64")
        dx = coords[:, 0] - point_from.x
        dy = coords[:, 1] - point_from.y
        res = float(np.sqrt(dx * dx + dy * dy).max())
        return round(res, 1)
    except Exception as e:
        print(view_polygon)
        raise e


def _visibility_accurate(point_from: Point, obstacles: gpd.GeoDataFrame, view_distance: float) -> Polygon:
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
    logger.debug("Calculation vis polygon")
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
    logger.debug("Done calculating!")
    vis_poly = point_buffer.difference(unary_union(polygons + obstacles.geometry.to_list()))

    if isinstance(vis_poly, MultiPolygon):
        vis_poly = list(vis_poly.geoms)
        for polygon in vis_poly:
            if polygon.intersects(point_from):
                vis_poly = polygon
                break
    return vis_poly


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

    Compared to ``_visibility_accurate()``, this method is much faster but
    less precise, especially in highly complex or detailed urban scenes.

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
    elif method == "simple":
        return _visibility_simple(point_geom, obstacles, view_distance, resolution)
    else:
        raise ValueError("method must be one of: 'accurate', 'fast'")


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
    * Optionally parallelizes computation across points using processes and
      can display a tqdm progress bar.

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

            * ``"accurate"`` – slower but more precise, based on obstacle
              boundaries and visibility wedges.
            * ``"simple"`` – faster approximation with radial rays and
              obstacle cutting.

        resolution (int, optional):
            Resolution parameter for the simple method. Passed as ``quad_segs``
            to ``Point.buffer()``; ignored when ``method="accurate"``.
        parallel (bool, optional):
            If ``True``, compute visibility polygons for multiple points in
            parallel using processes. If ``False``, process points
            sequentially in the current process.
        max_workers (int | None, optional):
            Maximum number of worker processes when ``parallel=True``. If
            ``None``, the default from ``ProcessPoolExecutor`` is used.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the same index and attributes as ``point_from``,
            but with the geometry column replaced by visibility polygons. The
            result is returned in the original CRS of ``point_from``.

    Raises:
        TypeError:
            If ``point_from`` is not a GeoDataFrame.
        ValueError:
            If ``point_from`` is empty, has no CRS, or neither
            ``view_distance`` nor the ``"visibility_distance"`` column is
            provided.

    Notes:
        * If a column named ``"visibility_distance"`` is present in
          ``point_from``, its values are used as per-point view distances and
          the ``view_distance`` argument is ignored.
        * When ``parallel=True`` and ``enable_tqdm`` is True in the global
          config, a progress bar is displayed using
          ``tqdm.contrib.concurrent.process_map`` during parallel execution.

    Differences between methods
    ---------------------------
    * ``method="accurate"``:
        Uses obstacle boundaries and angular visibility wedges. More precise,
        especially in dense environments and around corners, but slower.
    * ``method="simple"``:
        Uses radial rays and line splitting. Much faster and suitable for
        large batches or rough estimates, but may produce less detailed
        visibility shapes.
    """
    if not isinstance(point_from, gpd.GeoDataFrame):
        raise TypeError("point_from must be a GeoDataFrame with a point geometry")

    if point_from.empty:
        raise ValueError("GeoDataFrame 'point_from' is empty.")

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
    else:
        obstacles_local = None

    tasks = [(geom, obstacles_local, dist, method, resolution) for geom, dist in zip(points_local.geometry, distances)]
    logger.info("started")
    if not parallel:
        results = [_visibility_worker(t) for t in tasks]
        logger.info("done seq ")
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


# def get_visibilities_from_points(
#     points: gpd.GeoDataFrame,
#     obstacles: gpd.GeoDataFrame,
#     view_distance: int,
#     sectors_n=None,
#     max_workers: int = cpu_count(),
# ) -> list[Polygon]:
#     """
#     Calculate visibility polygons from a set of points considering obstacles within a specified view distance.
#
#     Args:
#         points (gpd.GeoDataFrame):
#             GeoDataFrame containing the points from which visibility is calculated.
#         obstacles (gpd.GeoDataFrame):
#             GeoDataFrame containing the obstacles that block visibility.
#         view_distance (int):
#             The maximum distance from each point within which visibility is calculated.
#         sectors_n (int, optional):
#             Number of sectors to divide the view into for more detailed visibility calculations. Defaults to None.
#         max_workers (int, optional):
#             Maximum workers in multiproccesing, multipocessing.cpu_count() by default.
#
#     Returns:
#         (list[Polygon]):
#             A list of visibility polygons for each input point.
#
#     Notes:
#         This function uses `get_visibility_accurate()` in multiprocessing way.
#
#     """
#     if points.crs != obstacles.crs:
#         raise ValueError(f"CRS mismatch, points crs:{points.crs} != obstacles crs:{obstacles.crs}")
#     if points.crs.is_geographic:
#         logger.warning("Points crs is geographic, it may produce invalid results")
#     # remove points inside polygons
#     joined = gpd.sjoin(points, obstacles, how="left", predicate="intersects")
#     points = joined[joined.index_right.isnull()]
#
#     # remove unused obstacles
#     points_view = points.geometry.buffer(view_distance).union_all()
#     s = obstacles.intersects(points_view)
#     buildings_in_buffer = obstacles.loc[s[s].index].reset_index(drop=True)
#
#     buildings_in_buffer.geometry = buildings_in_buffer.geometry.apply(
#         lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
#     )
#     args = [(point, buildings_in_buffer, view_distance, sectors_n) for point in points.geometry]
#     all_visions = process_map(
#         _multiprocess_get_vis,
#         args,
#         chunksize=5,
#         desc="Calculating Visibility Catchment Area from each Point, it might take a while for a "
#         "big amount of points",
#         max_workers=max_workers,
#     )
#
#     # could return sectorized visions if sectors_n is set
#     return all_visions


# def calculate_visibility_catchment_area(
#     points: gpd.GeoDataFrame, obstacles: gpd.GeoDataFrame, view_distance: int | float, max_workers: int = cpu_count()
# ) -> gpd.GeoDataFrame:  # pragma: no cover
#     """
#     Calculate visibility catchment areas for a large urban area based on given points and obstacles.
#     This function is designed to work with at least 1000 points spaced 10-20 meters apart for optimal results.
#     Points can be generated using a road graph.
#
#     Args:
#         points (gpd.GeoDataFrame): GeoDataFrame containing the points from which visibility is calculated.
#         obstacles (gpd.GeoDataFrame): GeoDataFrame containing the obstacles that block visibility.
#         view_distance (int | float): The maximum distance from each point within which visibility is calculated.
#         max_workers (int): Maximum workers in multiproccesing, multipocessing.cpu_count() by default.
#
#     Returns:
#         (gpd.GeoDataFrame): GeoDataFrame containing the calculated visibility catchment areas.
#     """
#
#     def filter_geoms(x):
#         if x.geom_type == "GeometryCollection":
#             return MultiPolygon([y for y in x.geoms if y.geom_type in ["Polygon", "MultiPolygon"]])
#         return x
#
#     def calc_group_factor(x):
#         return np.mean(x.new_ratio) * x.count_n
#
#     def unary_union_groups(x):
#         return unary_union(MultiPolygon(list(x["geometry"])).buffer(0))
#
#     raise NotImplementedError("This method is temporarily unsupported.")
#
#     local_crs = obstacles.estimate_utm_crs()
#     obstacles = obstacles.to_crs(local_crs)
#     points = points.to_crs(local_crs)
#
#     sectors_n = 12
#     logger.info("Calculating Visibility Catchment Area from each point")
#     all_visions_sectorized = get_visibilities_from_points(points, obstacles, view_distance, sectors_n, max_workers)
#     all_visions_sectorized = gpd.GeoDataFrame(
#         geometry=[item for sublist in all_visions_sectorized for item in sublist], crs=local_crs
#     )
#     logger.info("Calculating non-vision part...")
#     all_visions_unary = all_visions_sectorized.union_all()
#     convex = all_visions_unary.convex_hull
#     dif = convex.difference(all_visions_unary)
#
#     del convex, all_visions_unary
#
#     buf_area = (math.pi * view_distance**2) / sectors_n
#     all_visions_sectorized["ratio"] = all_visions_sectorized.area / buf_area
#     all_visions_sectorized["ratio"] = min_max_normalization(
#         all_visions_sectorized["ratio"].values, new_min=1, new_max=10
#     )
#     groups = all_visions_sectorized.sample(frac=1).groupby(all_visions_sectorized.index // 6000)
#     groups = [group for _, group in groups]
#
#     del all_visions_sectorized
#
#     groups_result = process_map(
#         _process_group,
#         groups,
#         desc="Counting intersections in each group...",
#         max_workers=max_workers,
#     )
#     logger.info("Calculating all groups intersection...")
#     all_in = combine_geometry(gpd.GeoDataFrame(data=pd.concat(groups_result), geometry="geometry", crs=local_crs))
#
#     del groups_result
#
#     all_in["count_n"] = all_in["index_right"].apply(len)
#
#     logger.info("Calculating intersection's parameters")
#     # all_in["factor"] = all_in.parallel_apply(calc_group_factor, axis=1) # TODO replace pandarallel methods
#     threshold = all_in["factor"].quantile(0.3)
#     all_in = all_in[all_in["factor"] > threshold]
#
#     all_in["factor_normalized"] = np.round(
#         min_max_normalization(np.sqrt(all_in["factor"].values), new_min=1, new_max=5)
#     ).astype(int)
#     logger.info("Calculating normalized groups geometry...")
#     all_in = (
#         all_in.groupby("factor_normalized").parallel_apply(unary_union_groups).reset_index()
#     )  # TODO replace pandarallel methods
#     all_in = gpd.GeoDataFrame(data=all_in.rename(columns={0: "geometry"}), geometry="geometry", crs=32636)
#
#     all_in = all_in.explode(index_parts=True).reset_index(drop=True)
#     all_in["area"] = all_in.area
#     threshold = all_in["area"].quantile(0.9)
#     all_in = all_in[all_in["area"] > threshold]
#     all_in = all_in.groupby("factor_normalized").apply(unary_union_groups).reset_index()
#     all_in = gpd.GeoDataFrame(data=all_in.rename(columns={0: "geometry"}), geometry="geometry", crs=32636)
#
#     all_in.geometry = all_in.geometry.buffer(20).buffer(-20).difference(dif)
#
#     all_in.sort_values(by="factor_normalized", ascending=False, inplace=True)
#     all_in.reset_index(drop=True, inplace=True)
#     logger.info("Smoothing normalized groups geometry...")
#     for ind, row in all_in.iloc[:-1].iterrows():
#         for ind2 in range(ind + 1, len(all_in)):
#             current_geometry = all_in.at[ind2, "geometry"]
#             all_in.at[ind2, "geometry"] = current_geometry.difference(row.geometry)
#             all_in["geometry"] = all_in["geometry"].apply(filter_geoms)
#
#     all_in = all_in.explode(index_parts=True)
#     logger.info("Done!")
#     return all_in
#
#
# def _multiprocess_get_vis(args):  # pragma: no cover
#     point, buildings, view_distance, sectors_n = args
#     result = get_visibility_accurate(point, buildings, view_distance)
#
#     if sectors_n is not None:
#         sectors = []
#
#         cx, cy = point.x, point.y
#
#         angle_increment = 2 * math.pi / sectors_n
#         view_distance = math.sqrt((view_distance**2) * (1 + (math.tan(angle_increment / 2) ** 2)))
#         for i in range(sectors_n):
#             angle1 = i * angle_increment
#             angle2 = (i + 1) * angle_increment
#
#             x1, y1 = cx + view_distance * math.cos(angle1), cy + view_distance * math.sin(angle1)
#             x2, y2 = cx + view_distance * math.cos(angle2), cy + view_distance * math.sin(angle2)
#
#             sector_triangle = Polygon([point, (x1, y1), (x2, y2)])
#             sector = result.intersection(sector_triangle)
#
#             if not sector.is_empty:
#                 sectors.append(sector)
#         result = sectors
#     return result
#
#
# def _process_group(group):  # pragma: no cover
#     geom = group
#     combined_geometry = combine_geometry(geom)
#     combined_geometry.drop(columns=["index", "index_right"], inplace=True)
#     combined_geometry["count_n"] = combined_geometry["ratio"].apply(len)
#     combined_geometry["new_ratio"] = combined_geometry.apply(
#         lambda x: np.power(np.prod(x.ratio), 1 / x.count_n) * x.count_n, axis=1
#     )
#
#     threshold = combined_geometry["new_ratio"].quantile(0.25)
#     combined_geometry = combined_geometry[combined_geometry["new_ratio"] > threshold]
#
#     combined_geometry["new_ratio_normalized"] = min_max_normalization(
#         combined_geometry["new_ratio"].values, new_min=1, new_max=10
#     )
#
#     combined_geometry["new_ratio_normalized"] = np.round(combined_geometry["new_ratio_normalized"]).astype(int)
#
#     result_union = (
#         combined_geometry.groupby("new_ratio_normalized")
#         .agg({"geometry": lambda x: unary_union(MultiPolygon(list(x)).buffer(0))})
#         .reset_index(drop=True)
#     )
#     result_union.set_geometry("geometry", inplace=True)
#     result_union.set_crs(geom.crs, inplace=True)
#
#     result_union = result_union.explode("geometry", index_parts=False).reset_index(drop=True)
#
#     representative_points = combined_geometry.copy()
#     representative_points["geometry"] = representative_points["geometry"].representative_point()
#
#     joined = gpd.sjoin(result_union, representative_points, how="inner", predicate="contains").reset_index()
#     joined = joined.groupby("index").agg({"geometry": "first", "new_ratio": lambda x: np.mean(list(x))})
#
#     joined.set_geometry("geometry", inplace=True)
#     joined.set_crs(geom.crs, inplace=True)
#     return joined
