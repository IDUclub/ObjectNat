import concurrent.futures
import math
import time
from collections import deque

import geopandas as gpd
import pandas as pd
from shapely import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union
from tqdm import tqdm

from objectnat import config
from objectnat.methods.noise.noise_reduce import dist_to_target_db, green_noise_reduce_db
from objectnat.methods.noise.noise_simulation_simplified import _eval_donuts_gdf
from objectnat.methods.utils.geom_utils import (
    gdf_to_circle_zones_from_point,
    get_point_from_a_thorough_b,
    polygons_to_multilinestring,
)
from objectnat.methods.visibility.visibility_analysis import _visibility_accurate, dist_to_furthest_point

logger = config.logger

MAX_DB_VALUE = 194
ANGLE_EPS = 1e-9
SOURCE_VERTEX_EPS = 1e-7


def simulate_noise(
    source_points: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame,
    source_noise_db: float = None,
    geometric_mean_freq_hz: float = None,
    **kwargs,
):
    """
    Simulates noise propagation from a set of source points considering obstacles, trees, and environmental factors.

    Args:
        source_points (gpd.GeoDataFrame):
            A GeoDataFrame with one or more point geometries representing noise sources.
            Optionally, it can include 'source_noise_db' and 'geometric_mean_freq_hz' columns for per-point simulation.
        obstacles (gpd.GeoDataFrame):
            A GeoDataFrame representing obstacles in the environment. If a column with sound absorption coefficients
            is present, its name should be provided in the `absorb_ratio_column` argument.
            Missing values will be filled with the `standart_absorb_ratio`.
        source_noise_db  (float, optional):
            Default noise level (dB) to use if not specified per-point. Decibels are logarithmic units used to measure
            sound intensity. A value of 20 dB represents a barely audible whisper, while 140 dB is comparable to the
            noise of jet engines.
        geometric_mean_freq_hz (float, optional):
            Default frequency (Hz) to use if not specified per-point. This parameter influences the sound wave's
            propagation and scattering in the presence of trees. Lower frequencies travel longer distances than higher
            frequencies. It's recommended to use values between 63 Hz and 8000 Hz; values outside this range will be
            clamped to the nearest boundary for the sound absorption coefficient calculation.

    Keyword Args:
        absorb_ratio_column (str, optional): The name of the column in the `obstacles` GeoDataFrame that contains the
            sound absorption coefficients for each obstacle. Default is None. If not specified, all obstacles will have
            the `standart_absorb_ratio`.
        standart_absorb_ratio (float, optional): The default sound absorption coefficient to use for obstacles without
            specified values in the `absorb_ratio_column`. Default is 0.05, which is a typical value for concrete walls.
        trees (gpd.GeoDataFrame, optional): A GeoDataFrame containing trees or dense vegetation along the sound wave's
            path. Trees will scatter and absorb sound waves.
        tree_resolution (int, optional): A resolution parameter for simulating tree interactions with sound waves.
            Recommended values are between 2 and 16, with higher values providing more accurate simulation results.
        air_temperature (float, optional): The air temperature in degrees Celsius. The recommended range is from 0 to
            30 degrees Celsius, as temperatures outside this range will be clipped. Temperature affects the sound
            propagation in the air.
        target_noise_db (float, optional): The target noise level (in dB) for the simulation. Default is 40 dB.
            Lower values may not be relevant for further analysis, as they are near the threshold of human hearing.
        db_sim_step (float, optional): The step size in decibels for the noise simulation. Default is 1. For more
            precise analysis, this can be adjusted. If the difference between `source_noise_db` and `target_noise_db`
            is not divisible by the step size, the function will raise an error.
        reflection_n (int, optional): The maximum number of reflections (bounces) to simulate for each sound wave.
            Recommended values are between 1 and 3. Larger values will result in longer simulation times.
        dead_area_r (float, optional): A debugging parameter that defines the radius of the "dead zone" for reflections.
            Points within this area will not generate reflections. This is useful to prevent the algorithm from getting
            stuck in corners or along building walls.
        source_position_buffer_r (float, optional): Radius in meters to cut from a containing obstacle or tree around
            the initial source point. Default is 0, which keeps source geometries unchanged.
        use_parallel (bool, optional): Whether to use ProcessPool for task distribution or not. Default is True.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the noise simulation results, including noise levels and geometries
            of the affected areas. Each point's simulation results will be merged into a single GeoDataFrame.
    """
    # Obstacles args
    absorb_ratio_column = kwargs.get("absorb_ratio_column", None)
    standart_absorb_ratio = kwargs.get("standart_absorb_ratio", 0.05)

    # Trees args
    trees = kwargs.get("trees", None)
    tree_res = kwargs.get("tree_resolution", 4)

    # Simulation conditions
    air_temperature = kwargs.get("air_temperature", 20)
    target_noise_db = kwargs.get("target_noise_db", 40)

    # Simulation params
    db_sim_step = kwargs.get("db_sim_step", 1)
    reflection_n = kwargs.get("reflection_n", 3)
    dead_area_r = kwargs.get("dead_area_r", 5)
    source_position_buffer_r = kwargs.get("source_position_buffer_r", 0)

    # Use paralleling
    use_parallel = kwargs.get("use_parallel", True)

    if source_position_buffer_r < 0:
        raise ValueError("source_position_buffer_r must be non-negative.")

    # Validate optional columns or default values
    use_column_db = False
    if "source_noise_db" in source_points.columns:
        if (source_points["source_noise_db"] > MAX_DB_VALUE).any():
            raise ValueError(
                f"One or more values in 'source_noise_db' column exceed the physical limit of {MAX_DB_VALUE} dB."
            )
        if source_points["source_noise_db"].isnull().any():
            raise ValueError("Column 'source_noise_db' contains missing (NaN) values")
        use_column_db = True

    use_column_freq = False
    if "geometric_mean_freq_hz" in source_points.columns:
        if source_points["geometric_mean_freq_hz"].isnull().any():
            raise ValueError("Column 'geometric_mean_freq_hz' contains missing (NaN) values")
        use_column_freq = True

    if not use_column_db:
        if source_noise_db is None:
            raise ValueError(
                "Either `source_noise_db` must be provided or the `source_points` must contain a 'source_noise_db' column."
            )
        if source_noise_db > MAX_DB_VALUE:
            raise ValueError(
                f"source_noise_db ({source_noise_db} dB) exceeds the physical limit of {MAX_DB_VALUE} dB in air."
            )

    if not use_column_freq:
        if geometric_mean_freq_hz is None:
            raise ValueError(
                "Either `geometric_mean_freq_hz` must be provided or the `source_points` must contain a 'geometric_mean_freq_hz' column."
            )
    if not use_column_db and not use_column_freq and len(source_points) > 1:
        logger.warning(
            "`source_noise_db` and `geometric_mean_freq_hz` will be used for all points. Per-point simulation parameters not found."
        )

    original_crs = source_points.crs
    source_points = source_points.copy()

    source_points = source_points.copy()
    if len(obstacles) > 0:
        obstacles = obstacles.copy()
        local_crs = obstacles.estimate_utm_crs()
        obstacles.to_crs(local_crs, inplace=True)
        obstacles.geometry = obstacles.geometry.simplify(tolerance=1)
        source_points.to_crs(local_crs, inplace=True)
    else:
        local_crs = source_points.estimate_utm_crs()
        source_points.to_crs(local_crs, inplace=True)
        source_points.reset_index(drop=True)
        source_points.geometry = source_points.centroid

    # Simplifying trees
    if trees is not None:
        trees = trees.copy()
        trees.to_crs(local_crs, inplace=True)
        trees.geometry = trees.geometry.simplify(tolerance=1)
    else:
        trees = gpd.GeoDataFrame()

    if absorb_ratio_column is None:
        obstacles["absorb_ratio"] = standart_absorb_ratio
    else:
        obstacles["absorb_ratio"] = obstacles[absorb_ratio_column].fillna(standart_absorb_ratio)
    obstacles = obstacles[["absorb_ratio", "geometry"]]

    # creating initial task and simulating for each point
    task_queue = deque()
    dead_area_dict = {}
    for ind, row in source_points.iterrows():
        source_point = row.geometry
        local_db = row["source_noise_db"] if use_column_db else source_noise_db
        local_freq = row["geometric_mean_freq_hz"] if use_column_freq else geometric_mean_freq_hz

        # calculating layer dist and db values
        dist_db = [(0, local_db)]
        cur_db = local_db - db_sim_step
        while cur_db > target_noise_db - db_sim_step:
            if cur_db - db_sim_step < target_noise_db:
                cur_db = target_noise_db
            max_dist = dist_to_target_db(local_db, cur_db, local_freq, air_temperature)
            dist_db.append((max_dist, cur_db))
            cur_db -= db_sim_step

        task_obstacles = _cut_source_buffer_from_containing_geometries(
            obstacles, source_point, source_position_buffer_r
        )
        task_trees = _cut_source_buffer_from_containing_geometries(trees, source_point, source_position_buffer_r)

        args = (source_point, task_obstacles, task_trees, 0, 0, dist_db)
        kwargs = {
            "reflection_n": reflection_n,
            "geometric_mean_freq_hz": local_freq,
            "tree_res": tree_res,
            "min_db": target_noise_db,
            "simulation_ind": ind,
        }
        task_queue.append((_noise_from_point_task, args, kwargs))
        dead_area_dict[ind] = source_point.buffer(dead_area_r, quad_segs=2)

    noise_gdf = _recursive_simulation_queue(
        task_queue, dead_area_dict=dead_area_dict, dead_area_r=dead_area_r, use_parallel=use_parallel
    )

    noise_gdf = gpd.GeoDataFrame(pd.concat(noise_gdf, ignore_index=True), crs=local_crs)
    polygons = gpd.GeoDataFrame(
        geometry=list(polygonize(noise_gdf.geometry.apply(polygons_to_multilinestring).union_all())), crs=local_crs
    )
    polygons_points = polygons.copy()
    polygons_points.geometry = polygons.representative_point()
    sim_result = polygons_points.sjoin(noise_gdf, predicate="within").reset_index()
    sim_result = sim_result.groupby("index").agg({"noise_level": "max"})
    sim_result["geometry"] = polygons
    sim_result = (
        gpd.GeoDataFrame(sim_result, geometry="geometry", crs=local_crs).dissolve(by="noise_level").reset_index()
    )

    return sim_result.to_crs(original_crs)


def _iter_polygonal_geometries(geometry):
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return [geometry]
    if isinstance(geometry, GeometryCollection):
        return [geom for geom in geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
    return []


def _cut_source_buffer_from_containing_geometries(
    geometries: gpd.GeoDataFrame, point_from: Point, source_position_buffer_r: float
) -> gpd.GeoDataFrame:
    if source_position_buffer_r <= 0 or len(geometries) == 0:
        return geometries

    source_buffer = point_from.buffer(source_position_buffer_r, quad_segs=8)
    candidate_positions = list(geometries.sindex.query(source_buffer, predicate="intersects"))
    if not candidate_positions:
        return geometries

    candidates = geometries.iloc[candidate_positions]
    containing_index = candidates.index[candidates.geometry.covers(point_from)]
    if containing_index.empty:
        return geometries

    result = geometries.copy()
    geometry_column = result.geometry.name
    result.loc[containing_index, geometry_column] = result.loc[containing_index, geometry_column].difference(
        source_buffer
    )
    result = result[result.geometry.notna()]
    result = result[~result.geometry.is_empty]
    return result


def _angle_from_point(point_from: Point, point_to: Point) -> float:
    angle = math.atan2(point_to.y - point_from.y, point_to.x - point_from.x)
    return angle if angle >= 0 else angle + 2 * math.pi


def _tree_angular_span(tree_geom: Polygon, point_from: Point) -> tuple[Point, Point, float] | None:
    points_with_angle = []
    for coords in tree_geom.exterior.coords:
        point = Point(coords)
        point_dist = point.distance(point_from)
        if point_dist <= SOURCE_VERTEX_EPS:
            continue
        points_with_angle.append((point, _angle_from_point(point_from, point), point_dist))

    if len(points_with_angle) < 2:
        return None

    points_with_angle.sort(key=lambda item: (item[1], item[2]))
    gaps = []
    for i, (_, angle, _) in enumerate(points_with_angle):
        next_i = (i + 1) % len(points_with_angle)
        next_angle = points_with_angle[next_i][1]
        if next_i == 0:
            next_angle += 2 * math.pi
        gaps.append((next_angle - angle, i))

    _, gap_start_i = max(gaps, key=lambda item: item[0])
    start_i = (gap_start_i + 1) % len(points_with_angle)
    end_i = gap_start_i
    start_point, start_angle, _ = points_with_angle[start_i]
    end_point, end_angle, _ = points_with_angle[end_i]
    delta_angle = (end_angle - start_angle) % (2 * math.pi)
    if delta_angle <= ANGLE_EPS:
        return None
    return start_point, end_point, delta_angle


def _build_tree_reduce_polygon(tree_geom: Polygon, point_from: Point, dist: float, vis_poly: Polygon):
    angular_span = _tree_angular_span(tree_geom, point_from)
    if angular_span is None:
        return None, None

    start_point, end_point, delta_angle = angular_span
    if delta_angle >= math.pi:
        projection_dist = dist * 2.1
    else:
        projection_dist = math.sqrt((dist**2) * (1 + (math.tan(delta_angle / 2) ** 2))) * 1.05

    p1 = get_point_from_a_thorough_b(point_from, start_point, projection_dist)
    p2 = get_point_from_a_thorough_b(point_from, end_point, projection_dist)
    shadow_sector = Polygon([start_point, p1, p2, end_point])
    if not shadow_sector.is_valid:
        shadow_sector = shadow_sector.buffer(0)
    if shadow_sector.is_empty:
        return None, None

    red_polygon = unary_union([shadow_sector.intersection(vis_poly), tree_geom])
    polygonal_geometries = _iter_polygonal_geometries(red_polygon)
    if not polygonal_geometries:
        return None, None

    if isinstance(red_polygon, GeometryCollection):
        red_polygon = max(polygonal_geometries, key=lambda geom: geom.area)
    if isinstance(red_polygon, MultiPolygon):
        red_polygon = red_polygon.buffer(0.1, quad_segs=1).buffer(-0.1, quad_segs=1)
    if isinstance(red_polygon, MultiPolygon):
        red_polygon = max(red_polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(red_polygon, Polygon) or red_polygon.is_empty:
        return None, None

    return Polygon(red_polygon.exterior), delta_angle


def _append_layered_reduce_polygons(
    reduce_polygons, red_polygon, tree_geom, point_from, dist_values, geometric_mean_freq_hz, r_tree
):
    if r_tree <= 0 or red_polygon.is_empty:
        return

    max_layer_dist = max((dist for dist, _ in dist_values), default=0)
    tree_entry_dist = point_from.distance(tree_geom)
    if max_layer_dist <= 0 or tree_entry_dist >= max_layer_dist:
        return

    tree_exit_dist = tree_entry_dist + r_tree
    split_distances = {0, max_layer_dist, min(tree_entry_dist, max_layer_dist), min(tree_exit_dist, max_layer_dist)}
    split_distances.update(dist for dist, _ in dist_values)
    split_distances = sorted({round(dist, 6) for dist in split_distances if 0 <= dist <= max_layer_dist})

    prev_dist = split_distances[0]
    prev_buffer = point_from.buffer(prev_dist)
    for cur_dist in split_distances[1:]:
        if cur_dist <= prev_dist:
            continue

        cur_buffer = point_from.buffer(cur_dist)
        layer_mid_dist = (prev_dist + cur_dist) / 2
        layer_tree_depth = min(max(layer_mid_dist - tree_entry_dist, 0), r_tree)
        if layer_tree_depth > 0:
            noise_reduce = green_noise_reduce_db(geometric_mean_freq_hz, layer_tree_depth)
            reduce_geometry = red_polygon.intersection(cur_buffer.difference(prev_buffer))
            for geom in _iter_polygonal_geometries(reduce_geometry):
                if not geom.is_empty and geom.area >= 0.01:
                    reduce_polygons.append((geom, noise_reduce))

        prev_dist = cur_dist
        prev_buffer = cur_buffer


def _noise_from_point_task(task, **kwargs) -> tuple[gpd.GeoDataFrame, list[tuple] | None]:
    # Unpacking task
    point_from, obstacles, trees_orig, passed_dist, deep, dist_db = task

    def donuts_dist_values(dist_db, passed_dist, max_view_dist):
        new_dist_db = dist_db + [(passed_dist, None), (max_view_dist + passed_dist, None)]
        new_dist_db = sorted(new_dist_db, key=lambda x: x[0])
        start = None
        end = None
        for i, (dist, db) in enumerate(new_dist_db[:-1]):
            if db is None:
                if start is None:
                    new_dist_db[i] = (dist, new_dist_db[i - 1][1])
                    start = i
                else:
                    new_dist_db[i] = (dist, new_dist_db[i + 1][1])
                    end = i + 1
                    break
        return [(dist - passed_dist, db) for dist, db in new_dist_db[start:end]]

    max_dist = max(dist_db, key=lambda x: x[0])[0]
    min_db = kwargs.get("min_db")
    reflection_n = kwargs.get("reflection_n")
    geometric_mean_freq_hz = kwargs.get("geometric_mean_freq_hz")
    tree_res = kwargs.get("tree_res")
    local_crs = obstacles.crs
    dist = round(max_dist - passed_dist, 1)

    obstacles = obstacles[obstacles.intersects(point_from.buffer(dist, quad_segs=8))]

    if len(obstacles) == 0:
        obstacles_union = Polygon()
    else:
        obstacles_union = obstacles.union_all()

    vis_poly = _visibility_accurate(point_from, obstacles, dist)
    max_view_dist = dist_to_furthest_point(point_from, vis_poly)
    donuts_dist_values = donuts_dist_values(dist_db, passed_dist, max_view_dist)

    allowed_geom_types = ["MultiPolygon", "Polygon"]

    # Trees noise reduce
    reduce_polygons = []
    if len(trees_orig) > 0:
        trees_orig = trees_orig[trees_orig.intersects(point_from.buffer(dist, quad_segs=8))]
        if len(trees_orig) > 0:
            try:
                trees = gdf_to_circle_zones_from_point(trees_orig, point_from, dist, resolution=tree_res)
                trees = trees.clip(vis_poly, keep_geom_type=True).explode(index_parts=False)
            except TypeError:
                trees = gpd.GeoDataFrame()

            for _, row in trees.iterrows():
                tree_geom = row.geometry
                if tree_geom.area < 1:
                    continue
                dist_to_centroid = tree_geom.centroid.distance(point_from)
                red_polygon, delta_angle = _build_tree_reduce_polygon(tree_geom, point_from, dist, vis_poly)
                if red_polygon is not None:
                    sin_half_angle = math.sin(delta_angle / 2)
                    if dist_to_centroid <= 0 or sin_half_angle <= 0:
                        continue
                    r_tree_new = tree_geom.area / (2 * dist_to_centroid * sin_half_angle)
                    _append_layered_reduce_polygons(
                        reduce_polygons,
                        red_polygon,
                        tree_geom,
                        point_from,
                        donuts_dist_values,
                        geometric_mean_freq_hz,
                        r_tree_new,
                    )

    noise_from_point = _eval_donuts_gdf(point_from, donuts_dist_values, local_crs, vis_poly)
    # intersect noise poly with noise reduce
    if len(reduce_polygons) > 0:
        reduce_polygons = gpd.GeoDataFrame(
            reduce_polygons, columns=["geometry", "reduce"], geometry="geometry", crs=local_crs
        )

        all_lines = (
            reduce_polygons.geometry.apply(polygons_to_multilinestring).tolist()
            + noise_from_point.geometry.apply(polygons_to_multilinestring).tolist()
        )

        cutted_polygons = gpd.GeoDataFrame(geometry=list(polygonize(unary_union(all_lines))), crs=local_crs)

        cutted_polygons_points = cutted_polygons.copy()
        cutted_polygons_points.geometry = cutted_polygons.representative_point()

        joined = (
            cutted_polygons_points.sjoin(noise_from_point, predicate="within", how="left")
            .drop(columns="index_right")
            .sjoin(reduce_polygons, predicate="within", how="left")
            .drop(columns="index_right")
        )
        joined.geometry = cutted_polygons.geometry
        joined = (
            joined.reset_index().groupby("index").agg({"geometry": "first", "reduce": "sum", "noise_level": "first"})
        )
        joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=local_crs)
        noise_from_point = joined.copy()

        noise_from_point = noise_from_point.dropna(subset=["noise_level"])

        noise_from_point["reduce"] = noise_from_point["reduce"].fillna(0)
        noise_from_point["noise_level"] = noise_from_point["noise_level"] - noise_from_point["reduce"]
    else:
        noise_from_point["reduce"] = 0
    noise_from_point = noise_from_point[noise_from_point.geom_type.isin(allowed_geom_types)]
    noise_from_point = noise_from_point[noise_from_point["noise_level"] >= min_db]
    noise_from_point = noise_from_point[noise_from_point.area >= 0.01]
    if deep == reflection_n:
        return noise_from_point, None

    if isinstance(vis_poly, Polygon):
        vis_poly_points = [Point(coords) for coords in vis_poly.exterior.coords]
    else:
        vis_poly_points = [Point(coords) for geom in vis_poly.geoms for coords in geom.exterior.coords]
    vis_poly_points = gpd.GeoDataFrame(geometry=vis_poly_points, crs=local_crs)

    # Generating reflection points
    vis_poly_points["point"] = vis_poly_points["geometry"].copy()
    vis_poly_points.geometry = vis_poly_points.geometry.buffer(1, resolution=1)
    vis_poly_points = vis_poly_points.sjoin(obstacles, predicate="intersects").drop(columns="index_right")
    vis_poly_points = vis_poly_points[~vis_poly_points.index.duplicated(keep="first")]
    vis_poly_points.dropna(subset=["absorb_ratio"], inplace=True)
    nearby_poly = point_from.buffer(1.1, quad_segs=2)
    try:
        vis_poly_points.geometry = (
            vis_poly_points.difference(vis_poly).difference(obstacles_union).difference(nearby_poly)
        )
    except GEOSException:
        return noise_from_point, None
    vis_poly_points = vis_poly_points[~vis_poly_points.is_empty]
    vis_poly_points = vis_poly_points[vis_poly_points.area >= 0.01]
    vis_poly_points["geometry"] = vis_poly_points["point"]
    vis_poly_points["dist"] = vis_poly_points.distance(point_from)
    vis_poly_points = vis_poly_points[vis_poly_points["dist"] < max_dist - 5]
    vis_poly_points = vis_poly_points.sjoin(noise_from_point, predicate="intersects", how="left")

    if len(vis_poly_points) == 0:
        return noise_from_point, None

    new_obs = pd.concat([obstacles, gpd.GeoDataFrame(geometry=[vis_poly], crs=local_crs)], ignore_index=True)

    # Creating new reflection tasks
    new_tasks = []
    for _, loc in vis_poly_points.iterrows():
        if not isinstance(loc.geometry, Point):
            continue
        new_passed_dist = round(loc.dist + passed_dist, 2)
        dist_last = max_dist - new_passed_dist
        if dist_last > 1:
            db_change = loc["reduce"]
            dist_change = loc["absorb_ratio"] * dist_last
            new_dist_db = [(dist - dist_change, db - db_change) for dist, db in dist_db]
            task_obs = new_obs.copy()
            task_obs.geometry = task_obs.difference(loc.geometry.buffer(1, quad_segs=1))
            new_tasks.append(
                (
                    _noise_from_point_task,
                    (loc.geometry, task_obs, trees_orig, new_passed_dist, deep + 1, new_dist_db),
                    kwargs,
                )
            )

    return noise_from_point, new_tasks


def _recursive_simulation_queue(task_queue: deque, dead_area_dict: dict, dead_area_r: int, use_parallel: bool):
    results = []
    total_tasks = len(task_queue)

    with tqdm(total=total_tasks, desc="Simulating noise") as pbar:
        if use_parallel:
            executor_class = concurrent.futures.ProcessPoolExecutor()
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor()
        with executor_class as executor:
            max_workers = executor._max_workers  # pylint: disable=protected-access
            future_to_task = {}
            while True:
                while task_queue and len(future_to_task) < max_workers:
                    func, task, kwargs = task_queue.popleft()
                    future = executor.submit(func, task, **kwargs)
                    future_to_task[future] = kwargs["simulation_ind"]
                done, _ = concurrent.futures.wait(future_to_task.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    simulation_ind = future_to_task.pop(future)
                    result, new_tasks = future.result()
                    if new_tasks:
                        new_tasks_n = 0
                        local_dead_area = dead_area_dict.get(simulation_ind)
                        new_dead_area_points = [local_dead_area]
                        for func, new_task, new_kwargs in new_tasks:
                            new_point = new_task[0]
                            if not local_dead_area.covers(new_point):
                                task_queue.append((func, new_task, new_kwargs))
                                new_dead_area_points.append(new_point.buffer(dead_area_r, quad_segs=2))
                                new_tasks_n += 1
                        dead_area_dict[simulation_ind] = unary_union(new_dead_area_points)
                        total_tasks += new_tasks_n
                        pbar.total = total_tasks
                        pbar.refresh()
                    results.append(result)
                    pbar.update(1)
                time.sleep(0.01)
                if not future_to_task and not task_queue:
                    break
    return results
