# simplified version
import geopandas as gpd
import pandas as pd
from shapely.ops import polygonize, unary_union
from tqdm.auto import tqdm

from objectnat.methods.noise.noise_reduce import dist_to_target_db
from objectnat.methods.utils.geom_utils import (
    distribute_points_on_linestrings,
    distribute_points_on_polygons,
    polygons_to_multilinestring,
)
from objectnat.methods.visibility.visibility_analysis import get_visibility

MAX_DB_VALUE = 194


def calculate_simplified_noise_frame(
    noise_sources: gpd.GeoDataFrame, obstacles: gpd.GeoDataFrame, air_temperature, **kwargs
) -> gpd.GeoDataFrame:
    """
    Calculates a simplified environmental noise frame using static noise source geometries without simulating
    full sound wave propagation or reflections.

    This function provides a fast approximation of noise dispersion from a variety of source geometries, including
    points (e.g., traffic noise measurement points), lines (e.g., roads or railways), and polygons (e.g., industrial
    zones or buildings). Instead of simulating detailed wave interactions and reflections, it constructs an
    envelope of potential noise exposure by buffering the source geometry and applying simplified decay formulas
    based on sound power, frequency and temperature.

    Args:
        noise_sources (gpd.GeoDataFrame): A GeoDataFrame containing geometries of noise sources (Point, LineString,
            or Polygon). Each feature must have the following two columns:

            - 'source_noise_db': Initial sound level at the source, in decibels (dB).
            - 'geometric_mean_freq_hz': Characteristic sound frequency (Hz) used to model distance-based attenuation.

            Values in 'source_noise_db' must not exceed the physical maximum of 194 dB. Missing or NaN values in
            required fields will raise an error.

        obstacles (gpd.GeoDataFrame): A GeoDataFrame representing physical obstructions in the environment
            (e.g., buildings, walls, terrain). These are used to build visibility masks that affect where sound can
            propagate. Geometry will be simplified for performance using a default tolerance of 1 unit.

        air_temperature (float): The ambient air temperature in degrees Celsius. This value influences the
            attenuation model of sound in the atmosphere. Temperatures significantly outside the typical 0–30°C
            range may lead to inaccurate results.

    Keyword Args:
        target_noise_db (float, optional): The minimum sound level threshold (in dB) to be modeled. Any value below
            this threshold is considered insignificant and will be excluded from the resulting noise frame.
            Default is 40 dB.
        db_sim_step (float, optional): The simulation step size (in dB) used to discretize sound levels into
            spatial layers. Default is 5. Smaller values produce more detailed output but increase computation time.
        linestring_point_radius (float, optional): The spacing radius (in meters) used when converting LineString
            geometries into distributed point sources for simulation. Default is 30. Reducing this value improves
            detail along long lines.
        polygon_point_radius (float, optional): The point spacing (in meters) for distributing sources within
            Polygon geometries. Default is 15. Points are sampled across the polygon’s surface and perimeter to
            represent the full sound-emitting area.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame representing simplified noise distribution areas. The output geometries
            are polygons where each polygon is associated with the maximum sound level (in dB) present in that area,
            as derived from overlapping source zones. The resulting data is dissolved by noise level and returned in
            the original coordinate reference system (CRS) of the input sources.

    Notes:
        - The function does not model reflections or complex diffraction effects. It uses straight-line
          visibility (line-of-sight) and a layered distance-decay approach for rapid estimation.
        - Obstacles are used for visibility masking only, not as reflectors or absorbers.
        - Output resolution and accuracy depend heavily on the geometry type and point distribution settings.
        - Results are useful for quick noise mapping or for generating initial noise envelopes prior to more
          detailed simulations.
    """
    target_noise_db = kwargs.get("target_noise_db", 40)
    db_sim_step = kwargs.get("db_sim_step", 5)
    linestring_point_radius = kwargs.get("linestring_point_radius", 30)
    polygon_point_radius = kwargs.get("polygon_point_radius", 15)

    required_columns = ["source_noise_db", "geometric_mean_freq_hz"]
    for col in required_columns:
        if col not in noise_sources.columns:
            raise ValueError(f"'{col}' column is missing in provided GeoDataFrame")
        if noise_sources[col].isnull().any():
            raise ValueError(f"Column '{col}' contains missing (NaN) values")
    if (noise_sources["source_noise_db"] > MAX_DB_VALUE).any():
        raise ValueError(
            f"One or more values in 'source_noise_db' column exceed the physical limit of {MAX_DB_VALUE} dB."
        )
    original_crs = noise_sources.crs
    if len(obstacles) > 0:
        obstacles = obstacles.copy()
        obstacles.geometry = obstacles.geometry.simplify(tolerance=1)
        local_crs = obstacles.estimate_utm_crs()
        obstacles.to_crs(local_crs, inplace=True)
        noise_sources.to_crs(local_crs, inplace=True)
    else:
        local_crs = noise_sources.estimate_utm_crs()
        noise_sources.to_crs(local_crs, inplace=True)
        noise_sources.reset_index(drop=True)

    noise_sources = noise_sources.explode(ignore_index=True)
    noise_sources["geom_type"] = noise_sources.geom_type

    grouped_sources = noise_sources.groupby(by=["source_noise_db", "geometric_mean_freq_hz", "geom_type"])

    frame_result = []
    total_tasks = 0
    with tqdm(total=total_tasks, desc="Simulating noise") as pbar:
        for (source_db, freq_hz, geom_type), group_gdf in grouped_sources:
            # calculating layer dist and db values
            dist_db = [(0, source_db)]
            cur_db = source_db - db_sim_step
            max_dist = 0
            while cur_db > target_noise_db - db_sim_step:
                if cur_db - db_sim_step < target_noise_db:
                    cur_db = target_noise_db
                max_dist = dist_to_target_db(source_db, cur_db, freq_hz, air_temperature)
                dist_db.append((max_dist, cur_db))
                cur_db -= db_sim_step

            # increasing max_dist for extra view
            max_dist = max_dist * 1.2

            if geom_type == "Point":
                total_tasks += len(group_gdf)
                pbar.total = total_tasks
                pbar.refresh()
                for _, row in group_gdf.iterrows():
                    point_from = row.geometry
                    point_buffer = point_from.buffer(max_dist, resolution=16)
                    local_obstacles = obstacles[obstacles.intersects(point_buffer)]
                    vis_poly = get_visibility(point_from, obstacles=local_obstacles, view_distance=max_dist)
                    noise_from_feature = _eval_donuts_gdf(point_from, dist_db, local_crs, vis_poly)
                    frame_result.append(noise_from_feature)
                    pbar.update(1)

            elif geom_type == "LineString":
                layer_points = distribute_points_on_linestrings(
                    group_gdf, radius=linestring_point_radius, lloyd_relax_n=1
                )
                total_tasks += len(layer_points)
                pbar.total = total_tasks
                pbar.refresh()
                noise_from_feature = _process_lines_or_polygons(
                    group_gdf, max_dist, obstacles, layer_points, dist_db, local_crs, pbar
                )
                frame_result.append(noise_from_feature)
            elif geom_type == "Polygon":
                group_gdf.geometry = group_gdf.buffer(0.1, resolution=1)
                layer_points = distribute_points_on_polygons(
                    group_gdf, only_exterior=False, radius=polygon_point_radius, lloyd_relax_n=1
                )
                total_tasks += len(layer_points)
                pbar.total = total_tasks
                pbar.refresh()
                noise_from_feature = _process_lines_or_polygons(
                    group_gdf, max_dist, obstacles, layer_points, dist_db, local_crs, pbar
                )
                frame_result.append(noise_from_feature)
            else:
                pass

    noise_gdf = gpd.GeoDataFrame(pd.concat(frame_result, ignore_index=True), crs=local_crs)
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


def _process_lines_or_polygons(
    group_gdf, max_dist, obstacles, layer_points, dist_db, local_crs, pbar
) -> gpd.GeoDataFrame:
    features_vision_polys = []
    layer_buffer = group_gdf.buffer(max_dist, resolution=16).union_all()
    local_obstacles = obstacles[obstacles.intersects(layer_buffer)]
    for _, row in layer_points.iterrows():
        point_from = row.geometry
        vis_poly = get_visibility(point_from, obstacles=local_obstacles, view_distance=max_dist)
        features_vision_polys.append(vis_poly)
        pbar.update(1)
    features_vision_polys = unary_union(features_vision_polys)
    return _eval_donuts_gdf(group_gdf.union_all(), dist_db, local_crs, features_vision_polys)


def _eval_donuts_gdf(initial_geometry, dist_db, local_crs, clip_poly) -> gpd.GeoDataFrame:
    donuts = []
    don_values = []
    to_cut_off = initial_geometry
    for i in range(len(dist_db[:-1])):
        cur_buffer = initial_geometry.buffer(dist_db[i + 1][0])
        donuts.append(cur_buffer.difference(to_cut_off))
        don_values.append(dist_db[i][1])
        to_cut_off = cur_buffer
    noise_from_feature = (
        gpd.GeoDataFrame(geometry=donuts, data={"noise_level": don_values}, crs=local_crs)
        .clip(clip_poly, keep_geom_type=True)
        .explode(ignore_index=True)
    )
    return noise_from_feature
