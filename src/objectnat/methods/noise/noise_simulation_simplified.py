# simplified version
import geopandas as gpd
import pandas as pd
from shapely.ops import polygonize, unary_union

from objectnat.methods.noise.noise_reduce import dist_to_target_db
from objectnat.methods.utils.geom_utils import (
    distribute_points_on_linestrings,
    distribute_points_on_polygons,
    polygons_to_multilinestring,
)
from objectnat.methods.visibility.visibility_analysis import get_visibility

MAX_DB_VALUE = 194


def calculate_simplified_noise_frame(
    noise_sources: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame,
    air_temperature,
    *,
    target_noise_db: int = 40,
    db_sim_step: int = 5,
    linestring_point_radius: int = 15,
    polygon_point_radius: int = 5,
    visibility_parallel: bool = True,
    visibility_max_workers: int | None = None,
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
        noise_sources (gpd.GeoDataFrame):
            A GeoDataFrame containing geometries of noise sources (Point, LineString,
            or Polygon). Each feature must have the following two columns:

            - 'source_noise_db': Initial sound level at the source, in decibels (dB).
            - 'geometric_mean_freq_hz': Characteristic sound frequency (Hz) used to model distance-based attenuation.

            Values in 'source_noise_db' must not exceed the physical maximum of 194 dB. Missing or NaN values in
            required fields will raise an error.

        obstacles (gpd.GeoDataFrame):
            A GeoDataFrame representing physical obstructions in the environment
            (e.g., buildings, walls, terrain). These are used to build visibility masks that affect where sound can
            propagate. Geometry will be simplified for performance using a default tolerance of 1 unit.

        air_temperature (float):
            The ambient air temperature in degrees Celsius. This value influences the
            attenuation model of sound in the atmosphere. Temperatures significantly outside the typical 0–30°C
            range may lead to inaccurate results.

        target_noise_db (float, default=40):
            The minimum sound level threshold (in dB) to be modeled. Any value below
            this threshold is considered insignificant and will be excluded from the resulting noise frame.
            Default is 40 dB.

        db_sim_step (float, default=5):
            The simulation step size (in dB) used to discretize sound levels into
            spatial layers. Default is 5. Smaller values produce more detailed output but increase computation time.

        linestring_point_radius (float,  default=15):
            The spacing radius (in meters) used when converting LineString
            geometries into distributed point sources for simulation. Default is 30. Reducing this value improves
            detail along long lines.

        polygon_point_radius (float,  default=5):
            The point spacing (in meters) for distributing sources within Polygon geometries.
            Default is 15. Points are sampled across the polygon’s surface and perimeter to
            represent the full sound-emitting area.

        visibility_parallel (bool, optional):
            If True, visibility polygons for all distributed sample points are computed in parallel using
            multiprocessing. Recommended when the number of sample points is large. Default is True.

        visibility_max_workers (int | None, optional):
            Maximum number of parallel worker processes for visibility computation. None uses the system default.

    Returns:
        gpd.GeoDataFrame:
            A GeoDataFrame representing simplified noise distribution areas. The output geometries
            are polygons where each polygon is associated with the maximum sound level (in dB) present in that area,
            as derived from overlapping source zones. The resulting data is dissolved by noise level and returned in
            the original coordinate reference system (CRS) of the input sources.

    Notes:
        - The function does not model reflections or complex diffraction effects. It uses straight-line
          visibility (line-of-sight) and a layered distance-decay approach for rapid estimation.
        - Obstacles are used for visibility masking only, not as reflectors or absorbers.
        - Output resolution and accuracy depend heavily on the geometry type and point distribution settings.
        - Parallel visibility significantly improves performance for line and
          polygon sources with many sample points.
        - Results are useful for quick noise mapping or for generating initial noise envelopes prior to more
          detailed simulations.
    """

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

    frame_result: list[gpd.GeoDataFrame] = []
    groups_meta: dict[int, dict] = {}  # group_id -> {geom_type, dist_db, view_radius, union_geometry}
    vis_points_records: list[dict] = []
    group_id_counter = 0

    for (source_db, freq_hz, geom_type), group_gdf in grouped_sources:
        dist_db: list[tuple[float, float]] = [(0, source_db)]
        cur_db = source_db - db_sim_step
        max_dist = 0.0
        while cur_db > target_noise_db - db_sim_step:
            if cur_db - db_sim_step < target_noise_db:
                cur_db = target_noise_db
            max_dist = dist_to_target_db(source_db, cur_db, freq_hz, air_temperature)
            dist_db.append((max_dist, cur_db))
            cur_db -= db_sim_step

        view_radius = max_dist * 1.2

        gid = group_id_counter
        group_id_counter += 1

        meta = {
            "group_id": gid,
            "geom_type": geom_type,
            "dist_db": dist_db,
            "view_radius": view_radius,
            "union_geometry": None,
        }

        if geom_type == "Point":
            for idx, row in group_gdf.iterrows():
                vis_points_records.append(
                    {
                        "geometry": row.geometry,
                        "group_id": gid,
                        "mode": "point",
                        "source_index": idx,
                        "visibility_distance": view_radius,
                    }
                )

        elif geom_type == "LineString":
            layer_points = distribute_points_on_linestrings(group_gdf, radius=linestring_point_radius, lloyd_relax_n=1)
            for _, row in layer_points.iterrows():
                vis_points_records.append(
                    {
                        "geometry": row.geometry,
                        "group_id": gid,
                        "mode": "line",
                        "visibility_distance": view_radius,
                    }
                )
            meta["union_geometry"] = group_gdf.union_all()

        elif geom_type == "Polygon":
            group_gdf.geometry = group_gdf.buffer(0.1, resolution=1)
            layer_points = distribute_points_on_polygons(
                group_gdf, only_exterior=False, radius=polygon_point_radius, lloyd_relax_n=1
            )
            for _, row in layer_points.iterrows():
                vis_points_records.append(
                    {
                        "geometry": row.geometry,
                        "group_id": gid,
                        "mode": "polygon",
                        "visibility_distance": view_radius,
                    }
                )
            meta["union_geometry"] = group_gdf.union_all()
        else:
            continue

        groups_meta[gid] = meta

    if not vis_points_records:
        return gpd.GeoDataFrame(columns=["noise_level", "geometry"])

    all_points_gdf = gpd.GeoDataFrame(vis_points_records, geometry="geometry", crs=local_crs)

    vis_gdf = get_visibility(
        point_from=all_points_gdf,
        obstacles=obstacles,
        view_distance=None,
        method="accurate",
        parallel=visibility_parallel,
        max_workers=visibility_max_workers,
    )

    vision_polys_by_group: dict[int, list] = {
        gid: [] for gid, meta in groups_meta.items() if meta["geom_type"] in ("LineString", "Polygon")
    }

    for rec, vis_row in zip(vis_points_records, vis_gdf.itertuples()):
        gid = rec["group_id"]
        meta = groups_meta[gid]
        vis_poly = vis_row.geometry

        if meta["geom_type"] == "Point":
            point_from = rec["geometry"]
            noise_from_feature = _eval_donuts_gdf(point_from, meta["dist_db"], local_crs, vis_poly)
            frame_result.append(noise_from_feature)
        else:
            vision_polys_by_group[gid].append(vis_poly)

    for gid, meta in groups_meta.items():
        if meta["geom_type"] not in ("LineString", "Polygon"):
            continue

        polys = vision_polys_by_group.get(gid, [])
        if not polys:
            continue

        features_vision_polys = unary_union(polys)
        initial_geometry = meta["union_geometry"]
        noise_from_feature = _eval_donuts_gdf(initial_geometry, meta["dist_db"], local_crs, features_vision_polys)
        frame_result.append(noise_from_feature)

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
