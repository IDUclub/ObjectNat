import math
import os
from collections import deque

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from shapely import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from objectnat import calculate_simplified_noise_frame, config, simulate_noise
from objectnat.methods.noise import get_air_resist_ratio
from objectnat.methods.noise.noise_simulation import (
    _angle_from_point,
    _cut_source_buffer_from_containing_geometries,
    _iter_polygonal_geometries,
    _recursive_simulation_queue,
    _tree_angular_span,
)
from tests.conftest import output_dir

logger = config.logger


def _queued_noise_task(task, simulation_ind):
    point = task[0]
    if point.x == 0:
        next_task = (_queued_noise_task, (Point(10, 0),), {"simulation_ind": simulation_ind})
        return "first", [next_task]
    return "second", None


def test_recursive_simulation_queue_adds_reflection_tasks():
    source = Point(0, 0)
    tasks = deque([(_queued_noise_task, (source,), {"simulation_ind": 0})])
    dead_areas = {0: source.buffer(1)}

    results = _recursive_simulation_queue(tasks, dead_areas, dead_area_r=1, use_parallel=False)

    assert results == ["first", "second"]
    assert not tasks
    assert dead_areas[0].covers(Point(10, 0))


def test_iter_polygonal_geometries():
    poly = box(0, 0, 1, 1)
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    collection = GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 1)])])

    assert _iter_polygonal_geometries(poly) == [poly]
    assert _iter_polygonal_geometries(multi) == [multi]
    # Only the polygonal member of the collection survives.
    assert _iter_polygonal_geometries(collection) == [box(0, 0, 1, 1)]
    # Non-polygonal geometry yields nothing.
    assert _iter_polygonal_geometries(LineString([(0, 0), (1, 1)])) == []


def test_angle_from_point_cardinal_directions():
    origin = Point(0, 0)
    assert _angle_from_point(origin, Point(1, 0)) == pytest.approx(0.0)
    assert _angle_from_point(origin, Point(0, 1)) == pytest.approx(math.pi / 2)
    assert _angle_from_point(origin, Point(-1, 0)) == pytest.approx(math.pi)
    # Due south normalizes into [0, 2*pi) rather than staying negative.
    assert _angle_from_point(origin, Point(0, -1)) == pytest.approx(3 * math.pi / 2)


def test_source_position_buffer_cuts_containing_geometry():
    source_point = Point(0, 0)
    geometries = gpd.GeoDataFrame(geometry=[box(-5, -5, 5, 5), box(10, 10, 12, 12)], crs=3857)

    result = _cut_source_buffer_from_containing_geometries(geometries, source_point, 1)

    assert len(result) == len(geometries)
    assert not result.iloc[0].geometry.covers(source_point)
    assert result.iloc[1].geometry.equals(geometries.iloc[1].geometry)


def test_source_position_buffer_non_positive_radius_is_noop():
    source_point = Point(0, 0)
    geometries = gpd.GeoDataFrame(geometry=[box(-5, -5, 5, 5)], crs=3857)
    # radius <= 0 short-circuits and returns the same object untouched.
    assert _cut_source_buffer_from_containing_geometries(geometries, source_point, 0) is geometries


def test_source_position_buffer_no_containing_geometry_is_noop():
    source_point = Point(100, 100)
    geometries = gpd.GeoDataFrame(geometry=[box(-5, -5, 5, 5)], crs=3857)
    # Source is not inside any geometry, so nothing is cut.
    result = _cut_source_buffer_from_containing_geometries(geometries, source_point, 1)
    assert result.iloc[0].geometry.equals(geometries.iloc[0].geometry)


def test_tree_angular_span_degenerate_returns_none():
    source_point = Point(0, 0)
    # All exterior vertices sit within SOURCE_VERTEX_EPS of the source -> fewer than 2 usable points.
    tiny_tree = Polygon([(1e-9, 0), (2e-9, 0), (1e-9, 1e-9)])
    assert _tree_angular_span(tiny_tree, source_point) is None


def test_tree_angular_span_handles_zero_angle_seam():
    source_point = Point(0, 0)
    tree_sector = Polygon([source_point, Point(10, -1), Point(20, 0), Point(10, 1), source_point])

    span = _tree_angular_span(tree_sector, source_point)

    assert span is not None
    start_point, end_point, delta_angle = span
    assert start_point.distance(source_point) > 0
    assert end_point.distance(source_point) > 0
    assert delta_angle == pytest.approx(2 * math.atan(0.1))


def test_source_position_buffer_rejects_negative_radius(gdf_1point, buildings_data):
    with pytest.raises(ValueError, match="source_position_buffer_r must be non-negative"):
        simulate_noise(
            source_points=gdf_1point,
            obstacles=buildings_data,
            source_noise_db=90,
            geometric_mean_freq_hz=2000,
            source_position_buffer_r=-1,
        )


def test_basic_functionality(gdf_1point, buildings_data, trees_data):
    gdf_1point = gdf_1point.to_crs(4326)
    source_noise_db = 90
    target_noise_db = 40
    reflection_n = 2
    geometric_mean_freq_hz = 2000
    result = simulate_noise(
        source_points=gdf_1point,
        obstacles=buildings_data,
        source_noise_db=source_noise_db,
        geometric_mean_freq_hz=2000,
        standart_absorb_ratio=0.05,
        trees=trees_data,
        tree_resolution=8,
        air_temperature=20,
        target_noise_db=target_noise_db,
        db_sim_step=1,
        reflection_n=reflection_n,
        dead_area_r=5,
        use_parallel=False,
    )

    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty
    assert "geometry" in result.columns
    assert "noise_level" in result.columns
    assert result["noise_level"].max() <= source_noise_db
    assert result["noise_level"].min() >= target_noise_db
    assert result.crs == gdf_1point.crs

    plot_simulation_result(
        result_gdf=result,
        source_points=gdf_1point,
        buildings=buildings_data,
        trees=trees_data,
        source_db_range=(source_noise_db, target_noise_db),
        reflection_n=reflection_n,
        frequency_desc=geometric_mean_freq_hz,
        output_filename="noise_simulation_1point",
    )


def test_multiple_sources(buildings_data, trees_data):
    p1 = Point(30.27060176, 59.93546846)
    p2 = Point(30.27303864, 59.9362777)
    p3 = Point(30.26804078, 59.93474246)
    gdf = gpd.GeoDataFrame(
        {
            "source_noise_db": [85, 90, 95],
            "geometric_mean_freq_hz": [500, 1000, 2000],
            "geometry": [p1, p2, p3],
        },
        crs=4326,
    )

    target_noise_db = 50
    reflection_n = 2
    result = simulate_noise(
        source_points=gdf,
        obstacles=buildings_data,
        standart_absorb_ratio=0.05,
        trees=trees_data,
        tree_resolution=4,
        air_temperature=20,
        target_noise_db=target_noise_db,
        db_sim_step=1,
        reflection_n=reflection_n,
        dead_area_r=5,
    )

    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty
    assert "geometry" in result.columns
    assert "noise_level" in result.columns
    assert result["noise_level"].max() <= gdf["source_noise_db"].max()
    assert result["noise_level"].min() >= 40
    assert result.crs == gdf.crs

    plot_simulation_result(
        result_gdf=result,
        source_points=gdf,
        buildings=buildings_data,
        trees=trees_data,
        source_db_range=(gdf["source_noise_db"].max(), target_noise_db),
        reflection_n=reflection_n,
        frequency_desc="Mixed",
        output_filename="noise_simulation_3points",
    )


def test_wrong_db_value(gdf_1point, buildings_data):
    gdf_1point = gdf_1point.to_crs(4326)
    with pytest.raises(ValueError) as _:
        _ = simulate_noise(
            source_points=gdf_1point,
            obstacles=buildings_data,
            source_noise_db=350,
            geometric_mean_freq_hz=2000,
            db_sim_step=4,
        )


def test_out_of_range_values(gdf_1point, buildings_data):
    out_of_range_hz = 10
    out_of_range_temperature = 40
    in_middle_hz = 1500
    in_middle_temperature = 15
    res = get_air_resist_ratio(out_of_range_temperature, out_of_range_hz, True)
    logger.info(f"Out of range result: {res}")
    res = get_air_resist_ratio(in_middle_temperature, in_middle_hz, True)
    logger.info(f"Between values result: {res}")
    res = get_air_resist_ratio(in_middle_temperature, 2000, True)
    logger.info(f"Between values result: {res}")
    res = get_air_resist_ratio(10, in_middle_hz, True)
    logger.info(f"Between values result: {res}")


@pytest.mark.network
def test_noise_frame_calculator(gdf_1point, buildings_data, intermodal_osm_1114252_edges_gdf):
    local_crs = buildings_data.estimate_utm_crs()
    buildings_data = buildings_data.to_crs(local_crs)
    intermodal_osm_1114252_edges_gdf = intermodal_osm_1114252_edges_gdf[
        ~intermodal_osm_1114252_edges_gdf["type"].isin(["walk", "boarding"])
    ]
    intermodal_osm_1114252_edges_gdf = intermodal_osm_1114252_edges_gdf.to_crs(local_crs)
    gdf_1point = gdf_1point.to_crs(local_crs)

    minx, miny, maxx, maxy = gdf_1point.buffer(250).total_bounds
    buffer_territory = box(minx, miny, maxx, maxy)

    builds_in_buffer = buildings_data.clip(buffer_territory, keep_geom_type=True)

    sample_size = max(1, int(0.02 * len(builds_in_buffer)))
    # Случайная выборка
    sampled_buildings = builds_in_buffer.sample(n=sample_size, random_state=42)
    sampled_buildings["source_noise_db"] = 90

    graph_in_buffer = intermodal_osm_1114252_edges_gdf.clip(buffer_territory, keep_geom_type=True)
    graph_in_buffer["source_noise_db"] = 75

    gdf_1point["source_noise_db"] = 95
    noise_sources = pd.concat([gdf_1point, graph_in_buffer, sampled_buildings], ignore_index=True)
    noise_sources["geometric_mean_freq_hz"] = 2000
    noise_frame = calculate_simplified_noise_frame(noise_sources, buildings_data, 20)
    assert isinstance(noise_frame, gpd.GeoDataFrame)
    assert not noise_frame.empty
    assert "geometry" in noise_frame.columns
    assert "noise_level" in noise_frame.columns
    assert noise_frame["noise_level"].max() <= noise_sources["source_noise_db"].max()
    assert noise_frame["noise_level"].min() >= 40
    assert noise_frame.crs == noise_frame.crs

    plot_simulation_result(
        result_gdf=noise_frame,
        source_points=noise_sources,
        buildings=buildings_data,
        trees=gpd.GeoDataFrame(),
        source_db_range=(95, 40),
        reflection_n="No reflections",
        frequency_desc=2000,
        output_filename="noise_frame",
        image_title="Noise frame (from roads,points,buildings)",
    )


def plot_simulation_result(
    result_gdf,
    source_points,
    buildings,
    trees,
    source_db_range,
    reflection_n,
    frequency_desc,
    output_filename,
    image_title="Noise propagation",
):

    source_db, target_db = source_db_range
    local_crs = result_gdf.estimate_utm_crs()
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    minx, miny, maxx, maxy = result_gdf.to_crs(local_crs).total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    result_gdf.to_crs(local_crs).plot(
        ax=ax,
        column="noise_level",
        cmap="RdYlGn_r",
        legend=True,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.1,
        vmin=30,
        vmax=100,
        legend_kwds={"label": "Noise level, Decibels", "shrink": 0.5},
    )

    buildings.to_crs(local_crs).plot(ax=ax, facecolor="gray", edgecolor="black", linewidth=0.5, label="Buildings")
    if len(trees) > 0:
        trees.to_crs(local_crs).plot(ax=ax, edgecolor="green", facecolor="none", linewidth=1.5, label="Trees")
    source_points.to_crs(local_crs).plot(ax=ax, color="red", markersize=10, label="Noise sources")

    ax.set_title(
        f"{image_title} {source_db}dB -> {target_db}dB\n"
        f"Frequency: {frequency_desc}, Reflection count: {reflection_n}, Temperature: 20°C"
    )
    ax.legend()
    ax.set_axis_off()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
