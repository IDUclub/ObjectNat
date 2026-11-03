import os

import geopandas as gpd
import numpy as np
import pytest
from matplotlib import pyplot as plt
from shapely import LineString, Point, Polygon, box

from objectnat import get_viewshed, get_viewshed_catchment, get_visibility
from objectnat.methods.visibility._native import _largest_part_at
from tests.conftest import output_dir

CRS = 32636
X0, Y0 = 400_000.0, 6_600_000.0


@pytest.fixture(scope="module")
def gdf_1point_special():
    return gpd.GeoDataFrame(geometry=[Point(30.2312112, 59.9482336)], crs=4326)


def test_empty_obstacles_give_the_view_disc(gdf_1point_special):
    result = get_visibility(gdf_1point_special, gpd.GeoDataFrame(), 500)

    assert result.crs == gdf_1point_special.crs
    local = result.to_crs(result.estimate_utm_crs()).geometry.iloc[0]
    assert local.area == pytest.approx(Point(0, 0).buffer(500).area, rel=1e-6)


def test_visibility_on_example_buildings(gdf_1point_special, buildings_data):
    radius = 800
    result = get_visibility(gdf_1point_special, buildings_data, radius)

    assert result.crs == gdf_1point_special.crs
    assert result.geometry.is_valid.all()
    assert not result.is_empty.any()

    local_crs = buildings_data.estimate_utm_crs()
    buildings = buildings_data.to_crs(local_crs)
    visible = result.to_crs(local_crs)
    point = gdf_1point_special.to_crs(local_crs)
    assert visible.geometry.iloc[0].area < point.buffer(radius).area.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 10))
    minx, miny, maxx, maxy = visible.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    buildings.plot(ax=ax, color="lightgray", alpha=0.7, edgecolor="gray", linewidth=0.5)
    visible.plot(ax=ax, color="#1f77b4", alpha=0.5, edgecolor="#0d3d66", linewidth=1)
    point.plot(ax=ax, color="purple", markersize=20, edgecolor="black")
    ax.set_title(f"Visibility, radius {radius} m")
    ax.set_axis_off()
    plt.savefig(os.path.join(output_dir, "visibility_example.png"), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()


def test_multiple_visibility(gdf_3points, buildings_data):
    result = get_visibility(gdf_3points, buildings_data, 800)
    assert len(result) == len(gdf_3points)


def test_single_wall_gives_exact_half_disc():
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 + 30, Y0 - 500, X0 + 31, Y0 + 500)], crs=CRS)
    point = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)

    result = get_visibility(point, obstacles, 100).geometry.iloc[0]

    expected = Point(X0, Y0).buffer(100).intersection(box(X0 - 200, Y0 - 200, X0 + 30, Y0 + 200))
    assert result.symmetric_difference(expected).area < 1e-6


def test_line_obstacles_and_per_point_radius():
    wall = LineString([(X0 + 20, Y0 - 300), (X0 + 20, Y0 + 300)])
    obstacles = gpd.GeoDataFrame(geometry=[wall], crs=CRS)
    points = gpd.GeoDataFrame({"visibility_distance": [50.0, 10.0]}, geometry=[Point(X0, Y0)] * 2, crs=CRS)

    result = get_visibility(points, obstacles, max_workers=1).geometry

    expected = Point(X0, Y0).buffer(50).intersection(box(X0 - 100, Y0 - 100, X0 + 20, Y0 + 100))
    assert result.iloc[0].symmetric_difference(expected).area < 1e-6
    assert result.iloc[1].area == pytest.approx(Point(X0, Y0).buffer(10).area)


def test_observer_inside_obstacle_gets_empty_polygon():
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 - 5, Y0 - 5, X0 + 5, Y0 + 5)], crs=CRS)
    points = gpd.GeoDataFrame(geometry=[Point(X0, Y0), Point(X0 + 50, Y0)], crs=CRS)

    result = get_visibility(points, obstacles, 30)

    assert result.geometry.iloc[0].is_empty
    assert not result.geometry.iloc[1].is_empty


def test_simple_method_is_deprecated_and_exact():
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 + 30, Y0 - 500, X0 + 31, Y0 + 500)], crs=CRS)
    point = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)
    with pytest.warns(DeprecationWarning):
        simple = get_visibility(point, obstacles, 100, "simple").geometry.iloc[0]
    assert simple.equals(get_visibility(point, obstacles, 100).geometry.iloc[0])
    with pytest.raises(ValueError):
        get_visibility(point, obstacles, 100, "fast")


def test_invalid_ring_repair_keeps_the_part_around_the_observer():
    # near-coincident walls can leave a micro bow-tie in the kernel's ring; make_valid splits it
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    repaired = _largest_part_at(bowtie, Point(1, 5))
    assert repaired.geom_type == "Polygon"
    assert repaired.is_valid
    assert repaired.area == pytest.approx(25.0)


def test_isovist_metrics():
    observers = gpd.GeoDataFrame(geometry=[Point(X0, Y0), Point(X0 + 500, Y0), Point(X0 + 25, Y0)], crs=CRS)
    building = gpd.GeoDataFrame(geometry=[box(X0 + 20, Y0 - 10, X0 + 30, Y0 + 10)], crs=CRS)
    result = get_visibility(observers, building, 100, isovist_metrics=True)
    shadowed, free, inside = (result.iloc[i] for i in range(3))

    assert free["n_occlusions"] == 0 and free["open_share"] == pytest.approx(1)
    assert free["compactness"] == pytest.approx(1, abs=1e-3) and free["drift"] == pytest.approx(0, abs=1e-6)

    # two shadow edges from the near corners (20, ±10) out to the view limit
    assert shadowed["n_occlusions"] == 2
    assert shadowed["min_radial"] == pytest.approx(20) and shadowed["max_radial"] == pytest.approx(100)
    assert shadowed["occlusivity"] == pytest.approx(2 * (100 - np.hypot(20, 10)), abs=0.2)
    # exact radial moments agree with sampling the distance to the boundary along rays
    polygon = result.geometry.iloc[0]
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    rays = [LineString([(X0, Y0), (X0 + 200 * np.cos(a), Y0 + 200 * np.sin(a))]) for a in angles]
    radial = np.array([ray.intersection(polygon).length for ray in rays])
    assert shadowed["mean_radial"] == pytest.approx(radial.mean(), abs=0.05)
    assert shadowed["std_radial"] == pytest.approx(radial.std(), abs=0.05)

    assert np.isnan(inside["area"]) and np.isnan(inside["mean_radial"])


def test_viewshed_sees_over_low_buildings_and_hides_behind_hills():
    # a 3 m garage from x = 10 to 20 m east of an observer looking from 10 m up (a balcony)
    observer = gpd.GeoDataFrame({"observer_height": [10.0]}, geometry=[Point(X0, Y0)], crs=CRS)
    garage = gpd.GeoDataFrame({"height": [3.0]}, geometry=[box(X0 + 10, Y0 - 5, X0 + 20, Y0 + 5)], crs=CRS)
    seen = get_viewshed(observer, garage, 60, cell_size=0.5, include_roofs=True).geometry.iloc[0]
    # the line over the garage's far wall (3 m at 20 m) reaches the ground at 20 * 10 / 7 = 28.6 m
    assert not seen.covers(Point(X0 + 25, Y0))
    assert seen.covers(Point(X0 + 32, Y0))
    assert seen.covers(Point(X0 + 15, Y0))  # its roof, seen from above
    flat = get_visibility(observer, garage, 60).geometry.iloc[0]
    assert not flat.covers(Point(X0 + 32, Y0))  # 2D visibility treats the garage as infinitely high

    # a 30 m ridge 100 m north hides the valley behind it, without any buildings
    x = np.arange(-300, 301, 10.0)
    y = np.arange(300, -301, -10.0)
    heights = np.where(np.abs(y - 100)[:, None] <= 10, 30.0, 0.0) * np.ones_like(x)[None, :]
    terrain = (heights, (10.0, 0.0, X0 - 305, 0.0, -10.0, Y0 + 305), CRS)
    result = get_viewshed(observer.assign(observer_height=1.6), None, 250, terrain=terrain, cell_size=5)
    seen = result.geometry.iloc[0]
    assert seen.covers(Point(X0, Y0 + 50)) and not seen.covers(Point(X0, Y0 + 200))
    assert 0 < result["visible_share"].iloc[0] < 1


def test_viewshed_catchment_from_a_hilltop():
    # four observers on top of a 40 m hill; a 10 m house 100 m east of them, at the foot of the hill
    observers = gpd.GeoDataFrame(
        geometry=[Point(X0 + dx, Y0 + dy) for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1))], crs=CRS
    )
    house = gpd.GeoDataFrame({"height": [10.0]}, geometry=[box(X0 + 100, Y0 - 30, X0 + 110, Y0 + 30)], crs=CRS)
    x = np.arange(-300, 301, 10.0)
    y = np.arange(300, -301, -10.0)
    hill = np.maximum(0.0, 40 - 0.5 * np.hypot(x[None, :], y[:, None]))
    terrain = (hill, (10.0, 0.0, X0 - 305, 0.0, -10.0, Y0 + 305), CRS)
    kwargs = {"view_distance": 200, "cell_size": 5, "classes": [1]}

    from_hill = get_viewshed_catchment(observers, house, terrain=terrain, **kwargs)
    from_ground = get_viewshed_catchment(observers, house, **kwargs)
    behind = Point(X0 + 170, Y0)  # the sight line clears the far wall (10 m at 110 m) from about 147 m on
    # eyes stand 1.6 m above the hilltop, so the ground behind the house shows over its roof
    assert from_hill.union_all().covers(behind)
    assert not from_ground.union_all().covers(behind)
    assert from_hill["count_max"].max() == 4
