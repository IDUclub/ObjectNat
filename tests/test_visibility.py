import os

import geopandas as gpd
import pytest
from matplotlib import pyplot as plt
from shapely import LineString, Point, Polygon, box

from objectnat import get_visibility
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
