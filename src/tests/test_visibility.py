import os

import geopandas as gpd
import pytest
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from shapely import Point, Polygon

from objectnat import get_visibility
from tests.conftest import output_dir


@pytest.fixture(scope="module")
def gdf_1point_special():
    return gpd.GeoDataFrame(geometry=[Point(30.2312112, 59.9482336)], crs=4326)


def test_empty_obstacles_point_gdf(gdf_1point_special):
    radius = 500

    simple = get_visibility(gdf_1point_special, gpd.GeoDataFrame(), radius, "simple")
    accurate = get_visibility(gdf_1point_special, gpd.GeoDataFrame(), radius, "accurate")

    assert simple.crs == gdf_1point_special.crs
    assert accurate.crs == gdf_1point_special.crs


def test_compare_visibility_methods(gdf_1point_special, buildings_data):

    radius = 800
    simple = get_visibility(gdf_1point_special, buildings_data, radius, "simple")
    accurate = get_visibility(gdf_1point_special, buildings_data, radius, "accurate")

    assert simple.crs == gdf_1point_special.crs
    assert accurate.crs == gdf_1point_special.crs

    local_crs = buildings_data.estimate_utm_crs()
    buildings = buildings_data.to_crs(local_crs)
    point = gdf_1point_special.to_crs(local_crs)
    simple = simple.to_crs(local_crs)
    accurate = accurate.to_crs(local_crs)

    simple_only = gpd.overlay(simple, accurate, how="difference")
    accurate_only = gpd.overlay(accurate, simple, how="difference")
    common_area = gpd.overlay(simple, accurate, how="intersection")

    fig, ax = plt.subplots(figsize=(12, 10))

    minx, miny, maxx, maxy = accurate.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    buildings.plot(ax=ax, color="lightgray", alpha=0.7, edgecolor="gray", linewidth=0.5, label="Buildings")

    point.plot(ax=ax, color="purple", markersize=20, edgecolor="black", label="Viewpoint")
    legend_elements = []

    if not common_area.empty:
        style = dict(color="#1f77b4", alpha=0.5, edgecolor="#0d3d66")
        common_area.plot(ax=ax, **style, linewidth=1)
        legend_elements.append(Patch(**style, label="Agreement Area (both methods)"))

    if not simple_only.empty:
        style = dict(color="#d62728", alpha=0.6, edgecolor="#8b0000")
        simple_only.plot(ax=ax, **style, linewidth=1)
        legend_elements.append(Patch(**style, label="False Positive (Simple method)"))

    if not accurate_only.empty:
        style = dict(color="#2ca02c", alpha=0.6, edgecolor="#006400")
        accurate_only.plot(ax=ax, **style, linewidth=1)
        legend_elements.append(Patch(**style, label="Advantage (Accurate method)"))

    ax.set_title(f"Visibility comparison\n" f"Radius: {radius}m")
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_axis_off()

    output_path = os.path.join(output_dir, "visibility_comparison_methods.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()

    assert not simple.is_empty.all()
    assert not accurate.is_empty.all()


def test_multiple_visibility(gdf_3points, buildings_data):
    result = get_visibility(gdf_3points, buildings_data, 800)
    assert len(result) == len(gdf_3points)
