import os

import pytest
from matplotlib import pyplot as plt

from objectnat import get_graph_isochrones, get_stepped_graph_isochrones
from tests.conftest import output_dir

pytestmark = pytest.mark.network


def test_1point_isochrone_radius(intermodal_osm_1114252, gdf_1point, buildings_data):
    weight_value = 15
    isochrones = get_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_1point,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="radius",
    )
    assert isochrones is not None
    assert len(isochrones) == 1
    visualize_isochrones(
        isochrones,
        gdf_1point,
        buildings_data,
        title_suffix=f"(radius mode, {weight_value} minutes)",
        filename_suffix=f"radius_{weight_value}_min",
    )


def test_1point_isochrone_ways(intermodal_osm_1114252, gdf_1point, buildings_data):
    gdf_1point = gdf_1point.to_crs(4326)
    weight_value = 15
    isochrones = get_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_1point,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="ways",
    )
    assert isochrones is not None
    assert len(isochrones) == 1
    assert isochrones.crs == gdf_1point.crs
    visualize_isochrones(
        isochrones,
        gdf_1point,
        buildings_data,
        title_suffix=f"(ways mode, {weight_value} minutes)",
        filename_suffix=f"ways_{weight_value}_min",
    )


def test_3point_isochrone_radius(intermodal_osm_1114252, gdf_3points, buildings_data):
    weight_value = 8
    isochrones = get_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_3points,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="radius",
    )
    assert isochrones is not None
    assert len(isochrones) == 3
    visualize_isochrones(
        isochrones,
        gdf_3points,
        buildings_data,
        title_suffix=f"(3 points radius mode, {weight_value} minutes)",
        filename_suffix=f"3points_radius_{weight_value}_min",
    )


def test_3point_isochrone_ways(intermodal_osm_1114252, gdf_3points):
    isochrones = get_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_3points,
        weight_value_cutoff=5,
        weight_type="time_min",
        geometry_type="ways",
    )
    assert isochrones is not None
    assert len(isochrones) == 3


def test_isochrone_stepped_radius(intermodal_osm_1114252, gdf_1point, buildings_data):
    weight_value = 15
    stepped_iso = get_stepped_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_1point,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="radius",
        step=3,
    )
    assert stepped_iso is not None
    assert not stepped_iso.empty

    visualize_stepped_isochrones(
        stepped_iso,
        gdf_1point,
        buildings_data,
        title_suffix=f"(radius mode, {weight_value} minutes)",
        filename_suffix=f"radius_{weight_value}_min",
    )


def test_isochrone_stepped_ways(intermodal_osm_1114252, gdf_1point, buildings_data):
    weight_value = 15
    stepped_iso = get_stepped_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_1point,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="ways",
        step=3,
    )
    assert stepped_iso is not None
    assert not stepped_iso.empty

    visualize_stepped_isochrones(
        stepped_iso,
        gdf_1point,
        buildings_data,
        title_suffix=f"(ways mode, {weight_value} minutes)",
        filename_suffix=f"ways_{weight_value}_min",
    )


def test_isochrone_stepped_separate(intermodal_osm_1114252, gdf_1point, buildings_data):
    weight_value = 15
    stepped_iso = get_stepped_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_1point,
        weight_value_cutoff=weight_value,
        weight_type="time_min",
        geometry_type="separate",
        step=3,
    )
    assert stepped_iso is not None
    assert not stepped_iso.empty
    visualize_stepped_isochrones(
        stepped_iso,
        gdf_1point,
        buildings_data,
        title_suffix=f"(separate mode, {weight_value} minutes)",
        filename_suffix=f"separate_{weight_value}_min",
    )


def test_multipoint_in_stepped(intermodal_osm_1114252, gdf_3points):
    stepped_iso = get_stepped_graph_isochrones(
        intermodal_osm_1114252,
        gdf_origins=gdf_3points,
        weight_value_cutoff=15,
        weight_type="time_min",
        geometry_type="radius",
        step=3,
    )
    assert stepped_iso is not None
    assert not stepped_iso.empty


def visualize_isochrones(isochrones, point_from, buildings_data, title_suffix="", filename_suffix=""):
    local_crs = buildings_data.estimate_utm_crs()

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    isochrones = isochrones.to_crs(local_crs)
    buildings_data = buildings_data.to_crs(local_crs)
    point_from = point_from.to_crs(local_crs)

    minx, miny, maxx, maxy = buildings_data.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    buildings_data.plot(ax=ax, edgecolor="gray", facecolor="none", linewidth=0.5)
    isochrones.reset_index(drop=True).reset_index().plot(
        ax=ax,
        alpha=0.8,
        column="index",
        cmap="tab20",
        linewidth=0.8,
        categorical=True,
        label="Isochrones",
    )
    point_from.plot(ax=ax, color="red", markersize=50, label="Start point")
    ax.set_title(f"Isochrone {title_suffix}")
    ax.legend()
    ax.set_axis_off()

    output_path = os.path.join(output_dir, f"isochrone_{filename_suffix}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize_stepped_isochrones(stepped_isochrones, point_from, buildings_data, title_suffix="", filename_suffix=""):

    local_crs = buildings_data.estimate_utm_crs()

    stepped_isochrones = stepped_isochrones.to_crs(local_crs)
    buildings_data = buildings_data.to_crs(local_crs)
    point_from = point_from.to_crs(local_crs)

    fig, ax = plt.subplots(figsize=(12, 10))

    minx, miny, maxx, maxy = buildings_data.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    buildings_data.plot(ax=ax, edgecolor="gray", facecolor="none", linewidth=0.5)

    stepped_isochrones.plot(
        ax=ax,
        column="dist",
        cmap="viridis",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.2,
        legend=True,
        legend_kwds={"label": "Distance (meters)", "shrink": 0.5},
        label="Stepped isochrone",
    )
    point_from.plot(ax=ax, color="red", markersize=50, label="Start point")

    ax.set_title(f"Stepped isochrone {title_suffix}")
    ax.set_axis_off()

    output_path = os.path.join(output_dir, f"stepped_isochrone_{filename_suffix}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
