import os

import pytest
from matplotlib import pyplot as plt
from pyproj.exceptions import CRSError
from objectnat import get_graph_coverage, get_radius_coverage
import geopandas as gpd
from tests.conftest import output_dir


def test_graph_time_min(services_data, buildings_data, intermodal_osm_1114252, boundary_osm_1114252):
    zone = gpd.GeoDataFrame(geometry=[boundary_osm_1114252], crs=4326)
    weight = 10
    result = get_graph_coverage(
        gdf_from=services_data,
        nx_graph=intermodal_osm_1114252,
        weight_type="time_min",
        weight_value_cutoff=weight,
        zone=zone,
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(services_data)

    visualize_coverage_zones(
        result,
        buildings_data,
        services_data,
        title_suffix=f"(Time cutoff {weight} minutes)",
        filename_suffix="time_10min",
    )


def test_graph_length_meter(services_data, buildings_data, intermodal_osm_1114252):
    weight = 600
    result = get_graph_coverage(
        gdf_from=services_data, nx_graph=intermodal_osm_1114252, weight_type="length_meter", weight_value_cutoff=weight
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(services_data)

    visualize_coverage_zones(
        result,
        buildings_data,
        services_data,
        title_suffix=f"(Distance cutoff {weight} meters)",
        filename_suffix="distance_600m",
    )


def test_graph_same_crs(services_data, intermodal_osm_1114252):
    services_data = services_data.to_crs(3857)
    result = get_graph_coverage(
        gdf_from=services_data, nx_graph=intermodal_osm_1114252, weight_type="length_meter", weight_value_cutoff=600
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(services_data)
    assert result.crs == services_data.crs


def test_wrong_graph_crs(services_data, intermodal_osm_1114252):
    wrong_graph = intermodal_osm_1114252.copy()
    wrong_graph.graph["crs"] = "Wrong CRS"
    with pytest.raises(CRSError) as _:
        _ = get_graph_coverage(
            gdf_from=services_data, nx_graph=wrong_graph, weight_type="length_meter", weight_value_cutoff=600
        )
    wrong_graph.graph = {}
    with pytest.raises(ValueError) as _:
        _ = get_graph_coverage(
            gdf_from=services_data, nx_graph=wrong_graph, weight_type="length_meter", weight_value_cutoff=600
        )


def test_radius_coverage(services_data):
    services_data = services_data.to_crs(4326)
    result = get_radius_coverage(services_data, radius=1000)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(services_data)
    assert result.crs == services_data.crs


def visualize_coverage_zones(coverage_gdf, buildings_data, services_data, title_suffix="", filename_suffix=""):
    local_crs = buildings_data.estimate_utm_crs()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    coverage_gdf = coverage_gdf.to_crs(local_crs)

    buildings_data = buildings_data.to_crs(local_crs)
    services_data = services_data.to_crs(local_crs)

    minx, miny, maxx, maxy = buildings_data.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    buildings_data.plot(ax=ax, edgecolor="gray", facecolor="none", linewidth=0.5)
    coverage_gdf.plot(
        ax=ax,
        column="name",
        cmap="tab20",
        legend=True,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.2,
        label="Coverage zones",
    )
    services_data.plot(ax=ax, color="red", markersize=15, edgecolor="white", linewidth=0.3, label="Services")
    ax.set_title(f"Coverage zones {title_suffix}")
    ax.legend()
    ax.set_axis_off()

    output_path = os.path.join(output_dir, f"coverage_zones_{filename_suffix}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    return output_path
