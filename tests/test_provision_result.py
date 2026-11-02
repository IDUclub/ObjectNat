"""Unit tests for the provision result-formatting layer.

Covers ``_validate_provision_result``, the buildings/services join helpers, link
geometry construction, and ``clip_provision`` — all on small hand-built results,
without the network graph or parquet fixtures.
"""

import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely import Point

from objectnat import (
    ProvisionResult,
    clip_provision,
    get_provision_buildings,
    get_provision_links,
    get_provision_services,
)
from objectnat.methods.provision.provision_calculating import _summarize_flow
from objectnat.methods.provision.provision_result import (
    _drop_overwritten_columns,
    _validate_provision_result,
)

BUILDING_INDEX = pd.Index([1, 2])
SERVICE_INDEX = pd.Index([10, 11])
MATRIX = pd.DataFrame([[4.0, 20.0], [30.0, 12.0]], index=BUILDING_INDEX, columns=SERVICE_INDEX)


def _make_flow(array):
    return pd.DataFrame.sparse.from_spmatrix(
        sparse.coo_matrix(np.array(array)),
        index=BUILDING_INDEX,
        columns=SERVICE_INDEX,
    ).astype(pd.SparseDtype("int", 0))


def _make_result(flow_array):
    flow = _make_flow(flow_array)
    demand = pd.Series([10, 5], index=BUILDING_INDEX)
    capacity = pd.Series([8, 6], index=SERVICE_INDEX)
    demand_rows, capacity_rows = _summarize_flow(flow, demand, capacity, MATRIX, 10.0)
    return ProvisionResult(flow, demand_rows, capacity_rows, MATRIX, 10.0)


@pytest.fixture
def result():
    # building 1 -> service 10 (dist 4), building 2 -> service 11 (dist 12)
    return _make_result([[6, 0], [0, 4]])


@pytest.fixture
def buildings_gdf():
    return gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(0, 0), Point(10, 0)],
        index=BUILDING_INDEX,
        crs=3857,
    )


@pytest.fixture
def services_gdf():
    return gpd.GeoDataFrame(
        {"kind": ["school", "clinic"]},
        geometry=[Point(0, 10), Point(10, 10)],
        index=SERVICE_INDEX,
        crs=3857,
    )


# --------------------------------------------------------------------------- #
# _validate_provision_result
# --------------------------------------------------------------------------- #
def test_validate_result_accepts_valid(result):
    _validate_provision_result(result)  # should not raise


def test_validate_result_type_error():
    with pytest.raises(TypeError):
        _validate_provision_result({"flow": None})


def test_validate_result_duplicate_flow_index(result):
    bad_flow = result.flow.copy()
    bad_flow.index = pd.Index([1, 1])
    bad = dataclasses.replace(result, flow=bad_flow)
    with pytest.raises(ValueError, match="flow.index must be unique"):
        _validate_provision_result(bad)


def test_validate_result_misaligned_index(result):
    bad_flow = result.flow.copy()
    bad_flow.index = pd.Index([1, 99])  # unique but no longer matches distance_matrix
    bad = dataclasses.replace(result, flow=bad_flow)
    with pytest.raises(ValueError, match="distance_matrix.index"):
        _validate_provision_result(bad)


def test_validate_result_misaligned_columns(result):
    bad_flow = result.flow.copy()
    bad_flow.columns = pd.Index([10, 99])
    bad = dataclasses.replace(result, flow=bad_flow)
    with pytest.raises(ValueError, match="distance_matrix.columns"):
        _validate_provision_result(bad)


# --------------------------------------------------------------------------- #
# _drop_overwritten_columns
# --------------------------------------------------------------------------- #
def test_drop_overwritten_columns_drops_intersection():
    df = pd.DataFrame({"a": [1], "b": [2], "demand": [3]})
    out = _drop_overwritten_columns(df, pd.Index(["demand", "provision_value"]), "df")
    assert list(out.columns) == ["a", "b"]


def test_drop_overwritten_columns_noop_returns_same_object():
    df = pd.DataFrame({"a": [1]})
    assert _drop_overwritten_columns(df, pd.Index(["demand"]), "df") is df


# --------------------------------------------------------------------------- #
# get_provision_buildings / get_provision_services
# --------------------------------------------------------------------------- #
def test_get_provision_buildings_join(result, buildings_gdf):
    joined = get_provision_buildings(buildings_gdf, result)
    assert len(joined) == len(buildings_gdf)
    assert {"demand", "provision_value", "min_dist"}.issubset(joined.columns)
    assert joined.loc[1, "demand"] == 10
    assert joined.loc[1, "name"] == "a"  # original columns preserved


def test_get_provision_buildings_type_error(result):
    with pytest.raises(TypeError):
        get_provision_buildings([1, 2, 3], result)


def test_get_provision_buildings_duplicate_index(result, buildings_gdf):
    dup = pd.concat([buildings_gdf, buildings_gdf.iloc[[0]]])
    with pytest.raises(ValueError, match="buildings_gdf.index must be unique"):
        get_provision_buildings(dup, result)


def test_get_provision_buildings_overwrites_existing_columns(result, buildings_gdf):
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["demand"] = 999  # stale column should be dropped and replaced
    joined = get_provision_buildings(buildings_gdf, result)
    assert joined.loc[1, "demand"] == 10


def test_get_provision_services_join(result, services_gdf):
    joined = get_provision_services(services_gdf, result)
    assert {"capacity", "service_load", "capacity_left"}.issubset(joined.columns)
    assert joined.loc[10, "capacity"] == 8
    assert joined.loc[10, "kind"] == "school"


def test_get_provision_services_type_error(result):
    with pytest.raises(TypeError):
        get_provision_services("not a dataframe", result)


# --------------------------------------------------------------------------- #
# get_provision_links
# --------------------------------------------------------------------------- #
def test_get_provision_links_geometry(result, buildings_gdf, services_gdf):
    links = get_provision_links(buildings_gdf, services_gdf, result)

    assert isinstance(links, gpd.GeoDataFrame)
    assert {"building_index", "service_index", "demand", "distance", "geometry"}.issubset(links.columns)
    assert links.crs == buildings_gdf.crs
    assert (links.geometry.geom_type == "LineString").all()
    assert len(links) == 2  # two non-zero flows

    by_pair = links.set_index(["building_index", "service_index"])
    assert by_pair.loc[(1, 10), "demand"] == 6
    assert by_pair.loc[(1, 10), "distance"] == 4.0


def test_get_provision_links_empty_flow(buildings_gdf, services_gdf):
    result = _make_result([[0, 0], [0, 0]])
    links = get_provision_links(buildings_gdf, services_gdf, result)
    assert links.empty


def test_get_provision_links_reprojects_services(result, buildings_gdf, services_gdf):
    services_4326 = services_gdf.to_crs(4326)
    links = get_provision_links(buildings_gdf, services_4326, result)
    assert not links.empty
    assert links.crs == buildings_gdf.crs


def test_get_provision_links_filters_unknown_building(result, buildings_gdf, services_gdf):
    # Drop building 2: its flow link must be filtered out without raising KeyError.
    partial = buildings_gdf.drop(index=2)
    links = get_provision_links(partial, services_gdf, result)
    assert (links["building_index"] == 1).all()


def test_get_provision_links_requires_geodataframe(result, services_gdf):
    with pytest.raises(TypeError):
        get_provision_links(pd.DataFrame({"a": [1, 2]}, index=BUILDING_INDEX), services_gdf, result)


# --------------------------------------------------------------------------- #
# clip_provision
# --------------------------------------------------------------------------- #
def test_clip_provision_crs_mismatch(result, buildings_gdf, services_gdf):
    links = get_provision_links(buildings_gdf, services_gdf, result)
    zone = gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(5)], crs=4326)
    with pytest.raises(ValueError):
        clip_provision(buildings_gdf, services_gdf, links, zone)


def test_clip_provision_keeps_intersecting(result, buildings_gdf, services_gdf):
    build_prov = get_provision_buildings(buildings_gdf, result)
    services_prov = get_provision_services(services_gdf, result)
    links = get_provision_links(buildings_gdf, services_gdf, result)

    # Zone tightly around building 1 at (0, 0) only.
    zone = gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1)], crs=3857)
    clipped_buildings, clipped_services, clipped_links = clip_provision(build_prov, services_prov, links, zone)

    assert list(clipped_buildings.index) == [1]
    assert (clipped_links["building_index"] == 1).all()
    # Only services referenced by the surviving links are kept.
    assert set(clipped_services.index) == set(clipped_links["service_index"])
