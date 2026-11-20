"""Exact reference comparisons for scene preparation fast paths (no network)."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from objectnat import get_facade_insolation, get_insolation
from objectnat.methods.utils import scene as scene_module
from objectnat.methods.utils._building_details import (
    _exposed_undersides,
    osm_direction,
    osm_directions,
    osm_number,
    osm_numbers,
    polygon_arrays,
)
from objectnat.methods.utils.scene import prepare_scene

DATA = Path(__file__).resolve().parents[1] / "docs/methods/examples/examples_data/buildings_osm3d.parquet"


def _old_polygon_arrays(polygons, shift):
    coords, rings, polys = [], [0], [0]
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            xy = np.asarray(ring.coords, dtype=float)[:-1, :2] - shift
            coords.extend(xy)
            rings.append(len(coords))
        polys.append(len(rings) - 1)
    return np.asarray(coords, dtype=float).reshape(-1, 2), np.asarray(rings, np.int64), np.asarray(polys, np.int64)


def _old_exposed(polygons, bottom, eave, raised):
    keep = raised.copy()
    indices = np.flatnonzero(raised)
    if not len(indices):
        return keep
    target, support = shapely.STRtree(polygons).query(polygons[indices], predicate="intersects")
    valid = (bottom[support] < bottom[indices[target]]) & (eave[support] >= bottom[indices[target]])
    target, support = target[valid], support[valid]
    order = np.argsort(target, kind="stable")
    target, support = target[order], support[order]
    starts = np.r_[0, np.flatnonzero(np.diff(target)) + 1, len(target)]
    for start, end in zip(starts[:-1], starts[1:]):
        if start == end:
            continue
        i = indices[target[start]]
        covered = shapely.union_all(polygons[support[start:end]])
        if shapely.area(shapely.difference(polygons[i], covered)) <= 1e-6 * shapely.area(polygons[i]):
            keep[i] = False
    return keep


@pytest.mark.parametrize("dtype", [object, "float64", "float32", "float16", "Float64", "Int64", bool])
def test_number_column_matches_scalar(dtype):
    tricky = [
        None,
        np.nan,
        pd.NA,
        "inf",
        "nan",
        "1e400",
        " 5 m",
        "5,5",
        ".5",
        "1;2",
        "+3",
        "0x10",
        "",
        True,
        False,
        12,
        -3.25,
        np.float32(1.2345678),
        np.int64(23),
        "NNE",
        " across ",
    ]
    numeric = (
        [1.23456789, -0.0, np.inf, np.nan] if dtype in ("float64", "float32", "float16", "Float64") else [1, 0, 23]
    )
    values = pd.Series(tricky if dtype is object else numeric, dtype=dtype)
    before = values.copy(deep=True)
    for scalar, vector in ((osm_number, osm_numbers), (osm_direction, osm_directions)):
        np.testing.assert_array_equal(vector(values), [scalar(v) for v in values])
    pd.testing.assert_series_equal(values, before)


@pytest.mark.parametrize(
    "polygons",
    [
        [],
        [shapely.Polygon()],
        [
            shapely.Polygon([(0, 0), (9, 0), (9, 9), (0, 9)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]]),
            shapely.box(20, 20, 21, 21),
        ],
    ],
)
def test_polygon_arrays_reference(polygons):
    for actual, expected in zip(
        polygon_arrays(polygons, [123.25, -0.5]), _old_polygon_arrays(polygons, [123.25, -0.5])
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("container", ["series", "geometry_array", "readonly"])
def test_polygon_arrays_writable_copy(container):
    series = gpd.GeoSeries(
        [
            shapely.box(0, 0, 5, 5).difference(shapely.box(1, 1, 2, 2)),
            shapely.box(8, 8, 9, 9),
        ]
    )
    if container == "series":
        polygons = series
    elif container == "geometry_array":
        polygons = series.values
    else:
        polygons = np.array(series, dtype=object, copy=True)
        polygons.setflags(write=False)
    for actual, expected in zip(polygon_arrays(polygons, [0.5, 1.25]), _old_polygon_arrays(series, [0.5, 1.25])):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("gap", [0, 0.25, 1e-6 - 1e-10, 1e-6, 1e-6 + 1e-10])
def test_undersides_reference(gap):
    polygons = np.array(
        [
            shapely.box(0, 0, 1, 1),
            shapely.box(0, 0, 0.5, 1),
            shapely.box(0.5 + gap, 0, 1, 1),
            shapely.box(0, 0, 1, 1),
            shapely.box(3, 3, 4, 4),
            shapely.box(3, 3, 5, 5),
        ]
    )
    bottom = np.array([2, 0, 0, 4, 2, 0])
    eave = np.array([4, 2, 2, 6, 4, 2])
    np.testing.assert_array_equal(
        _exposed_undersides(polygons, bottom, eave, bottom > 0),
        _old_exposed(polygons, bottom, eave, bottom > 0),
    )


def test_example_columns_and_geometry_reference():
    if not DATA.exists():
        pytest.skip("local OSM 3D example not installed")
    rows = gpd.read_parquet(DATA).explode(index_parts=False)
    for column in rows.columns.drop("geometry"):
        values = rows[column]
        np.testing.assert_array_equal(osm_numbers(values), [osm_number(v) for v in values])
        np.testing.assert_array_equal(osm_directions(values), [osm_direction(v) for v in values])
    polygons = rows.geometry.to_numpy()
    for actual, expected in zip(polygon_arrays(polygons, [0, 0]), _old_polygon_arrays(polygons, [0, 0])):
        np.testing.assert_array_equal(actual, expected)
    from objectnat.methods.utils._building_details import _heights, _volume_rows

    volumes = _volume_rows(rows)
    bottom, eave, _, _ = _heights(volumes, "height", None, 3.0, 10.0, 30.0)
    polygons = volumes.geometry.to_numpy()
    np.testing.assert_array_equal(
        _exposed_undersides(polygons, bottom, eave, bottom > 0),
        _old_exposed(polygons, bottom, eave, bottom > 0),
    )


X, Y, CRS = 350_000.0, 6_650_000.0, 32636


def _clip_buildings():
    return gpd.GeoDataFrame(
        {
            "height": [12.0, 4.0, 1200.0, 8.0, 5.0, 8.0],
            "building:part": [None, None, None, None, "yes", "yes"],
            "min_height": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            "roof:shape": ["gabled", "flat", "flat", "flat", "flat", "gabled"],
        },
        geometry=[
            shapely.box(X, Y, X + 10, Y + 10),
            shapely.box(X - 100_000, Y, X - 99_990, Y + 10),
            shapely.box(X - 50, Y - 510, X + 50, Y - 500),
            shapely.box(X + 12, Y, X + 18, Y + 6),
            shapely.box(X + 12, Y, X + 18, Y + 6),
            shapely.box(X + 12, Y, X + 18, Y + 6),
        ],
        index=["house", "far_low", "tall", "outline", "support", "raised"],
        crs=CRS,
    )


def _clip_scene(buildings, *, detailed, reach=None, terrain=None):
    return prepare_scene(
        buildings,
        None,
        terrain,
        "height",
        5,
        "height",
        None,
        3.0,
        None,
        building_details=detailed,
        reach=reach,
        _context=(buildings.crs, None) if terrain is None else None,
    )


@pytest.mark.parametrize("detailed", [False, True])
def test_reach_drops_low_retains_tall_and_mesh_arithmetic(detailed):
    buildings = _clip_buildings()
    area = shapely.box(X - 10, Y - 10, X + 20, Y + 20)
    full = _clip_scene(buildings, detailed=detailed)
    clipped = _clip_scene(buildings, detailed=detailed, reach=(area, 0.5))
    assert "far_low" not in clipped.blds.index
    assert "tall" in clipped.blds.index
    expected = full.blds.index.isin(clipped.blds.index)
    pd.testing.assert_frame_equal(clipped.blds, full.blds.loc[expected], check_exact=True)
    for label in clipped.blds.index:
        old_owner = np.flatnonzero(full.blds.index == label)
        new_owner = np.flatnonzero(clipped.blds.index == label)
        np.testing.assert_array_equal(
            clipped.segments[np.isin(clipped.owner, new_owner)],
            full.segments[np.isin(full.owner, old_owner)],
        )
        np.testing.assert_array_equal(
            clipped.tops[np.isin(clipped.owner, new_owner)],
            full.tops[np.isin(full.owner, old_owner)],
        )
    if detailed:
        for actual, reference in zip(clipped.undersides, full.undersides):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("terrain_enabled", [False, True])
@pytest.mark.parametrize("trees_enabled", [False, True])
@pytest.mark.parametrize("facade,elevation", [(False, 0.0), (False, 10.0), (True, 10.0), (True, 0.0)])
def test_reach_public_results(monkeypatch, detailed, terrain_enabled, trees_enabled, facade, elevation):
    buildings = _clip_buildings()
    area = gpd.GeoSeries([shapely.box(X - 10, Y - 10, X + 20, Y + 20)], crs=CRS)
    terrain = None
    if terrain_enabled:
        terrain = (
            np.arange(25, dtype=float).reshape(5, 5) - 10,
            (100_000, 0, X - 250_000, 0, 100_000, Y - 250_000),
            CRS,
        )
    trees = gpd.GeoDataFrame(geometry=[shapely.Point(X + 5, Y - 5)], crs=CRS) if trees_enabled else None
    call = get_facade_insolation if facade else get_insolation
    kwargs = dict(
        area=area,
        terrain=terrain,
        trees=trees,
        building_details=detailed,
        min_sun_elevation=elevation,
        time_step_min=60,
        day_step=1,
    )
    kwargs.update({"spacing": 5, "floors": [1, 2]} if facade else {"grid_step": 5})
    monkeypatch.setattr(scene_module, "_REACH_CLIP_ENABLED", False)
    full = call(buildings, ("2026-03-20", "2026-03-22"), **kwargs)
    monkeypatch.setattr(scene_module, "_REACH_CLIP_ENABLED", True)
    clipped = call(buildings, ("2026-03-20", "2026-03-22"), **kwargs)
    assert clipped.attrs == full.attrs
    weighted = ["sun_hours", "min_sun_hours", "max_continuous_hours"] if trees_enabled else []
    pd.testing.assert_frame_equal(clipped.drop(columns=weighted), full.drop(columns=weighted), check_exact=True)
    if weighted:
        np.testing.assert_allclose(clipped[weighted], full[weighted], atol=1e-9, rtol=0)


@pytest.mark.parametrize("detailed", [False, True])
def test_tall_far_building_shades(detailed):
    from objectnat._rust import insolation_day_stats
    from objectnat.methods.utils.scene import kernel_scene

    buildings = _clip_buildings().loc[["far_low", "tall"]]
    area = shapely.box(X - 1, Y - 1, X + 1, Y + 1)
    scene = _clip_scene(buildings, detailed=detailed, reach=(area, 0.5))
    assert list(scene.blds.index) == ["tall"]
    segments, tops, kwargs = kernel_scene(scene, np.array([X, Y]))
    counts, _ = insolation_day_stats(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, -1.0, 0.5]]),
        np.array([0, 1], dtype=np.int64),
        segments,
        tops,
        **kwargs,
    )
    assert counts[0, 0] == 0
    empty = _clip_scene(
        buildings.iloc[:0],
        detailed=detailed,
        terrain=(
            np.zeros((2, 2)),
            (1, 0, X, 0, 1, Y),
            CRS,
        ),
    )
    segments, tops, kwargs = kernel_scene(empty, np.array([X, Y]))
    counts, _ = insolation_day_stats(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, -1.0, 0.5]]),
        np.array([0, 1], dtype=np.int64),
        segments,
        tops,
        **kwargs,
    )
    assert counts[0, 0] == 1


def test_underside_exposure_uses_dropped_supports():
    buildings = gpd.GeoDataFrame(
        {"height": [10.0, 200.0, 1.0], "min_height": [0.0, 10.0, 0.0]},
        geometry=[shapely.box(X + 100, Y, X + 110, Y + 10)] * 2 + [shapely.box(X + 10_000, Y, X + 10_010, Y + 10)],
        index=["low_support", "raised", "far_low"],
        crs=CRS,
    )
    full = _clip_scene(buildings, detailed=True)
    clipped = _clip_scene(buildings, detailed=True, reach=(shapely.box(X, Y, X + 1, Y + 1), 1.0))
    assert list(clipped.blds.index) == ["raised"]
    for actual, reference in zip(clipped.undersides, full.undersides):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("detailed", [False, True])
def test_detailed_groups_and_independent_flat_rows_with_duplicate_labels(detailed):
    buildings = gpd.GeoDataFrame(
        {"height": [5.0, 5.0, 200.0, 3.0], "building:part": [None, "yes", "yes", None]},
        geometry=[
            shapely.box(X + 100, Y, X + 140, Y + 40),
            shapely.box(X + 101, Y + 1, X + 110, Y + 10),
            shapely.box(X + 120, Y + 20, X + 130, Y + 30),
            shapely.box(X + 10000, Y, X + 10010, Y + 10),
        ],
        index=["duplicate"] * 4,
        crs=CRS,
    )
    scene = _clip_scene(buildings, detailed=detailed, reach=(shapely.box(X, Y, X + 1, Y + 1), 1.0))
    # Detailed: both parts plus the outline remainder the parts do not cover.
    assert len(scene.blds) == (3 if detailed else 1)
    assert scene.blds["height"].tolist() == ([5.0, 200.0, 5.0] if detailed else [200.0])


def test_facade_region_covers_whole_target_building():
    from objectnat.methods.utils.scene import _reach_mask

    buildings = gpd.GeoDataFrame(
        geometry=[shapely.box(X, Y, X + 1000, Y + 10), shapely.box(X + 1001, Y, X + 1002, Y + 10)],
        crs=CRS,
    )
    mask = _reach_mask(
        buildings,
        np.array([5.0, 5.0]),
        np.zeros(2),
        None,
        (shapely.box(X - 1, Y - 1, X + 1, Y + 1), 1.0),
        groups=np.arange(2),
        parent_geometries=buildings.geometry.to_numpy(),
        facade=True,
    )
    np.testing.assert_array_equal(mask, [True, True])


@pytest.mark.parametrize("facade,elevation,area_present", [(False, 0, False), (True, 10, False), (True, 0, True)])
def test_no_reach_without_area_or_positive_facade_cutoff(monkeypatch, facade, elevation, area_present):
    import importlib

    module = importlib.import_module("objectnat.methods.insolation.insolation")
    original = module.prepare_scene
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs.get("reach"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_scene", capture)
    buildings = _clip_buildings().iloc[:1]
    area = gpd.GeoSeries([shapely.box(X - 10, Y - 10, X + 20, Y + 20)], crs=CRS) if area_present else None
    call = get_facade_insolation if facade else get_insolation
    call(buildings, "2026-06-21", area=area, min_sun_elevation=elevation, time_step_min=60)
    assert calls == [None]


@pytest.mark.parametrize("facade", [False, True])
def test_reach_global_bound_exits_before_geometry_and_groups(monkeypatch, facade):
    class BoundsOnly:
        total_bounds = np.array([-20.0, -20.0, 20.0, 20.0])

        def __len__(self):
            return 2

        @property
        def geometry(self):
            pytest.fail("global shortcut must not inspect individual geometries")

    def unexpected(*args, **kwargs):
        pytest.fail("global shortcut must not reduce groups or test intersections")

    monkeypatch.setattr(np, "unique", unexpected)
    monkeypatch.setattr(shapely, "intersects", unexpected)
    mask = scene_module._reach_mask(
        BoundsOnly(),
        np.array([1.0, 50.0]),
        np.zeros(2),
        None,
        (shapely.box(-1, -1, 1, 1), 1.0),
        groups=np.array([0, 1]),
        facade=facade,
    )
    np.testing.assert_array_equal(mask, [True, True])


def _envelope_buildings():
    # The L's envelope contains the receiver region, while its actual geometry is far away.
    polygon = shapely.Polygon([(0, 0), (100, 0), (100, 100), (90, 100), (90, 10), (0, 10)])
    polygon = shapely.transform(polygon, lambda xy: xy + [X, Y])
    return gpd.GeoDataFrame(
        {"height": [1.0, 1.0]},
        geometry=[polygon, shapely.box(X + 500, Y + 500, X + 510, Y + 510)],
        index=["near_bbox", "far_bbox"],
        crs=CRS,
    )


@pytest.mark.parametrize("facade", [False, True])
def test_envelope_distance_keeps_far_geometry_conservatively(monkeypatch, facade):
    buildings = _envelope_buildings()
    area = shapely.box(X + 45, Y + 45, X + 46, Y + 46)
    assert shapely.distance(buildings.geometry.iloc[0], area) > 30

    def unexpected(*args, **kwargs):
        pytest.fail("reach clipping must not compute polygon distances, unions or buffers")

    for name in ("distance", "union_all", "buffer"):
        monkeypatch.setattr(shapely, name, unexpected)
    original = shapely.intersects
    candidate_counts = []

    def intersects(polygons, region):
        candidate_counts.append(len(polygons))
        return original(polygons, region)

    monkeypatch.setattr(shapely, "intersects", intersects)
    mask = scene_module._reach_mask(buildings, np.ones(2), np.zeros(2), None, (area, 1.0), facade=facade)
    np.testing.assert_array_equal(mask, [True, False])
    assert candidate_counts == [1]


@pytest.mark.parametrize("detailed", [False, True])
def test_envelope_conservative_retention_preserves_results(monkeypatch, detailed):
    buildings = _envelope_buildings()
    area = gpd.GeoSeries([shapely.box(X + 45, Y + 45, X + 46, Y + 46)], crs=CRS)
    kwargs = dict(area=area, min_sun_elevation=10, time_step_min=60, grid_step=0.5, building_details=detailed)
    monkeypatch.setattr(scene_module, "_REACH_CLIP_ENABLED", False)
    full = get_insolation(buildings, "2026-06-21", **kwargs)
    monkeypatch.setattr(scene_module, "_REACH_CLIP_ENABLED", True)
    clipped = get_insolation(buildings, "2026-06-21", **kwargs)
    pd.testing.assert_frame_equal(clipped, full, check_exact=True)
    assert clipped.attrs == full.attrs


def test_flat_clip_does_not_search_for_parents(monkeypatch):
    import objectnat.methods.utils._building_details as details

    def unexpected(*args, **kwargs):
        pytest.fail("flat rows must not compute building parents")

    monkeypatch.setattr(details, "_building_parents", unexpected)
    scene = _clip_scene(_clip_buildings(), detailed=False, reach=(shapely.box(X, Y, X + 1, Y + 1), 1.0))
    assert "far_low" not in scene.blds.index
