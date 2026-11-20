"""Native point filters versus the original GEOS path, without network data."""

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from objectnat import get_facade_insolation, get_insolation
from objectnat._rust import classify_grid, classify_points
from objectnat.methods.insolation import _points
from objectnat.methods.insolation.insolation import _detailed_facade_points
from objectnat.methods.utils._building_details import _exposed_undersides, polygon_arrays
from objectnat.methods.utils.scene import prepare_scene

DATA = Path(__file__).resolve().parents[1] / "docs/methods/examples/examples_data/buildings_osm3d.parquet"
X, Y, CRS = 350_000.0, 6_650_000.0, 32636


def _pair_set(pairs):
    return set(map(tuple, pairs.T))


def _assert_filters(geometries, gx, gy):
    xy = np.column_stack([np.tile(gx, len(gy)), np.repeat(gy, len(gx))])
    expected = shapely.STRtree(geometries).query(shapely.points(xy), predicate="within")
    actual = _points.within_pairs(geometries, xy, gx=gx, gy=gy)
    assert _pair_set(actual) == _pair_set(expected)
    assert _pair_set(_points.within_pairs(geometries, xy)) == _pair_set(expected)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (X, Y), (1e12, -1e12)])
def test_boundaries_holes_multipart_and_tiny_polygons(shift):
    x, y = shift
    shell = shapely.box(x, y, x + 12, y + 12).difference(shapely.box(x + 3, y + 3, x + 9, y + 9))
    geometries = np.array(
        [
            shell,
            shapely.MultiPolygon([shapely.box(x + 15, y, x + 18, y + 3), shapely.box(x + 15, y + 6, x + 18, y + 9)]),
            shapely.box(x + 0.2, y + 0.2, x + 0.3, y + 0.3),
            shapely.Polygon(),
        ],
        dtype=object,
    )
    gx, gy = x + np.arange(-1.0, 20.0), y + np.arange(-1.0, 15.0)
    _assert_filters(geometries, gx, gy)
    arrays = polygon_arrays([shell], np.zeros(2))
    states, _ = classify_grid(*arrays, gx, gy, union=True)
    xy = np.column_stack([np.tile(gx, len(gy)), np.repeat(gy, len(gx))])
    points = shapely.points(xy)
    certain = states != 2
    np.testing.assert_array_equal(states[certain] == 1, shapely.covers(shell, points[certain]))
    boundary = shapely.intersects(shapely.boundary(shell), points)
    assert (states[boundary] == 2).all()
    inside, uncertain = classify_points(*arrays, xy)
    assert _pair_set(inside).isdisjoint(_pair_set(uncertain))
    assert shapely.within(points[inside[0]], shell).all()


def test_random_concave_polygons_with_holes():
    rng = np.random.default_rng(42)
    geometries = []
    for _ in range(30):
        angle = np.arange(23) * (2 * np.pi / 23)
        radius = rng.uniform(8, 15, len(angle))
        outer = np.column_stack([np.cos(angle) * radius, np.sin(angle) * radius])
        centre = rng.uniform(-30, 30, 2) + [X, Y]
        hole = np.column_stack([np.cos(angle), np.sin(angle)]) * 2
        geometries.append(shapely.Polygon(outer + centre, holes=[hole + centre]))
    geometries = np.array(geometries)
    _assert_filters(geometries, X + np.arange(-50, 51, 0.75), Y + np.arange(-50, 51, 0.75))
    xy = rng.uniform(-50, 50, (2000, 2)) + [X, Y]
    # Include near-vertex and one-ULP offsets, not only random points far from boundaries.
    vertices = shapely.get_coordinates(geometries)
    xy = np.concatenate([xy, vertices, np.nextafter(vertices, np.inf), vertices + [0.0, 1e-8]])
    expected = shapely.STRtree(geometries).query(shapely.points(xy), predicate="within")
    assert _pair_set(_points.within_pairs(geometries, xy)) == _pair_set(expected)


def test_nearly_horizontal_crossings_and_reversed_rings():
    polygon = shapely.Polygon([(X, Y), (X + 100, Y + 1e-6), (X + 100, Y + 10), (X, Y + 10)])
    geometries = np.array([polygon, shapely.reverse(polygon)])
    _assert_filters(geometries, X + np.arange(-1, 102, 0.5), Y + np.array([0, 1e-8, 5e-7, 1e-6, 0.5, 10]))


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("empty_grid", [False, True])
def test_ground_masks_match_original_on_all_area_types(monkeypatch, detailed, empty_grid):
    blds = gpd.GeoDataFrame(geometry=[shapely.box(2, 2, 6, 6), shapely.box(7, 2, 9, 5)], crs=CRS)
    scene = SimpleNamespace(
        blds=blds,
        base=np.array([0.0, 1.0]),
        volumes=pd.DataFrame({"B": [0.0, 2.0]}) if detailed else None,
        ground_at=lambda xy: np.floor(xy[:, 0] / 3),
    )
    polygon = shapely.box(0, 0, 10, 10).difference(shapely.box(3, 3, 4, 4))
    areas = [
        polygon,
        shapely.MultiPolygon([polygon, shapely.box(12, 0, 14, 3)]),
        shapely.GeometryCollection([polygon, shapely.MultiPolygon([shapely.box(12, 0, 14, 3)])]),
        shapely.GeometryCollection([polygon, shapely.LineString([(0, 12), (14, 12)])]),
        shapely.LineString([(0, 0), (14, 14)]),
        shapely.Polygon(),
        shapely.GeometryCollection(),
        shapely.Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)]),  # invalid: keep GEOS semantics
    ]
    gx = np.array([]) if empty_grid else np.arange(-1.0, 16.0)
    gy = np.arange(-1.0, 16.0)
    for area in areas:
        for receiver_height in [0.0, 1.0, 3.0]:
            monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", True)
            points, xy = _points.grid_receivers(scene, area, gx, gy, receiver_height)
            monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", False)
            old_points, old_xy = _points.grid_receivers(scene, area, gx, gy, receiver_height)
            np.testing.assert_array_equal(xy.view(np.uint64), old_xy.view(np.uint64))
            np.testing.assert_array_equal(shapely.to_wkb(points), shapely.to_wkb(old_points))


def test_empty_polygon_batches_and_points():
    arrays = polygon_arrays([], np.zeros(2))
    states, (inside, uncertain) = classify_grid(*arrays, np.arange(3.0), np.arange(2.0), union=True)
    np.testing.assert_array_equal(states, np.zeros(6, np.uint8))
    assert inside.shape == uncertain.shape == (2, 0)
    assert all(p.shape == (2, 0) for p in classify_points(*arrays, np.empty((0, 2))))
    _assert_filters(np.array([], dtype=object), np.arange(3.0), np.arange(2.0))


@pytest.mark.parametrize("case", ["decreasing", "nan", "shape", "offsets"])
def test_classifier_validates_arrays(case):
    arrays = list(polygon_arrays([shapely.box(0, 0, 1, 1)], np.zeros(2)))
    if case == "decreasing":
        with pytest.raises(ValueError, match="non-decreasing"):
            classify_grid(*arrays, np.array([1.0, 0.0]), np.array([0.0]))
    elif case == "nan":
        with pytest.raises(ValueError, match="finite"):
            classify_points(*arrays, np.array([[np.nan, 0.0]]))
    elif case == "shape":
        with pytest.raises(ValueError, match="shape"):
            classify_points(*arrays, np.zeros((1, 3)))
    else:
        arrays[1] = np.array([1, 4], dtype=np.int64)
        with pytest.raises(ValueError, match="ring_offsets"):
            classify_grid(*arrays, np.array([0.0]), np.array([0.0]))


def _buildings():
    return gpd.GeoDataFrame(
        {
            "height": [12.0, 6.0, 12.0, 7.0],
            "min_height": [0.0, 0.0, 6.0, 0.0],
            "roof:shape": ["gabled", "flat", "skillion", "flat"],
        },
        geometry=[
            shapely.box(X, Y, X + 10, Y + 10),
            shapely.box(X + 10.05, Y, X + 14, Y + 10),
            shapely.box(X + 9, Y - 1, X + 15, Y + 11),
            shapely.box(X - 10, Y, X - 2, Y + 9),
        ],
        index=["gable", "low_neighbour", "raised", "separate"],
        crs=CRS,
    )


@pytest.mark.parametrize("floors", [None, [1, 3, 5], []])
def test_facade_floor_expansion_matches_original(monkeypatch, floors):
    scene = prepare_scene(_buildings(), None, None, "height", 5, "height", None, 3.0, None, building_details=True)
    args = (scene, np.arange(len(scene.blds)), 2.0, 3.0, 1.2, floors)
    monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", True)
    actual = _detailed_facade_points(*args)
    monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", False)
    expected = _detailed_facade_points(*args)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", [get_insolation, get_facade_insolation])
@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("terrain_and_trees", [False, True])
def test_complete_outputs_are_exact(monkeypatch, method, detailed, terrain_and_trees):
    buildings = _buildings()
    kwargs = dict(
        area=shapely.box(X - 15, Y - 15, X + 20, Y + 15),
        building_details=detailed,
        time_step_min=120,
        min_sun_elevation=8,
    )
    kwargs.update({"grid_step": 2} if method is get_insolation else {"spacing": 2, "floors": [1, 2, 4]})
    if terrain_and_trees:
        heights = np.arange(144, dtype=float).reshape(12, 12) / 50
        kwargs["terrain"] = (heights, (5.0, 0.0, X - 25, 0.0, -5.0, Y + 25), CRS)
        kwargs["trees"] = gpd.GeoDataFrame(geometry=[shapely.Point(X - 5, Y - 5)], crs=CRS)
    monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", True)
    actual = method(buildings, ("2026-06-20", "2026-06-22"), **kwargs)
    monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", False)
    expected = method(buildings, ("2026-06-20", "2026-06-22"), **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(shapely.to_wkb(actual.geometry), shapely.to_wkb(expected.geometry))
    assert actual.attrs == expected.attrs


def test_example_receiver_and_facade_filters(monkeypatch):
    if not DATA.exists():
        pytest.skip("local OSM 3D example not installed")
    buildings = gpd.read_parquet(DATA)
    centre = buildings.geometry.iloc[0].representative_point()
    nearby = buildings[shapely.distance(buildings.geometry.values, centre) < 100]
    area = shapely.buffer(centre, 80)
    for detailed in [False, True]:
        for method in [get_insolation, get_facade_insolation]:
            kwargs = dict(area=area, building_details=detailed, default_height=10.0, time_step_min=180)
            kwargs.update({"grid_step": 4} if method is get_insolation else {"spacing": 4})
            monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", True)
            actual = method(nearby, "2026-06-21", **kwargs)
            monkeypatch.setattr(_points, "_USE_NATIVE_POINTS", False)
            expected = method(nearby, "2026-06-21", **kwargs)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            np.testing.assert_array_equal(shapely.to_wkb(actual.geometry), shapely.to_wkb(expected.geometry))
            assert actual.attrs == expected.attrs


def test_small_supports_skip_intersections(monkeypatch):
    polygons = np.array([shapely.box(0, 0, 100, 100), shapely.box(1, 1, 2, 2), shapely.box(3, 3, 4, 4)])
    original = shapely.intersection
    pair_counts = []

    def intersection(a, b, **kwargs):
        pair_counts.append(len(a))
        return original(a, b, **kwargs)

    monkeypatch.setattr(shapely, "intersection", intersection)
    actual = _exposed_undersides(polygons, np.array([2, 0, 0]), np.array([4, 2, 2]), np.array([True, False, False]))
    np.testing.assert_array_equal(actual, [True, False, False])
    assert sum(pair_counts) == 0


@pytest.mark.parametrize("gap", [0.0, 0.25, 1e-6 - 1e-10, 1e-6, 1e-6 + 1e-10, 1e-6 + 1e-8])
@pytest.mark.parametrize("shift", [0.0, 1e6])
def test_support_area_bound_preserves_tolerance_decision(gap, shift):
    polygons = np.array(
        [
            shapely.box(shift, shift, shift + 1, shift + 1),
            shapely.box(shift, shift, shift + 0.5, shift + 1),
            shapely.box(shift + 0.5 + gap, shift, shift + 1, shift + 1),
        ]
    )
    covered = shapely.union_all(polygons[1:])
    exposed = shapely.area(shapely.difference(polygons[0], covered)) > 1e-6 * shapely.area(polygons[0])
    actual = _exposed_undersides(polygons, np.array([2, 0, 0]), np.array([4, 2, 2]), np.array([True, False, False]))
    np.testing.assert_array_equal(actual, [exposed, False, False])
