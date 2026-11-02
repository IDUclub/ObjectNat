"""Tests for the optional Rust visibility core and its Python bridge."""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely import LineString, Point, Polygon, box

from objectnat import config, get_visibility
from objectnat.methods.visibility._native import (
    _largest_part_at,
    _polygon_parts,
    geometries_to_segments,
    native_available,
)
from objectnat.methods.visibility.visibility_analysis import _visibility_accurate

requires_core = pytest.mark.skipif(not native_available(), reason="objectnat-core is not installed")

CRS = 32636
X0, Y0 = 400_000.0, 6_600_000.0


def _iou(a, b) -> float:
    union = shapely.union(a, b).area
    return 1.0 if union == 0 else shapely.intersection(a, b).area / union


@pytest.fixture
def python_only():
    previous = config.enable_rust_core
    config.set_enable_rust_core(False)
    yield
    config.set_enable_rust_core(previous)


# --- geometries_to_segments (pure Python, always runs) -------------------------------------------


def test_segments_from_polygon_with_hole_are_oriented_interior_left():
    outer = [(0, 0), (0, 10), (10, 10), (10, 0)]  # clockwise on purpose
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]  # counter-clockwise on purpose
    segments, one_sided = geometries_to_segments([Polygon(outer, [hole])])

    assert segments.shape == (8, 4)
    assert one_sided.all()
    polygon = Polygon(outer, [hole])
    for x1, y1, x2, y2 in segments:
        # a point just left of the wall's midpoint must be inside the polygon body
        mx, my, dx, dy = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        left = Point(mx - dy / length * 0.01, my + dx / length * 0.01)
        assert polygon.contains(left), "polygon interior must lie on the left of every wall"


def test_segments_from_lines_and_invalid_polygons_are_two_sided():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    segments, one_sided = geometries_to_segments([LineString([(0, 0), (1, 0), (1, 1)]), bowtie])
    assert len(segments) == 2 + 4
    assert not one_sided.any()


def test_segments_ignore_points_and_empty_geometries():
    segments, one_sided = geometries_to_segments([Point(0, 0), Polygon(), None])
    assert segments.shape == (0, 4)
    assert one_sided.shape == (0,)


# --- Rust core -------------------------------------------------------------------------------------


@requires_core
def test_single_wall_matches_exact_half_disc():
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 + 30, Y0 - 500, X0 + 31, Y0 + 500)], crs=CRS)
    point = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)

    result = get_visibility(point, obstacles, 100).geometry.iloc[0]

    expected = Point(X0, Y0).buffer(100).intersection(box(X0 - 200, Y0 - 200, X0 + 30, Y0 + 200))
    assert result.symmetric_difference(expected).area < 1e-6


@requires_core
def test_rust_matches_python_on_example_buildings(buildings_data, python_only):  # pylint: disable=unused-argument
    rng = np.random.default_rng(7)
    minx, miny, maxx, maxy = buildings_data.total_bounds
    union = buildings_data.union_all()
    xy = []
    while len(xy) < 8:
        p = Point(rng.uniform(minx + 1500, maxx - 1500), rng.uniform(miny + 1000, maxy - 1000))
        if not union.contains(p):
            xy.append(p)
    points = gpd.GeoDataFrame(geometry=xy, crs=buildings_data.crs)

    python_result = get_visibility(points, buildings_data, 250)
    config.set_enable_rust_core(True)
    rust_result = get_visibility(points, buildings_data, 250)

    assert rust_result.crs == points.crs
    assert rust_result.geometry.is_valid.all()
    for a, b in zip(python_result.geometry, rust_result.geometry):
        assert _iou(a, b) > 0.99999


@requires_core
def test_observer_inside_obstacle_gets_empty_polygon():
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 - 5, Y0 - 5, X0 + 5, Y0 + 5)], crs=CRS)
    points = gpd.GeoDataFrame(geometry=[Point(X0, Y0), Point(X0 + 50, Y0)], crs=CRS)

    result = get_visibility(points, obstacles, 30)

    assert result.geometry.iloc[0].is_empty
    assert not result.geometry.iloc[1].is_empty


def test_repair_keeps_the_part_around_the_observer():
    # a bow-tie: make_valid splits it into a MultiPolygon (possibly inside a collection)
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    repaired = _largest_part_at(bowtie, Point(1, 5))
    assert repaired.geom_type == "Polygon"
    assert repaired.is_valid
    assert repaired.distance(Point(1, 5)) == 0
    assert repaired.area == pytest.approx(25.0)


def test_polygon_parts_flattens_nested_collections():
    a, b = box(0, 0, 1, 1), box(2, 2, 3, 3)
    nested = shapely.GeometryCollection([shapely.MultiPolygon([a, b]), LineString([(0, 0), (1, 1)])])
    assert len(_polygon_parts(nested)) == 2


@requires_core
def test_line_obstacles_cast_shadows():
    wall = LineString([(X0 + 20, Y0 - 300), (X0 + 20, Y0 + 300)])
    obstacles = gpd.GeoDataFrame(geometry=[wall], crs=CRS)
    point = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)

    result = get_visibility(point, obstacles, 50).geometry.iloc[0]

    expected = Point(X0, Y0).buffer(50).intersection(box(X0 - 100, Y0 - 100, X0 + 20, Y0 + 100))
    assert result.symmetric_difference(expected).area < 1e-6


@requires_core
def test_private_accurate_dispatch_matches_python_with_holes_and_walls(python_only):
    # courtyard building with a gate + a free-standing wall; the observer stands in the courtyard.
    # The wall is a thin polygon because the Python implementation does not accept lines here.
    courtyard = Polygon(
        [(X0 - 40, Y0 - 40), (X0 + 40, Y0 - 40), (X0 + 40, Y0 + 40), (X0 - 40, Y0 + 40)],
        [[(X0 - 20, Y0 - 20), (X0 + 20, Y0 - 20), (X0 + 20, Y0 + 20), (X0 - 20, Y0 + 20)]],
    )
    gate = box(X0 + 19, Y0 - 3, X0 + 41, Y0 + 3)
    building = courtyard.difference(gate)
    wall = box(X0 + 60, Y0 - 30, X0 + 61, Y0 + 30)
    obstacles = gpd.GeoDataFrame(geometry=[building, wall], crs=CRS)
    observer = Point(X0 - 10, Y0)

    python_poly = _visibility_accurate(observer, obstacles, 150)
    config.set_enable_rust_core(True)
    rust_poly = _visibility_accurate(observer, obstacles, 150)

    assert rust_poly.is_valid
    assert _iou(python_poly, rust_poly) > 0.99999


def test_rust_core_is_not_called_when_disabled(python_only, monkeypatch):  # pylint: disable=unused-argument
    def fail(*args, **kwargs):
        raise AssertionError("the Rust core must not be used when disabled")

    monkeypatch.setattr("objectnat.methods.visibility.visibility_analysis.native_visibility_polygons", fail)
    monkeypatch.setattr("objectnat.methods.visibility.visibility_analysis.native_visibility_single", fail)
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 + 10, Y0 - 10, X0 + 12, Y0 + 10)], crs=CRS)
    point = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)
    result = get_visibility(point, obstacles, 50)
    assert not result.geometry.iloc[0].is_empty


@requires_core
def test_core_validates_shapes():
    from objectnat_core import visibility_polygons  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError):
        visibility_polygons(np.zeros((2, 3)), np.ones(2), np.zeros((0, 4)))
    with pytest.raises(ValueError):
        visibility_polygons(np.zeros((2, 2)), np.ones(3), np.zeros((0, 4)))
    with pytest.raises(ValueError):
        visibility_polygons(np.zeros((1, 2)), np.ones(1), np.zeros((2, 4)), one_sided=np.ones(3, dtype=bool))
