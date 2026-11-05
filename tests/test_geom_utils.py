"""Unit tests for the pure geometry helpers in ``utils/geom_utils``.

All in-memory shapely/geopandas — no network graph or data fixtures.
"""

import geopandas as gpd
import pytest
from shapely import LineString, MultiPolygon, Point, Polygon, box

from objectnat.methods.utils.geom_utils import (
    distribute_points_on_linestrings,
    distribute_points_on_polygons,
    get_point_from_a_thorough_b,
    polygons_to_multilinestring,
    remove_inner_geom,
)


# --------------------------------------------------------------------------- #
# polygons_to_multilinestring
# --------------------------------------------------------------------------- #
def test_polygons_to_multilinestring_simple_polygon():
    result = polygons_to_multilinestring(box(0, 0, 1, 1))
    assert result.geom_type == "MultiLineString"
    assert len(result.geoms) == 1


def test_polygons_to_multilinestring_polygon_with_hole():
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    result = polygons_to_multilinestring(poly)
    # exterior ring + one interior ring
    assert len(result.geoms) == 2


def test_polygons_to_multilinestring_multipolygon():
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    result = polygons_to_multilinestring(multi)
    assert len(result.geoms) == 2


def test_polygons_to_multilinestring_rejects_non_polygon():
    with pytest.raises(TypeError):
        polygons_to_multilinestring(Point(0, 0))


# --------------------------------------------------------------------------- #
# get_point_from_a_thorough_b
# --------------------------------------------------------------------------- #
def test_get_point_from_a_thorough_b_along_axes():
    east = get_point_from_a_thorough_b(Point(0, 0), Point(1, 0), 5)
    assert (east.x, east.y) == pytest.approx((5.0, 0.0))

    north = get_point_from_a_thorough_b(Point(0, 0), Point(0, 2), 3)
    assert (north.x, north.y) == pytest.approx((0.0, 3.0))


# --------------------------------------------------------------------------- #
# remove_inner_geom
# --------------------------------------------------------------------------- #
def test_remove_inner_geom_drops_holes():
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    result = remove_inner_geom(poly)
    assert len(result.interiors) == 0
    assert result.area == pytest.approx(100.0)


def test_remove_inner_geom_multipolygon():
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    result = remove_inner_geom(multi)
    assert isinstance(result, MultiPolygon)
    assert len(result.geoms) == 2


def test_remove_inner_geom_non_polygon_returns_empty():
    result = remove_inner_geom(LineString([(0, 0), (1, 1)]))
    assert result.is_empty


# --------------------------------------------------------------------------- #
# distribute_points_on_linestrings / distribute_points_on_polygons
# --------------------------------------------------------------------------- #
def test_distribute_points_on_linestrings_smoke():
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(30.300, 59.90), (30.303, 59.90)])],
        crs=4326,
    )
    points = distribute_points_on_linestrings(lines, radius=20)
    assert isinstance(points, gpd.GeoDataFrame)
    assert not points.empty
    assert points.crs == lines.crs
    assert (points.geom_type == "Point").all()


def test_distribute_points_on_polygons_smoke():
    polygons = gpd.GeoDataFrame(
        geometry=[box(30.300, 59.900, 30.303, 59.902)],
        crs=4326,
    )
    points = distribute_points_on_polygons(polygons, radius=20)
    assert isinstance(points, gpd.GeoDataFrame)
    assert not points.empty
    assert points.crs == polygons.crs
