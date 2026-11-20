"""Offline tree parsing, expansion, bindings and public insolation integration."""

import datetime as dt
import importlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, box

from objectnat import get_facade_insolation, get_insolation
from objectnat._rust import insolation_day_stats, insolation_day_stats_trees, line_of_sight
from objectnat.methods.insolation._trees import _expand, leaf_state, prepare_trees, validate_tree_parameters
from objectnat.methods.utils.scene import Scene

CRS = 32636
X0, Y0 = 350_000.0, 6_650_000.0
INSOLATION = importlib.import_module("objectnat.methods.insolation.insolation")


def _scene(terrain=None):
    buildings = gpd.GeoDataFrame({"height": []}, geometry=[], crs=CRS)
    return Scene(CRS, buildings, np.empty(0), np.empty(0), terrain, np.empty((0, 4)), np.empty(0), np.empty(0, int))


def _prepare(trees, *, scene=None, footprints=None, spacing=None):
    scene = _scene() if scene is None else scene
    footprints = scene.blds if footprints is None else footprints
    return prepare_trees(scene, trees, footprints, 10.0, 6.0, 0.3, (0.2, 0.6), spacing)


def test_tree_numeric_defaults_shapes_cycles_and_terrain(monkeypatch):
    messages = []
    monkeypatch.setattr("objectnat.methods.insolation._trees.logger.warning", messages.append)
    trees = gpd.GeoDataFrame(
        {
            "height": ["12,5 m", "bad", None, "8;9", "0", "-2", "10"],
            "est_height": [None, "8 m", None, None, None, None, None],
            "diameter_crown": ["5,0 m", None, "bad", "4;6", "6", "6", "0"],
            "leaf_type": ["broadleaved", "needleleaved", "mixed", "needleleaved", None, None, None],
            "leaf_cycle": [None, None, "evergreen", "semi_evergreen", None, None, None],
        },
        geometry=[Point(k * 10, 0) for k in range(7)],
        crs=CRS,
    )
    terrain = (np.full((3, 3), 20.0), -100.0, -100.0, 100.0, 100.0)
    crowns = _prepare(trees, scene=_scene(terrain)).trees
    assert crowns.shape == (4, 8)
    np.testing.assert_array_equal(crowns[:, 7], [0, 1, 0, 1])
    np.testing.assert_allclose(crowns[:, 2], [20 + 0.65 * 12.5, 20 + 0.3 * 8, 26.5, 22.4])
    np.testing.assert_allclose(crowns[:, 3], [2.5, 3, 3, 2])
    np.testing.assert_allclose(crowns[:, 4], [0.35 * 12.5, 0.7 * 8, 3.5, 5.6])
    np.testing.assert_allclose(crowns[:, 5], -np.log(0.2) / np.array([5, 6, 6, 4]))
    np.testing.assert_allclose(crowns[[1, 2], 6], crowns[[1, 2], 5])
    np.testing.assert_allclose(crowns[[0, 3], 6], -np.log(0.6) / np.array([5, 4]))
    assert len(messages) == 1 and "3 tree features" in messages[0]


def test_leaf_cycle_inference_and_explicit_overrides():
    trees = gpd.GeoDataFrame(
        {
            "leaf_type": [None, "other", "needleleaved", "needleleaved", "broadleaved", "needleleaved"],
            "leaf_cycle": [None, "semi_deciduous", None, "deciduous", "evergreen", "semi_evergreen"],
        },
        geometry=[Point(k, 0) for k in range(6)],
        crs=CRS,
    )
    crowns = _prepare(trees).trees
    np.testing.assert_array_equal(crowns[:, 5] == crowns[:, 6], [False, False, True, False, True, False])


def test_point_and_multipart_expansion_and_unsupported_geometry_warning(monkeypatch):
    messages = []
    monkeypatch.setattr("objectnat.methods.insolation._trees.logger.warning", messages.append)
    geometries = np.array(
        [Point(0, 0), MultiPoint([(1, 0), (2, 0)]), GeometryCollection([Point(99, 99)]), Point()], dtype=object
    )
    xy, owner = _expand(geometries, np.full(4, 6.0))
    np.testing.assert_array_equal(xy, [[0, 0], [1, 0], [2, 0]])
    np.testing.assert_array_equal(owner, [0, 1, 1])
    assert len(messages) == 1 and "1 unsupported" in messages[0]


def test_row_expansion_exact_spacing_centred_and_short_lines():
    lines = np.array(
        [LineString([(0, 0), (20, 0)]), MultiLineString([[(0, 10), (12, 10)], [(0, 20), (2, 20)]])],
        dtype=object,
    )
    xy, owner = _expand(lines, np.array([6.0, 4.0]))
    np.testing.assert_allclose(xy, [[4, 0], [10, 0], [16, 0], [2, 10], [6, 10], [10, 10], [1, 20]])
    np.testing.assert_array_equal(owner, [0, 0, 0, 1, 1, 1, 1])


def test_polygon_hexagonal_spacing_holes_and_small_multipart_fallback():
    polygon = box(0, 0, 20, 20).difference(box(6, 6, 14, 14))
    tiny = MultiPolygon([box(30, 0, 30.1, 0.1), box(40, 0, 40.1, 0.1)])
    xy, owner = _expand(np.array([polygon, tiny], dtype=object), np.array([4.0, 6.0]))
    large = xy[owner == 0]
    assert len(large) > 10
    assert shapely.covers(polygon, shapely.points(large)).all()
    distances = np.linalg.norm(large[:, None] - large[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_allclose(distances.min(axis=1), 4.0)
    assert (owner == 1).sum() == 2
    assert shapely.covers(tiny, shapely.points(xy[owner == 1])).all()
    regular, _ = _expand(np.array([box(0, 0, 8, 8)], dtype=object), np.array([4.0]))
    assert len(regular) == 4


def test_footprint_drop_warns_unique_count_and_keeps_courtyard(monkeypatch):
    messages = []
    monkeypatch.setattr("objectnat.methods.insolation._trees.logger.warning", messages.append)
    footprint = box(-5, -5, 5, 5).difference(box(-1, -1, 1, 1))
    buildings = gpd.GeoDataFrame(geometry=[footprint, footprint], crs=CRS)
    trees = gpd.GeoDataFrame(geometry=[MultiPoint([(0, 0), (3, 0), (10, 0)])], crs=CRS)
    crowns = _prepare(trees, footprints=buildings).trees
    np.testing.assert_array_equal(crowns[:, :2], [[0, 0], [10, 0]])
    assert len(messages) == 1 and "1 trees" in messages[0]


def test_tree_crs_reprojection():
    trees = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)
    np.testing.assert_allclose(_prepare(trees.to_crs(4326)).trees, _prepare(trees).trees, atol=1e-8)


def test_leaf_season_boundaries_wrap_and_leap_day():
    days = [dt.date(2024, month, day) for month, day in [(2, 29), (4, 30), (5, 1), (10, 15), (10, 16), (12, 31)]]
    np.testing.assert_array_equal(leaf_state(days, ("05-01", "10-15")), [0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(leaf_state(days, ("10-15", "05-01")), [1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(leaf_state(days, ("02-29", "02-29")), [1, 0, 0, 0, 0, 0])
    assert leaf_state([dt.date(2026, 7, 1)], ("10-15", "05-01"))[0] == 0


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("height", 0),
        ("height", np.inf),
        ("height", None),
        ("diameter", -1),
        ("diameter", np.nan),
        ("base", -0.1),
        ("base", 1),
        ("base", np.nan),
        ("base", "bad"),
        ("transmittance", (0, 0.6)),
        ("transmittance", (0.2, 1.1)),
        ("transmittance", (np.nan, 0.6)),
        ("transmittance", (0.2,)),
        ("spacing", 0),
        ("spacing", np.inf),
        ("threshold", -0.1),
        ("threshold", 1.1),
        ("threshold", np.nan),
        ("leaf_on", ("02-30", "10-15")),
        ("leaf_on", ("5-1", "10-15")),
        ("leaf_on", ("05-01",)),
    ],
)
def test_tree_parameter_validation(parameter, value):
    parameters = dict(
        height=10.0,
        diameter=6.0,
        base=0.3,
        transmittance=(0.2, 0.6),
        leaf_on=("05-01", "10-15"),
        spacing=None,
        threshold=0.5,
    )
    parameters[parameter] = value
    with pytest.raises(ValueError):
        validate_tree_parameters(**parameters)


@pytest.mark.parametrize("bad", [None, pd.DataFrame(), gpd.GeoDataFrame(geometry=[Point(0, 0)])])
def test_trees_require_geodataframe_with_crs(bad):
    with pytest.raises(ValueError, match="trees must be"):
        _prepare(bad)


def _fixed_sun(days, *_args):
    return np.tile([0.0, -1.0, 0.1], (len(days), 1)), list(range(len(days) + 1))


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize("facade", [False, True])
def test_public_tree_shading_exact_diameter_all_modes(monkeypatch, details, facade):
    monkeypatch.setattr(INSOLATION, "_sun_positions", _fixed_sun)
    building = box(X0 - 1, Y0 if facade else Y0 + 1, X0 + 1, Y0 + 3)
    buildings = gpd.GeoDataFrame({"height": [9.0]}, geometry=[building], crs=CRS)
    function = get_facade_insolation if facade else get_insolation
    options = dict(building_details=details, time_step_min=60.0)
    if facade:
        options.update(floors=[3], window_height=0.0, spacing=10.0)
    else:
        options.update(receiver_height=6.0, area=box(X0 - 0.01, Y0 - 0.01, X0 + 0.01, Y0 + 0.01), grid_step=2.0)
    tree_y = Y0 - 10.1 if facade else Y0 - 10.0
    trees = gpd.GeoDataFrame(geometry=[Point(X0, tree_y)], crs=CRS)
    baseline = function(buildings, "2026-06-21", **options)
    explicit_none = function(buildings, "2026-06-21", trees=None, **options)
    pd.testing.assert_frame_equal(baseline, explicit_none)
    assert baseline.attrs == explicit_none.attrs
    assert "sun_hours_no_trees" not in baseline and "trees" not in baseline.attrs
    shaded = function(buildings, "2026-06-21", trees=trees, tree_crown_base=0.4, **options)
    np.testing.assert_array_equal(shaded["sun_hours_no_trees"], baseline["sun_hours"])
    assert shaded.attrs["trees"] == 1
    assert len(shaded) == len(baseline)
    selected = (shaded["facade_azimuth"] - 180).abs() < 5 if facade else np.ones(len(shaded), bool)
    np.testing.assert_allclose(shaded.loc[selected, "sun_hours"], 0.2, atol=1e-8)
    np.testing.assert_allclose(shaded.loc[selected, "min_sun_hours"], 0.2, atol=1e-8)
    np.testing.assert_array_equal(shaded.loc[selected, "max_continuous_hours"], 0.0)
    winter = function(buildings, "2026-01-21", trees=trees, tree_crown_base=0.4, **options)
    np.testing.assert_allclose(winter.loc[selected, "sun_hours"], 0.6, atol=1e-8)
    np.testing.assert_array_equal(winter.loc[selected, "max_continuous_hours"], 1.0)
    period = function(
        buildings, ("2026-01-21", "2026-06-21"), trees=trees, tree_crown_base=0.4, day_step=1000, **options
    )
    np.testing.assert_allclose(period.loc[selected, "sun_hours"], 0.4, atol=1e-8)
    np.testing.assert_allclose(period.loc[selected, "min_sun_hours"], 0.2, atol=1e-8)
    np.testing.assert_array_equal(period["sun_hours_no_trees"], baseline["sun_hours"])
    empty = function(buildings, "2026-06-21", trees=trees.iloc[:0], **options)
    assert empty.attrs["trees"] == 0
    pd.testing.assert_frame_equal(empty[baseline.columns], baseline)
    np.testing.assert_array_equal(empty["sun_hours_no_trees"], baseline["sun_hours"])


def _kernel_inputs():
    return dict(
        receivers=np.array([[0.0, 0.0, 0.0]]),
        suns=np.array([[1.0, 0.0, 1.0]]),
        day_offsets=np.array([0, 1], dtype=np.int64),
        segments=np.empty((0, 4)),
        tops=np.empty(0),
        trees=np.array([[4.0, 0.0, 4.0, 1.0, 1.0, -np.log(0.2) / 2, -np.log(0.6) / 2, 0.0]]),
        leaf_on=np.array([1], dtype=np.uint8),
    )


def test_tree_binding_stats_and_visibility_unchanged():
    inputs = _kernel_inputs()
    weighted, baseline, longest = insolation_day_stats_trees(**inputs)
    assert weighted.dtype == np.float64 and baseline.dtype == longest.dtype == np.uint32
    assert weighted[0, 0] == pytest.approx(0.2)
    assert baseline[0, 0] == 1 and longest[0, 0] == 0
    del inputs["trees"], inputs["leaf_on"]
    old, _ = insolation_day_stats(**inputs)
    np.testing.assert_array_equal(old, baseline)
    assert (
        line_of_sight(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[8.0, 0.0, 8.0]]),
            np.array([0], dtype=np.int64),
            inputs["segments"],
            inputs["tops"],
        )[0]
        == 1
    )


@pytest.mark.parametrize(
    "parameter,value,message",
    [
        ("trees", np.empty((0, 7)), "trees"),
        ("trees", np.array([[0, 0, 0, 0, 1, 1, 1, 0.0]]), "trees"),
        ("trees", np.array([[0, 0, 0, 1, 1, 1, 1, 2.0]]), "trees"),
        ("trees", np.array([[0, 0, 0, 1, 1, -1, 1, 0.0]]), "trees"),
        ("trees", np.array([[np.nan, 0, 0, 1, 1, 1, 1, 0.0]]), "trees"),
        ("leaf_on", np.array([], dtype=np.uint8), "leaf_on"),
        ("leaf_on", np.array([2], dtype=np.uint8), "leaf_on"),
        ("sunlit_threshold", np.nan, "sunlit_threshold"),
        ("sunlit_threshold", 1.1, "sunlit_threshold"),
        ("day_offsets", np.array([1, 0], dtype=np.int64), "day_offsets"),
        ("normals", np.zeros((2, 2)), "normals"),
    ],
)
def test_tree_binding_rejects_invalid_arrays(parameter, value, message):
    inputs = _kernel_inputs()
    inputs[parameter] = value
    with pytest.raises(ValueError, match=message):
        insolation_day_stats_trees(**inputs)


def test_empty_receivers_and_empty_days():
    inputs = _kernel_inputs()
    inputs["receivers"] = np.empty((0, 3))
    assert all(array.shape == (0, 1) for array in insolation_day_stats_trees(**inputs))
    inputs["receivers"] = np.array([[0.0, 0.0, 0.0]])
    inputs["day_offsets"] = np.array([0, 0], dtype=np.int64)
    assert all(array[0, 0] == 0 for array in insolation_day_stats_trees(**inputs))


def test_no_tree_dispatch_and_tree_coordinate_shift(monkeypatch):
    native = importlib.import_module("objectnat._rust")
    called = []

    def record(*args, **kwargs):
        called.append((args, kwargs))
        return np.ones((1, 1)), np.ones((1, 1), dtype=np.uint32), np.ones((1, 1), dtype=np.uint32)

    monkeypatch.setattr(native, "insolation_day_stats_trees", record)
    scene = _scene()
    xyz = np.array([[X0, Y0, 0.0]])
    suns = np.array([[0.0, -1.0, 1.0]])
    assert len(INSOLATION._day_stats(scene, xyz, suns, [0, 1])) == 2
    assert not called
    crowns = np.array([[X0, Y0 - 4, 4.0, 1.0, 1.0, 0.8, 0.3, 0.0]])
    scene = scene._replace(trees=crowns)
    result = INSOLATION._day_stats(scene, xyz, suns, [0, 1], days=[dt.date(2026, 6, 21)])
    assert len(result) == 3 and len(called) == 1
    args, _ = called[0]
    np.testing.assert_array_equal(args[0], [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(args[5][:, :2], [[0.0, -4.0]])
    np.testing.assert_array_equal(args[6], [1])
    np.testing.assert_array_equal(scene.trees, crowns)


def test_transparent_crowns_preserve_hours():
    inputs = _kernel_inputs()
    inputs["trees"][:, 5:7] = 0.0  # tau = 1
    weighted, opaque, runs = insolation_day_stats_trees(**inputs, sunlit_threshold=1.0)
    np.testing.assert_array_equal(weighted, opaque)
    np.testing.assert_array_equal(runs, opaque)
