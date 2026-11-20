"""Offline S3DB integration, legacy compatibility and an independent ray/triangle oracle."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal
from pyproj import Transformer
from shapely import MultiPolygon, Polygon, box

from objectnat import _rust, get_facade_insolation, get_insolation
from objectnat.methods.insolation.insolation import _detailed_facade_points
from objectnat.methods.utils._building_details import (
    _boundary_pieces,
    _exposed_undersides,
    _facade_boundaries,
    _grid_north_azimuths,
    _heights,
    _volume_rows,
    osm_direction,
    osm_number,
    polygon_arrays,
)
from objectnat.methods.utils.scene import Scene, grid_north_azimuth, kernel_scene, prepare_scene

CRS = 32636
X0, Y0 = 350_000.0, 6_650_000.0
DATA = Path(__file__).resolve().parents[1] / "docs/methods/examples/examples_data/buildings_osm3d.parquet"


def _buildings(data, geometries=None, index=None):
    if geometries is None:
        geometries = [box(X0, Y0, X0 + 20, Y0 + 10)]
    return gpd.GeoDataFrame(data, geometry=geometries, index=index, crs=CRS)


def _scene(buildings, **kwargs):
    args = dict(
        buildings=buildings,
        area=None,
        terrain=None,
        terrain_height_column="height",
        terrain_cell=5,
        height_column="height",
        storeys_column=None,
        storey_height=3,
        default_height=None,
        building_details=True,
    )
    args.update(kwargs)
    return prepare_scene(**args)


@pytest.mark.parametrize(
    "value,expected", [("12", 12), ("12 m", 12), ("12,5", 12.5), ("3;4", 3), (" -2.5m ", -2.5), (3.2, 3.2), (".5", 0.5)]
)
def test_pure_python_osm_numbers(value, expected):
    assert osm_number(value) == expected


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, "garbage", "abc12", "inf", ""])
def test_pure_python_missing_numbers(value):
    assert np.isnan(osm_number(value))


@pytest.mark.parametrize(
    "value,expected", [("N", 0), (" nne ", 22.5), ("SW", 225), ("NNW", 337.5), ("180", 180), ("90,5", 90.5)]
)
def test_pure_python_cardinal_bearings(value, expected):
    assert osm_direction(value) == expected


def test_pure_python_parts_parents_and_inherited_heights():
    blds = _buildings(
        {
            "building:part": [None, "yes", "yes", "yes"],
            "height": ["18 m", None, "9", None],
            "building:levels": [None] * 4,
            "min_height": ["8", None, None, "3"],
            "roof:height": ["5", None, None, None],
            "custom": [None, "7", None, "12"],
        },
        [
            box(X0, Y0, X0 + 30, Y0 + 30),
            box(X0, Y0, X0 + 10, Y0 + 10),
            box(X0 + 10, Y0, X0 + 20, Y0 + 10),
            box(X0 + 50, Y0, X0 + 60, Y0 + 10),
        ],
        ["outline", "inherited", "own", "isolated"],
    )
    rows = _volume_rows(blds)
    assert rows.index.tolist() == ["inherited", "own", "isolated", "outline"]
    assert rows["parent"].tolist() == ["outline", "outline", "isolated", "outline"]
    bottom, eave, top, _ = _heights(rows, "custom", None, 3, None, 30)
    np.testing.assert_array_equal(top, [18, 9, 12, 18])
    np.testing.assert_array_equal(eave, top)  # neither roof nor min_height is inherited
    np.testing.assert_array_equal(bottom, [0, 0, 3, 8])
    blds.loc["outline", ["height", "building:levels"]] = [None, "4"]
    rows = _volume_rows(blds)
    assert _heights(rows, "custom", None, 3, None, 30)[2].tolist() == [12, 9, 12, 12]
    # Own levels take priority over the outline's height; own roof tags still apply.
    blds.loc["inherited", ["building:levels", "roof:height"]] = ["2", "3"]
    assert _heights(_volume_rows(blds), "custom", None, 3, None, 30)[2][0] == 9


def test_pure_python_levels_roof_levels_angle_and_clamping():
    rows = _buildings(
        {
            "building:levels": ["3;4"] * 4,
            "building:min_level": ["1"] * 4,
            "roof:shape": ["gabled", "skillion", "gabled", None],
            "roof:levels": ["1,5", None, None, None],
            "roof:angle": [None, "45", "45", None],
        },
        [box(X0, Y0, X0 + 20, Y0 + 10)] * 4,
    )
    bottom, eave, top, codes = _heights(rows, "height", None, 3, None, 30)
    np.testing.assert_array_equal(bottom, [3] * 4)
    np.testing.assert_array_equal(eave, [9] * 4)
    np.testing.assert_allclose(top, [13.5, 19, 14, 9])
    assert codes.tolist() == [2, 1, 2, 0]
    rows["height"] = "10"
    rows["roof:height"] = "50"
    bottom, eave, top, _ = _heights(rows, "height", None, 3, None, 30)
    np.testing.assert_array_equal(eave, bottom)
    np.testing.assert_array_equal(top, [10] * 4)
    rows["roof:height"] = None
    rows["roof:levels"] = None
    rows["roof:angle"] = None
    assert _heights(rows, "height", None, 3, None, 30)[1][0] == pytest.approx(10 - 5 / np.sqrt(3))


@pytest.mark.parametrize("gap,expected_rows", [(10, 2), (0.2, 1), (0, 1)])
def test_outline_remainders(gap, expected_rows):
    outline = box(X0, Y0, X0 + 20, Y0 + 20)
    blds = _buildings(
        {
            "building:part": [None, "yes"],
            "height": [12, 18],
            "roof:shape": ["gabled", "gabled"],
            "roof:height": [4, 3],
            "roof:levels": [1, 1],
            "roof:angle": [30, 30],
            "roof:direction": [90, 90],
            "roof:orientation": ["across", "across"],
        },
        [outline, box(X0, Y0, X0 + 20 - gap, Y0 + 20)],
        ["outline", "part"],
    )
    rows, groups = _volume_rows(blds, return_groups=True)
    assert len(rows) == expected_rows
    assert groups.tolist() == [0] * expected_rows
    if gap == 10:
        assert shapely.union_all(rows.geometry).equals(outline)
        remainder = rows.loc[["outline"]]
        assert remainder.parent.tolist() == ["outline"]
        assert remainder.parent_geometry.iloc[0].equals(outline)
        assert remainder.filter(like="roof:").isna().all().all()
        bottom, eave, top, shapes = _heights(remainder, "height", None, 3, None, 30)
        assert (bottom[0], eave[0], top[0], shapes[0]) == (0, 12, 12, 0)
        ground = get_insolation(
            blds, "2026-06-21", area=outline.buffer(5), building_details=True, grid_step=2, time_step_min=60
        )
        # Receivers exactly on a wall line (including the shared part/remainder edge) are kept, as before.
        assert len(ground)
        for volume in rows.geometry:
            assert not ground.within(volume).any()
        facades = get_facade_insolation(blds, "2026-06-21", building_details=True, spacing=5, time_step_min=60)
        assert set(facades.part) == {"outline", "part"}
        assert set(facades.building) == {"outline"}
    else:
        assert rows.index.tolist() == ["part"]
        assert rows.geometry.iloc[0].equals(blds.geometry.iloc[1])


def test_pure_python_multipolygons_and_holes():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (2, 4), (4, 4), (4, 2)]])
    rows = _volume_rows(_buildings({"height": [10]}, [MultiPolygon([polygon, box(20, 0, 25, 5)])], ["multi"]))
    assert rows.index.tolist() == ["multi", "multi"]
    xy, rings, polys = polygon_arrays(rows.geometry, np.array([1, 2]))
    assert xy.shape == (12, 2) and rings.tolist() == [0, 4, 8, 12] and polys.tolist() == [0, 2, 3]
    assert rings.dtype == polys.dtype == np.int64


def test_pure_python_nested_outlines_and_tuple_labels():
    frame = _buildings(
        {"building:part": [None, None, "yes"], "height": [24, 12, None]},
        [box(0, 0, 30, 30), box(0, 0, 20, 20), box(1, 1, 10, 10)],
        pd.MultiIndex.from_tuples([("outline", 1), ("outline", 2), ("part", 1)]),
    )
    rows = _volume_rows(frame)
    assert rows.index.tolist() == [("part", 1), ("outline", 1), ("outline", 2)]
    assert rows.parent.tolist() == [("outline", 2), ("outline", 1), ("outline", 2)]
    assert _heights(rows, "height", None, 3, None, 30)[2].tolist() == [12, 24, 12]


def test_pure_python_height_fallbacks_and_unknown_shapes():
    frame = _buildings(
        {
            "custom_height": ["15 m", None, None],
            "storeys": ["2", "4;5", None],
            "roof:shape": ["alien", "many", None],
            "roof:height": [3, 2, None],
        },
        [box(0, 0, 20, 10)] * 3,
    )
    bottom, eave, top, codes = _heights(frame, "custom_height", "storeys", 3, 9, 30)
    np.testing.assert_array_equal(bottom, [0, 0, 0])
    np.testing.assert_array_equal(eave, [12, 10, 9])
    np.testing.assert_array_equal(top, [15, 12, 9])
    assert codes.tolist() == [255, 16, 0]


def test_pure_python_empty_details_scene(monkeypatch):
    def empty_mesh(*args, **kwargs):
        assert args[0].shape == (0, 2)
        assert args[1].tolist() == args[2].tolist() == [0]
        return (
            (np.empty((0, 3)), np.array([0], dtype=np.int64)),
            np.empty(0, dtype=np.int64),
            np.empty((0, 6)),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        )

    monkeypatch.setattr(_rust, "roof_mesh", empty_mesh)
    frame = _buildings({"height": []}, [])
    scene = _scene(frame, area=box(X0, Y0, X0 + 10, Y0 + 10))
    segments, tops, kwargs = kernel_scene(scene, np.zeros(2))
    assert segments.shape == (0, 4) and tops.shape == (0, 2)
    assert kwargs["under_xy"].shape == (0, 2) and kwargs["bottoms"].size == 0


def test_pure_python_boundary_normals_and_kernel_shift():
    polygon = box(0, 0, 10, 10).difference(box(3, 3, 7, 7))
    edges = np.array([[0, 0, 5, 10, 0, 8], [3, 3, 5, 7, 3, 5], [0, 0, 5, 10, 10, 8]], dtype=float)
    walls, normals = _boundary_pieces(np.array([polygon], dtype=object), edges, np.zeros(3, np.int64))
    assert walls.tolist() == [0, 1]
    np.testing.assert_array_equal(normals, [[0, -1], [0, 1]])
    blds = _buildings({"height": [8]}, [polygon])
    under = (*polygon_arrays([polygon], np.zeros(2)), np.array([3.0]))
    scene = Scene(
        CRS,
        blds,
        np.array([8.0]),
        np.array([0.0]),
        None,
        edges[:, [0, 1, 3, 4]],
        edges[:, [2, 5]],
        np.zeros(3, np.int64),
        bottoms=np.full(3, 3.0),
        undersides=under,
    )
    segments, tops, kwargs = kernel_scene(scene, np.array([100.0, 200.0]))
    np.testing.assert_array_equal(segments[0], [-100, -200, -90, -200])
    np.testing.assert_array_equal(tops, scene.tops)
    np.testing.assert_array_equal(kwargs["under_xy"], under[0] - [100, 200])
    np.testing.assert_array_equal(scene.undersides[0], under[0])
    old = Scene(*scene[:8])._replace(tops=np.full(3, 8.0))
    assert kernel_scene(old, np.zeros(2))[1].ndim == 1
    assert kernel_scene(old, np.zeros(2))[2] == {}


def test_pure_python_batched_roof_arrays_direction_terrain_and_skipped_volumes(monkeypatch):
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        # Test the Python array contract independently of native roof construction.
        xy, _, _, _, _, top = args
        edges = np.column_stack([xy, np.full(len(xy), top[0]), np.roll(xy, -1, axis=0), np.full(len(xy), top[0])])
        return (
            (np.empty((0, 3)), np.array([0], dtype=np.int64)),
            np.empty(0, dtype=np.int64),
            edges,
            np.zeros(len(edges), dtype=np.int64),
            np.array([0], dtype=np.uint8),
            np.array([0], dtype=np.uint8),
        )

    monkeypatch.setattr(_rust, "roof_mesh", capture)
    blds = _buildings(
        {
            "height": ["12 m", "2"],
            "min_height": ["3", "3"],
            "roof:shape": ["gabled", None],
            "roof:height": ["4", None],
            "roof:direction": ["N", None],
            "roof:orientation": ["across", None],
        },
        [box(X0, Y0, X0 + 20, Y0 + 10), box(X0 + 30, Y0, X0 + 40, Y0 + 10)],
    )
    terrain = (np.full((20, 20), 7.0), (10, 0, X0 - 50, 0, -10, Y0 + 100), CRS)
    scene = _scene(blds, terrain=terrain)
    assert len(calls) == 1 and len(scene.volumes) == 1
    assert scene.boundary_walls is None and scene.boundary_normals is None
    args, kwargs = calls[0]
    assert abs(args[0]).max() < 30  # coordinates were shifted before the Rust call
    np.testing.assert_array_equal(args[4], [15])
    np.testing.assert_array_equal(args[5], [19])
    lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(X0 + 10, Y0 + 5)
    assert kwargs["direction"][0] == pytest.approx(-np.radians(grid_north_azimuth(CRS, lon, lat)))
    assert kwargs["across"].tolist() == [True]
    assert np.all(scene.bottoms == 10) and scene.undersides[3].tolist() == [10]


def _kernel_suns(scene, xyz, suns):
    shift = xyz[:, :2].min(axis=0)
    segments, tops, kwargs = kernel_scene(scene, shift)
    xyz = xyz.copy()
    xyz[:, :2] -= shift
    counts, _ = _rust.insolation_day_stats(
        xyz, np.asarray(suns, dtype=float), np.arange(len(suns) + 1, dtype=np.int64), segments, tops, **kwargs
    )
    return counts.astype(bool)


def test_gabled_shadow_ridge_and_eave():
    scene = _scene(_buildings({"height": [10], "roof:height": [4], "roof:shape": ["gabled"]}))
    # At tan(elevation)=0.5 the ridge at y=5 casts to y=-15; the eave to y=-12.
    xyz = np.array([[X0 + 10, Y0 - 14, 0], [X0 + 10, Y0 - 16, 0]], dtype=float)
    np.testing.assert_array_equal(_kernel_suns(scene, xyz, [[0, 1, 0.5]]).ravel(), [False, True])


def test_arch_underside_and_ground_receivers():
    blds = _buildings({"height": [10], "min_height": [5], "building:part": ["yes"]})
    scene = _scene(blds)
    xyz = np.array([[X0 + 10, Y0 + 5, 0], [X0 + 10, Y0 + 5, 0]], dtype=float)
    # A low ray escapes below the walls; a steep ray crosses the underside before any wall.
    result = _kernel_suns(scene, xyz, [[1, 0, 0.1], [1, 0, 2.0]])
    np.testing.assert_array_equal(result, [[True, False], [True, False]])
    low = get_insolation(blds, "2026-06-21", building_details=True, grid_step=2, time_step_min=60)
    assert low.within(blds.geometry.iloc[0]).any()
    high = get_insolation(blds, "2026-06-21", building_details=True, receiver_height=5, grid_step=2, time_step_min=60)
    assert not high.within(blds.geometry.iloc[0]).any()


def test_facades_gables_parents_and_height_aware_neighbours():
    blds = _buildings(
        {
            "height": [10, None, 10],
            "building:part": [None, "yes", "yes"],
            "min_height": [None, None, 6],
            "roof:shape": [None, "gabled", None],
            "roof:height": [None, 4, None],
        },
        [box(X0, Y0, X0 + 20, Y0 + 10), box(X0, Y0, X0 + 20, Y0 + 10), box(X0 + 20, Y0, X0 + 25, Y0 + 10)],
        ["outline", "house", "bridge"],
    )
    scene = _scene(blds)
    xy, normals, owners, floors, heights = _detailed_facade_points(scene, np.arange(2), 1, 3, 1.2, None)
    house = owners == 0
    east = house & (xy[:, 0] > X0 + 20)
    assert np.any(east & (heights < 6))
    assert not np.any(east & (heights >= 6))  # only the upper part of the shared wall is hidden
    assert np.any(house & (heights > 6))  # gable windows above the eave
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    assert np.all(heights[owners == 1] > 6)
    assert np.all(heights == (floors - 1) * 3 + 1.2)
    result = get_facade_insolation(blds, "2026-06-21", building_details=True, spacing=2, time_step_min=60)
    assert set(result["part"]) == {"house", "bridge"}
    assert set(result.loc[result.part == "house", "building"]) == {"outline"}
    assert set(result.loc[result.part == "bridge", "building"]) == {"bridge"}


def test_pure_python_facade_height_filter_and_gable_windows():
    frame = _buildings(
        {"B": [0, 6], "H": [10, 10]}, [box(X0, Y0, X0 + 20, Y0 + 10), box(X0 + 20, Y0, X0 + 25, Y0 + 10)]
    )
    # Explicit east gable pieces and a neighbouring raised volume, independent of the roof generator.
    segments = np.array([[20, 0, 20, 5], [20, 5, 20, 10], [0, 0, 0, 5], [0, 5, 0, 10]], dtype=float)
    segments += np.tile([X0, Y0], 2)
    scene = Scene(
        CRS,
        frame,
        np.array([10.0, 10.0]),
        np.zeros(2),
        None,
        segments,
        np.array([[6, 10], [10, 6], [6, 10], [10, 6]], dtype=float),
        np.zeros(4, dtype=np.int64),
        volumes=frame,
        boundary_walls=np.arange(4),
        boundary_normals=np.array([[1, 0], [1, 0], [-1, 0], [-1, 0]]),
    )
    xy, normals, owner, floor, height = _detailed_facade_points(scene, [0], 1, 3, 1.2, None)
    east = normals[:, 0] == 1
    assert (height[east] < 6).all() and np.any(height[~east] > 6)
    assert np.any(height[~east] == 7.2) and np.all(owner == 0)
    assert np.all(height == (floor - 1) * 3 + 1.2)
    assert not frame.geometry.iloc[0].contains(shapely.Point(xy[0]))
    selected = _detailed_facade_points(scene, [0], 1, 3, 1.2, [2])
    assert (selected[3] == 2).all()


@pytest.mark.parametrize("facades", [False, True])
def test_details_false_is_exactly_legacy(facades):
    blds = _buildings(
        {
            "height": [12.0],
            "building:part": ["yes"],
            "min_height": [6.0],
            "roof:shape": ["gabled"],
            "roof:height": [4.0],
        }
    )
    dropped = blds[["height", "geometry"]]
    detailed_off, legacy = _scene(blds, building_details=False), _scene(dropped, building_details=False)
    for name in ("building_height", "base", "segments", "tops", "owner"):
        np.testing.assert_array_equal(getattr(detailed_off, name), getattr(legacy, name))
    method = get_facade_insolation if facades else get_insolation
    kwargs = {"time_step_min": 60, "building_details": False}
    if not facades:
        kwargs["area"] = blds.geometry.iloc[0].buffer(10)
    actual, expected = method(blds, "2026-06-21", **kwargs), method(dropped, "2026-06-21", **kwargs)
    assert_geodataframe_equal(actual, expected)
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual.drop(columns="geometry")), pd.DataFrame(expected.drop(columns="geometry")), check_exact=True
    )
    np.testing.assert_array_equal(actual.geometry.to_wkb(), expected.geometry.to_wkb())
    assert actual.attrs == expected.attrs
    assert "part" not in actual


def _example_subset():
    frame = gpd.read_parquet(DATA)
    chosen = []
    # Deterministic diversity, including parts, raised parts, holes and every available roof shape.
    groups = [frame[frame["building:part"].notna()], frame[frame["min_height"].map(osm_number) > 0]]
    groups.extend(group for _, group in frame.groupby("roof:shape", sort=True))
    for group in groups:
        chosen.extend(group.index[:2])
    chosen = list(dict.fromkeys(chosen))
    chosen.extend(i for i in frame.index if i not in chosen)
    subset = frame.loc[chosen[:40]].copy()
    # Bring containing outlines with their parts, rather than silently losing the S3DB relationship.
    parts = subset[subset["building:part"].notna()]
    outlines = frame[frame["building:part"].isna()]
    if len(parts):
        _, parents = shapely.STRtree(outlines.geometry.values).query(
            parts.representative_point().values, predicate="within"
        )
        subset = frame.loc[list(dict.fromkeys([*subset.index, *outlines.index[parents]]))]
    return subset


def _planar_triangles(polygon, z_at):
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
    return [
        np.column_stack([xy, z_at(xy)])
        for triangle in triangles
        if (xy := shapely.get_coordinates(triangle)[:3]).shape == (3, 2)
    ]


def _triangles(scene, shift):
    triangles = []
    xyz, offsets, _ = scene.roof_faces
    for start, end in zip(offsets[:-1], offsets[1:]):
        face = xyz[start:end].copy()
        face[:, :2] -= shift
        # Roof faces are graphs of a plane; least squares also handles extra collinear vertices.
        coeff = np.linalg.lstsq(np.column_stack([face[:, :2], np.ones(len(face))]), face[:, 2], rcond=None)[0]
        triangles.extend(_planar_triangles(Polygon(face[:, :2]), lambda xy: xy @ coeff[:2] + coeff[2]))
    for wall in _facade_boundaries(scene)[0]:
        owner = scene.owner[wall]
        a, b = scene.segments[wall].reshape(2, 2) - shift
        za, zb = scene.tops[wall]
        bottom = scene.base[owner] + scene.volumes.B.iloc[owner]
        quad = np.array([[*a, bottom], [*b, bottom], [*b, zb], [*a, za]])
        triangles.extend([quad[[0, 1, 2]], quad[[0, 2, 3]]])
    for i, row in enumerate(scene.volumes.itertuples()):
        if row.B > 0:
            polygon = shapely.transform(row.geometry, lambda xy: xy - shift)
            z = scene.base[i] + row.B
            triangles.extend(_planar_triangles(polygon, lambda xy: np.full(len(xy), z)))
    triangles = np.asarray(triangles)
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return triangles[area2 > 1e-10]


def _ray_triangle_hits(receivers, direction, triangles):
    """Vectorised Moller-Trumbore; independently flag hits within 1e-6 m of triangle edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    p = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    valid = np.abs(det) > 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    delta = receivers[:, None, :] - a
    u = np.einsum("nmi,mi->nm", delta, p) * inv
    q = np.cross(delta, edge1)
    v = np.einsum("i,nmi->nm", direction, q) * inv
    t = np.einsum("mi,nmi->nm", edge2, q) * inv
    area2 = np.linalg.norm(np.cross(edge1, edge2), axis=1)
    altitudes = area2[:, None] / np.column_stack(
        [np.linalg.norm(edge2, axis=1), np.linalg.norm(edge1, axis=1), np.linalg.norm(c - b, axis=1)]
    )
    distance = np.stack([u, v, 1 - u - v], axis=-1) * altitudes
    hits = valid & (t > 1e-8) & (distance.min(axis=-1) >= -1e-8)
    grazing = (valid & (t > 1e-8) & (distance.min(axis=-1) >= -1e-6) & (np.abs(distance).min(axis=-1) <= 1e-6)).any(
        axis=1
    )
    return hits.any(axis=1), grazing


def test_pure_python_triangle_oracle_hit_miss_and_grazing():
    triangles = np.array([[[0, 0, 5], [10, 0, 5], [0, 10, 5]]], dtype=float)
    receivers = np.array([[1, 1, 0], [9, 9, 0], [0, 0, 0], [1, 1, 6]], dtype=float)
    hits, grazing = _ray_triangle_hits(receivers, np.array([0.0, 0.0, 1.0]), triangles)
    np.testing.assert_array_equal(hits, [True, False, True, False])
    np.testing.assert_array_equal(grazing, [False, False, True, False])


def test_pure_python_vectorized_boundary_pieces_match_scalar():
    polygons = np.array(
        [
            Polygon([(0, 0), (12, 0), (12, 12), (0, 12)], [[(3, 3), (9, 3), (9, 9), (3, 9)]]),
            shapely.affinity.rotate(box(20, 0, 28, 8), 31),
        ],
        dtype=object,
    )
    edges, owners = [], []
    for owner, polygon in enumerate(polygons):
        for ring in (polygon.exterior, *polygon.interiors):
            for a, b in zip(ring.coords[:-1], ring.coords[1:]):
                a, b = np.array(a), np.array(b)
                mid = (a + b) / 2
                edges.extend([[*a, 5, *mid, 8], [*b, 8, *mid, 5]])  # both windings, split pieces
                owners.extend([owner, owner])
        edges.extend([[*polygon.exterior.coords[0], 5, *polygon.exterior.coords[2], 8], [0, 0, 5, 0, 0, 5]])
        owners.extend([owner, owner])
    edges, owners = np.array(edges), np.array(owners, dtype=np.int64)
    expected, normals = [], []
    for i, (edge, owner) in enumerate(zip(edges, owners)):
        a, b = edge[:2], edge[3:5]
        vec = b - a
        length = np.linalg.norm(vec)
        if length <= 1e-6 or not polygons[owner].boundary.buffer(1e-7).covers(shapely.LineString([a, b])):
            continue
        normal = vec[[1, 0]] * [1, -1] / length
        if polygons[owner].contains(shapely.Point((a + b) / 2 + normal * 1e-5)):
            normal *= -1
        expected.append(i)
        normals.append(normal)
    walls, actual = _boundary_pieces(polygons, edges, owners)
    np.testing.assert_array_equal(walls, expected)
    np.testing.assert_allclose(actual, normals, rtol=1e-15, atol=1e-15)
    empty = _boundary_pieces(np.empty(0, object), np.empty((0, 6)), np.empty(0, np.int64))
    assert empty[0].shape == (0,) and empty[1].shape == (0, 2)


@pytest.mark.parametrize("targets", [[1], [2, 0], []])
def test_pure_python_facade_boundaries_subset(targets, monkeypatch):
    from objectnat.methods.utils import _building_details

    polygons = [box(i * 20, 0, i * 20 + 10, 10) for i in range(3)]
    blds = _buildings({"height": [8] * 3}, polygons)
    segments, owners = [], []
    for owner, polygon in enumerate(polygons):
        coords = list(polygon.exterior.coords)
        segments.extend([[*a, *b] for a, b in zip(coords[:-1], coords[1:])])
        segments.append([owner * 20, 0, owner * 20 + 10, 10])  # interior mesh edge
        owners.extend([owner] * 5)
    scene = Scene(
        CRS,
        blds,
        np.full(3, 8.0),
        np.zeros(3),
        None,
        np.asarray(segments),
        np.full((15, 2), 8.0),
        np.asarray(owners),
        volumes=blds,
    )
    walls, normals = _facade_boundaries(scene)
    selected = np.isin(scene.owner[walls], targets)
    original = _building_details._boundary_pieces

    def checked(polygons, edges, owner):
        assert len(polygons) == len(set(targets))
        assert len(edges) == 5 * len(set(targets))
        return original(polygons, edges, owner)

    monkeypatch.setattr(_building_details, "_boundary_pieces", checked)
    for candidate in (scene, scene._replace(boundary_walls=walls, boundary_normals=normals)):
        actual_walls, actual_normals = _facade_boundaries(candidate, targets)
        np.testing.assert_array_equal(actual_walls, walls[selected])
        np.testing.assert_array_equal(actual_normals, normals[selected])


def test_boundary_extraction_is_lazy(monkeypatch):
    from objectnat.methods.utils import _building_details

    def unexpected(*_args):
        raise AssertionError("boundary extraction requested")

    monkeypatch.setattr(_building_details, "_boundary_pieces", unexpected)
    scene = _scene(_buildings({"height": [10], "roof:shape": ["gabled"]}))
    assert scene.boundary_walls is None
    with pytest.raises(AssertionError, match="boundary extraction requested"):
        _detailed_facade_points(scene, [0], 2, 3, 1.2, None)


def test_pure_python_vectorized_widths_and_grid_bearings():
    geometries = [shapely.affinity.rotate(box(X0, Y0, X0 + 20, Y0 + 10), angle) for angle in [0, 13, 47, 90]]
    geometries.extend([Polygon([(0, 0), (1, 0), (2, 0)]), Polygon()])
    rows = _buildings({"height": [100] * 6, "roof:shape": ["gabled"] * 6, "roof:angle": [45] * 6}, geometries)
    _, eave, _, _ = _heights(rows, "height", None, 3, None, 30)
    widths = []
    for polygon in geometries:
        rectangle = polygon.minimum_rotated_rectangle
        widths.append(
            np.linalg.norm(np.diff(shapely.get_coordinates(rectangle), axis=0), axis=1).min()
            if rectangle.geom_type == "Polygon" and not rectangle.is_empty
            else 0.0
        )
    np.testing.assert_array_equal(eave, 100 - np.asarray(widths) / 2)
    lon, lat = np.array([27, 30, 33, 35.5]), np.array([50, 55, 60, 65])
    expected = [grid_north_azimuth(CRS, x, y) for x, y in zip(lon, lat)]
    np.testing.assert_array_equal(_grid_north_azimuths(CRS, lon, lat), expected)
    assert _grid_north_azimuths(CRS, np.empty(0), np.empty(0)).shape == (0,)


def test_pure_python_redundant_undersides_union_holes_and_absolute_heights():
    upper = box(0, 0, 10, 10)
    polygons = np.array([box(0, 0, 5, 10), box(5, 0, 10, 10), upper], dtype=object)
    bottom, eave = np.array([10.0, 12.0, 16.0]), np.array([17.0, 18.0, 20.0])
    raised = np.array([False, True, True])
    np.testing.assert_array_equal(_exposed_undersides(polygons, bottom, eave, raised), [False, True, False])
    # A high ridge cannot substitute for an eave below B; equal bottoms cannot support one another.
    np.testing.assert_array_equal(_exposed_undersides(polygons, bottom, [15.0, 18.0, 20.0], raised), raised)
    np.testing.assert_array_equal(_exposed_undersides(polygons, [16.0, 12.0, 16.0], eave, raised), raised)
    holed = np.array([upper.difference(box(4, 4, 6, 6)), upper], dtype=object)
    np.testing.assert_array_equal(
        _exposed_undersides(holed, [0.0, 6.0], [7.0, 10.0], np.array([False, True])), [False, True]
    )
    assert not _exposed_undersides(np.empty(0, object), np.empty(0), np.empty(0), np.empty(0, bool)).any()
    # The specified relative-area tolerance only permits tiny uncovered slivers.
    for gap, expected in [(5e-6, False), (2e-5, True)]:
        polygons = np.array([box(0, 0, 10 - gap, 10), upper], dtype=object)
        assert bool(_exposed_undersides(polygons, [0.0, 6.0], [7.0, 10.0], np.array([False, True]))[1]) == expected


def test_stacked_parts_pruned_undersides_match_all_undersides():
    blds = _buildings(
        {
            "building:part": ["yes"] * 4,
            "height": [7, 7, 13, 12],
            "min_height": [0, 2, 6, 5],
            "roof:shape": ["flat", "flat", "gabled", "flat"],
            "roof:height": [0, 0, 4, 0],
        },
        [
            box(X0, Y0, X0 + 10, Y0 + 10),
            box(X0 + 10, Y0, X0 + 20, Y0 + 10),
            box(X0, Y0, X0 + 20, Y0 + 10),
            box(X0 + 30, Y0, X0 + 50, Y0 + 10),
        ],
    )
    scene = _scene(blds)
    assert scene.undersides[3].tolist() == [2, 5]
    raised = scene.volumes.B.to_numpy() > 0
    all_scene = scene._replace(
        undersides=(
            *polygon_arrays(scene.volumes.geometry.values[raised], np.zeros(2)),
            (scene.base + scene.volumes.B.to_numpy())[raised],
        )
    )
    rng = np.random.default_rng(8521)
    xyz = rng.uniform([X0 - 10, Y0 - 10, 0.2], [X0 + 60, Y0 + 20, 16], (600, 3))
    inside, volume = shapely.STRtree(scene.volumes.geometry.values).query(
        shapely.points(xyz[:, :2]), predicate="within"
    )
    keep = np.ones(len(xyz), bool)
    occupied = (scene.volumes.B.to_numpy()[volume] <= xyz[inside, 2]) & (
        xyz[inside, 2] <= scene.volumes.H.to_numpy()[volume]
    )
    keep[inside[occupied]] = False
    xyz = xyz[keep]
    bearing = rng.uniform(0, 2 * np.pi, 40)
    suns = np.column_stack([np.cos(bearing), np.sin(bearing), rng.uniform(0.03, 5, 40)])
    np.testing.assert_array_equal(_kernel_suns(scene, xyz, suns), _kernel_suns(all_scene, xyz, suns))
    targets = np.roll(xyz, 1, axis=0)
    for eyes, ends in [(xyz, targets), (targets, xyz), (xyz, xyz + [0, 0, 30])]:
        results = []
        for candidate in (scene, all_scene):
            segments, tops, kwargs = kernel_scene(candidate, np.zeros(2))
            results.append(
                _rust.line_of_sight(eyes, ends, np.arange(len(eyes), dtype=np.int64), segments, tops, **kwargs)
            )
        np.testing.assert_array_equal(*results)


def test_osm_example_triangle_oracle():
    scene = _scene(_example_subset(), default_height=12)
    assert len(scene.volumes) >= 30
    assert scene.volumes.B.gt(0).any() and scene.volumes.used_shape.nunique() >= 4
    rng = np.random.default_rng(73021)
    centers = shapely.get_coordinates(scene.volumes.representative_point().values)
    choice = rng.integers(0, len(centers), 320)
    xy = centers[choice] + rng.uniform(-35, 35, (320, 2))
    z = rng.uniform(0.2, 4, len(xy))
    inside, volumes = shapely.STRtree(scene.blds.geometry.values).query(shapely.points(xy), predicate="within")
    keep = np.ones(len(xy), bool)
    keep[
        inside[(scene.volumes.B.to_numpy()[volumes] <= z[inside]) & (z[inside] <= scene.volumes.H.to_numpy()[volumes])]
    ] = False
    xyz = np.column_stack([xy[keep], z[keep]])
    bearings = rng.uniform(0, 2 * np.pi, 28)
    suns = np.column_stack([np.sin(bearings), np.cos(bearings), np.tan(rng.uniform(0.05, 1.35, 28))])
    actual = _kernel_suns(scene, xyz, suns)
    shift = xyz[:, :2].min(axis=0)
    local = xyz.copy()
    local[:, :2] -= shift
    triangles = _triangles(scene, shift)
    checked = blocked_count = 0
    for i, sun in enumerate(suns):
        blocked, grazing = _ray_triangle_hits(local, sun, triangles)
        np.testing.assert_array_equal(actual[~grazing, i], ~blocked[~grazing], err_msg=f"sun {i}")
        checked += (~grazing).sum()
        blocked_count += blocked[~grazing].sum()
    assert checked > 4000 and blocked_count > 50


def test_osm_example_api_smoke():
    blds = _example_subset()
    area = blds.geometry.iloc[0].buffer(20)
    kwargs = dict(area=area, building_details=True, default_height=12, time_step_min=60)
    ground = get_insolation(blds, "2026-06-21", grid_step=5, **kwargs)
    facade = get_facade_insolation(blds, "2026-06-21", spacing=5, **kwargs)
    assert len(ground) and len(facade)
    assert "part" in facade and ground.crs == facade.crs == blds.crs
    assert (ground.sun_hours >= 0).all() and (facade.sun_hours >= 0).all()
