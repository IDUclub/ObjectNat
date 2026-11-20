"""Private Rust bindings: array contracts and compatibility for roof and raised-volume scenes."""

import numpy as np
import pytest

from objectnat import _rust


@pytest.mark.parametrize("shape", [0, 1, 2, 3, 4, 6, 9])
def test_vectorized_facade_boundary_indices_and_outward_normals(shape):
    import shapely

    from objectnat.methods.utils._building_details import _boundary_pieces, polygon_arrays

    polygons = np.array(
        [
            shapely.box(0, 0, 12, 8),
            shapely.box(20, 0, 36, 16).difference(shapely.box(26, 6, 30, 10)),
        ],
        dtype=object,
    )
    _, _, edges, owners, _, _ = _rust.roof_mesh(
        *polygon_arrays(polygons, np.zeros(2)),
        np.full(2, shape, np.uint8),
        np.full(2, 10.0),
        np.full(2, 14.0),
    )
    walls, normals = _boundary_pieces(polygons, edges, owners)
    assert len(walls) and len(walls) <= len(edges)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    mid = (edges[walls, :2] + edges[walls, 3:5]) / 2
    outside, inside = mid + normals * 1e-5, mid - normals * 1e-5
    assert not shapely.contains_xy(polygons[owners[walls]], outside[:, 0], outside[:, 1]).any()
    assert shapely.contains_xy(polygons[owners[walls]], inside[:, 0], inside[:, 1]).all()
    lengths = np.linalg.norm(edges[walls, 3:5] - edges[walls, :2], axis=1)
    for owner, polygon in enumerate(polygons):
        assert lengths[owners[walls] == owner].sum() == pytest.approx(polygon.length)


def _inputs():
    return {
        "xy": np.array([[0, 0], [12, 0], [12, 8], [0, 8]], dtype=np.float64),
        "ring_offsets": np.array([0, 4], dtype=np.int64),
        "poly_offsets": np.array([0, 1], dtype=np.int64),
        "shape": np.array([2], dtype=np.uint8),
        "eave": np.array([10.0]),
        "ridge": np.array([14.0]),
    }


def test_roof_mesh_contract_and_thread_counts():
    args = _inputs()
    serial = _rust.roof_mesh(**args, n_threads=1)
    for n_threads in (None, 0, 2):
        result = _rust.roof_mesh(**args, n_threads=n_threads)
        for a, b in zip((*serial[0], *serial[1:]), (*result[0], *result[1:])):
            np.testing.assert_array_equal(a, b)
    (xyz, offsets), owner, edges, edge_owner, status, used = serial
    assert xyz.shape == (8, 3) and xyz.dtype == np.float64
    assert offsets.tolist() == [0, 4, 8] and offsets.dtype == np.int64
    assert owner.tolist() == [0, 0] and owner.dtype == np.int64
    assert edges.shape == (7, 6) and edges.dtype == np.float64
    assert edge_owner.tolist() == [0] * 7 and edge_owner.dtype == np.int64
    assert status.tolist() == [0] and status.dtype == np.uint8
    assert used.tolist() == [2] and used.dtype == np.uint8
    assert xyz[:, 2].max() == pytest.approx(14.0)


def test_empty_batch_and_polygon_owners():
    (xyz, offsets), owner, edges, edge_owner, status, used = _rust.roof_mesh(
        np.empty((0, 2)),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.uint8),
        np.empty(0),
        np.empty(0),
    )
    assert xyz.shape == (0, 3) and edges.shape == (0, 6)
    assert offsets.tolist() == [0]
    assert all(a.size == 0 for a in (owner, edge_owner, status, used))
    args = _inputs()
    args.update(
        xy=np.concatenate([args["xy"], args["xy"] + 20]),
        ring_offsets=np.array([0, 4, 8], dtype=np.int64),
        poly_offsets=np.array([0, 1, 2], dtype=np.int64),
        shape=np.array([0, 2], dtype=np.uint8),
        eave=np.array([10.0, 10.0]),
        ridge=np.array([14.0, 14.0]),
    )
    _, owner, _, edge_owner, status, used = _rust.roof_mesh(**args)
    assert owner.tolist() == [0, 1, 1]
    assert edge_owner.tolist() == [0] * 4 + [1] * 7
    assert status.tolist() == [0, 0] and used.tolist() == [0, 2]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("xy", np.zeros((4, 3)), "xy"),
        ("xy", np.array([[0.0, np.nan]] * 4), "finite"),
        ("ring_offsets", np.array([-1, 4], dtype=np.int64), "ring_offsets"),
        ("ring_offsets", np.array([0, 5], dtype=np.int64), "ring_offsets"),
        ("ring_offsets", np.array([0, 4, 2, 4], dtype=np.int64), "ring_offsets"),
        ("ring_offsets", np.array([0, 2, 4], dtype=np.int64), "at least 3"),
        ("poly_offsets", np.array([0, 2], dtype=np.int64), "poly_offsets"),
        ("poly_offsets", np.array([0, 0, 1], dtype=np.int64), "exterior"),
        ("poly_offsets", np.array([], dtype=np.int64), "poly_offsets"),
        ("shape", np.array([], dtype=np.uint8), "shape"),
        ("eave", np.array([np.nan]), "finite"),
        ("ridge", np.array([np.inf]), "finite"),
        ("direction", np.array([np.inf]), "direction"),
        ("across", np.array([], dtype=bool), "across"),
    ],
)
def test_malformed_roof_arrays_raise_value_error(field, value, match):
    args = _inputs()
    args[field] = value
    if field == "ring_offsets" and match == "at least 3":
        args["poly_offsets"] = np.array([0, 2], dtype=np.int64)
    with pytest.raises(ValueError, match=match):
        _rust.roof_mesh(**args)


def test_roof_mesh_accepts_noncontiguous_views_and_reports_approximation():
    args = _inputs()
    for name in ("xy", "ring_offsets", "poly_offsets", "shape", "eave", "ridge"):
        value = args[name]
        bigger = np.repeat(value, 2, axis=0)
        args[name] = bigger[::2]
    args["shape"][:] = 13
    result = _rust.roof_mesh(**args)
    assert result[-2].tolist() == [1] and result[-1].tolist() == [3]


def _raised_details():
    return {
        "under_xy": np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]], dtype=float),
        "under_ring_offsets": np.array([0, 4], dtype=np.int64),
        "under_poly_offsets": np.array([0, 1], dtype=np.int64),
        "under_z": np.array([5.0]),
    }


def test_raised_volume_bindings_and_sloped_top_compatibility():
    segments = np.array([[0, -5, 0, 5]], dtype=float)
    eyes = np.array([[-10, 0, 4], [-10, 0, 6], [-10, 0, 12]], dtype=float)
    targets = np.array([[10, 0, 4], [10, 0, 6], [10, 0, 12]], dtype=float)
    owners = np.arange(3, dtype=np.int64)
    args = (eyes, targets, owners, segments)
    original = _rust.line_of_sight(*args, np.array([10.0]))
    detailed = _rust.line_of_sight(*args, np.array([[10.0, 10.0]]))
    np.testing.assert_array_equal(original, detailed)
    np.testing.assert_array_equal(
        _rust.line_of_sight(*args, np.array([[8.0, 12.0]]), bottoms=np.array([5.0])), [1, 0, 1]
    )
    # An overhead underside blocks even without any wall segments.
    counts, _ = _rust.insolation_day_stats(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 1.0]]),
        np.array([0, 1], dtype=np.int64),
        np.empty((0, 4)),
        np.empty((0, 2)),
        **_raised_details(),
    )
    assert counts.tolist() == [[0]]
    seen = _rust.line_of_sight(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 8.0]]),
        np.array([0], dtype=np.int64),
        np.empty((0, 4)),
        np.empty(0),
        **_raised_details(),
    )
    assert seen.tolist() == [0]


def test_viewshed_bindings_accept_raised_volume_arrays():
    args = {
        "eyes": np.array([[0.5, 0.5, 0.0]]),
        "radii": np.array([100.0]),
        "grid": (0.0, 0.0, 1.0, 2, 2),
        "target_z": np.full((2, 2), 8.0),
        "segments": np.empty((0, 4)),
        "tops": np.empty((0, 2)),
        **_raised_details(),
    }
    cells, offsets = _rust.viewshed_cells(**args)
    assert cells.size == 0 and offsets.tolist() == [0, 0]
    counts, sums = _rust.viewshed_count_grid(**args, weights=np.array([2.0]))
    np.testing.assert_array_equal(counts, np.zeros((2, 2)))
    np.testing.assert_array_equal(sums, np.zeros((2, 2)))


@pytest.mark.parametrize("bad", ["bottoms", "tops", "partial_underside", "under_z", "offsets"])
def test_scene_array_validation(bad):
    args = {
        "eyes": np.array([[0.0, 0.0, 0.0]]),
        "targets": np.array([[1.0, 0.0, 8.0]]),
        "owner": np.array([0], dtype=np.int64),
        "segments": np.empty((0, 4)),
        "tops": np.empty(0),
    }
    if bad == "bottoms":
        args["bottoms"] = np.array([5.0])
    elif bad == "tops":
        args["tops"] = np.empty((0, 3))
    elif bad == "partial_underside":
        args["under_z"] = np.array([5.0])
    else:
        args.update(_raised_details())
        if bad == "under_z":
            args["under_z"] = np.array([np.nan])
        else:
            args["under_poly_offsets"] = np.array([0, 2], dtype=np.int64)
    with pytest.raises(ValueError):
        _rust.line_of_sight(**args)
