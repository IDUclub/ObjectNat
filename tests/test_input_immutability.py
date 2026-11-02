"""Regression tests for noise/visibility defects; none of them need Overpass."""

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString, Point, box

from objectnat import calculate_simplified_noise_frame, get_visibility, simulate_noise
from objectnat.methods.visibility import visibility_analysis
from objectnat.methods.visibility.visibility_analysis import dist_to_furthest_point

SOURCE_LON_LAT = (30.27, 59.935)


@pytest.fixture
def source_point_4326():
    # Non-default index: the function must not rely on (or alter) the caller's index.
    return gpd.GeoDataFrame(geometry=[Point(*SOURCE_LON_LAT)], index=[7], crs=4326)


@pytest.fixture
def single_obstacle_4326(source_point_4326):
    local_crs = source_point_4326.estimate_utm_crs()
    p = source_point_4326.to_crs(local_crs).geometry.iloc[0]
    obstacle = box(p.x + 5, p.y - 5, p.x + 15, p.y + 5)
    return gpd.GeoDataFrame({"name": ["wall"]}, geometry=[obstacle], crs=local_crs).to_crs(4326)


@pytest.fixture
def empty_obstacles_4326():
    return gpd.GeoDataFrame(geometry=[], crs=4326)


def _run_simplified(noise_sources, obstacles):
    return calculate_simplified_noise_frame(
        noise_sources, obstacles, 20, target_noise_db=50, db_sim_step=5, visibility_parallel=False
    )


def _run_simulate(source_points, obstacles):
    return simulate_noise(
        source_points,
        obstacles,
        source_noise_db=60,
        geometric_mean_freq_hz=2000,
        target_noise_db=50,
        db_sim_step=5,
        reflection_n=1,
        use_parallel=False,
    )


@pytest.mark.parametrize("obstacles_fixture", ["single_obstacle_4326", "empty_obstacles_4326"])
def test_simplified_noise_frame_does_not_mutate_inputs(request, source_point_4326, obstacles_fixture):
    obstacles = request.getfixturevalue(obstacles_fixture)
    noise_sources = source_point_4326.assign(source_noise_db=60, geometric_mean_freq_hz=2000)
    sources_before = noise_sources.copy()
    obstacles_before = obstacles.copy()

    result = _run_simplified(noise_sources, obstacles)

    assert_geodataframe_equal(noise_sources, sources_before)
    assert_geodataframe_equal(obstacles, obstacles_before)
    assert noise_sources.crs == sources_before.crs
    assert not result.empty
    assert result.crs == sources_before.crs


@pytest.mark.parametrize("obstacles_fixture", ["single_obstacle_4326", "empty_obstacles_4326"])
def test_simulate_noise_does_not_mutate_inputs(request, source_point_4326, obstacles_fixture):
    obstacles = request.getfixturevalue(obstacles_fixture)
    sources_before = source_point_4326.copy()
    obstacles_before = obstacles.copy()

    result = _run_simulate(source_point_4326, obstacles)

    assert_geodataframe_equal(source_point_4326, sources_before)
    assert_geodataframe_equal(obstacles, obstacles_before)
    assert "absorb_ratio" not in obstacles.columns
    assert not result.empty
    assert result.crs == sources_before.crs


def test_simulate_noise_without_obstacles_handles_duplicate_index():
    # Two sources sharing an index label: without reset_index their dead areas would collide in one dict key.
    local = gpd.GeoDataFrame(geometry=[Point(*SOURCE_LON_LAT)], crs=4326).estimate_utm_crs()
    p = gpd.GeoSeries([Point(*SOURCE_LON_LAT)], crs=4326).to_crs(local).iloc[0]
    sources = gpd.GeoDataFrame(geometry=[p, Point(p.x + 500, p.y)], index=[0, 0], crs=local).to_crs(4326)
    sources_before = sources.copy()

    result = _run_simulate(sources, gpd.GeoDataFrame(geometry=[], crs=4326))

    assert_geodataframe_equal(sources, sources_before)
    result_local = result.to_crs(local)
    # Both sources must produce their own noise zone.
    assert result_local.union_all().contains(Point(p.x, p.y))
    assert result_local.union_all().contains(Point(p.x + 500, p.y))


def test_dist_to_furthest_point_value():
    assert dist_to_furthest_point(Point(0, 0), box(-3, -4, 3, 4)) == pytest.approx(5.0)


def test_dist_to_furthest_point_error_is_not_printed(capsys):
    # A LineString has no exterior: the error must propagate without a debug print to stdout.
    with pytest.raises(AttributeError):
        dist_to_furthest_point(Point(0, 0), LineString([(0, 0), (1, 1)]))
    assert capsys.readouterr().out == ""


def test_get_visibility_rejects_invalid_method_before_work(monkeypatch, source_point_4326):
    def fail(*_args, **_kwargs):
        raise AssertionError("no work must be done for an invalid method")

    monkeypatch.setattr(visibility_analysis, "_visibility_accurate_batch", fail)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", fail)
    monkeypatch.setattr(gpd.GeoDataFrame, "estimate_utm_crs", fail)

    with pytest.raises(ValueError, match="'accurate', 'simple'"):
        get_visibility(source_point_4326, gpd.GeoDataFrame(), 100, method="fast")
