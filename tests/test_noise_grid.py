import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, Point, box

from objectnat import simulate_noise_grid
from objectnat.methods.noise import dist_to_target_db

CRS = 32636
X0, Y0 = 400_000.0, 6_600_000.0


def _source(*xy, level=70, freq=1000):
    return gpd.GeoDataFrame(
        {"source_noise_db": [level] * len(xy), "geometric_mean_freq_hz": [freq] * len(xy)},
        geometry=[Point(X0 + x, Y0 + y) for x, y in xy],
        crs=CRS,
    )


def _level_at(receivers, x, y):
    distance = receivers.distance(Point(X0 + x, Y0 + y))
    return receivers.loc[distance.idxmin(), "noise_level"]


def test_free_field_follows_the_distance_law():
    bands, receivers = simulate_noise_grid(_source((0, 0)), None, grid_step=1.0, output="both")

    assert receivers.crs == bands.crs
    assert _level_at(receivers, 10, 0) == pytest.approx(70 - 20 - 10 * 0.0052, abs=0.05)
    outer = bands.union_all().convex_hull.exterior.distance(Point(X0, Y0))
    assert outer == pytest.approx(dist_to_target_db(70, 40, 1000, 20), abs=1.5)
    assert bands["noise_level"].is_monotonic_decreasing


def test_two_equal_sources_add_three_db():
    _, one = simulate_noise_grid(_source((0, 0)), None, grid_step=1.0, output="both")
    _, two = simulate_noise_grid(_source((0, 0), (0, 0)), None, grid_step=1.0, output="both")
    assert _level_at(two, 20, 0) - _level_at(one, 20, 0) == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_building_shadow_reflection_and_diffraction():
    building = gpd.GeoDataFrame(geometry=[box(X0 + 20, Y0 - 10, X0 + 30, Y0 + 10)], crs=CRS)
    _, direct = simulate_noise_grid(
        _source((0, 0)), building, grid_step=1.0, reflections=False, diffraction=False, output="both"
    )
    _, reflected = simulate_noise_grid(_source((0, 0)), building, grid_step=1.0, diffraction=False, output="both")

    # nothing reaches right behind the building without diffraction
    assert direct.distance(Point(X0 + 40, Y0)).min() > 1
    # around the upper corner the shadow gets the free-field level less 5 to 20 dB (Maekawa, ISO cap)
    loud = _source((0, 0), level=90)
    silent = simulate_noise_grid(
        loud, building, grid_step=1.0, reflections=False, diffraction=False, output="receivers"
    )
    bent = simulate_noise_grid(loud, building, grid_step=1.0, reflections=False, output="receivers")
    assert silent.distance(Point(X0 + 45, Y0 + 15)).min() > 1
    free = 90 - 20 * np.log10(np.hypot(45, 15)) - np.hypot(45, 15) * 0.0052
    assert free - 20 <= _level_at(bent, 45, 15) <= free - 10 * np.log10(3) + 0.5
    # in front of the wall the reflection makes it louder than the direct sound alone
    # (direct path 10 m, reflected path 30 m: +0.4 dB)
    assert _level_at(reflected, 10, 0) > _level_at(direct, 10, 0) + 0.3
    # no receivers inside the building
    assert not reflected.within(building.geometry.iloc[0]).any()  # facade receivers are allowed


def test_line_source_follows_the_line_law():
    road = gpd.GeoDataFrame(
        {"source_noise_db": [60], "geometric_mean_freq_hz": [1000]},
        geometry=[LineString([(X0 - 300, Y0), (X0 + 300, Y0)])],
        crs=CRS,
    )
    _, receivers = simulate_noise_grid(road, None, grid_step=1.0, source_spacing=2.0, output="both")
    near_middle = receivers[receivers.geometry.x.between(X0 - 20, X0 + 20)]
    at_10m = near_middle[(near_middle.geometry.y - Y0 - 10).abs() < 0.5]["noise_level"].mean()
    assert at_10m == pytest.approx(60 - 10 - 10 * 0.0052, abs=0.5)
