import geopandas as gpd
import numpy as np
import pytest
from shapely import Point, box

from objectnat import get_facade_insolation, get_insolation, rasterize
from objectnat.methods.insolation._sun import day_samples, solar_position

CRS = 32636
X0, Y0 = 350_000.0, 6_650_000.0  # about 59.97 N, 30.3 E


def _sun_hours_at(points, x, y):
    return points.loc[points.distance(Point(X0 + x, Y0 + y)).idxmin(), "sun_hours"]


def test_sun_position_saint_petersburg_midsummer():
    times = day_samples(__import__("datetime").date(2026, 6, 21), 30.31, 1)
    azimuth, elevation = solar_position(times, 59.94, 30.31)
    assert elevation.max() == pytest.approx(90 - 59.94 + 23.44, abs=0.05)
    assert azimuth[elevation.argmax()] == pytest.approx(180, abs=0.5)
    assert (elevation > 0).sum() / 60 == pytest.approx(18.7, abs=0.15)  # hours above the horizon


def test_courtyard_stays_in_shadow_in_winter():
    # a 30 m high ring building around a 10 x 10 m courtyard, and an open lot to the south
    ring = box(X0 - 20, Y0 - 20, X0 + 20, Y0 + 20).difference(box(X0 - 5, Y0 - 5, X0 + 5, Y0 + 5))
    buildings = gpd.GeoDataFrame({"height": [30.0]}, geometry=[ring], crs=CRS)
    area = box(X0 - 60, Y0 - 120, X0 + 60, Y0 + 60)
    bands, points = get_insolation(buildings, "2026-12-21", area=area, grid_step=1, output="both")

    day = points.attrs["day_length_hours"]
    assert 5 < day < 6.5  # midwinter day at 60 N
    assert _sun_hours_at(points, 0, 0) == 0  # the courtyard gets no direct sun at all
    assert _sun_hours_at(points, 0, -110) == pytest.approx(day)  # far south, nothing in the way
    assert _sun_hours_at(points, 0, 30) < 0.5 * day  # right north of the building
    assert bands.iloc[0]["label"] == "0" and not bands.geometry.iloc[0].is_empty
    assert not points.within(ring).any()

    # the receivers rasterize onto their own grid, also from degrees; footprint cells stay empty
    raster = rasterize(points.to_crs(4326), "sun_hours")
    assert raster.cell_size == 1 and np.isfinite(raster.values).sum() == len(points)
    a, _, c, _, e, f = raster.transform
    assert raster.values[int((Y0 - f) / e), int((X0 - c) / a)] == 0  # the courtyard
    assert np.isnan(raster.values[int((Y0 + 12 - f) / e), int((X0 - c) / a)])  # inside the ring


@pytest.mark.parametrize("as_polygons", [False, True])
def test_terrain_ridge_shades_its_north_slope(as_polygons):
    # flat ground with a 40 m ridge running east-west, no buildings
    if as_polygons:
        terrain = gpd.GeoDataFrame(
            {"height": [0.0, 40.0]},
            geometry=[box(X0 - 200, Y0 - 200, X0 + 200, Y0 + 200), box(X0 - 200, Y0 - 20, X0 + 200, Y0 + 20)],
            crs=CRS,
        )
    else:
        x = np.arange(-200, 201, 10.0)
        y = np.arange(200, -201, -10.0)  # north-up raster rows
        heights = np.where(np.abs(y)[:, None] <= 20, 40.0, 0.0) * np.ones_like(x)[None, :]
        terrain = (heights, (10.0, 0.0, X0 - 205, 0.0, -10.0, Y0 + 205), CRS)
    no_buildings = gpd.GeoDataFrame({"height": []}, geometry=[], crs=CRS)
    area = box(X0 - 50, Y0 - 150, X0 + 50, Y0 + 150)
    points = get_insolation(no_buildings, "2026-12-21", area=area, terrain=terrain, grid_step=5, output="receivers")
    assert _sun_hours_at(points, 0, 40) == 0  # low winter sun from the south: north foot of the ridge in shadow
    assert _sun_hours_at(points, 0, -100) == pytest.approx(points.attrs["day_length_hours"])


def test_period_reports_mean_and_worst_day():
    building = gpd.GeoDataFrame({"height": [30.0]}, geometry=[box(X0 - 20, Y0 - 10, X0 + 20, Y0 + 10)], crs=CRS)
    area = box(X0 - 40, Y0 - 80, X0 + 40, Y0 + 60)
    points = get_insolation(
        building, ("2026-03-21", "2026-06-21"), day_step=30, area=area, grid_step=2, output="receivers"
    )
    assert points.attrs["days"] == ["2026-03-21", "2026-04-20", "2026-05-20", "2026-06-19", "2026-06-21"]
    assert (points["min_sun_hours"] <= points["sun_hours"] + 1e-9).all()
    assert (points["max_continuous_hours"] <= points["min_sun_hours"] + 1e-9).all()
    north = points.distance(Point(X0, Y0 + 20)).idxmin()
    # north of the building the high summer sun gets over it, the equinox sun does not: worst day < mean
    assert points.at[north, "min_sun_hours"] < points.at[north, "sun_hours"]


def test_facade_insolation_by_floor():
    house = box(X0 - 10, Y0 - 5, X0 + 10, Y0 + 5)  # 5 floors, long facades face north and south
    tower = box(X0 - 15, Y0 - 30, X0 + 15, Y0 - 20)  # 15 m high, 15 m south of the house
    annex = box(X0 + 10, Y0 - 5, X0 + 16, Y0 + 5)  # adjoins the house's east wall
    buildings = gpd.GeoDataFrame(
        {"height": [15.0, 15.0, 6.0]}, geometry=[house, tower, annex], index=["house", "tower", "annex"], crs=CRS
    )
    points = get_facade_insolation(buildings, "2026-03-21", area=house.buffer(-1), norm_hours=2)

    assert set(points["building"]) == {"house"}
    assert sorted(points["floor"].unique()) == [1, 2, 3, 4, 5]
    assert not (points.distance(annex) < 0.5).any()  # no points on the wall shared with the annex
    # grid north is 2.3 degrees off true north here (meridian convergence of UTM 36)
    south = points[(points["facade_azimuth"] - 180).abs() < 5]
    north = points[(points["facade_azimuth"] - 0).abs() < 5]
    assert len(south) and len(north)
    assert north["sun_hours"].max() < 0.5  # at the equinox the sun stays south of east-west
    by_floor = south.groupby("floor")["sun_hours"].mean()
    assert by_floor.is_monotonic_increasing and by_floor[1] < by_floor[5]  # the tower shades lower floors
    assert points["meets_norm"].dtype == bool
