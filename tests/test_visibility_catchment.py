import geopandas as gpd
import numpy as np
import pytest
from iduedu import UrbanGraph
from shapely import LineString, Point, box

from objectnat import get_observer_points, get_visibility_catchment, rasterize

CRS = 32636
X0, Y0 = 400_000.0, 6_600_000.0


def _observers(*xy, **columns):
    return gpd.GeoDataFrame(columns, geometry=[Point(X0 + x, Y0 + y) for x, y in xy], crs=CRS)


WALL = gpd.GeoDataFrame(geometry=[box(X0 - 0.5, Y0 - 50, X0 + 0.5, Y0 + 50)], crs=CRS)


def test_single_observer_without_obstacles_sees_its_disc():
    observers = _observers((0, 0)).to_crs(4326)
    cells = get_visibility_catchment(observers, None, view_distance=100, cell_size=2)

    assert cells.crs == observers.crs
    assert (cells["count"] == 1).all()
    # the seen cells cover the disc up to one cell along its perimeter
    raster = rasterize(cells, "count")
    area = np.nansum(raster.values) * raster.cell_size**2
    assert area == pytest.approx(np.pi * 100**2, abs=2 * np.pi * 100 * 2)


@pytest.mark.parametrize("obstacles, most_observers", [(None, 2), (WALL, 1)])
def test_wall_separates_the_observers(obstacles, most_observers):
    observers = _observers((-10, 0), (10, 0))
    cells = get_visibility_catchment(observers, obstacles, view_distance=30, cell_size=1)

    assert sorted(cells["count"].unique()) == list(range(1, most_observers + 1))


def test_weights_are_summed():
    observers = _observers((-10, 0), (10, 0), flow=[1.0, 3.0])
    cells = get_visibility_catchment(observers, None, view_distance=30, cell_size=1, weight_column="flow")

    assert sorted(cells["weight"].unique()) == [1.0, 3.0, 4.0]
    assert sorted(cells["count"].unique()) == [1, 2]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "view_distance"),
        ({"view_distance": 10, "cell_size": 0}, "cell_size"),
        ({"view_distance": 10, "weight_column": "missing"}, "weight_column"),
    ],
)
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_visibility_catchment(_observers((0, 0)), None, **kwargs)


def test_observer_points_along_streets():
    nodes = gpd.GeoDataFrame(
        {"node": [0, 1, 2]}, geometry=[Point(X0, Y0), Point(X0 + 100, Y0), Point(X0 + 100, Y0 + 30)], crs=CRS
    ).set_index("node")
    lines = [LineString([(X0, Y0), (X0 + 100, Y0)]), LineString([(X0 + 100, Y0), (X0 + 100, Y0 + 30)])]
    edges = gpd.GeoDataFrame(
        {
            "u": [0, 1, 1, 2],
            "v": [1, 0, 2, 1],
            "type": "walk",
            "length_meter": [100.0, 100.0, 30.0, 30.0],
            "time_min": [1.0, 1.0, 0.3, 0.3],
        },
        # both directions of each street, the reverse one stored reversed
        geometry=[lines[0], lines[0].reverse(), lines[1], lines[1].reverse()],
        crs=CRS,
    )
    graph = UrbanGraph(nodes, edges, is_multigraph=False, is_directed=True, crs=CRS, graph_type="walk")

    points = get_observer_points(graph, spacing=10)

    assert points.crs == graph.crs
    xy = np.round(points.get_coordinates().to_numpy() - [X0, Y0], 6)
    assert sorted(map(tuple, xy)) == sorted([(x, 0.0) for x in range(5, 100, 10)] + [(100.0, y) for y in (5, 15, 25)])
