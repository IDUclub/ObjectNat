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
    result, cells = get_visibility_catchment(observers, None, view_distance=100, cell_size=2, output="both")

    assert result.crs == observers.crs
    assert result["visibility_class"].tolist() == [1]
    assert result[["count_min", "count_max"]].values.tolist() == [[1, 1]]
    zone = result.to_crs(CRS).geometry.iloc[0]
    assert zone.is_valid
    # the zone follows the cell grid: its area differs from the disc by at most one cell along the perimeter
    assert zone.area == pytest.approx(np.pi * 100**2, abs=2 * np.pi * 100 * 2)
    # the seen cells are the zone, cell by cell
    assert (cells["count"] == 1).all() and (cells["visibility_class"] == 1).all()
    raster = rasterize(cells, "count")
    assert np.nansum(raster.values) * raster.cell_size**2 == pytest.approx(zone.area)


@pytest.mark.parametrize("obstacles, expected_classes", [(None, [1, 2]), (WALL, [1])])
def test_wall_separates_the_observers(obstacles, expected_classes):
    observers = _observers((-10, 0), (10, 0))
    result = get_visibility_catchment(observers, obstacles, view_distance=30, cell_size=1, classes=[1, 2])

    assert result["visibility_class"].tolist() == expected_classes
    assert result["count_max"].max() == len(expected_classes)


def test_weights_are_summed():
    observers = _observers((-10, 0), (10, 0), flow=[1.0, 3.0])
    result = get_visibility_catchment(
        observers, None, view_distance=30, cell_size=1, weight_column="flow", classes=[1, 2, 4]
    )

    assert result[["visibility_class", "weight_min", "weight_max"]].values.tolist() == [
        [1, 1.0, 1.0],
        [2, 3.0, 3.0],
        [3, 4.0, 4.0],
    ]
    assert result.geometry.is_valid.all()


def test_quantile_classes_split_the_values():
    observers = _observers(*[(x, 0) for x in range(0, 50, 5)])
    result = get_visibility_catchment(observers, None, view_distance=40, cell_size=2, classes=3)

    assert result["visibility_class"].tolist() == [1, 2, 3]
    assert (result["count_min"].to_numpy()[1:] > result["count_max"].to_numpy()[:-1]).all()
    assert result["count_max"].max() == len(observers)
    assert not result.geometry.iloc[0].intersects(result.geometry.iloc[2].buffer(-0.1))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "view_distance"),
        ({"view_distance": 10, "cell_size": 0}, "cell_size"),
        ({"view_distance": 10, "classes": [3, 1]}, "ascending"),
        ({"view_distance": 10, "classes": 0}, "classes"),
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
