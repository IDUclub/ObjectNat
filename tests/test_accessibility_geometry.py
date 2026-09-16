import geopandas as gpd
import pandas as pd
import pytest
from iduedu import UrbanGraph
from shapely.geometry import LineString, Point

from objectnat import get_graph_coverage, get_stepped_graph_isochrones

CRS = 32636
SPACING = 100.0


class _WalkingGrid:
    """A 5x5 walking grid in which every walk edge takes one minute, ready for transit to be added."""

    def __init__(self):
        self.nodes, self.edges, self.cells = [], [], {}
        for i in range(5):
            for j in range(5):
                self.cells[(i, j)] = self.add_node(Point(i * SPACING, j * SPACING))
        for (i, j), node in self.cells.items():
            for neighbour in ((i + 1, j), (i, j + 1)):
                if neighbour in self.cells:
                    self.add_edge(node, self.cells[neighbour], 1.0, "walk", oneway=False)

    def add_node(self, geometry, node_type=None):
        self.nodes.append({"node": len(self.nodes), "type": node_type, "geometry": geometry})
        return len(self.nodes) - 1

    def add_route_node(self, cell):
        """A bus route node standing exactly on the walking node of ``cell``."""
        return self.add_node(self.nodes[self.cells[cell]]["geometry"], "bus")

    def add_edge(self, u, v, time_min, edge_type, oneway=True):
        line = LineString([self.nodes[u]["geometry"], self.nodes[v]["geometry"]])
        self.edges.append(
            {
                "u": u,
                "v": v,
                "geometry": line,
                "length_meter": line.length,
                "time_min": time_min,
                "type": edge_type,
                "oneway": oneway,
            }
        )

    def graph(self):
        nodes_gdf = gpd.GeoDataFrame(self.nodes, geometry="geometry", crs=CRS).set_index("node")
        edges_gdf = gpd.GeoDataFrame(self.edges, geometry="geometry", crs=CRS)
        edges_gdf["k"] = edges_gdf.groupby(["u", "v"]).cumcount()
        return UrbanGraph(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            is_multigraph=True,
            is_directed=True,
            edge_direction_column="oneway",
            crs=CRS,
            graph_type="intermodal",
        )


def _overlap_areas(polygons: gpd.GeoDataFrame) -> dict:
    return {
        (a, b): polygons.loc[a, "geometry"].intersection(polygons.loc[b, "geometry"]).area
        for i, a in enumerate(polygons.index)
        for b in polygons.index[i + 1 :]
    }


@pytest.mark.parametrize("geometry_type", [None, "radius", "ways"])
def test_stepped_isochrone_bands_ignore_transit_nodes_on_a_stop(geometry_type):
    grid = _WalkingGrid()
    stop = grid.cells[(3, 2)]
    # Two routes call at a stop one minute's walk from the origin; boarding them takes 8 and 11 minutes.
    for wait in (8.0, 11.0):
        route_node = grid.add_route_node((3, 2))
        grid.add_edge(stop, route_node, wait, "boarding")
        grid.add_edge(route_node, stop, 0.0, "alighting")

    result = get_stepped_graph_isochrones(
        grid.graph(),
        origin_nodes=[grid.cells[(2, 2)]],
        weight_type="time_min",
        geometry_type=geometry_type,
        weight_value_cutoff=15,
        step=3,
    )

    bands = result.dissolve(by="dist")
    overlaps = _overlap_areas(bands)
    assert all(area < 1.0 for area in overlaps.values()), overlaps
    assert pd.Series(bands.index).is_monotonic_increasing
    # The stop belongs to the band it is walked to, not to the bands of the waits for its buses.
    stop_point = grid.nodes[stop]["geometry"].buffer(1.0)
    assert result.loc[result.intersects(stop_point), "dist"].unique().tolist() == [3.0]


@pytest.mark.parametrize("geometry_type", [None, "radius", "ways"])
def test_coverage_zones_ignore_transit_nodes_on_a_stop(geometry_type):
    grid = _WalkingGrid()
    near_service, far_service = grid.cells[(0, 2)], grid.cells[(4, 4)]
    stop = grid.cells[(1, 2)]
    # A bus from the stop next to the near service reaches the far one in half a minute. The route
    # node on the stop is therefore nearest to the far service, while the stop itself, where the
    # boarding wait applies, is nearest to the near one.
    boarding_node, alighting_node = grid.add_route_node((1, 2)), grid.add_route_node((4, 4))
    grid.add_edge(stop, boarding_node, 8.0, "boarding")
    grid.add_edge(boarding_node, stop, 0.0, "alighting")
    grid.add_edge(boarding_node, alighting_node, 0.5, "bus")
    grid.add_edge(alighting_node, far_service, 0.0, "alighting")

    zones = get_graph_coverage(
        grid.graph(),
        destination_nodes=[near_service, far_service],
        weight_type="time_min",
        geometry_type=geometry_type,
        weight_value_cutoff=15,
    )

    overlaps = _overlap_areas(zones)
    assert all(area < 1.0 for area in overlaps.values()), overlaps
    stop_point = grid.nodes[stop]["geometry"].buffer(1.0)
    assert zones.index[zones.intersects(stop_point)].tolist() == [near_service]
