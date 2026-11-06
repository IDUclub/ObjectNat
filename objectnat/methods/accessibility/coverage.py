from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from iduedu import UrbanGraph, multi_source_dijkstra_nearest_source
from shapely import concave_hull

from objectnat.methods.accessibility._utils import (
    SteppedGeometryType,
    build_radius_clip_geometry,
    build_voronoi_cells,
    build_ways_clip_geometry,
    check_buffer_args,
    edge_speed_m_per_min,
    node_points,
    select_geometry_nodes,
    stepped_accessibility,
)


def get_graph_coverage(
    urban_graph: UrbanGraph,
    *,
    gdf_destinations: gpd.GeoDataFrame | None = None,
    destination_nodes: Iterable[Any] | None = None,
    graph_node_column: str = "graph_node_id",
    weight_type: Literal["time_min", "length_meter"] = "time_min",
    geometry_type: Literal["radius", "ways"] | None = None,
    weight_value_cutoff: float | None = None,
    zone: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    buffer_factor: float = 0.7,
    road_buffer_size: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    Calculate coverage zones of destinations (services) on an urban graph using
    Dijkstra reachability and Voronoi partitioning.

    Coverage answers *"which area is served by each destination"*. The function:

    1. Snaps each destination to its nearest graph node (or uses the provided
       ``destination_nodes``).
    2. Runs a multi-source Dijkstra search on the reversed graph, so every
       reachable node is labelled with its nearest destination within
       ``weight_value_cutoff``.
    3. Builds Voronoi polygons around the graph nodes.
    4. Dissolves the reachable Voronoi cells per source into one zone each.
    5. Clips the result to ``zone``, to a residual-radius / road geometry
       (``geometry_type``), or to the concave hull of the reachable nodes.

    Args:
        urban_graph:
            City graph with node (``nodes_gdf``) and edge (``edges_gdf``) tables.

        gdf_destinations (gpd.GeoDataFrame, optional):
            Destinations the coverage is measured *to*. If the table already
            contains ``graph_node_column`` those node ids are used directly;
            otherwise each geometry is snapped to its nearest graph node. Pass
            either this or ``destination_nodes``.

        destination_nodes (Iterable, optional):
            Node ids of the destinations, used instead of ``gdf_destinations``.

        graph_node_column:
            Name of the column with graph node ids in ``gdf_destinations``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: edge travel time in minutes
            - ``"length_meter"``: edge length in metres

        geometry_type:
            Optional refinement of the coverage shape:

            - ``None``: raw Voronoi cells clipped to ``zone`` or the concave hull
            - ``"radius"``: additionally clip to residual-radius buffers around
              reachable nodes (remaining budget converted to distance)
            - ``"ways"``: additionally clip to buffered road geometry inside the
              residual radii (walk edges only on intermodal/walk graphs)

            Requires ``weight_value_cutoff`` when set.

        weight_value_cutoff (float, optional):
            Maximum path cost, e.g. max travel time or distance. Units depend on
            ``weight_type``.

        zone (gpd.GeoDataFrame | gpd.GeoSeries, optional):
            Boundary polygon to clip the resulting zones. If ``None`` and no
            ``geometry_type`` is given, the concave hull of the reachable nodes
            is used.

        buffer_factor:
            Multiplier for the residual radius when ``geometry_type`` is set
            (default 0.7).

        road_buffer_size:
            Buffer applied to graph edges for ``geometry_type="ways"``, in metres
            (default 5.0).

    Returns:
        gpd.GeoDataFrame:
            One coverage polygon per destination, returned in the CRS of
            ``gdf_destinations`` (or the graph CRS when node ids are passed). The
            index matches ``gdf_destinations`` / ``destination_nodes``.

    Notes:
        - For a directed graph the search runs on the reversed edges, so a zone
          describes the area from which its destination can be reached.
        - On walk and intermodal graphs the zones are built from nodes reachable on foot,
          so route nodes sharing a stop cannot give one place to several destinations.
        - An empty ``GeoDataFrame`` is returned when nothing is reachable.
    """

    if geometry_type is not None and geometry_type not in {"radius", "ways"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if geometry_type is not None and weight_value_cutoff is None:
        raise ValueError("weight_value_cutoff is required when geometry_type is set")
    check_buffer_args(buffer_factor, road_buffer_size)

    original_crs = gdf_destinations.crs if gdf_destinations is not None else urban_graph.nodes_gdf.crs
    local_crs = urban_graph.nodes_gdf.crs

    nearest_sources = multi_source_dijkstra_nearest_source(
        urban_graph,
        source_nodes=destination_nodes,
        gdf_sources=gdf_destinations,
        graph_node_column=graph_node_column,
        weight=weight_type,
        cutoff=weight_value_cutoff,
        reverse=True,
    )
    source_nodes_s = nearest_sources.attrs["source_nodes"]
    distances = nearest_sources["dist"].sparse.to_dense()
    reachable_mask = distances < np.inf
    reachable_nodes = pd.DataFrame(
        {
            "node_to": nearest_sources.loc[reachable_mask, "source_node"],
            "dist": distances.loc[reachable_mask].astype(float),
        },
        index=pd.Index(nearest_sources.index[reachable_mask], name="node"),
    )

    if reachable_nodes.empty:
        return gpd.GeoDataFrame()
    reachable_graph_nodes_gdf = select_geometry_nodes(urban_graph, node_points(urban_graph, reachable_nodes))

    if geometry_type == "radius":
        speed_m_per_min = edge_speed_m_per_min(urban_graph) if weight_type == "time_min" else None
        clip_zone = build_radius_clip_geometry(
            reachable_graph_nodes_gdf,
            weight_value=float(weight_value_cutoff),
            weight_type=weight_type,
            speed_m_per_min=speed_m_per_min,
        )
    elif geometry_type == "ways":
        speed_m_per_min = edge_speed_m_per_min(urban_graph) if weight_type == "time_min" else None
        clip_zone = build_ways_clip_geometry(
            urban_graph,
            reachable_graph_nodes_gdf,
            weight_value=float(weight_value_cutoff),
            weight_type=weight_type,
            speed_m_per_min=speed_m_per_min,
            road_buffer_size=float(road_buffer_size),
        )
    elif zone is None:
        clip_zone = concave_hull(reachable_graph_nodes_gdf.union_all(), ratio=0.5)
    else:
        clip_zone = zone.to_crs(local_crs)

    if geometry_type is not None and zone is not None:
        clip_zone = clip_zone.intersection(zone.to_crs(local_crs).union_all())

    cells = build_voronoi_cells(
        urban_graph,
        reachable_graph_nodes_gdf,
        value_column="node_to",
        local_crs=local_crs,
        clip_geom=clip_zone if geometry_type is not None else None,
    )
    result = cells.dissolve(by="node_to").reset_index().drop(columns=["node"])
    zone_coverages = result.clip(clip_zone, keep_geom_type=True)

    source_index_by_node = pd.DataFrame(
        {
            "node_to": source_nodes_s.to_numpy(),
            "__source_index": source_nodes_s.index,
        }
    )
    zone_coverages = zone_coverages.merge(source_index_by_node, on="node_to", how="inner")
    zone_coverages = zone_coverages.drop(columns="node_to").set_index("__source_index")
    zone_coverages.index.name = source_nodes_s.index.name

    return zone_coverages.to_crs(original_crs)


def get_stepped_graph_coverage(
    urban_graph: UrbanGraph,
    *,
    gdf_destinations: gpd.GeoDataFrame | None = None,
    destination_nodes: Iterable[Any] | None = None,
    graph_node_column: str = "graph_node_id",
    weight_type: Literal["time_min", "length_meter"] = "time_min",
    geometry_type: SteppedGeometryType = "radius",
    weight_value_cutoff: float | None = None,
    zone: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    step: float | None = None,
    buffer_factor: float = 0.7,
    road_buffer_size: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    Calculate stepped coverage zones of destinations (services), combining graph
    reachability with banded (stepped) isochrone geometry.

    Like :func:`get_graph_coverage`, but instead of one zone per destination the
    reachable area is split into concentric bands of width ``step``. The
    function:

    1. Snaps each destination to its nearest graph node (or uses the provided
       ``destination_nodes``).
    2. Runs a multi-source Dijkstra search on the reversed graph, labelling
       every reachable node with the cost to its nearest destination.
    3. Buckets the nodes into steps and builds banded geometry with the
       selected ``geometry_type``.
    4. Optionally clips the bands to ``zone``.

    Args:
        urban_graph:
            City graph with node and edge tables.

        gdf_destinations (gpd.GeoDataFrame, optional):
            Destinations the coverage is measured *to*. If the table contains
            ``graph_node_column`` those ids are used; otherwise geometries are
            snapped to nearest nodes. Pass either this or ``destination_nodes``.

        destination_nodes (Iterable, optional):
            Node ids of the destinations, used instead of ``gdf_destinations``.

        graph_node_column:
            Name of the graph node id column in ``gdf_destinations``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: edge travel time in minutes
            - ``"length_meter"``: edge length in metres

        geometry_type:
            Method used to build each step's geometry:

            - ``None``: Voronoi cells around graph nodes
            - ``"radius"``: Voronoi cells clipped to residual-radius buffers
            - ``"ways"``: Voronoi cells clipped to buffered road geometry
              (walk edges only on intermodal/walk graphs)
            - ``"separate"``: independent circular buffers per step

        weight_value_cutoff (float, optional):
            Maximum path cost limiting the coverage extent. If ``None``, the
            farthest reachable node defines the extent.

        zone (gpd.GeoDataFrame | gpd.GeoSeries, optional):
            Boundary polygon to clip the resulting stepped zones.

        step (float, optional):
            Width of each step, in units of ``weight_type``. Defaults to
            100 metres for ``length_meter`` and 1 minute for ``time_min``.

        buffer_factor:
            Residual-radius multiplier for ``"radius"``, ``"ways"`` and
            ``"separate"`` (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in metres (default 5.0).

    Returns:
        gpd.GeoDataFrame:
            Stepped coverage polygons with a ``dist`` column (the upper bound of
            each step, in units of ``weight_type``) and ``geometry``, returned in
            the CRS of ``gdf_destinations`` (or the graph CRS).

    Notes:
        - For a directed graph the search runs on the reversed edges; for an
          undirected graph on the original symmetric adjacency.
        - On walk and intermodal graphs the bands are built from nodes reachable on foot,
          so transit route nodes at a stop, whose cost includes the wait, do not move it to a later band.
        - An empty ``GeoDataFrame`` is returned when nothing is reachable.
    """

    return stepped_accessibility(
        urban_graph,
        source_nodes=destination_nodes,
        gdf_sources=gdf_destinations,
        graph_node_column=graph_node_column,
        weight_type=weight_type,
        weight_value_cutoff=weight_value_cutoff,
        step=step,
        geometry_type=geometry_type,
        zone=zone,
        buffer_factor=buffer_factor,
        road_buffer_size=road_buffer_size,
        reverse=True,
    )
