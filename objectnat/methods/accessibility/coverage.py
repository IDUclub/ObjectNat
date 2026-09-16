from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from iduedu import UrbanGraph, multi_source_dijkstra_nearest_source, multi_source_dijkstra_path_length
from shapely import concave_hull

from objectnat.methods.accessibility._utils import (
    SteppedGeometryType,
    build_radius_clip_geometry,
    build_stepped_accessibility_geometry,
    build_voronoi_cells,
    build_ways_clip_geometry,
    edge_speed_m_per_min,
    select_geometry_nodes,
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
    Calculate coverage zones from source objects through a graph network using
    Dijkstra reachability and Voronoi partitioning.

    Coverage answers *"which area is served by each source object"*. The function:

    1. Snaps each source object to its nearest graph node (or uses the provided
       ``destination_nodes``).
    2. Runs a multi-source Dijkstra search on the reversed graph, so every
       reachable node is labelled with its nearest source within
       ``weight_value_cutoff``.
    3. Builds Voronoi polygons around the graph nodes.
    4. Dissolves the reachable Voronoi cells per source into one zone each.
    5. Clips the result to ``zone``, to a residual-radius / road geometry
       (``geometry_type``), or to the concave hull of the reachable nodes.

    Args:
        urban_graph:
            City graph with node (``nodes_gdf``) and edge (``edges_gdf``) tables.

        gdf_destinations (gpd.GeoDataFrame, optional):
            Source objects the coverage is measured *to*. If the table already
            contains ``graph_node_column`` those node ids are used directly;
            otherwise each geometry is snapped to its nearest graph node. Pass
            either this or ``destination_nodes``.

        destination_nodes (Iterable, optional):
            Ready-made source node ids, used instead of ``gdf_destinations``.

        graph_node_column:
            Name of the column holding graph node ids in ``gdf_destinations``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: edge travel time in minutes
            - ``"length_meter"``: edge length in meters

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
            Buffer applied to graph edges for ``geometry_type="ways"``, in meters
            (default 5.0).

    Returns:
        gpd.GeoDataFrame:
            One coverage polygon per source object, returned in the CRS of
            ``gdf_destinations`` (or the graph CRS when node ids are passed). The
            index matches ``gdf_destinations`` / ``destination_nodes``.

    Notes:
        - For a directed graph the search runs on the reversed edges, so a zone
          describes the area from which its source can be reached.
        - On walk and intermodal graphs the zones are built from nodes reachable on foot,
          so route nodes sharing a stop cannot give one place to several sources.
        - An empty ``GeoDataFrame`` is returned when nothing is reachable.
    """

    if geometry_type is not None and geometry_type not in {"radius", "ways"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if geometry_type is not None and weight_value_cutoff is None:
        raise ValueError("weight_value_cutoff is required when geometry_type is set")
    if buffer_factor <= 0:
        raise ValueError(f"buffer_factor must be > 0, got {buffer_factor}")
    if road_buffer_size <= 0:
        raise ValueError(f"road_buffer_size must be > 0, got {road_buffer_size}")

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

    reachable_graph_nodes_gdf = urban_graph.nodes_gdf.loc[pd.Index(reachable_nodes.index), ["geometry"]].join(
        reachable_nodes, how="left"
    )

    if reachable_graph_nodes_gdf.empty:
        return gpd.GeoDataFrame()

    reachable_graph_nodes_gdf = select_geometry_nodes(urban_graph, reachable_graph_nodes_gdf)

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
    Calculate stepped coverage zones from source objects, combining graph
    reachability with banded (stepped) isochrone geometry.

    Like :func:`get_graph_coverage`, but instead of one zone per source the
    reachable area is split into concentric bands of width ``step``. The
    function:

    1. Snaps each source object to its nearest graph node (or uses the provided
       ``destination_nodes``).
    2. Runs a multi-source Dijkstra search on the reversed graph, labelling
       every reachable node with the distance to its nearest source.
    3. Buckets the nodes into steps and builds banded geometry with the
       selected ``geometry_type``.
    4. Optionally clips the bands to ``zone``.

    Args:
        urban_graph:
            City graph with node and edge tables.

        gdf_destinations (gpd.GeoDataFrame, optional):
            Source objects the coverage is measured *to*. If the table contains
            ``graph_node_column`` those ids are used; otherwise geometries are
            snapped to nearest nodes. Pass either this or ``destination_nodes``.

        destination_nodes (Iterable, optional):
            Ready-made source node ids, used instead of ``gdf_destinations``.

        graph_node_column:
            Name of the graph node id column in ``gdf_destinations``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: edge travel time in minutes
            - ``"length_meter"``: edge length in meters

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
            100 meters for ``length_meter`` and 1 minute for ``time_min``.

        buffer_factor:
            Residual-radius multiplier for ``"radius"``, ``"ways"`` and
            ``"separate"`` (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in meters (default 5.0).

    Returns:
        gpd.GeoDataFrame:
            Stepped coverage polygons with a ``dist`` column (the upper bound of
            each step, in units of ``weight_type``) and ``geometry``, returned in
            the CRS of ``gdf_destinations`` (or the graph CRS).

    Notes:
        - For a directed graph the search runs on the reversed edges; for an
          undirected graph on the original symmetric adjacency.
        - On walk and intermodal graphs the bands are built from nodes reachable on foot,
          so transit route nodes on a stop do not paint it with a later band.
        - An empty ``GeoDataFrame`` is returned when nothing is reachable.
    """

    if geometry_type is not None and geometry_type not in {"radius", "ways", "separate"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if step is None:
        step = 100.0 if weight_type == "length_meter" else 1.0
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if buffer_factor <= 0:
        raise ValueError(f"buffer_factor must be > 0, got {buffer_factor}")
    if road_buffer_size <= 0:
        raise ValueError(f"road_buffer_size must be > 0, got {road_buffer_size}")

    original_crs = gdf_destinations.crs if gdf_destinations is not None else urban_graph.nodes_gdf.crs
    local_crs = urban_graph.nodes_gdf.crs

    distances = multi_source_dijkstra_path_length(
        urban_graph,
        source_nodes=destination_nodes,
        gdf_sources=gdf_destinations,
        graph_node_column=graph_node_column,
        weight=weight_type,
        cutoff=weight_value_cutoff,
        reverse=True,
    )
    distances = distances.sparse.to_dense()
    reachable_nodes = pd.DataFrame(
        {"dist": distances.loc[distances < np.inf].astype(float)},
        index=pd.Index(distances.index[distances < np.inf], name="node"),
    )
    if reachable_nodes.empty:
        return gpd.GeoDataFrame()

    if weight_value_cutoff is None:
        weight_value_cutoff = float(reachable_nodes["dist"].max())

    result = build_stepped_accessibility_geometry(
        urban_graph,
        urban_graph.nodes_gdf.loc[pd.Index(reachable_nodes.index), ["geometry"]].join(reachable_nodes, how="left"),
        geometry_type=geometry_type,
        weight_value=float(weight_value_cutoff),
        weight_type=weight_type,
        step=float(step),
        local_crs=local_crs,
        zone=zone,
        buffer_factor=float(buffer_factor),
        road_buffer_size=float(road_buffer_size),
    )
    return result.to_crs(original_crs)
