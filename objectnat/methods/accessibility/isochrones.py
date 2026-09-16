from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from iduedu import UrbanGraph, dijkstra_path_length_parallel, multi_source_dijkstra_path_length

from objectnat.methods.accessibility._utils import (
    SteppedGeometryType,
    build_radius_clip_geometry,
    build_separated_dist_polygons,
    build_stepped_accessibility_geometry,
    build_ways_clip_geometry,
    edge_speed_m_per_min,
    select_geometry_nodes,
)


def get_graph_isochrones(
    urban_graph: UrbanGraph,
    *,
    weight_value_cutoff: float,
    gdf_origins: gpd.GeoDataFrame | None = None,
    origin_nodes: Iterable[Any] | None = None,
    graph_node_column: str = "graph_node_id",
    weight_type: Literal["time_min", "length_meter"] = "time_min",
    geometry_type: Literal["radius", "ways", "separate"] = "radius",
    zone: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    buffer_factor: float = 0.7,
    road_buffer_size: float = 5.0,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Calculate accessibility isochrones from origin objects over an urban graph.

    An isochrone is the area reachable from an origin within ``weight_value_cutoff``.
    Unlike coverage, each origin gets its own independent isochrone, but all
    shortest-path searches run in a single parallel Numba call. The function:

    1. Snaps each origin to its nearest graph node (or uses the provided
       ``origin_nodes``).
    2. Runs one Dijkstra search per origin in parallel.
    3. Builds a reachability polygon for each origin with the selected
       ``geometry_type``.
    4. Optionally clips each isochrone to ``zone``.

    Args:
        urban_graph:
            City graph with node and edge tables.

        weight_value_cutoff (float):
            Maximum travel time (minutes) or distance (meters) defining the
            isochrone extent.

        gdf_origins (gpd.GeoDataFrame, optional):
            Origin objects. If the table contains ``graph_node_column`` those ids
            are used; otherwise geometries are snapped to nearest nodes. Pass
            either this or ``origin_nodes``.

        origin_nodes (Iterable, optional):
            Ready-made origin node ids, used instead of ``gdf_origins``.

        graph_node_column:
            Name of the graph node id column in ``gdf_origins``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: time-based accessibility in minutes
            - ``"length_meter"``: distance-based accessibility in meters

        geometry_type:
            Method used to build each isochrone:

            - ``"radius"``: union of residual-radius buffers around reachable nodes
            - ``"ways"``: buffered reachable road geometry (walk edges only on
              intermodal/walk graphs)
            - ``"separate"``: a single merged buffer-based isochrone

        zone (gpd.GeoDataFrame | gpd.GeoSeries, optional):
            Boundary polygon to clip the resulting isochrones.

        buffer_factor:
            Buffer multiplier used by ``geometry_type="separate"`` (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in meters (default 5.0).

        max_workers (int, optional):
            Number of parallel Numba threads. Defaults to the Numba runtime.

    Returns:
        gpd.GeoDataFrame:
            One isochrone polygon per origin with a ``source_node`` column and
            ``geometry``, returned in the CRS of ``gdf_origins`` (or the graph
            CRS). The index matches ``gdf_origins`` / ``origin_nodes``.

    Notes:
        - For ``geometry_type="ways"`` the search budget is slightly extended so
          road edges are not clipped short of the boundary.
        - Origins with no reachable nodes are dropped from the result.
        - On walk and intermodal graphs the geometry is built from nodes reachable on
          foot; transit route nodes, whose distance includes the wait to board, only
          carry the search.
    """

    if geometry_type not in {"radius", "ways", "separate"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if weight_value_cutoff < 0:
        raise ValueError(f"weight_value_cutoff must be >= 0, got {weight_value_cutoff}")
    if buffer_factor <= 0:
        raise ValueError(f"buffer_factor must be > 0, got {buffer_factor}")
    if road_buffer_size <= 0:
        raise ValueError(f"road_buffer_size must be > 0, got {road_buffer_size}")
    original_crs = gdf_origins.crs if gdf_origins is not None else urban_graph.nodes_gdf.crs
    local_crs = urban_graph.nodes_gdf.crs

    calculation_cutoff = float(weight_value_cutoff)
    if geometry_type == "ways":
        calculation_cutoff += 100.0 if weight_type == "length_meter" else 1.0

    path_lengths = dijkstra_path_length_parallel(
        urban_graph,
        source_nodes=origin_nodes,
        gdf_sources=gdf_origins,
        graph_node_column=graph_node_column,
        weight=weight_type,
        cutoff=calculation_cutoff,
        max_workers=max_workers,
    )
    origin_nodes_s = path_lengths.attrs["source_nodes"]

    speed_m_per_min = edge_speed_m_per_min(urban_graph) if weight_type == "time_min" else None
    zone_geom = zone.to_crs(local_crs).union_all() if zone is not None else None
    geometries = []
    result_index = []
    result_source_nodes = []
    source_nodes = origin_nodes_s.to_numpy()
    for row_i, (source_index, distances) in enumerate(path_lengths.iterrows()):
        distances = pd.Series(distances.to_numpy(dtype=float), index=distances.index)
        reachable_nodes = pd.DataFrame(
            {"dist": distances.loc[distances < np.inf].astype(float)},
            index=pd.Index(distances.index[distances < np.inf], name="node"),
        )
        if reachable_nodes.empty:
            continue

        reachable_graph_nodes_gdf = urban_graph.nodes_gdf.loc[pd.Index(reachable_nodes.index), ["geometry"]].join(
            reachable_nodes, how="left"
        )
        reachable_graph_nodes_gdf = select_geometry_nodes(urban_graph, reachable_graph_nodes_gdf)
        if geometry_type == "radius":
            geom = build_radius_clip_geometry(
                reachable_graph_nodes_gdf,
                weight_value=float(weight_value_cutoff),
                weight_type=weight_type,
                speed_m_per_min=speed_m_per_min,
            )
        elif geometry_type == "ways":
            geom = build_ways_clip_geometry(
                urban_graph,
                reachable_graph_nodes_gdf,
                weight_value=float(weight_value_cutoff),
                weight_type=weight_type,
                speed_m_per_min=speed_m_per_min,
                road_buffer_size=float(road_buffer_size),
            )
        else:
            geom = build_separated_dist_polygons(
                reachable_graph_nodes_gdf,
                weight_value=float(weight_value_cutoff),
                weight_type=weight_type,
                step=float(weight_value_cutoff) if weight_value_cutoff > 0 else 1.0,
                speed_m_per_min=speed_m_per_min,
                buffer_ratio=float(buffer_factor),
            ).union_all()

        if zone_geom is not None:
            geom = geom.intersection(zone_geom)
        geometries.append(geom)
        result_index.append(source_index)
        result_source_nodes.append(source_nodes[row_i])

    result = gpd.GeoDataFrame(
        {"source_node": result_source_nodes},
        geometry=geometries,
        index=pd.Index(result_index, name=origin_nodes_s.index.name),
        crs=local_crs,
    )
    result = result[result.geometry.notna() & ~result.geometry.is_empty]
    return result.to_crs(original_crs)


def get_stepped_graph_isochrones(
    urban_graph: UrbanGraph,
    *,
    gdf_origins: gpd.GeoDataFrame | None = None,
    origin_nodes: Iterable[Any] | None = None,
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
    Calculate stepped accessibility isochrones from origin objects over an urban
    graph, splitting each reachable area into concentric bands of width ``step``.

    Unlike coverage, the search runs outward from the origins (along the original
    edge direction on a directed graph). The function:

    1. Snaps each origin to its nearest graph node (or uses the provided
       ``origin_nodes``).
    2. Runs a multi-source Dijkstra search outward from the origins.
    3. Buckets reachable nodes into steps and builds banded geometry with the
       selected ``geometry_type``.
    4. Optionally clips the bands to ``zone``.

    Args:
        urban_graph:
            City graph with node and edge tables.

        gdf_origins (gpd.GeoDataFrame, optional):
            Origin objects. If the table contains ``graph_node_column`` those ids
            are used; otherwise geometries are snapped to nearest nodes. Pass
            either this or ``origin_nodes``.

        origin_nodes (Iterable, optional):
            Ready-made origin node ids, used instead of ``gdf_origins``.

        graph_node_column:
            Name of the graph node id column in ``gdf_origins``.

        weight_type:
            Type of edge weight used for path calculation:

            - ``"time_min"``: time-based accessibility in minutes
            - ``"length_meter"``: distance-based accessibility in meters

        geometry_type:
            Method used to build each step's geometry:

            - ``None``: Voronoi cells around graph nodes
            - ``"radius"``: Voronoi cells clipped to residual-radius buffers
            - ``"ways"``: Voronoi cells clipped to buffered road geometry
              (walk edges only on intermodal/walk graphs)
            - ``"separate"``: independent circular buffers per step

        weight_value_cutoff (float, optional):
            Maximum travel time or distance. If ``None``, the isochrone is built
            out to the farthest reachable node.

        zone (gpd.GeoDataFrame | gpd.GeoSeries, optional):
            Boundary polygon to clip the resulting stepped isochrones.

        step (float, optional):
            Width of each step, in units of ``weight_type``. Defaults to
            100 meters for ``length_meter`` and 1 minute for ``time_min``.

        buffer_factor:
            Residual-radius multiplier for geometry construction (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in meters (default 5.0).

    Returns:
        gpd.GeoDataFrame:
            Stepped isochrone polygons with a ``dist`` column (the upper bound of
            each step, in units of ``weight_type``) and ``geometry``, returned in
            the CRS of ``gdf_origins`` (or the graph CRS).

    Notes:
        - Multiple origins are processed together: the bands describe the
          combined reachability of all origins (each node keeps the distance to
          its nearest origin).
        - On walk and intermodal graphs the bands are built from nodes reachable on
          foot, so a stop is drawn in the band it is walked to, not in the bands of the
          waits for the vehicles that call there.
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

    original_crs = gdf_origins.crs if gdf_origins is not None else urban_graph.nodes_gdf.crs
    local_crs = urban_graph.nodes_gdf.crs

    distances = multi_source_dijkstra_path_length(
        urban_graph,
        source_nodes=origin_nodes,
        gdf_sources=gdf_origins,
        graph_node_column=graph_node_column,
        weight=weight_type,
        cutoff=weight_value_cutoff,
        reverse=False,
    )
    distances = distances.sparse.to_dense()
    reachable_nodes = pd.DataFrame(
        {"dist": distances.loc[distances < np.inf].astype(float)},
        index=pd.Index(distances.index[distances < np.inf], name="node"),
    )
    if reachable_nodes.empty:
        return gpd.GeoDataFrame()

    weight_value = float(reachable_nodes["dist"].max()) if weight_value_cutoff is None else float(weight_value_cutoff)
    result = build_stepped_accessibility_geometry(
        urban_graph,
        urban_graph.nodes_gdf.loc[pd.Index(reachable_nodes.index), ["geometry"]].join(reachable_nodes, how="left"),
        geometry_type=geometry_type,
        weight_value=weight_value,
        weight_type=weight_type,
        step=float(step),
        local_crs=local_crs,
        zone=zone,
        buffer_factor=float(buffer_factor),
        road_buffer_size=float(road_buffer_size),
    )
    return result.to_crs(original_crs)
