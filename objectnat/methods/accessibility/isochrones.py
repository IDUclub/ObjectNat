import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from iduedu import UrbanGraph, dijkstra_parallel, single_source_dijkstra

from objectnat.methods.accessibility._utils import (
    SteppedGeometryType,
    build_separated_dist_polygons,
    check_buffer_args,
    edge_speed_m_per_min,
    finish_ways_clip_geometry,
    node_points,
    radius_buffers,
    select_geometry_nodes,
    stepped_accessibility,
    ways_road_buffers,
)


def _reachable_per_origin(
    urban_graph: UrbanGraph,
    *,
    origin_nodes,
    gdf_origins,
    graph_node_column: str,
    weight_type: str,
    cutoff: float,
    max_workers: int | None,
):
    """
    Origins resolved to graph nodes, and for each origin the costs of the nodes it reaches.

    Uses one single-source search per origin: IduEdu's batch call returns an origins x nodes sparse frame whose
    rows take longer to read back than the searches themselves. Falls back to the batch call if IduEdu's node
    resolution helper is not available.
    """
    try:
        from iduedu.graph.graph_inputs import (  # pylint: disable=import-outside-toplevel
            resolve_graph_nodes_input,
        )
    except ImportError:  # pragma: no cover - older/newer IduEdu layouts
        path_lengths = dijkstra_parallel(
            urban_graph,
            source_nodes=origin_nodes,
            gdf_sources=gdf_origins,
            graph_node_column=graph_node_column,
            weight=weight_type,
            cutoff=cutoff,
            max_workers=max_workers,
            output="distances",
        )
        rows = ((index, row[row < np.inf]) for index, row in path_lengths.iterrows())
        return path_lengths.attrs["source_nodes"], rows

    origin_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=origin_nodes,
        gdf=gdf_origins,
        graph_node_column=graph_node_column,
        nodes_name="origin_nodes",
        gdf_name="gdf_origins",
    )
    rows = (
        (index, single_source_dijkstra(urban_graph, node, weight=weight_type, cutoff=cutoff, output="distances"))
        for index, node in origin_nodes_s.items()
    )
    return origin_nodes_s, rows


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
            Maximum travel time (minutes) or distance (metres) defining the
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
            - ``"length_meter"``: distance-based accessibility in metres

        geometry_type:
            Method used to build each isochrone:

            - ``"radius"``: union of residual-radius buffers around reachable nodes
            - ``"ways"``: buffered reachable road geometry (walk edges only on
              intermodal/walk graphs)
            - ``"separate"``: buffers of ``buffer_factor`` × residual distance around the reachable nodes,
              merged into one polygon

        zone (gpd.GeoDataFrame | gpd.GeoSeries, optional):
            Boundary polygon to clip the resulting isochrones.

        buffer_factor:
            Buffer multiplier used by ``geometry_type="separate"`` (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in metres (default 5.0).

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
          take part in the search.
    """

    if geometry_type not in {"radius", "ways", "separate"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if weight_value_cutoff < 0:
        raise ValueError(f"weight_value_cutoff must be >= 0, got {weight_value_cutoff}")
    check_buffer_args(buffer_factor, road_buffer_size)
    original_crs = gdf_origins.crs if gdf_origins is not None else urban_graph.nodes_gdf.crs
    local_crs = urban_graph.nodes_gdf.crs

    calculation_cutoff = float(weight_value_cutoff)
    if geometry_type == "ways":
        calculation_cutoff += 100.0 if weight_type == "length_meter" else 1.0

    origin_nodes_s, reachable_per_origin = _reachable_per_origin(
        urban_graph,
        origin_nodes=origin_nodes,
        gdf_origins=gdf_origins,
        graph_node_column=graph_node_column,
        weight_type=weight_type,
        cutoff=calculation_cutoff,
        max_workers=max_workers,
    )

    speed_m_per_min = edge_speed_m_per_min(urban_graph) if weight_type == "time_min" else None
    zone_geom = zone.to_crs(local_crs).union_all() if zone is not None else None
    source_nodes = origin_nodes_s.to_numpy()

    # 1. per origin, with pandas (sequential): reachable nodes and the polygons to merge
    prepared = []
    for row_i, (source_index, distances) in enumerate(reachable_per_origin):
        distances = distances[np.isfinite(distances.to_numpy(dtype=float))]
        if distances.empty:
            continue
        reachable = pd.DataFrame(
            {"dist": distances.to_numpy(dtype=float)}, index=pd.Index(distances.index, name="node")
        )
        reachable_graph_nodes_gdf = select_geometry_nodes(urban_graph, node_points(urban_graph, reachable))
        parts = []
        if geometry_type in ("radius", "ways"):
            parts.append(
                radius_buffers(
                    reachable_graph_nodes_gdf,
                    weight_value=float(weight_value_cutoff),
                    weight_type=weight_type,
                    speed_m_per_min=speed_m_per_min,
                )
            )
        if geometry_type == "ways":
            parts.append(ways_road_buffers(urban_graph, reachable_graph_nodes_gdf, float(road_buffer_size)))
        if geometry_type == "separate":
            parts.append(
                build_separated_dist_polygons(
                    reachable_graph_nodes_gdf,
                    weight_value=float(weight_value_cutoff),
                    weight_type=weight_type,
                    step=float(weight_value_cutoff) if weight_value_cutoff > 0 else 1.0,
                    speed_m_per_min=speed_m_per_min,
                    buffer_ratio=float(buffer_factor),
                ).geometry.values
            )
        prepared.append((source_index, source_nodes[row_i], reachable_graph_nodes_gdf, parts))

    # 2. the unions, the costly part, in threads (shapely releases the GIL; no shared pandas objects)
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as pool:
        merged = list(pool.map(lambda item: [shapely.union_all(p) for p in item[3]], prepared))

    # 3. finishing touches (sequential)
    geometries = []
    result_index = []
    result_source_nodes = []
    for (source_index, source_node, reachable_graph_nodes_gdf, _), unions in zip(prepared, merged):
        if geometry_type == "ways":
            geom = finish_ways_clip_geometry(unions[0], unions[1], reachable_graph_nodes_gdf, local_crs)
        else:
            geom = unions[0]
        if zone_geom is not None:
            geom = geom.intersection(zone_geom)
        geometries.append(geom)
        result_index.append(source_index)
        result_source_nodes.append(source_node)

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
            - ``"length_meter"``: distance-based accessibility in metres

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
            100 metres for ``length_meter`` and 1 minute for ``time_min``.

        buffer_factor:
            Residual-radius multiplier for geometry construction (default 0.7).

        road_buffer_size:
            Edge buffer for ``geometry_type="ways"``, in metres (default 5.0).

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

    return stepped_accessibility(
        urban_graph,
        source_nodes=origin_nodes,
        gdf_sources=gdf_origins,
        graph_node_column=graph_node_column,
        weight_type=weight_type,
        weight_value_cutoff=weight_value_cutoff,
        step=step,
        geometry_type=geometry_type,
        zone=zone,
        buffer_factor=buffer_factor,
        road_buffer_size=road_buffer_size,
        reverse=False,
    )
