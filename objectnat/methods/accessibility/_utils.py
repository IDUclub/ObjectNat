from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from iduedu import UrbanGraph, multi_source_dijkstra
from shapely import convex_hull
from shapely.geometry import GeometryCollection

from objectnat.methods.utils.geom_utils import overlay_by_value, remove_inner_geom

SteppedGeometryType = Literal["radius", "ways", "separate"] | None
_WALK_ONLY_GRAPH_TYPES = {"intermodal", "walk"}


def check_buffer_args(buffer_factor: float, road_buffer_size: float, off_network_distance: float | None = None) -> None:
    """Validate the buffer arguments shared by the accessibility functions."""
    if buffer_factor <= 0:
        raise ValueError(f"buffer_factor must be > 0, got {buffer_factor}")
    if road_buffer_size <= 0:
        raise ValueError(f"road_buffer_size must be > 0, got {road_buffer_size}")
    if off_network_distance is not None and not off_network_distance > 0:
        raise ValueError(f"off_network_distance must be > 0 or None, got {off_network_distance}")


def node_points(urban_graph: UrbanGraph, table: pd.DataFrame) -> gpd.GeoDataFrame:
    """Graph node points for the rows of ``table`` (indexed by node id), with its columns."""
    return urban_graph.nodes_gdf.loc[pd.Index(table.index), ["geometry"]].join(table, how="left")


def stepped_accessibility(
    urban_graph: UrbanGraph,
    *,
    source_nodes,
    gdf_sources,
    graph_node_column: str,
    weight_type: str,
    weight_value_cutoff: float | None,
    step: float | None,
    geometry_type: SteppedGeometryType,
    zone,
    buffer_factor: float,
    road_buffer_size: float,
    off_network_distance: float | None,
    reverse: bool,
) -> gpd.GeoDataFrame:
    """
    Bands of width ``step`` of the cost to the nearest source: stepped isochrones search from the origins
    (``reverse=False``), stepped coverage towards the services (``reverse=True``).
    """
    if geometry_type is not None and geometry_type not in {"radius", "ways", "separate"}:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")
    if step is None:
        step = 100.0 if weight_type == "length_meter" else 1.0
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    check_buffer_args(buffer_factor, road_buffer_size, off_network_distance)

    distances = multi_source_dijkstra(
        urban_graph,
        source_nodes=source_nodes,
        gdf_sources=gdf_sources,
        graph_node_column=graph_node_column,
        weight=weight_type,
        cutoff=weight_value_cutoff,
        reverse=reverse,
        output="distances",
    ).sparse.to_dense()
    reachable = distances < np.inf
    reachable_nodes = pd.DataFrame(
        {"dist": distances[reachable].astype(float)}, index=pd.Index(distances.index[reachable], name="node")
    )
    if reachable_nodes.empty:
        return gpd.GeoDataFrame()

    weight_value = float(reachable_nodes["dist"].max()) if weight_value_cutoff is None else float(weight_value_cutoff)
    result = build_stepped_accessibility_geometry(
        urban_graph,
        node_points(urban_graph, reachable_nodes),
        geometry_type=geometry_type,
        weight_value=weight_value,
        weight_type=weight_type,
        step=float(step),
        local_crs=urban_graph.nodes_gdf.crs,
        zone=zone,
        buffer_factor=float(buffer_factor),
        road_buffer_size=float(road_buffer_size),
        off_network_distance=off_network_distance,
    )
    original_crs = gdf_sources.crs if gdf_sources is not None else urban_graph.nodes_gdf.crs
    return result.to_crs(original_crs)


def geometry_edges(urban_graph: UrbanGraph) -> gpd.GeoDataFrame:
    """Return the edges that shape accessibility geometry.

    Walk and intermodal graphs use their pedestrian (``type == "walk"``) edges; other
    graphs, or graphs without a ``type`` column, use every edge.
    """
    edges = urban_graph.edges_gdf
    if urban_graph.type in _WALK_ONLY_GRAPH_TYPES and "type" in edges.columns:
        walk_edges = edges[edges["type"] == "walk"]
        if not walk_edges.empty:
            return walk_edges
    return edges


def select_geometry_nodes(urban_graph: UrbanGraph, reachable_graph_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep the reachable nodes that shape geometry, one per location.

    Only endpoints of :func:`geometry_edges` are kept, unless none of them is reachable,
    and nodes sharing a location are collapsed onto the smallest ``dist``. Several route
    nodes can sit on one stop; a Voronoi cell shared by them would otherwise be assigned
    to every band they fall in.
    """
    edges = geometry_edges(urban_graph)
    nodes = reachable_graph_nodes_gdf
    on_geometry_edges = nodes.index.isin(edges["u"]) | nodes.index.isin(edges["v"])
    if on_geometry_edges.any():
        nodes = nodes.loc[on_geometry_edges]
    ordered = nodes.iloc[np.argsort(nodes["dist"].to_numpy(dtype=float), kind="stable")]
    location = pd.MultiIndex.from_arrays([ordered.geometry.x.to_numpy(), ordered.geometry.y.to_numpy()])
    return ordered.iloc[np.flatnonzero(~location.duplicated())]


def build_voronoi_cells(
    urban_graph: UrbanGraph,
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    value_column: str,
    local_crs: Any,
    clip_geom=None,
) -> gpd.GeoDataFrame:
    reachable_node_index = pd.Index(reachable_graph_nodes_gdf.index)
    edges = geometry_edges(urban_graph)
    if clip_geom is None:
        edges_near_reachable = edges[edges["u"].isin(reachable_node_index) | edges["v"].isin(reachable_node_index)]
        neighbor_nodes = pd.Index(edges_near_reachable["u"]).append(pd.Index(edges_near_reachable["v"])).unique()
        boundary_nodes = neighbor_nodes.difference(reachable_node_index)
        voronoi_node_index = reachable_node_index.append(boundary_nodes).unique()
    else:
        nodes = urban_graph.nodes_gdf
        site_mask = (nodes.index.isin(edges["u"]) | nodes.index.isin(edges["v"])) & nodes.geometry.intersects(clip_geom)
        voronoi_node_index = nodes.index[site_mask].append(reachable_node_index).unique()
    voronoi_nodes = urban_graph.nodes_gdf.loc[voronoi_node_index, ["geometry"]]
    voronois = gpd.GeoDataFrame(geometry=list(voronoi_nodes.geometry.voronoi_polygons()), crs=local_crs)
    reachable_cell_index = reachable_graph_nodes_gdf.reset_index(names="node").sjoin(voronois, how="inner")
    return gpd.GeoDataFrame(
        reachable_cell_index[["node", value_column]].reset_index(drop=True),
        geometry=voronois.geometry.iloc[reachable_cell_index["index_right"].to_numpy()].reset_index(drop=True),
        crs=local_crs,
    )


def edge_speed_m_per_min(urban_graph: UrbanGraph) -> float:
    edge_speeds = urban_graph.edges_gdf["length_meter"] / urban_graph.edges_gdf["time_min"]
    edge_speeds = edge_speeds.replace([np.inf, -np.inf], np.nan).dropna()
    edge_speeds = edge_speeds[edge_speeds > 0]
    if edge_speeds.empty:
        raise ValueError("Cannot calculate speed_m_per_min from edge length_meter/time_min")
    return float(edge_speeds.median())


def radius_buffers(
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    weight_value: float,
    weight_type: str,
    speed_m_per_min: float | None,
    max_distance: float | None = None,
) -> np.ndarray:
    """
    Buffers of the reachable nodes by their remaining budget as a distance (the parts of a radius isochrone),
    at most ``max_distance`` metres: the budget left at a node is spent off the network only that far.
    """
    buffer_size = (float(weight_value) - reachable_graph_nodes_gdf["dist"].to_numpy(dtype=float)).clip(min=0.0)

    if weight_type == "time_min":
        if speed_m_per_min is None:
            raise ValueError("speed_m_per_min is required for weight_type='time_min'")
        buffer_size = buffer_size * float(speed_m_per_min)
    if max_distance is not None:
        buffer_size = np.minimum(buffer_size, float(max_distance))
    return shapely.buffer(reachable_graph_nodes_gdf.geometry.values, buffer_size, quad_segs=4)


def build_radius_clip_geometry(
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    weight_value: float,
    weight_type: str,
    speed_m_per_min: float | None,
    max_distance: float | None = None,
):
    return shapely.union_all(
        radius_buffers(
            reachable_graph_nodes_gdf,
            weight_value=weight_value,
            weight_type=weight_type,
            speed_m_per_min=speed_m_per_min,
            max_distance=max_distance,
        )
    )


def ways_road_buffers(
    urban_graph: UrbanGraph, reachable_graph_nodes_gdf: gpd.GeoDataFrame, road_buffer_size: float
) -> np.ndarray:
    """
    Buffers of the geometry edges touching reachable nodes. Edges stored once per direction share
    their geometry; each is kept once, which halves the union work without changing its result.
    """
    reachable_node_index = pd.Index(reachable_graph_nodes_gdf.index)
    edges = geometry_edges(urban_graph)
    edge_mask = edges["u"].isin(reachable_node_index) | edges["v"].isin(reachable_node_index)
    lines = shapely.normalize(edges.geometry.values[edge_mask.to_numpy()])
    if len(lines) == 0:
        return lines
    _, first = np.unique(shapely.to_wkb(lines), return_index=True)
    return shapely.buffer(lines[np.sort(first)], float(road_buffer_size), quad_segs=1)


def finish_ways_clip_geometry(radius_geom, roads_geom, reachable_graph_nodes_gdf: gpd.GeoDataFrame, crs):
    """Keep the parts of the road buffers inside the radius isochrone that hold a reachable node."""
    if roads_geom is None or roads_geom.is_empty:
        return GeometryCollection()
    roads = gpd.GeoDataFrame(geometry=[roads_geom], crs=crs)
    clipped_roads = roads.clip(radius_geom, keep_geom_type=True).explode(ignore_index=True)
    if clipped_roads.empty:
        return GeometryCollection()

    geom_to_keep = clipped_roads.sjoin(reachable_graph_nodes_gdf[["geometry"]], how="inner").index.unique()
    if len(geom_to_keep) == 0:
        return GeometryCollection()
    return remove_inner_geom(clipped_roads.loc[geom_to_keep].union_all())


def build_ways_clip_geometry(
    urban_graph: UrbanGraph,
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    weight_value: float,
    weight_type: str,
    speed_m_per_min: float | None,
    road_buffer_size: float,
):
    radius_geom = build_radius_clip_geometry(
        reachable_graph_nodes_gdf,
        weight_value=weight_value,
        weight_type=weight_type,
        speed_m_per_min=speed_m_per_min,
    )
    road_buffers = ways_road_buffers(urban_graph, reachable_graph_nodes_gdf, road_buffer_size)
    if len(road_buffers) == 0:
        return GeometryCollection()
    return finish_ways_clip_geometry(
        radius_geom, shapely.union_all(road_buffers), reachable_graph_nodes_gdf, urban_graph.edges_gdf.crs
    )


def build_separated_dist_polygons(
    points: gpd.GeoDataFrame,
    *,
    weight_value: float,
    weight_type: str,
    step: float,
    speed_m_per_min: float | None = None,
    buffer_ratio: float = 0.7,
) -> gpd.GeoDataFrame:
    """
    Build independent per-step buffer polygons over reachable graph nodes.

    ``points`` must contain point geometry and a ``dist`` column in units of
    ``weight_type``. For ``time_min`` the buffer size is converted to metres via
    ``speed_m_per_min``.
    """

    def empty_result() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"dist": []}, geometry=[], crs=points.crs)

    if points.empty:
        return empty_result()
    if "dist" not in points.columns:
        raise KeyError("points must contain 'dist' column")
    if weight_value < 0:
        raise ValueError(f"weight_value must be >= 0, got {weight_value}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if weight_type == "time_min" and (speed_m_per_min is None or speed_m_per_min <= 0):
        raise ValueError("speed_m_per_min must be positive for weight_type='time_min'")

    points = points.loc[points["dist"].notna(), ["dist", points.geometry.name]].copy()
    if points.empty:
        return empty_result()

    dist_for_buffer = points["dist"].astype(float).clip(lower=0.1)
    upper_step = np.minimum((np.floor(dist_for_buffer / float(step)) + 1.0) * float(step), float(weight_value))
    buffer_size = np.maximum(upper_step - dist_for_buffer, 0.0) * float(buffer_ratio)
    if weight_type == "time_min":
        buffer_size = buffer_size * float(speed_m_per_min)

    # the first band is [0, step]: sources at cost 0 belong to it, not to a band of their own
    points["dist"] = np.minimum(
        np.maximum(np.ceil(points["dist"].astype(float).to_numpy() / float(step)), 1.0) * float(step),
        float(weight_value),
    )
    points.geometry = points.geometry.buffer(buffer_size)
    points = points[~points.geometry.is_empty]
    if points.empty:
        return empty_result()

    # a face covered by several bands is reached first in the earliest of them
    stepped = overlay_by_value(points.dissolve(by="dist", as_index=False), "dist", how="min")
    if stepped.empty:
        return empty_result()
    return stepped.explode(ignore_index=True)


def build_stepped_accessibility_geometry(
    urban_graph: UrbanGraph,
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    geometry_type: SteppedGeometryType,
    weight_value: float,
    weight_type: str,
    step: float,
    local_crs: Any,
    zone: gpd.GeoDataFrame | gpd.GeoSeries | None,
    buffer_factor: float,
    road_buffer_size: float,
    off_network_distance: float | None = None,
) -> gpd.GeoDataFrame:
    if reachable_graph_nodes_gdf.empty:
        return gpd.GeoDataFrame()

    reachable_graph_nodes_gdf = select_geometry_nodes(urban_graph, reachable_graph_nodes_gdf)
    speed_m_per_min = edge_speed_m_per_min(urban_graph) if weight_type == "time_min" else None

    if geometry_type == "separate":
        result = build_separated_dist_polygons(
            reachable_graph_nodes_gdf,
            weight_value=float(weight_value),
            weight_type=weight_type,
            step=float(step),
            speed_m_per_min=speed_m_per_min,
            buffer_ratio=buffer_factor,
        )
        if zone is not None and not result.empty:
            result = result.clip(zone.to_crs(local_crs))
        return result

    if geometry_type is None:
        if zone is None:
            clip_geom = convex_hull(reachable_graph_nodes_gdf.union_all())
        else:
            clip_geom = zone.to_crs(local_crs).union_all()
    elif geometry_type == "radius":
        clip_geom = build_radius_clip_geometry(
            reachable_graph_nodes_gdf,
            weight_value=float(weight_value),
            weight_type=weight_type,
            speed_m_per_min=speed_m_per_min,
            max_distance=off_network_distance,
        )
    elif geometry_type == "ways":
        clip_geom = build_ways_clip_geometry(
            urban_graph,
            reachable_graph_nodes_gdf,
            weight_value=float(weight_value),
            weight_type=weight_type,
            speed_m_per_min=speed_m_per_min,
            road_buffer_size=road_buffer_size,
        )
    else:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")

    if geometry_type is not None and zone is not None:
        clip_geom = clip_geom.intersection(zone.to_crs(local_crs).union_all())

    nodes_for_voronoi = reachable_graph_nodes_gdf.copy()
    # the first band is [0, step]: sources at cost 0 belong to it, not to a band of their own
    nodes_for_voronoi["dist"] = np.minimum(
        np.maximum(np.ceil(nodes_for_voronoi["dist"].astype(float).to_numpy() / float(step)), 1.0) * float(step),
        float(weight_value),
    )

    cells = build_voronoi_cells(
        urban_graph,
        nodes_for_voronoi,
        value_column="dist",
        local_crs=local_crs,
        clip_geom=clip_geom if geometry_type is not None else None,
    )
    result = cells.dissolve(by="dist", as_index=False).drop(columns=["node"])
    return result.clip(clip_geom)
