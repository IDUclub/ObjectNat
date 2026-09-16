from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from iduedu import UrbanGraph
from shapely import convex_hull
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from objectnat.methods.utils.geom_utils import polygons_to_multilinestring

SteppedGeometryType = Literal["radius", "ways", "separate"] | None
_WALK_ONLY_GRAPH_TYPES = {"intermodal", "walk"}


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


def build_radius_clip_geometry(
    reachable_graph_nodes_gdf: gpd.GeoDataFrame,
    *,
    weight_value: float,
    weight_type: str,
    speed_m_per_min: float | None,
):
    buffer_size = (float(weight_value) - reachable_graph_nodes_gdf["dist"].astype(float)).clip(lower=0.0)

    if weight_type == "time_min":
        if speed_m_per_min is None:
            raise ValueError("speed_m_per_min is required for weight_type='time_min'")
        buffer_size = buffer_size * float(speed_m_per_min)
    return reachable_graph_nodes_gdf.geometry.buffer(buffer_size, resolution=4).union_all()


def remove_inner_geom(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(polygon.exterior) for polygon in geom.geoms])
    if geom.geom_type == "GeometryCollection":
        polygon_geoms = [
            remove_inner_geom(part) for part in geom.geoms if part.geom_type in {"Polygon", "MultiPolygon"}
        ]
        if not polygon_geoms:
            return GeometryCollection()
        return unary_union(polygon_geoms)
    return geom


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
    reachable_node_index = pd.Index(reachable_graph_nodes_gdf.index)
    edges = geometry_edges(urban_graph)
    edge_mask = edges["u"].isin(reachable_node_index) | edges["v"].isin(reachable_node_index)
    if not edge_mask.any():
        return GeometryCollection()

    roads = gpd.GeoDataFrame(
        geometry=[edges.loc[edge_mask].geometry.buffer(float(road_buffer_size), resolution=1).union_all()],
        crs=urban_graph.edges_gdf.crs,
    )
    clipped_roads = roads.clip(radius_geom, keep_geom_type=True).explode(ignore_index=True)
    if clipped_roads.empty:
        return GeometryCollection()

    geom_to_keep = clipped_roads.sjoin(reachable_graph_nodes_gdf[["geometry"]], how="inner").index.unique()
    if len(geom_to_keep) == 0:
        return GeometryCollection()
    return remove_inner_geom(clipped_roads.loc[geom_to_keep].union_all())


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
    ``weight_type``. For ``time_min`` the buffer size is converted to meters via
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

    points["dist"] = np.minimum(
        np.ceil(points["dist"].astype(float).to_numpy() / float(step)) * float(step),
        float(weight_value),
    )
    points.geometry = points.geometry.buffer(buffer_size)
    points = points[~points.geometry.is_empty]
    if points.empty:
        return empty_result()

    dissolved = points.dissolve(by="dist", as_index=False)
    lines = dissolved.geometry.apply(polygons_to_multilinestring).union_all()
    polygons = gpd.GeoDataFrame(geometry=list(polygonize(lines)), crs=points.crs)
    if polygons.empty:
        return empty_result()

    polygon_points = polygons.copy()
    polygon_points.geometry = polygons.representative_point()
    stepped = polygon_points.sjoin(dissolved[["dist", "geometry"]], predicate="within")
    if stepped.empty:
        return empty_result()

    stepped_dist = stepped.groupby(level=0)["dist"].mean()
    stepped_dist = np.minimum(np.floor(stepped_dist / float(step)) * float(step), float(weight_value))
    stepped_polygons = gpd.GeoDataFrame(
        {"dist": stepped_dist.to_numpy()},
        geometry=polygons.geometry.loc[stepped_dist.index].to_numpy(),
        crs=points.crs,
    )
    return stepped_polygons.dissolve(by="dist", as_index=False).explode(ignore_index=True)


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
    nodes_for_voronoi["dist"] = np.minimum(
        np.ceil(nodes_for_voronoi["dist"].astype(float).to_numpy() / float(step)) * float(step),
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
