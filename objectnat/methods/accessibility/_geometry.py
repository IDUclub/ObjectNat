from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection

from objectnat._rust import keep_point_buffers  # pylint: disable=no-name-in-module

_TILE_SIZE = 2048
_COVERAGE_TILE_SIZE = 8192
_MERGE_FAN_IN = 4


class BufferGroup(NamedTuple):
    """GEOS buffer inputs for geometries that form one union."""

    geometries: np.ndarray
    distances: np.ndarray | float
    quad_segs: int


def point_buffer_group(points: np.ndarray, distances: np.ndarray, quad_segs: int) -> BufferGroup:
    coordinates = np.column_stack((shapely.get_x(points), shapely.get_y(points)))
    keep = keep_point_buffers(coordinates, distances, quad_segs)
    return BufferGroup(points[keep], distances[keep], quad_segs)


def _spatial_batches(geometries: np.ndarray, batch_size: int) -> list[np.ndarray]:
    indices = np.arange(len(geometries))
    if len(geometries) <= batch_size:
        return [indices]
    bounds = shapely.bounds(geometries)
    centres = (bounds[:, :2] + bounds[:, 2:]) * 0.5
    finite = np.isfinite(centres).all(axis=1)
    if not finite.any():
        return [indices]
    lower, upper = centres[finite].min(axis=0), centres[finite].max(axis=0)
    width = np.max(upper - lower)
    if not 0 < width < np.inf:
        return [indices]

    # Sort by spatial tile, then split dense tiles into batches without cutting any geometry.
    divisions = int(np.ceil(np.sqrt(len(geometries) / batch_size)))
    centres[~finite] = lower
    grid = np.minimum(((centres - lower) / width * divisions).astype(np.intp), divisions - 1)
    order = np.lexsort((centres[:, 0], grid[:, 0], grid[:, 1]))
    tile_ids = grid[order, 0] + divisions * grid[order, 1]
    cuts = np.flatnonzero(np.diff(tile_ids)) + 1
    return [
        tile[start : start + batch_size] for tile in np.split(order, cuts) for start in range(0, len(tile), batch_size)
    ]


def _tiles(group: BufferGroup) -> list[BufferGroup]:
    geometries, distances, quad_segs = group
    if len(geometries) <= _TILE_SIZE:
        return [group]
    return [
        BufferGroup(geometries[batch], distances[batch] if isinstance(distances, np.ndarray) else distances, quad_segs)
        for batch in _spatial_batches(geometries, _TILE_SIZE)
    ]


def _buffer_union(group: BufferGroup):
    return shapely.union_all(shapely.buffer(group.geometries, group.distances, quad_segs=group.quad_segs))


def _merge_tiles(groups, pool: ThreadPoolExecutor, merge=shapely.union_all):
    """Reduce adjacent tiles in parallel, keeping the order of each group at every level."""
    groups = list(groups)
    while any(len(group) > 1 for group in groups):
        batches = [
            [group[start : start + _MERGE_FAN_IN] for start in range(0, len(group), _MERGE_FAN_IN)] for group in groups
        ]
        merged = pool.map(merge, (batch for group in batches for batch in group if len(batch) > 1))
        groups = [[batch[0] if len(batch) == 1 else next(merged) for batch in group] for group in batches]
    return [group[0] if len(group) else GeometryCollection() for group in groups]


def _polygon_intersection(tile, mask):
    parts = shapely.get_parts(shapely.intersection(tile, mask))
    parts = shapely.get_parts(parts)
    return shapely.multipolygons(parts[shapely.get_type_id(parts) == shapely.GeometryType.POLYGON])


def _clip_tiles(tiles, mask, pool: ThreadPoolExecutor):
    # Only road tiles crossing the radius boundary need a polygon intersection.
    tiles = np.asarray(tiles, dtype=object)
    shapely.prepare(mask)
    inside = shapely.covers(mask, tiles)
    crossing = ~inside & shapely.intersects(mask, tiles)
    tiles[crossing] = list(pool.map(_polygon_intersection, tiles[crossing], repeat(mask)))
    return tiles[inside | crossing]


def _reachable_road_components(tiles, points):
    """Group the clipped road shells whose component contains a reachable point."""
    parts = shapely.get_parts(tiles)
    parts = parts[~shapely.is_empty(parts)]
    if parts.size == 0:
        return []

    # A GEOS polygon union joins parts through their interiors or along edges.
    # Parts touching at a point stay separate when testing reachability.
    left, right = shapely.STRtree(parts).query(parts, predicate="intersects")
    forward = left < right
    left, right = left[forward], right[forward]
    touching = shapely.touches(parts[left], parts[right])
    joined = ~touching
    joined[touching] = shapely.relate_pattern(parts[left[touching]], parts[right[touching]], "****1****")
    links = coo_matrix((np.ones(joined.sum(), dtype=bool), (left[joined], right[joined])), shape=(len(parts),) * 2)
    _, labels = connected_components(links, directed=False)
    reachable, _ = shapely.STRtree(points).query(parts, predicate="intersects")
    keep = np.isin(labels, labels[reachable])
    if not keep.any():
        return []

    # Each tile hole is inside the exterior of its eventual component. Fill
    # it now that reachability is known, so GEOS merges fewer road boundaries.
    shells = shapely.polygons(shapely.get_exterior_ring(parts[keep]))
    order = np.argsort(labels[keep], kind="stable")
    cuts = np.flatnonzero(np.diff(labels[keep][order])) + 1
    return np.split(shells[order], cuts)


def _road_exteriors(components):
    if not components:
        return GeometryCollection()
    polygons = shapely.polygons(shapely.get_exterior_ring(shapely.get_parts(components)))
    return polygons[0] if len(polygons) == 1 else shapely.multipolygons(polygons)


def buffer_unions(
    groups: list[BufferGroup],
    *,
    road_masks: dict[int, tuple[int, np.ndarray]] | None = None,
    max_workers: int | None = None,
) -> list:
    """Buffer and dissolve spatial tiles in one pool, preserving the order of the input groups."""
    road_masks = road_masks or {}
    tiles_by_group = [_tiles(group) for group in groups]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = iter(pool.map(_buffer_union, (tile for tiles in tiles_by_group for tile in tiles)))
        tile_unions = [list(next(parts) for _ in tiles) for tiles in tiles_by_group]
        unmasked = [index for index in range(len(groups)) if index not in road_masks]
        merged = [None] * len(groups)
        for index, geometry in zip(unmasked, _merge_tiles((tile_unions[i] for i in unmasked), pool)):
            merged[index] = geometry
        if road_masks:
            roads = [_clip_tiles(tile_unions[index], merged[mask], pool) for index, (mask, _) in road_masks.items()]
            components = list(
                pool.map(_reachable_road_components, roads, (points for _, points in road_masks.values()))
            )
            unions = iter(_merge_tiles((group for road in components for group in road), pool))
            for index, road in zip(road_masks, components):
                merged[index] = _road_exteriors([next(unions) for _ in road])
    return merged


def coverage_union(geometries):
    """Dissolve cells from one Voronoi diagram, with a fallback for an invalid coverage."""
    geometries = np.asarray(geometries)
    if len(geometries) == 1:
        return geometries[0]
    try:
        merged = shapely.coverage_union_all(geometries)
        cell_area = shapely.area(geometries).sum()
        if np.isclose(merged.area, cell_area, rtol=1e-12, atol=0.0) and merged.is_valid:
            return merged
    except GEOSException:
        pass
    return shapely.union_all(geometries)


def dissolve_cells(cells: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    groups = cells.groupby(column, sort=True, observed=False)
    values = groups.size().index
    geometries = cells.geometry.to_numpy()
    tiles = []
    for value in values:
        parts = geometries[groups.indices[value]]
        tiles.append([parts[batch] for batch in _spatial_batches(parts, _COVERAGE_TILE_SIZE)])
    with ThreadPoolExecutor() as pool:
        unions = pool.map(coverage_union, (tile for group in tiles for tile in group))
        tile_unions = [[next(unions) for _ in group] for group in tiles]
        dissolved = _merge_tiles(tile_unions, pool, merge=coverage_union)
    return gpd.GeoDataFrame({column: values}, geometry=dissolved, crs=cells.crs)
