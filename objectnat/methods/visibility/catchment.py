"""Visibility catchment: how many observers see each place, counted on a grid of cells."""

import geopandas as gpd
import numpy as np
import shapely
from iduedu import UrbanGraph

from objectnat import config
from objectnat._rust import visibility_count_grid  # pylint: disable=no-name-in-module
from objectnat.methods.accessibility._utils import geometry_edges
from objectnat.methods.utils.raster import grid_attrs
from objectnat.methods.visibility._native import DEFAULT_QUAD_SEGS, geometries_to_segments

logger = config.logger

MAX_GRID_CELLS = 200_000_000
OBSTACLE_GEOM_TYPES = ("Polygon", "MultiPolygon", "LineString", "MultiLineString")


def get_visibility_catchment(
    observers: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame | None,
    *,
    view_distance: float | None = None,
    cell_size: float = 5.0,
    weight_column: str | None = None,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Count, for every place around the observers, how many of them see it (the visibility catchment).

    The area within ``view_distance`` of the observers is split into square cells. A cell counts an observer when
    its centre lies inside the observer's exact visibility polygon (see :func:`get_visibility`); with
    ``weight_column`` it sums the observers' weights instead. With observers placed densely along streets (see
    :func:`get_observer_points`) the counts show where objects are seen from most often: the visual catchment of
    facades, squares and landmarks. :func:`rasterize` turns the result into a :class:`Raster`.

    Args:
        observers (gpd.GeoDataFrame):
            Observer points with a CRS.
        obstacles (gpd.GeoDataFrame | None):
            Obstacle geometries (Polygon, MultiPolygon, LineString, MultiLineString) in any CRS. ``None`` or an
            empty frame means no occlusion.
        view_distance (float | None, optional):
            View distance of every observer, in metres. Used when ``observers`` has no
            ``"visibility_distance"`` column; one of the two is required.
        cell_size (float, optional):
            Cell side in metres. Halving it gives four times as many cells, and memory and time grow with them.
        weight_column (str | None, optional):
            Column of non-negative observer weights (e.g. pedestrian flow). Cells then sum the weights of the
            observers that see them instead of counting them. Observers with zero or missing weight are
            ignored.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            The cells seen by at least one observer, as points at the cell centres in the CRS of ``observers``,
            with ``count`` (the number of observers that see the cell) and, with ``weight_column``, ``weight``
            (the sum of their weights). Cells seen by nobody, including cells inside obstacles, are left out.
            ``attrs`` record the grid, so :func:`rasterize` rebuilds it exactly.

    Raises:
        TypeError: If ``observers`` is not a GeoDataFrame.
        ValueError: If ``observers`` is empty or has no CRS, if no view distance is given, if ``cell_size`` is
            invalid, if ``weight_column`` is missing or contains negative values, or if the grid would exceed
            200 million cells.

    Notes:
        Visibility is symmetric, so the value of a cell is also the number of observers visible *from* it.
    """
    if not isinstance(observers, gpd.GeoDataFrame):
        raise TypeError("observers must be a GeoDataFrame with point geometries")
    if observers.empty:
        raise ValueError("GeoDataFrame 'observers' is empty.")
    if observers.crs is None:
        raise ValueError("GeoDataFrame 'observers' must have a CRS.")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError(f"cell_size must be a positive number; got {cell_size!r}")
    original_crs = observers.crs
    local_crs = observers.estimate_utm_crs()
    points_local = observers.to_crs(local_crs)
    points_local = points_local[points_local.geom_type == "Point"]
    if points_local.empty:
        raise ValueError("observers must contain point geometries")

    if "visibility_distance" in points_local.columns:
        radii = points_local["visibility_distance"].to_numpy(dtype="float64")
    elif view_distance is None:
        raise ValueError(
            "Either provide parameter view_distance or add column 'visibility_distance' to observers GeoDataFrame."
        )
    else:
        radii = np.full(len(points_local), float(view_distance))

    weights = None
    if weight_column is not None:
        if weight_column not in points_local.columns:
            raise ValueError(f"weight_column {weight_column!r} is not a column of observers")
        weights = points_local[weight_column].to_numpy(dtype="float64")
        if (weights < 0).any():
            raise ValueError(f"weight_column {weight_column!r} must not hold negative values")
        weights = np.nan_to_num(weights, nan=0.0)

    coords = shapely.get_coordinates(points_local.geometry.values)
    usable = np.isfinite(radii) & (radii > 0)
    coords, radii = coords[usable], radii[usable]
    if weights is not None:
        weights = weights[usable]
    value_name = "count" if weights is None else "weight"
    if len(coords) == 0:
        return empty_cells(value_name, original_crs)

    reach = float(radii.max())
    x0 = np.floor((coords[:, 0].min() - reach) / cell_size) * cell_size
    y0 = np.floor((coords[:, 1].min() - reach) / cell_size) * cell_size
    nx = int(np.ceil((coords[:, 0].max() + reach - x0) / cell_size))
    ny = int(np.ceil((coords[:, 1].max() + reach - y0) / cell_size))
    if nx * ny > MAX_GRID_CELLS:
        raise ValueError(
            f"The grid would have {nx * ny:,} cells (limit {MAX_GRID_CELLS:,}); increase cell_size or split the area."
        )

    segments, one_sided = _obstacle_segments(obstacles, local_crs, (x0, y0, x0 + nx * cell_size, y0 + ny * cell_size))
    logger.info(f"Counting visibility of {nx * ny:,} cells from {len(coords):,} observers")
    counts, weight_sums = visibility_count_grid(
        coords,
        radii,
        segments,
        (float(x0), float(y0)),
        float(cell_size),
        (ny, nx),
        weights=weights,
        quad_segs=DEFAULT_QUAD_SEGS,
        n_threads=max_workers,
        one_sided=one_sided,
    )
    return cell_points(counts, weight_sums, (float(x0), float(y0), float(cell_size)), local_crs, original_crs)


def get_observer_points(urban_graph: UrbanGraph, *, spacing: float = 15.0) -> gpd.GeoDataFrame:
    """
    Place observer points evenly along the streets of a graph.

    Each street segment (an edge, counted once for both directions) is split into pieces of about ``spacing``
    metres and gets one point in the middle of every piece, so points never pile up at intersections. Walk and
    intermodal graphs use their pedestrian (``type == "walk"``) edges only.

    Args:
        urban_graph (UrbanGraph):
            Street graph from IduEdu.
        spacing (float, optional):
            Target distance between neighbouring points in metres.

    Returns:
        gpd.GeoDataFrame:
            Points in the CRS of ``urban_graph``, ready for :func:`get_visibility_catchment`.

    Raises:
        ValueError: If ``spacing`` is not positive or the graph has no edge geometries.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"spacing must be a positive number; got {spacing!r}")
    edges = geometry_edges(urban_graph)
    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty]
    if edges.empty:
        raise ValueError("urban_graph has no edge geometries")
    crs = edges.crs
    local = (
        edges.geometry if crs is not None and not crs.is_geographic else edges.geometry.to_crs(edges.estimate_utm_crs())
    )
    lines = shapely.normalize(shapely.line_merge(local.values))
    _, unique = np.unique(shapely.to_wkb(lines), return_index=True)
    lines = lines[np.sort(unique)]
    lengths = shapely.length(lines)
    pieces = np.maximum(1, np.rint(lengths / spacing)).astype(int)
    owner = np.repeat(np.arange(len(lines)), pieces)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    position = (np.arange(len(owner)) - first + 0.5) / pieces[owner]
    points = shapely.line_interpolate_point(lines[owner], position, normalized=True)
    result = gpd.GeoDataFrame(geometry=points, crs=local.crs)
    return result.to_crs(crs) if crs is not None and result.crs != crs else result


def _obstacle_segments(obstacles, local_crs, bbox) -> tuple[np.ndarray, np.ndarray]:
    if obstacles is None or len(obstacles) == 0:
        return geometries_to_segments([])
    obstacles_local = obstacles.to_crs(local_crs)
    obstacles_local = obstacles_local[obstacles_local.geom_type.isin(OBSTACLE_GEOM_TYPES)]
    obstacles_local = obstacles_local.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    return geometries_to_segments(obstacles_local.geometry.values)


def cell_points(counts, weight_sums, grid, local_crs, crs) -> gpd.GeoDataFrame:
    """Centres of the cells seen by at least one observer, with their count (and weight sum)."""
    x0, y0, cell = grid
    j, i = np.nonzero(counts > 0)
    data = {"count": counts[j, i].astype("int64")}
    if weight_sums is not None:
        data["weight"] = weight_sums[j, i].astype("float64")
    points = shapely.points(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)
    cells = gpd.GeoDataFrame(data, geometry=points, crs=local_crs).to_crs(crs)
    cells.attrs.update(grid_attrs(local_crs, cell, (x0 + 0.5 * cell, y0 + 0.5 * cell)))
    return cells


def empty_cells(value_name: str, crs) -> gpd.GeoDataFrame:
    """No cell is seen (no usable observer)."""
    columns = {"count": np.array([], dtype="int64")}
    if value_name == "weight":
        columns["weight"] = np.array([], dtype="float64")
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
