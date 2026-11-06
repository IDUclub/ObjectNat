"""Visibility catchment areas: how many observers see each place, classified into zones."""

from collections.abc import Sequence
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from iduedu import UrbanGraph

from objectnat import config
from objectnat._rust import visibility_count_grid  # pylint: disable=no-name-in-module
from objectnat.methods.accessibility._utils import geometry_edges
from objectnat.methods.utils.raster import grid_attrs, label_zones, resolve_output
from objectnat.methods.visibility._native import DEFAULT_QUAD_SEGS, geometries_to_segments

logger = config.logger

MAX_GRID_CELLS = 200_000_000
MAX_CLASSES = 255
OBSTACLE_GEOM_TYPES = ("Polygon", "MultiPolygon", "LineString", "MultiLineString")


def get_visibility_catchment(
    observers: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame | None,
    *,
    view_distance: float | None = None,
    cell_size: float = 5.0,
    weight_column: str | None = None,
    classes: int | Sequence[float] = 5,
    output: Literal["zones", "cells", "both"] = "zones",
    max_workers: int | None = None,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build visibility catchment zones: places classified by how many observers can see them.

    The area around the observers is split into square cells. A cell is visible from an observer when its
    centre lies inside the observer's exact visibility polygon (see :func:`get_visibility`); each cell gets the
    number of such observers, or the sum of their weights. Cells seen by nobody (including cells inside
    obstacles) are left out, the rest are grouped into classes, and every class is returned as one polygonal
    zone.

    With observers placed densely along streets (see :func:`get_observer_points`) the zones show where
    objects can be seen from most often: the visual catchment of facades, squares and landmarks.

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
            Cell side in metres. Smaller cells give finer zone outlines at a quadratic memory and time cost.
        weight_column (str | None, optional):
            Column of non-negative observer weights (e.g. pedestrian flow). Cells then sum the weights of the
            observers that see them instead of counting them. Observers with zero or missing weight are
            ignored.
        classes (int | Sequence[float], optional):
            Either the number of classes, built from quantiles of the non-zero cell values (classes with equal
            bounds are merged, so fewer may be returned), or explicit ascending class lower bounds in observer
            counts (or weight sums), e.g. ``[1, 5, 20]``: cells below the first bound are left out. Explicit
            bounds make zones comparable between runs.
        output (str, optional):
            ``"zones"`` (default) for the class zones, ``"cells"`` for the cells seen by at least one observer as
            points at the cell centres (no zone polygons are built), ``"both"`` for the tuple ``(zones, cells)``.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            One (Multi)Polygon per class, in the CRS of ``observers``, sorted by ``visibility_class`` (``1`` is
            the least visible class). Columns ``count_min`` / ``count_max`` (or ``weight_min`` / ``weight_max``
            with ``weight_column``) give the range of cell values in the class. Zone outlines follow the cell
            grid. With ``output="cells"`` (or ``"both"``) the cells, with ``count`` (and
            ``weight`` with ``weight_column``) and ``visibility_class`` (``0`` for cells below the lowest class
            bound) of every seen cell; :func:`rasterize` turns a column into a :class:`Raster`.

    Raises:
        TypeError: If ``observers`` is not a GeoDataFrame.
        ValueError: If ``observers`` is empty or has no CRS, if no view distance is given, if ``cell_size`` or
            ``classes`` is invalid, if ``weight_column`` is missing or contains negative values, or if the grid
            would exceed 200 million cells.

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
    lower_bounds = _explicit_bounds(classes)
    output = resolve_output(output, "cells")

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
        return _finish(_empty_result(value_name, original_crs), None, output)

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
    grid = (float(x0), float(y0), float(cell_size))
    bounds = classes if lower_bounds is None else lower_bounds
    return count_grid_result(counts, weight_sums, bounds, grid, local_crs, original_crs, output)


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


def _explicit_bounds(classes) -> np.ndarray | None:
    """Validated explicit class bounds, or ``None`` for a class count."""
    if isinstance(classes, (int, np.integer)) and not isinstance(classes, bool):
        if not 1 <= classes <= MAX_CLASSES:
            raise ValueError(f"classes must be between 1 and {MAX_CLASSES}; got {classes}")
        return None
    bounds = np.asarray(classes, dtype="float64").reshape(-1)
    if bounds.size == 0 or bounds.size > MAX_CLASSES or not np.isfinite(bounds).all():
        raise ValueError(f"classes must be an int or 1..{MAX_CLASSES} finite class lower bounds; got {classes!r}")
    if (np.diff(bounds) <= 0).any():
        raise ValueError(f"class lower bounds must be strictly ascending; got {classes!r}")
    return bounds


def _classify(values: np.ndarray, seen: np.ndarray, classes) -> np.ndarray:
    """Class label of every cell (``0`` = left out) from a class count or explicit lower bounds."""
    if isinstance(classes, np.ndarray):
        bounds = classes
    else:
        visible = values[seen]
        if visible.size == 0:
            return np.zeros(values.shape, dtype="uint8")
        bounds = np.unique(np.quantile(visible, np.linspace(0, 1, classes + 1)[:-1], method="lower"))
    labels = np.searchsorted(bounds, values, side="right").astype("uint8")
    labels[~seen] = 0
    return labels


def _obstacle_segments(obstacles, local_crs, bbox) -> tuple[np.ndarray, np.ndarray]:
    if obstacles is None or len(obstacles) == 0:
        return geometries_to_segments([])
    obstacles_local = obstacles.to_crs(local_crs)
    obstacles_local = obstacles_local[obstacles_local.geom_type.isin(OBSTACLE_GEOM_TYPES)]
    obstacles_local = obstacles_local.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    return geometries_to_segments(obstacles_local.geometry.values)


def count_grid_result(counts, weight_sums, classes, grid, local_crs, crs, output: str):
    """Class zones and/or seen cells of a grid of observer counts (and weight sums), as ``output`` asks."""
    value_name = "count" if weight_sums is None else "weight"
    values = counts.astype("float64") if weight_sums is None else weight_sums
    labels = _classify(values, counts > 0, classes)
    cells = cell_points(counts, weight_sums, labels, grid, local_crs, crs) if output != "zones" else None
    if output == "cells":
        return cells
    if not labels.any():
        return _finish(_empty_result(value_name, crs), cells, output)
    x0, y0, cell = grid
    rows = []
    for label, geometry in label_zones(labels, (x0, y0), cell).items():
        in_class = values[labels == label]
        rows.append(
            {
                "visibility_class": label,
                f"{value_name}_min": float(in_class.min()),
                f"{value_name}_max": float(in_class.max()),
                "geometry": geometry,
            }
        )
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=local_crs)
    if value_name == "count":
        result = result.astype({"count_min": "int64", "count_max": "int64"})
    return _finish(result.to_crs(crs), cells, output)


def cell_points(counts, weight_sums, labels, grid, local_crs, crs) -> gpd.GeoDataFrame:
    """Centres of the cells seen by at least one observer, with their count, weight sum and class."""
    x0, y0, cell = grid
    j, i = np.nonzero(counts > 0)
    data = {"count": counts[j, i].astype("int64")}
    if weight_sums is not None:
        data["weight"] = weight_sums[j, i].astype("float64")
    data["visibility_class"] = labels[j, i].astype("int64")
    points = shapely.points(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)
    cells = gpd.GeoDataFrame(data, geometry=points, crs=local_crs).to_crs(crs)
    cells.attrs.update(grid_attrs(local_crs, cell, (x0 + 0.5 * cell, y0 + 0.5 * cell)))
    return cells


def _finish(zones: gpd.GeoDataFrame, cells: gpd.GeoDataFrame | None, output: str):
    if output == "zones":
        return zones
    if cells is None:
        cells = gpd.GeoDataFrame({"count": [], "visibility_class": []}, geometry=[], crs=zones.crs)
    return cells if output == "cells" else (zones, cells)


def _empty_result(value_name: str, crs) -> gpd.GeoDataFrame:
    columns = {"visibility_class": [], f"{value_name}_min": [], f"{value_name}_max": []}
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
