"""2.5D visibility: viewsheds and viewshed catchment zones among buildings of known height and terrain."""

import geopandas as gpd
import numpy as np
import shapely

from objectnat import config
from objectnat._rust import (  # pylint: disable=no-name-in-module
    rasterize_footprints,
    viewshed_cells,
    viewshed_count_grid,
)
from objectnat.methods.utils.raster import label_zones
from objectnat.methods.utils.scene import Scene, TerrainInput, kernel_scene, prepare_scene
from objectnat.methods.visibility.catchment import cell_points, empty_cells

logger = config.logger

MAX_GRID_CELLS = 30_000_000  # per target grid: ~250 MB of heights and footprint ids


def get_viewshed(
    point_from: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    view_distance: float | None = None,
    *,
    height_column: str | None = "height",
    storeys_column: str | None = None,
    storey_height: float = 3.0,
    default_height: float | None = None,
    terrain: TerrainInput | None = None,
    terrain_height_column: str = "height",
    terrain_cell: float = 5.0,
    observer_height: float = 1.6,
    target_height: float = 0.0,
    cell_size: float = 2.0,
    include_roofs: bool = False,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute 2.5D visibility: the ground (and optionally the roofs) each observer can see from its eye height,
    among buildings of known height and terrain.

    Unlike :func:`get_visibility`, which treats every obstacle as infinitely high, here a low building or a
    hill hides only what lies below the line of sight: an observer sees over a garage onto the street behind
    it, from a hill over the roofs, from a tower far across the district. The area within ``view_distance``
    is split into ``cell_size`` cells, and a cell is visible when the straight line from the eye to the cell
    centre (at ``target_height`` above the ground) passes above every building wall it crosses and above the
    terrain. Visible cells are merged into one polygon per observer. The eye is ``observer_height`` above the
    terrain; observers standing inside a building footprint are placed on its roof.

    Args:
        point_from (gpd.GeoDataFrame):
            Observer points in any CRS. Optional columns ``visibility_distance`` and ``observer_height``
            override ``view_distance`` and ``observer_height`` per point; other attributes are kept.
        buildings (gpd.GeoDataFrame | None):
            Building footprints with heights (see ``height_column``, ``storeys_column``); ``None`` for terrain
            only.
        view_distance (float | None, optional):
            View distance in metres, unless ``point_from`` has ``visibility_distance``.
        height_column, storeys_column, storey_height, default_height:
            Building heights, as in :func:`get_insolation`.
        terrain, terrain_height_column, terrain_cell:
            Ground heights, as in :func:`get_insolation`; without terrain the ground is flat.
        observer_height (float, optional):
            Eye height above the terrain (or the roof) in metres. Defaults to 1.6.
        target_height (float, optional):
            Height above the ground of what is looked at, in metres: 0 for the ground itself, 1.6 to see
            people's faces, more for the upper part of objects. Defaults to 0.
        cell_size (float, optional):
            Cell size in metres; time grows with ``(view_distance / cell_size) ** 2``. Defaults to 2.
        include_roofs (bool, optional):
            Also return the parts of roofs that can be seen (flat roofs at the building height). Defaults to
            ``False``: building footprints are left out.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            Same index and attributes as ``point_from`` with the geometry replaced by the visible area (cells
            merged into polygons, an empty polygon when nothing is visible) in the CRS of ``point_from``, plus
            ``visible_area`` (m²) and ``visible_share`` (the share of the target cells within the view distance
            that are visible).

    Raises:
        ValueError: If ``point_from`` is empty or has no CRS, or no view distance is given.
    """
    scene, observers, radius, eye_offset = _setup(
        point_from, buildings, view_distance, observer_height,
        height_column, storeys_column, storey_height, default_height,
        terrain, terrain_height_column, terrain_cell,
    )  # fmt: skip
    oxy = shapely.get_coordinates(observers.geometry.values)
    eye_z = scene.ground_at(oxy) + eye_offset
    if len(scene.blds):
        on_obs, on_bld = shapely.STRtree(scene.blds.geometry.values).query(shapely.points(oxy), predicate="within")
        eye_z[on_obs] = scene.base[on_bld] + scene.building_height[on_bld] + eye_offset[on_obs]

    geoms = np.empty(len(observers), dtype=object)
    geoms[:] = [shapely.Polygon()] * len(observers)
    share = np.zeros(len(observers))
    for group in _spatial_groups(oxy, radius, cell_size):
        grid, target_z, shift, kernel = _target_grid(
            scene, oxy[group], radius[group], cell_size, target_height, include_roofs
        )
        segments, tops, kwargs = kernel
        x0, y0, cell, nx, _ = grid
        flat, offsets = viewshed_cells(
            np.ascontiguousarray(np.column_stack([oxy[group] - shift, eye_z[group]]), dtype=float),
            np.ascontiguousarray(radius[group], dtype=float),
            (x0 - shift[0], y0 - shift[1], cell, grid[3], grid[4]),
            target_z,
            segments,
            tops,
            n_threads=max_workers,
            **kwargs,
        )
        targets = np.isfinite(target_z)
        for k, observer in enumerate(group):
            cells = flat[offsets[k] : offsets[k + 1]]
            share[observer] = len(cells) / max(_targets_in_disc(targets, grid, oxy[observer], radius[observer]), 1)
            if len(cells):
                geoms[observer] = _cells_polygon(cells, nx, x0, y0, cell)

    result = observers.copy()
    result["geometry"] = geoms
    result = result.set_geometry("geometry")
    result["visible_area"] = shapely.area(geoms)
    result["visible_share"] = share
    return result.to_crs(point_from.crs)


def get_viewshed_catchment(
    observers: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    *,
    view_distance: float | None = None,
    height_column: str | None = "height",
    storeys_column: str | None = None,
    storey_height: float = 3.0,
    default_height: float | None = None,
    terrain: TerrainInput | None = None,
    terrain_height_column: str = "height",
    terrain_cell: float = 5.0,
    observer_height: float = 1.6,
    target_height: float = 0.0,
    cell_size: float = 5.0,
    include_roofs: bool = False,
    weight_column: str | None = None,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Count, for every place around the observers, how many of them see it, over buildings of known height and
    terrain (the 2.5D visibility catchment).

    The 2.5D counterpart of :func:`get_visibility_catchment`: every cell around the observers gets the number
    of observers (or the sum of their weights) whose line of sight to the cell centre at ``target_height`` is
    clear, as in :func:`get_viewshed`. Observers stand on the ground with their eyes ``observer_height`` above
    the terrain; observers inside a building footprint are left out. With ``include_roofs`` roof cells are
    counted too, which shows the roofs and building tops seen from the streets.

    Args:
        observers (gpd.GeoDataFrame):
            Observer points with a CRS, e.g. from :func:`get_observer_points`. Optional columns
            ``visibility_distance`` and ``observer_height`` override ``view_distance`` and
            ``observer_height`` per point.
        buildings (gpd.GeoDataFrame | None):
            Building footprints with heights; ``None`` for terrain only.
        view_distance (float | None, optional):
            View distance in metres, unless ``observers`` has ``visibility_distance``.
        height_column, storeys_column, storey_height, default_height:
            Building heights, as in :func:`get_insolation`.
        terrain, terrain_height_column, terrain_cell:
            Ground heights, as in :func:`get_insolation`.
        observer_height (float, optional):
            Eye height above the terrain in metres. Defaults to 1.6.
        target_height (float, optional):
            Height above the ground of what is looked at, in metres. Defaults to 0.
        cell_size (float, optional):
            Cell side in metres. Defaults to 5.
        include_roofs (bool, optional):
            Count roof cells too. Defaults to ``False``.
        weight_column (str | None, optional):
            Observer weights, as in :func:`get_visibility_catchment`.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            The cells seen by at least one observer, as points at the cell centres in the CRS of ``observers``,
            with ``count`` (and ``weight``), as in :func:`get_visibility_catchment`; :func:`rasterize` turns
            them into a :class:`Raster`.

    Raises:
        ValueError: On empty observers, a missing view distance, bad weights or a grid larger than 30 million
            cells.
    """
    scene, points, radius, eye_offset = _setup(
        observers, buildings, view_distance, observer_height,
        height_column, storeys_column, storey_height, default_height,
        terrain, terrain_height_column, terrain_cell,
    )  # fmt: skip
    weights = None
    if weight_column is not None:
        if weight_column not in points.columns:
            raise ValueError(f"weight_column {weight_column!r} is not a column of observers")
        weights = points[weight_column].to_numpy(dtype=float)
        if (weights < 0).any():
            raise ValueError(f"weight_column {weight_column!r} must not hold negative values")
        weights = np.nan_to_num(weights, nan=0.0)
    value_name = "count" if weights is None else "weight"

    oxy = shapely.get_coordinates(points.geometry.values)
    usable = np.isfinite(radius) & (radius > 0)
    if len(scene.blds):
        inside, _ = shapely.STRtree(scene.blds.geometry.values).query(shapely.points(oxy), predicate="within")
        if len(inside):
            logger.info(f"{len(np.unique(inside))} observers inside buildings are left out")
        usable[inside] = False
    if weights is not None:
        usable &= weights > 0
    if not usable.any():
        return empty_cells(value_name, observers.crs)
    oxy, radius, eye_offset = oxy[usable], radius[usable], eye_offset[usable]
    weights = weights[usable] if weights is not None else None

    grid, target_z, shift, (segments, tops, kwargs) = _target_grid(
        scene, oxy, radius, cell_size, target_height, include_roofs, limit=True
    )
    x0, y0, cell, nx, ny = grid
    logger.info(f"Counting 2.5D visibility of {nx * ny:,} cells from {len(oxy):,} observers")
    counts, sums = viewshed_count_grid(
        np.ascontiguousarray(np.column_stack([oxy - shift, scene.ground_at(oxy) + eye_offset]), dtype=float),
        np.ascontiguousarray(radius, dtype=float),
        (x0 - shift[0], y0 - shift[1], cell, nx, ny),
        target_z,
        segments,
        tops,
        weights=None if weights is None else np.ascontiguousarray(weights, dtype=float),
        n_threads=max_workers,
        **kwargs,
    )
    return cell_points(counts, sums, (float(x0), float(y0), float(cell)), scene.local_crs, observers.crs)


def _setup(
    points, buildings, view_distance, observer_height,
    height_column, storeys_column, storey_height, default_height,
    terrain, terrain_height_column, terrain_cell,
):  # fmt: skip
    """Scene, observers in its CRS, their radii and eye heights above the ground."""
    if not isinstance(points, gpd.GeoDataFrame) or points.crs is None or points.empty:
        raise ValueError("observers must be a non-empty GeoDataFrame with a CRS")
    if buildings is None:
        buildings = gpd.GeoDataFrame({"height": []}, geometry=[], crs=points.crs)
        height_column, storeys_column = "height", None
    scene = prepare_scene(
        buildings, points, terrain, terrain_height_column, terrain_cell,
        height_column, storeys_column, storey_height, default_height,
    )  # fmt: skip
    local = points.to_crs(scene.local_crs)
    local = local[local.geom_type == "Point"]
    if local.empty:
        raise ValueError("observers must contain point geometries")
    if "visibility_distance" in local.columns:
        radius = local["visibility_distance"].to_numpy(dtype=float)
    elif view_distance is not None:
        radius = np.full(len(local), float(view_distance))
    else:
        raise ValueError("Provide view_distance or a 'visibility_distance' column in the observers")
    eye = (
        local["observer_height"].to_numpy(dtype=float)
        if "observer_height" in local.columns
        else np.full(len(local), float(observer_height))
    )
    return scene, local, radius, eye


def _spatial_groups(xy: np.ndarray, radius: np.ndarray, cell: float) -> list[np.ndarray]:
    """Observer groups whose target grids stay below ``MAX_GRID_CELLS`` cells."""
    reach = float(np.nanmax(radius)) if len(radius) else 0.0
    span = np.ptp(xy, axis=0) + 2 * reach if len(xy) else np.zeros(2)
    if (span[0] / cell + 2) * (span[1] / cell + 2) <= MAX_GRID_CELLS:
        return [np.arange(len(xy))]
    tile = np.sqrt(MAX_GRID_CELLS) * cell - 2 * reach - 2 * cell
    if tile <= 0:
        raise ValueError("view_distance is too large for cell_size; increase cell_size")
    keys = np.floor(xy / tile).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    return [np.flatnonzero(group.ravel() == g) for g in range(group.max() + 1)]


def _target_grid(scene: Scene, xy, radius, cell, target_height, include_roofs, limit=False):
    """Cell grid around the observers (aligned to multiples of ``cell``) with the absolute height of the
    target in every cell: the ground plus ``target_height``, the roof on footprints with ``include_roofs``,
    NaN on footprints otherwise. Also returns the coordinate shift and the kernel's scene arguments."""
    reach = float(np.nanmax(radius))
    x0 = np.floor((xy[:, 0].min() - reach) / cell) * cell
    y0 = np.floor((xy[:, 1].min() - reach) / cell) * cell
    nx = int(np.ceil((xy[:, 0].max() + reach - x0) / cell)) + 1
    ny = int(np.ceil((xy[:, 1].max() + reach - y0) / cell)) + 1
    if limit and nx * ny > MAX_GRID_CELLS:
        raise ValueError(
            f"The grid would have {nx * ny:,} cells (limit {MAX_GRID_CELLS:,}); increase cell_size or split the area."
        )
    shift = np.array([x0, y0])
    kernel = kernel_scene(scene, shift)
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    if scene.terrain_grid is None:
        target_z = np.full((ny, nx), float(target_height))
    else:
        centres = np.column_stack([np.tile(xs, ny), np.repeat(ys, nx)])
        target_z = (scene.ground_at(centres) + target_height).reshape(ny, nx)
    if len(scene.blds):
        roof = scene.base + scene.building_height
        footprint = rasterize_footprints(
            kernel[0], scene.owner, np.ascontiguousarray(roof, dtype=float), (0.0, 0.0, float(cell), nx, ny)
        )
        covered = footprint >= 0
        target_z[covered] = roof[footprint[covered]] if include_roofs else np.nan
    return (float(x0), float(y0), float(cell), nx, ny), np.ascontiguousarray(target_z), shift, kernel


def _targets_in_disc(targets: np.ndarray, grid, xy, radius) -> int:
    x0, y0, cell, nx, ny = grid
    i0, i1 = max(int((xy[0] - radius - x0) / cell), 0), min(int((xy[0] + radius - x0) / cell) + 1, nx)
    j0, j1 = max(int((xy[1] - radius - y0) / cell), 0), min(int((xy[1] + radius - y0) / cell) + 1, ny)
    cx = x0 + (np.arange(i0, i1) + 0.5) * cell - xy[0]
    cy = y0 + (np.arange(j0, j1) + 0.5) * cell - xy[1]
    disc = cx[None, :] ** 2 + cy[:, None] ** 2 <= radius**2
    return int((disc & targets[j0:j1, i0:i1]).sum())


def _cells_polygon(cells: np.ndarray, nx: int, x0: float, y0: float, cell: float):
    """Merge flat cell indices of a grid into one (Multi)Polygon."""
    j, i = np.divmod(cells.astype(np.int64), nx)
    i0, j0 = i.min(), j.min()
    labels = np.zeros((j.max() - j0 + 1, i.max() - i0 + 1), dtype=np.uint8)
    labels[j - j0, i - i0] = 1
    return label_zones(labels, (x0 + i0 * cell, y0 + j0 * cell), cell)[1]
