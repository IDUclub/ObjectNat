"""2.5D visibility (viewsheds): ground and roofs visible from an eye height among buildings and terrain."""

import geopandas as gpd
import numpy as np
import shapely

from objectnat import config
from objectnat._rust import line_of_sight  # pylint: disable=no-name-in-module
from objectnat.methods.utils.scene import TerrainInput, kernel_scene, prepare_scene
from objectnat.methods.visibility.catchment import _zones_from_labels

logger = config.logger

MAX_TARGETS_PER_BATCH = 4_000_000


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
    terrain. Visible cells are merged into one polygon per observer. Observers standing inside a building
    footprint are placed on its roof.

    Args:
        point_from (gpd.GeoDataFrame):
            Observer points in any CRS. Optional columns ``visibility_distance`` and ``observer_height``
            override ``view_distance`` and ``observer_height`` per point; other attributes are kept.
        buildings (gpd.GeoDataFrame | None):
            Building footprints with heights (see ``height_column``, ``storeys_column``); ``None`` for terrain
            only.
        view_distance (float | None, optional):
            Viewing radius in metres, unless ``point_from`` has ``visibility_distance``.
        height_column, storeys_column, storey_height, default_height:
            Building heights, as in :func:`get_insolation`.
        terrain, terrain_height_column, terrain_cell:
            Ground heights, as in :func:`get_insolation`; without terrain the ground is flat.
        observer_height (float, optional):
            Eye height above the ground (or the roof) in metres. Defaults to 1.6.
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
            ``visible_area`` (m²) and ``visible_share`` (visible part of the target cells within the radius).

    Raises:
        ValueError: If ``point_from`` is empty or has no CRS, or no viewing distance is given.
    """
    if not isinstance(point_from, gpd.GeoDataFrame) or point_from.crs is None or point_from.empty:
        raise ValueError("point_from must be a non-empty GeoDataFrame with a CRS")
    if buildings is None:
        buildings = gpd.GeoDataFrame({"height": []}, geometry=[], crs=point_from.crs)
        height_column, storeys_column = "height", None

    scene = prepare_scene(
        buildings, point_from, terrain, terrain_height_column, terrain_cell,
        height_column, storeys_column, storey_height, default_height,
    )  # fmt: skip
    observers = point_from.to_crs(scene.local_crs)
    if "visibility_distance" in observers.columns:
        radius = observers["visibility_distance"].to_numpy(dtype=float)
    elif view_distance is not None:
        radius = np.full(len(observers), float(view_distance))
    else:
        raise ValueError("Provide view_distance or a 'visibility_distance' column in point_from")
    eye_offset = (
        observers["observer_height"].to_numpy(dtype=float)
        if "observer_height" in observers.columns
        else np.full(len(observers), float(observer_height))
    )

    blds = scene.blds
    tree = shapely.STRtree(blds.geometry.values) if len(blds) else None
    roof = scene.base + scene.building_height
    oxy = shapely.get_coordinates(observers.geometry.values)
    eye_z = scene.ground_at(oxy) + eye_offset
    if tree is not None:
        on_obs, on_bld = tree.query(shapely.points(oxy), predicate="within")
        eye_z[on_obs] = roof[on_bld] + eye_offset[on_obs]

    shift = oxy.min(axis=0)
    segments, tops, kwargs = kernel_scene(scene, shift)
    eyes = np.column_stack([oxy - shift, eye_z])

    geoms = np.empty(len(observers), dtype=object)
    share = np.zeros(len(observers))
    batch, batch_size = [], 0
    for k in range(len(observers)):
        grid = _disc_cells(oxy[k], radius[k], cell_size)
        targets = _target_heights(scene, tree, roof, grid, target_height, include_roofs)
        batch.append((k, grid, targets))
        batch_size += len(targets[0])
        if batch_size >= MAX_TARGETS_PER_BATCH or k == len(observers) - 1:
            _run_batch(batch, eyes, shift, segments, tops, kwargs, max_workers, cell_size, geoms, share)
            batch, batch_size = [], 0

    result = observers.copy()
    result["geometry"] = geoms
    result = result.set_geometry("geometry")
    result["visible_area"] = shapely.area(geoms)
    result["visible_share"] = share
    return result.to_crs(point_from.crs)


def _disc_cells(xy: np.ndarray, radius: float, cell: float) -> tuple:
    """Cells of a global ``cell`` grid whose centres lie within ``radius`` of ``xy``:
    ``(i0, j0, nx, ny, flat indices in the local box, centres)``."""
    i0, j0 = int(np.floor((xy[0] - radius) / cell)), int(np.floor((xy[1] - radius) / cell))
    i1, j1 = int(np.floor((xy[0] + radius) / cell)), int(np.floor((xy[1] + radius) / cell))
    nx, ny = i1 - i0 + 1, j1 - j0 + 1
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    centres = np.column_stack([(i0 + ii.ravel() + 0.5) * cell, (j0 + jj.ravel() + 0.5) * cell])
    inside = np.hypot(centres[:, 0] - xy[0], centres[:, 1] - xy[1]) <= radius
    return i0, j0, nx, ny, np.flatnonzero(inside), centres[inside]


def _target_heights(scene, tree, roof, grid, target_height, include_roofs) -> tuple:
    """Flat cell indices and ``(m, 3)`` targets: ground cells at ``target_height``, roof cells on the roof."""
    _, _, _, _, flat, centres = grid
    z = scene.ground_at(centres) + target_height
    keep = np.ones(len(centres), dtype=bool)
    if tree is not None and len(centres):
        cell_idx, bld_idx = tree.query(shapely.points(centres), predicate="within")
        if include_roofs:
            z[cell_idx] = roof[bld_idx]
        else:
            keep[cell_idx] = False
    return flat[keep], np.column_stack([centres[keep], z[keep]])


def _run_batch(batch, eyes, shift, segments, tops, kwargs, max_workers, cell, geoms, share) -> None:
    flats = [targets[0] for _, _, targets in batch]
    points = np.concatenate([targets[1] for _, _, targets in batch]) if batch else np.empty((0, 3))
    owner = np.concatenate([np.full(len(f), k, dtype=np.int64) for (k, _, _), f in zip(batch, flats)])
    local = points.copy()
    local[:, :2] -= shift
    seen = line_of_sight(
        np.ascontiguousarray(eyes, dtype=float),
        np.ascontiguousarray(local, dtype=float),
        owner,
        segments,
        tops,
        n_threads=max_workers,
        **kwargs,
    ).astype(bool)
    start = 0
    for (k, grid, _), flat in zip(batch, flats):
        visible = seen[start : start + len(flat)]
        start += len(flat)
        i0, j0, nx, ny, _, _ = grid
        share[k] = visible.mean() if len(visible) else 0.0
        labels = np.zeros(nx * ny, dtype=np.uint8)
        labels[flat[visible]] = 1
        zones = _zones_from_labels(labels.reshape(ny, nx), (i0 * cell, j0 * cell), cell) if visible.any() else {}
        geoms[k] = zones.get(1, shapely.Polygon())
