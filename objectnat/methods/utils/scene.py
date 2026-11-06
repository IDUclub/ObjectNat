"""2.5D scene shared by insolation and 3D visibility: buildings with heights on optional terrain, in a local
metric CRS, as walls with absolute roof heights for ObjectNat's Rust kernels."""

import os
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from objectnat import config

logger = config.logger

TerrainInput = gpd.GeoDataFrame | str | os.PathLike | tuple
MAX_TERRAIN_CELLS = 20_000_000


class Scene(NamedTuple):
    """Buildings with heights on optional terrain in ``local_crs``, with their walls and roof tops."""

    local_crs: object
    blds: gpd.GeoDataFrame
    building_height: np.ndarray
    base: np.ndarray
    terrain_grid: tuple | None
    segments: np.ndarray
    tops: np.ndarray
    owner: np.ndarray  # building of every wall

    def ground_at(self, xy: np.ndarray) -> np.ndarray:
        if self.terrain_grid is None:
            return np.zeros(len(xy))
        return np.nan_to_num(sample_terrain(self.terrain_grid, xy), nan=0.0)


def prepare_scene(
    buildings, area, terrain, terrain_height_column, terrain_cell,
    height_column, storeys_column, storey_height, default_height,
) -> Scene:  # fmt: skip
    """Buildings with heights standing on the terrain, in a local metric CRS, and their walls."""
    if not isinstance(buildings, gpd.GeoDataFrame) or buildings.crs is None:
        raise ValueError("buildings must be a GeoDataFrame with a CRS")
    terrain_grid = None
    if isinstance(terrain, gpd.GeoDataFrame):
        local_crs = _estimate_local_crs(buildings, area, terrain)
        terrain_grid = _rasterize_terrain(terrain, terrain_height_column, terrain_cell, local_crs)
    elif terrain is not None:
        heights, transform, terrain_crs = _read_terrain(terrain)
        local_crs = gpd.GeoSeries(crs=terrain_crs).crs
        if not local_crs.is_projected:
            raise ValueError("terrain must be in a projected CRS (metres)")
        terrain_grid = _terrain_grid(heights, transform)
    else:
        local_crs = _estimate_local_crs(buildings, area, None)

    blds = buildings.to_crs(local_crs)
    blds = blds[blds.geom_type.isin(["Polygon", "MultiPolygon"]) & ~blds.is_empty]
    building_height = _building_heights(blds, height_column, storeys_column, storey_height, default_height)
    ok = ~np.isnan(building_height)
    blds, building_height = blds[ok], building_height[ok]
    if terrain_grid is not None:
        points = shapely.get_coordinates(blds.representative_point().values)
        base = np.nan_to_num(sample_terrain(terrain_grid, points), nan=0.0)
    else:
        base = np.zeros(len(blds))
    # imported here: the visibility package itself builds on this module
    from objectnat.methods.visibility._native import (  # pylint: disable=import-outside-toplevel
        geometries_to_segments,
    )

    segments, _, owner = geometries_to_segments(blds.geometry.values, return_owner=True)
    tops = (base + building_height)[owner] if len(segments) else np.empty(0)
    owner = np.asarray(owner if len(segments) else np.empty(0), dtype=np.int64)
    return Scene(local_crs, blds, building_height, base, terrain_grid, segments, tops, owner)


def _estimate_local_crs(buildings, area, terrain):
    """UTM zone of the first non-empty input among buildings, area and terrain polygons."""
    candidates = [buildings]
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        candidates.append(area)
    elif area is not None:
        candidates.append(gpd.GeoSeries([area], crs=buildings.crs))
    if terrain is not None:
        candidates.append(terrain)
    for frame in candidates:
        if len(frame) and not frame.is_empty.all():
            return frame.estimate_utm_crs()
    raise ValueError("Nothing to compute: the area and the buildings are empty")


def _building_heights(blds, height_column, storeys_column, storey_height, default_height) -> np.ndarray:
    has_height = height_column is not None and height_column in blds.columns
    has_storeys = storeys_column is not None and storeys_column in blds.columns
    if storeys_column is not None and not has_storeys:
        raise ValueError(f"buildings have no column {storeys_column!r}")
    if not has_height and not has_storeys and default_height is None:
        raise ValueError(
            f"buildings have no column {height_column!r}; pass height_column, storeys_column or default_height"
        )
    heights = np.full(len(blds), np.nan)
    if height_column is not None and height_column in blds.columns:
        heights = pd.to_numeric(blds[height_column], errors="coerce").to_numpy(dtype=float)
    if storeys_column is not None and storeys_column in blds.columns:
        storeys = pd.to_numeric(blds[storeys_column], errors="coerce").to_numpy(dtype=float)
        heights = np.where(np.isnan(heights), storeys * storey_height, heights)
    heights = np.where(heights > 0, heights, np.nan)
    if default_height is not None:
        heights = np.where(np.isnan(heights), float(default_height), heights)
    missing = int(np.isnan(heights).sum())
    if missing:
        logger.warning(f"{missing} buildings without a height are skipped; set default_height to keep them")
    return heights


def area_geometry(area, blds, buildings_crs, local_crs):
    if area is None:
        return shapely.box(*blds.total_bounds) if len(blds) else shapely.Polygon()
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return area.to_crs(local_crs).union_all()
    return gpd.GeoSeries([area], crs=buildings_crs).to_crs(local_crs).iloc[0]


def _read_terrain(terrain):
    if isinstance(terrain, tuple):
        if len(terrain) != 3:
            raise ValueError("terrain tuple must be (heights, transform, crs)")
        heights, transform, crs = terrain
        return np.asarray(heights, dtype=float), tuple(transform)[:6], crs
    try:
        import rasterio  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("Reading terrain from a file needs rasterio: pip install rasterio") from exc
    with rasterio.open(terrain) as src:
        heights = src.read(1, masked=True).astype(float).filled(np.nan)
        return heights, tuple(src.transform)[:6], src.crs


def _rasterize_terrain(terrain: gpd.GeoDataFrame, column: str, cell: float, local_crs) -> tuple:
    """Cell-centre grid ``(z, x0, y0, sx, sy)`` from terrain polygons with heights (highest where they
    overlap, NaN where none covers a cell)."""
    if terrain.crs is None:
        raise ValueError("terrain must have a CRS")
    if column not in terrain.columns:
        raise ValueError(f"terrain has no column {column!r}; set terrain_height_column")
    polys = terrain.to_crs(local_crs)
    polys = polys[polys.geom_type.isin(["Polygon", "MultiPolygon"]) & ~polys.is_empty]
    values = pd.to_numeric(polys[column], errors="coerce").to_numpy(dtype=float)
    polys, values = polys[~np.isnan(values)], values[~np.isnan(values)]
    if polys.empty:
        raise ValueError("terrain has no polygons with a height")
    minx, miny, maxx, maxy = polys.total_bounds
    cell = float(cell)
    cells = ((maxx - minx) / cell + 1) * ((maxy - miny) / cell + 1)
    if cells > MAX_TERRAIN_CELLS:
        cell *= float(np.sqrt(cells / MAX_TERRAIN_CELLS))
        logger.warning(f"terrain is large; rasterizing it with {cell:.1f} m cells")
    nx = max(int(np.ceil((maxx - minx) / cell)), 2)
    ny = max(int(np.ceil((maxy - miny) / cell)), 2)
    xs = minx + cell / 2 + np.arange(nx) * cell
    ys = miny + cell / 2 + np.arange(ny) * cell
    centres = shapely.points(np.tile(xs, ny), np.repeat(ys, nx))
    point_idx, poly_idx = shapely.STRtree(polys.geometry.values).query(centres, predicate="within")
    z = np.full(nx * ny, np.nan)
    np.fmax.at(z, point_idx, values[poly_idx])
    return z.reshape(ny, nx), xs[0], ys[0], cell, cell


def _terrain_grid(heights: np.ndarray, transform) -> tuple:
    """Cell-centre grid ``(z, x0, y0, sx, sy)`` with rows going north, from a north-up raster."""
    a, b, c, d, e, f = transform
    if b != 0 or d != 0:
        raise ValueError("rotated terrain rasters are not supported")
    if heights.ndim != 2 or min(heights.shape) < 2:
        raise ValueError("terrain must be a 2-D array of at least 2 x 2 cells")
    z = heights
    x0, sx = c + a / 2, a
    if e < 0:
        z = z[::-1]
        y0, sy = f + e * (heights.shape[0] - 0.5), -e
    else:
        y0, sy = f + e / 2, e
    if sx < 0:
        z = z[:, ::-1]
        x0, sx = c + a * (heights.shape[1] - 0.5), -sx
    return np.ascontiguousarray(z, dtype=float), x0, y0, sx, sy


def sample_terrain(terrain_grid, xy: np.ndarray) -> np.ndarray:
    """Bilinear terrain heights at points (NaN outside), as the Rust kernel samples them."""
    z, x0, y0, sx, sy = terrain_grid
    ny, nx = z.shape
    fx, fy = (xy[:, 0] - x0) / sx, (xy[:, 1] - y0) / sy
    out = np.full(len(xy), np.nan)
    ok = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    i = np.minimum(np.floor(fx[ok]).astype(int), nx - 2)
    j = np.minimum(np.floor(fy[ok]).astype(int), ny - 2)
    tx, ty = fx[ok] - i, fy[ok] - j
    out[ok] = (
        (1 - tx) * (1 - ty) * z[j, i]
        + tx * (1 - ty) * z[j, i + 1]
        + (1 - tx) * ty * z[j + 1, i]
        + tx * ty * z[j + 1, i + 1]
    )
    return out


def grid_north_azimuth(local_crs, lon: float, lat: float) -> float:
    """Azimuth of true north measured clockwise from grid north (meridian convergence), in degrees."""
    to_local = Transformer.from_crs(4326, local_crs, always_xy=True)
    (xa, xb), (ya, yb) = to_local.transform([lon, lon], [lat, lat + 0.001])
    return float(np.degrees(np.arctan2(xb - xa, yb - ya)))


def kernel_scene(scene: Scene, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Walls, roof tops and terrain keyword arguments of the Rust kernels in coordinates shifted by ``shift``."""
    segments = scene.segments.copy()
    if len(segments):
        segments[:, :2] -= shift
        segments[:, 2:] -= shift
    kwargs = {}
    if scene.terrain_grid is not None:
        z, tx0, ty0, sx, sy = scene.terrain_grid
        kwargs = {"terrain": np.ascontiguousarray(z), "terrain_grid": (tx0 - shift[0], ty0 - shift[1], sx, sy)}
    return np.ascontiguousarray(segments, dtype=float), np.ascontiguousarray(scene.tops, dtype=float), kwargs
