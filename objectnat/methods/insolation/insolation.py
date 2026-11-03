"""Direct-sun hours on the ground among buildings and terrain, computed in ObjectNat's Rust extension."""

import datetime as dt
import os
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from objectnat import config
from objectnat._rust import insolation_day_stats  # pylint: disable=no-name-in-module
from objectnat.methods.insolation._sun import day_samples, solar_position
from objectnat.methods.visibility._native import geometries_to_segments

logger = config.logger

TerrainInput = gpd.GeoDataFrame | str | os.PathLike | tuple
DateInput = dt.date | str
MAX_TERRAIN_CELLS = 20_000_000


def get_insolation(
    buildings: gpd.GeoDataFrame,
    date: DateInput | tuple[DateInput, DateInput],
    *,
    area: gpd.GeoDataFrame | gpd.GeoSeries | shapely.Geometry | None = None,
    height_column: str | None = "height",
    storeys_column: str | None = None,
    storey_height: float = 3.0,
    default_height: float | None = None,
    terrain: TerrainInput | None = None,
    terrain_height_column: str = "height",
    terrain_cell: float = 5.0,
    grid_step: float = 2.0,
    time_step_min: float = 10.0,
    day_step: int = 7,
    min_sun_elevation: float = 0.0,
    receiver_height: float = 0.0,
    hour_breaks: Sequence[float] = (1, 2, 3, 4, 6, 8),
    return_receivers: bool = False,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Hours of direct sunlight on the ground during a day or a period, among buildings of known height and
    terrain.

    The sun is followed through every simulated day in ``time_step_min`` steps (NOAA solar position for the
    date and the location of the area, with refraction), so its path changes with latitude and season. For
    every point of a regular grid and every sun position above ``min_sun_elevation`` a ray is cast towards
    the sun; the point is in the sun when the ray passes above every building wall it crosses and above the
    terrain. Buildings are extruded footprints with flat roofs standing on the terrain. The result shows how
    many hours each place gets and which places stay in shadow all the time.

    Args:
        buildings (gpd.GeoDataFrame):
            Building footprints (Polygon/MultiPolygon) in any CRS. Include the buildings around the area
            of interest too: shadows come from them.
        date (datetime.date | str | tuple):
            One day (``"2026-06-21"``) or a period ``(start, end)``, both days included
            (``("2026-04-22", "2026-08-22")``). Every day is covered from sunrise to sunset.
        area (gpd.GeoDataFrame | gpd.GeoSeries | shapely geometry | None, optional):
            Where to place receivers. Shapely geometries are taken in the CRS of ``buildings``.
            Defaults to the bounding box of the buildings.
        height_column (str | None, optional):
            Column with building heights in metres. Defaults to ``"height"``.
        storeys_column (str | None, optional):
            Column with the number of storeys, used (times ``storey_height``) where the height is missing.
        storey_height (float, optional):
            Height of one storey in metres. Defaults to 3.
        default_height (float | None, optional):
            Height for buildings with neither value. Without it such buildings are skipped with a warning.
        terrain (gpd.GeoDataFrame | str | os.PathLike | tuple | None, optional):
            Ground heights in metres, one of:

            - a GeoDataFrame of polygons with the height in ``terrain_height_column`` (e.g. elevation zones);
              it is rasterized with ``terrain_cell`` cells, overlapping polygons take the highest value and
              places no polygon covers count as no data;
            - a GeoTIFF path (needs ``rasterio``);
            - a tuple ``(heights, transform, crs)`` with a 2-D array, an affine transform ``(a, b, c, d, e,
              f)`` as in rasterio (``x = a * col + c``, ``y = e * row + f``, north-up, no rotation) and a
              projected CRS.

            Rasters set the CRS of the calculation; NaN marks missing data. Without terrain the ground is
            flat.
        terrain_height_column (str, optional):
            Height column of a terrain GeoDataFrame. Defaults to ``"height"``.
        terrain_cell (float, optional):
            Cell size in metres used to rasterize a terrain GeoDataFrame. Defaults to 5.
        grid_step (float, optional):
            Receiver spacing in metres. Defaults to 2.
        time_step_min (float, optional):
            Time step in minutes; it must divide a day evenly. Defaults to 10.
        day_step (int, optional):
            For a period, simulate every ``day_step``-th day (the last day of the period is always
            included). Defaults to 7.
        min_sun_elevation (float, optional):
            Sun positions lower than this (degrees) count as no sun, e.g. to leave out the low morning and
            evening sun as insolation norms do. Defaults to 0.
        receiver_height (float, optional):
            Height of the receivers above the ground in metres. Defaults to 0.
        hour_breaks (Sequence[float], optional):
            Bounds of the sun-hour classes of the result, in hours per day. Defaults to ``(1, 2, 3, 4, 6, 8)``.
        return_receivers (bool, optional):
            Also return the receiver points with their values.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            Zones of equal class of daily sun hours (the mean over the simulated days) in the CRS of
            ``buildings``, with ``sun_hours_min``, ``sun_hours_max`` (``inf`` for the last class) and
            ``label``. The zone ``0`` is ground that gets no direct sun on any simulated day. Buildings are
            cut out. With ``return_receivers=True`` also the receiver points with:

            - ``sun_hours``: mean hours of direct sun per day;
            - ``min_sun_hours``: hours on the worst day;
            - ``max_continuous_hours``: the longest uninterrupted sunny period on the worst day (the least
              of the daily longest periods), which insolation norms usually limit.

            ``attrs`` hold ``days`` (the simulated dates) and ``day_length_hours`` (mean time the sun is
            above ``min_sun_elevation``).

    Raises:
        ValueError: On missing CRS, empty input, unknown height columns, a bad time step or period.
        ImportError: If ``terrain`` is a file path and ``rasterio`` is not installed.
    """
    if not isinstance(buildings, gpd.GeoDataFrame) or buildings.crs is None:
        raise ValueError("buildings must be a GeoDataFrame with a CRS")
    days = _simulated_days(date, day_step)
    original_crs = buildings.crs

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
    blds, building_height = blds[~np.isnan(building_height)], building_height[~np.isnan(building_height)]

    area_geom = _area_geometry(area, blds, buildings.crs, local_crs)
    x0, y0, x1, y1 = area_geom.bounds
    if not np.all(np.isfinite([x0, y0, x1, y1])):
        raise ValueError("Nothing to compute: the area and the buildings are empty")
    gx = np.arange(np.floor(x0 / grid_step) * grid_step, x1 + grid_step / 2, grid_step)
    gy = np.arange(np.floor(y0 / grid_step) * grid_step, y1 + grid_step / 2, grid_step)
    xx, yy = np.meshgrid(gx, gy)
    receivers = shapely.points(xx.ravel(), yy.ravel())
    keep = shapely.covers(area_geom, receivers)
    if len(blds):
        inside, _ = shapely.STRtree(blds.geometry.values).query(receivers, predicate="within")
        keep[inside] = False
    receivers = receivers[keep]

    # heights: ground under receivers and buildings, roof tops of the walls
    rxy = shapely.get_coordinates(receivers)
    if terrain_grid is not None:
        ground = np.nan_to_num(_sample(terrain_grid, rxy), nan=0.0)
        base = np.nan_to_num(
            _sample(terrain_grid, shapely.get_coordinates(blds.representative_point().values)), nan=0.0
        )
    else:
        ground = np.zeros(len(rxy))
        base = np.zeros(len(blds))
    segments, _, owner = geometries_to_segments(blds.geometry.values, return_owner=True)
    tops = (base + building_height)[owner] if len(segments) else np.empty(0)

    # sun positions of every day; true north differs from grid north away from the central meridian
    lon, lat = Transformer.from_crs(local_crs, 4326, always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)
    grid_north = _grid_north_azimuth(local_crs, lon, lat)
    suns, offsets = [], [0]
    for day in days:
        azimuth, elevation = solar_position(day_samples(day, lon, time_step_min), lat, lon)
        up = elevation > max(min_sun_elevation, 0.0)
        angle = np.radians(azimuth[up] - grid_north)
        suns.append(np.column_stack([np.sin(angle), np.cos(angle), np.tan(np.radians(elevation[up]))]))
        offsets.append(offsets[-1] + int(up.sum()))
    suns = np.concatenate(suns)

    shift = np.array([gx[0], gy[0]])
    shifted = segments.copy()
    if len(shifted):
        shifted[:, :2] -= shift
        shifted[:, 2:] -= shift
    terrain_args = {}
    if terrain_grid is not None:
        z, tx0, ty0, sx, sy = terrain_grid
        terrain_args = {"terrain": np.ascontiguousarray(z), "terrain_grid": (tx0 - shift[0], ty0 - shift[1], sx, sy)}
    counts, longest = insolation_day_stats(
        np.ascontiguousarray(np.column_stack([rxy - shift, ground + receiver_height]), dtype=float),
        np.ascontiguousarray(suns, dtype=float),
        np.asarray(offsets, dtype=np.int64),
        np.ascontiguousarray(shifted, dtype=float),
        np.ascontiguousarray(tops, dtype=float),
        n_threads=max_workers,
        **terrain_args,
    )

    step_h = time_step_min / 60.0
    sun_hours = counts.mean(axis=1) * step_h
    attrs = {
        "days": [d.isoformat() for d in days],
        "day_length_hours": float(np.mean(np.diff(offsets))) * step_h,
    }
    footprint = shapely.union_all(blds.geometry.values) if len(blds) else shapely.Polygon()
    bands = _hour_bands(receivers, sun_hours, hour_breaks, grid_step, footprint)
    bands = gpd.GeoDataFrame(bands, geometry="geometry", crs=local_crs).to_crs(original_crs)
    bands.attrs.update(attrs)
    if not return_receivers:
        return bands
    points = gpd.GeoDataFrame(
        {
            "sun_hours": sun_hours,
            "min_sun_hours": counts.min(axis=1) * step_h,
            "max_continuous_hours": longest.min(axis=1) * step_h,
        },
        geometry=receivers,
        crs=local_crs,
    ).to_crs(original_crs)
    points.attrs.update(attrs)
    return bands, points


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


def _simulated_days(date, day_step: int) -> list[dt.date]:
    def as_date(value):
        return dt.date.fromisoformat(value) if isinstance(value, str) else value

    if not isinstance(date, (tuple, list)):
        return [as_date(date)]
    if len(date) != 2:
        raise ValueError("a period must be (start, end)")
    start, end = as_date(date[0]), as_date(date[1])
    if end < start:
        raise ValueError("the period ends before it starts")
    if int(day_step) < 1:
        raise ValueError("day_step must be at least 1")
    days = [start + dt.timedelta(days=k) for k in range(0, (end - start).days + 1, int(day_step))]
    if days[-1] != end:
        days.append(end)
    return days


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


def _area_geometry(area, blds, buildings_crs, local_crs):
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


def _sample(terrain_grid, xy: np.ndarray) -> np.ndarray:
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


def _grid_north_azimuth(local_crs, lon: float, lat: float) -> float:
    """Azimuth of true north measured clockwise from grid north (meridian convergence), in degrees."""
    to_local = Transformer.from_crs(4326, local_crs, always_xy=True)
    (xa, xb), (ya, yb) = to_local.transform([lon, lon], [lat, lat + 0.001])
    return float(np.degrees(np.arctan2(xb - xa, yb - ya)))


def _hour_bands(points, hours, breaks, cell, footprint) -> pd.DataFrame:
    """Merge receiver cells into one polygon per sun-hour class; class 0 is no direct sun at all."""
    edges = np.array(sorted({float(b) for b in breaks if b > 0}))
    lower = np.concatenate([[0.0, 0.0], edges])
    upper = np.concatenate([[0.0], edges, [np.inf]])
    eps = 1e-9
    label = np.where(hours <= eps, 0, 1 + np.searchsorted(edges, hours + eps, side="right"))
    xy = shapely.get_coordinates(points)
    half = cell / 2
    cells = shapely.box(xy[:, 0] - half, xy[:, 1] - half, xy[:, 0] + half, xy[:, 1] + half)
    rows = []
    for k in np.unique(label):
        geom = shapely.coverage_union_all(cells[label == k])
        if not footprint.is_empty:
            geom = geom.difference(footprint)
        if geom.is_empty:
            continue
        lo, hi = lower[k], upper[k]
        name = "0" if k == 0 else (f"<{hi:g}" if lo == 0 else (f"≥{lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}"))
        rows.append({"sun_hours_min": lo, "sun_hours_max": hi, "label": name, "geometry": geom})
    if not rows:
        return pd.DataFrame(columns=["sun_hours_min", "sun_hours_max", "label", "geometry"])
    return pd.DataFrame(rows)
