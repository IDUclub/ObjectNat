"""Direct-sun hours on the ground among buildings and terrain, computed in ObjectNat's Rust extension."""

import datetime as dt
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from objectnat._rust import insolation_day_stats  # pylint: disable=no-name-in-module
from objectnat.methods.insolation._sun import day_samples, solar_position
from objectnat.methods.utils.raster import grid_attrs
from objectnat.methods.utils.scene import (
    Scene,
    TerrainInput,
    area_geometry,
    grid_north_azimuth,
    kernel_scene,
    prepare_scene,
)

DateInput = dt.date | str


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
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
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
            - a GeoTIFF path (requires ``rasterio``);
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
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            Receiver points in the CRS of ``buildings``; points inside buildings are left out. Columns:

            - ``sun_hours``: mean hours of direct sun per day (``0`` for ground that gets no direct sun on any
              simulated day);
            - ``min_sun_hours``: hours on the worst day;
            - ``max_continuous_hours``: the longest uninterrupted sunlit period on the worst day (the least
              of the daily longest periods), for which insolation norms usually set a minimum.

            ``attrs`` contain ``days`` (the simulated dates) and ``day_length_hours`` (mean time the sun is
            above ``min_sun_elevation``) and record the grid, so :func:`rasterize` turns any column into a
            :class:`Raster`.

    Raises:
        ValueError: On missing CRS, empty input, unknown height columns, a bad time step or period.
        ImportError: If ``terrain`` is a file path and ``rasterio`` is not installed.
    """
    days = _simulated_days(date, day_step)
    scene = prepare_scene(
        buildings, area, terrain, terrain_height_column, terrain_cell,
        height_column, storeys_column, storey_height, default_height,
    )  # fmt: skip
    blds = scene.blds
    area_geom = area_geometry(area, blds, buildings.crs, scene.local_crs)
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
    rxy = shapely.get_coordinates(receivers)

    suns, offsets = _sun_positions(
        days, scene.local_crs, (x0 + x1) / 2, (y0 + y1) / 2, time_step_min, min_sun_elevation
    )
    xyz = np.column_stack([rxy, scene.ground_at(rxy) + receiver_height])
    counts, longest = _day_stats(scene, xyz, suns, offsets, max_workers=max_workers)

    step_h = time_step_min / 60.0
    sun_hours = counts.mean(axis=1) * step_h
    attrs = {"days": [d.isoformat() for d in days], "day_length_hours": float(np.mean(np.diff(offsets))) * step_h}
    points = gpd.GeoDataFrame(
        {
            "sun_hours": sun_hours,
            "min_sun_hours": counts.min(axis=1) * step_h,
            "max_continuous_hours": longest.min(axis=1) * step_h,
        },
        geometry=receivers,
        crs=scene.local_crs,
    ).to_crs(buildings.crs)
    points.attrs.update(attrs)
    points.attrs.update(grid_attrs(scene.local_crs, grid_step, (gx[0], gy[0])))
    return points


def get_facade_insolation(
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
    spacing: float = 3.0,
    window_height: float = 1.2,
    floors: Sequence[int] | None = None,
    time_step_min: float = 10.0,
    day_step: int = 7,
    min_sun_elevation: float = 0.0,
    min_facade_angle: float = 0.0,
    norm_hours: float | None = None,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Hours of direct sunlight on building facades, floor by floor: which windows get sun and which do not.

    Calculation points are placed along every wall of the buildings in ``area`` (outer walls and courtyard
    walls) every ``spacing`` metres, on every floor at ``window_height`` above the floor level, just in
    front of the wall. Walls shared with a neighbouring building get no points. A point is sunlit when the
    sun is in front of its facade, at least ``min_facade_angle`` away from the facade plane, and the ray
    towards the sun passes above every other wall and the terrain, as in :func:`get_insolation`. All
    buildings cast shadows, including those outside ``area``.

    Args:
        buildings (gpd.GeoDataFrame):
            Building footprints with heights, as in :func:`get_insolation`.
        date (datetime.date | str | tuple):
            One day or a period ``(start, end)``, as in :func:`get_insolation`.
        area (gpd.GeoDataFrame | gpd.GeoSeries | shapely geometry | None, optional):
            Buildings intersecting it get facade points (shapely geometries in the CRS of ``buildings``).
            Defaults to all buildings.
        height_column, storeys_column, storey_height, default_height:
            Building heights, as in :func:`get_insolation`. The number of floors is the storeys value when
            given, otherwise the height divided by ``storey_height`` (at least one floor).
        terrain, terrain_height_column, terrain_cell:
            Ground heights, as in :func:`get_insolation`.
        spacing (float, optional):
            Distance between points along a wall in metres; shorter walls get one point. Defaults to 3.
        window_height (float, optional):
            Height of the points above each floor level in metres. Defaults to 1.2.
        floors (Sequence[int] | None, optional):
            Only these floors (1 is the ground floor). Defaults to all floors.
        time_step_min, day_step, min_sun_elevation:
            Sun sampling, as in :func:`get_insolation`.
        min_facade_angle (float, optional):
            Sun closer than this angle (degrees, horizontal) to the facade plane does not count. Defaults to
            0: any sun in front of the facade counts.
        norm_hours (float | None, optional):
            Required continuous sunlight in hours; adds ``meets_norm`` (``max_continuous_hours >=
            norm_hours``).
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            One point per wall position and floor in the CRS of ``buildings`` with ``building`` (index label
            of the building), ``floor``, ``height`` (above the ground), ``facade_azimuth`` (direction the
            facade faces, degrees clockwise from true north), ``sun_hours``, ``min_sun_hours`` and
            ``max_continuous_hours`` (as in :func:`get_insolation`) and, with ``norm_hours``,
            ``meets_norm``. ``attrs`` contain ``days`` and ``day_length_hours``.
    """
    days = _simulated_days(date, day_step)
    scene = prepare_scene(
        buildings, area, terrain, terrain_height_column, terrain_cell,
        height_column, storeys_column, storey_height, default_height,
    )  # fmt: skip
    blds = scene.blds
    targets = np.arange(len(blds))
    if area is not None:
        area_geom = area_geometry(area, blds, buildings.crs, scene.local_crs)
        targets = targets[shapely.intersects(blds.geometry.values, area_geom)]
    if len(targets) == 0:
        raise ValueError("No buildings with a height in the area")

    xy, normals, owner = _facade_points(blds.geometry.values[targets], spacing)
    owner = targets[owner]
    if len(xy):
        on_neighbour, _ = shapely.STRtree(blds.geometry.values).query(shapely.points(xy), predicate="within")
        keep = np.ones(len(xy), dtype=bool)
        keep[on_neighbour] = False
        xy, normals, owner = xy[keep], normals[keep], owner[keep]

    if storeys_column is not None and storeys_column in blds.columns:
        storeys = pd.to_numeric(blds[storeys_column], errors="coerce").to_numpy(dtype=float)
    else:
        storeys = np.full(len(blds), np.nan)
    storeys = np.where(storeys >= 1, storeys, np.floor(scene.building_height / storey_height))
    storeys = np.maximum(storeys, 1).astype(int)
    floor_counts = storeys[owner]
    point_idx = np.repeat(np.arange(len(xy)), floor_counts)
    floor = np.concatenate([np.arange(1, n + 1) for n in floor_counts]) if len(xy) else np.empty(0, dtype=int)
    height = (floor - 1) * storey_height + window_height
    keep = height < scene.building_height[owner[point_idx]]
    if floors is not None:
        keep &= np.isin(floor, list(floors))
    point_idx, floor, height = point_idx[keep], floor[keep], height[keep]
    if len(point_idx) == 0:
        raise ValueError("No facade points: check spacing, window_height and floors")

    pxy, pnormal, powner = xy[point_idx], normals[point_idx], owner[point_idx]
    x0, y0 = pxy.min(axis=0)
    x1, y1 = pxy.max(axis=0)
    suns, offsets = _sun_positions(
        days, scene.local_crs, (x0 + x1) / 2, (y0 + y1) / 2, time_step_min, min_sun_elevation
    )
    xyz = np.column_stack([pxy, scene.base[powner] + height])
    counts, longest = _day_stats(
        scene, xyz, suns, offsets, normals=pnormal, min_facing=np.sin(np.radians(min_facade_angle)),
        max_workers=max_workers,
    )  # fmt: skip

    step_h = time_step_min / 60.0
    lon, lat = Transformer.from_crs(scene.local_crs, 4326, always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)
    azimuth = np.mod(
        np.degrees(np.arctan2(pnormal[:, 0], pnormal[:, 1])) + grid_north_azimuth(scene.local_crs, lon, lat), 360
    )
    result = gpd.GeoDataFrame(
        {
            "building": blds.index.to_numpy()[powner],
            "floor": floor,
            "height": height,
            "facade_azimuth": azimuth,
            "sun_hours": counts.mean(axis=1) * step_h,
            "min_sun_hours": counts.min(axis=1) * step_h,
            "max_continuous_hours": longest.min(axis=1) * step_h,
        },
        geometry=shapely.points(pxy),
        crs=scene.local_crs,
    ).to_crs(buildings.crs)
    if norm_hours is not None:
        result["meets_norm"] = result["max_continuous_hours"] >= float(norm_hours) - 1e-9
    result.attrs.update(
        {"days": [d.isoformat() for d in days], "day_length_hours": float(np.mean(np.diff(offsets))) * step_h}
    )
    return result


def _sun_positions(days, local_crs, cx, cy, time_step_min, min_sun_elevation) -> tuple[np.ndarray, list[int]]:
    """Sun directions in grid coordinates ``(t, 3)`` [dx, dy, tan(elevation)] of all days, and day offsets.
    True north differs from grid north away from the projection's central meridian."""
    lon, lat = Transformer.from_crs(local_crs, 4326, always_xy=True).transform(cx, cy)
    grid_north = grid_north_azimuth(local_crs, lon, lat)
    suns, offsets = [], [0]
    for day in days:
        azimuth, elevation = solar_position(day_samples(day, lon, time_step_min), lat, lon)
        up = elevation > max(min_sun_elevation, 0.0)
        angle = np.radians(azimuth[up] - grid_north)
        suns.append(np.column_stack([np.sin(angle), np.cos(angle), np.tan(np.radians(elevation[up]))]))
        offsets.append(offsets[-1] + int(up.sum()))
    return np.concatenate(suns), offsets


def _day_stats(scene: Scene, xyz, suns, offsets, *, normals=None, min_facing=0.0, max_workers=None):
    """Per point and day: sunlit sun positions and the longest sunlit run (Rust kernel, local coordinates)."""
    shift = xyz[:, :2].min(axis=0) if len(xyz) else np.zeros(2)
    segments, tops, kwargs = kernel_scene(scene, shift)
    if normals is not None:
        kwargs.update(normals=np.ascontiguousarray(normals, dtype=float), min_facing=float(min_facing))
    local = xyz.copy()
    local[:, :2] -= shift
    return insolation_day_stats(
        np.ascontiguousarray(local, dtype=float),
        np.ascontiguousarray(suns, dtype=float),
        np.asarray(offsets, dtype=np.int64),
        segments,
        tops,
        n_threads=max_workers,
        **kwargs,
    )


def _facade_points(geometries: np.ndarray, spacing: float, offset: float = 0.1):
    """Points along every wall of the polygons, ``offset`` metres in front of it, with the wall's outward
    unit normal and the index of the polygon."""
    parts, part_owner = shapely.get_parts(shapely.orient_polygons(geometries), return_index=True)
    rings, ring_part = [], []
    for k, part in enumerate(parts):
        for ring in [part.exterior, *part.interiors]:
            rings.append(ring)
            ring_part.append(k)
    coords, ring_idx = shapely.get_coordinates(np.array(rings, dtype=object), return_index=True)
    same = ring_idx[1:] == ring_idx[:-1]
    a, b, wall_owner = coords[:-1][same], coords[1:][same], part_owner[np.asarray(ring_part)[ring_idx[:-1][same]]]
    vec = b - a
    length = np.hypot(vec[:, 0], vec[:, 1])
    ok = length > 1e-6
    a, vec, length, wall_owner = a[ok], vec[ok], length[ok], wall_owner[ok]
    normal = np.column_stack([vec[:, 1], -vec[:, 0]]) / length[:, None]  # right of a CCW exterior / CW hole
    count = np.maximum(np.round(length / spacing).astype(int), 1)
    wall = np.repeat(np.arange(len(a)), count)
    position = np.concatenate([(np.arange(n) + 0.5) / n for n in count]) if len(count) else np.empty(0)
    xy = a[wall] + position[:, None] * vec[wall] + offset * normal[wall]
    return xy, normal[wall], wall_owner[wall]


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
