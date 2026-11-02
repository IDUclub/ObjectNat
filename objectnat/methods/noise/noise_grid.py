"""Receiver-grid noise model computed in ObjectNat's Rust extension."""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from objectnat import config
from objectnat._rust import noise_grid_energy  # pylint: disable=no-name-in-module
from objectnat.methods.noise.noise_reduce import dist_to_target_db, get_air_resist_ratio
from objectnat.methods.visibility._native import geometries_to_segments

logger = config.logger

MAX_DB_VALUE = 194
FLOOR_MARGIN_DB = 10.0  # paths this far below the target can still add up above it
LINE_FLOOR_MARGIN_DB = 30.0  # many weak samples of a long line add up: keep more of them


def _vegetation_db_per_m(freq_hz: float) -> float:
    """Foliage attenuation rate of ``green_noise_reduce_db`` (dB per metre of vegetation crossed)."""
    return 0.08 * (freq_hz ** (1 / 3)) / 8


def _cut_holes_around_sources(geometries: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """Cut a ``radius`` buffer around every source from the geometries that contain it."""
    if radius <= 0 or len(geometries) == 0:
        return geometries
    geometries = geometries.copy()
    tree = shapely.STRtree(geometries)
    point_idx, geom_idx = tree.query(points, predicate="within")
    for p, g in zip(point_idx, geom_idx):
        geometries[g] = geometries[g].difference(points[p].buffer(radius, quad_segs=8))
    return geometries


def simulate_noise_grid(
    source_points: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame | None,
    source_noise_db: float | None = None,
    geometric_mean_freq_hz: float | None = None,
    *,
    absorb_ratio_column: str | None = None,
    standard_absorb_ratio: float = 0.05,
    trees: gpd.GeoDataFrame | None = None,
    air_temperature: float = 20,
    target_noise_db: float = 40,
    db_sim_step: float = 1,
    reflections: bool = True,
    reflection_max_dist: float = 50.0,
    grid_step: float = 2.0,
    source_spacing: float | None = None,
    source_position_buffer_r: float = 0,
    return_receivers: bool = False,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Simulate noise from point, line and polygon sources on a regular grid of receivers.

    An alternative to :func:`simulate_noise` built on the receiver-grid approach used by engineering
    noise models. Every receiver collects sound energy from every source along:

    * the direct path, when no obstacle blocks it;
    * first-order reflections from obstacle walls (image-source method), when ``reflections`` is True.

    Each path loses ``20 lg r`` (geometric spreading, ``r`` in metres), air absorption ``k r`` (``k`` from
    temperature and frequency, as in :func:`simulate_noise`), a reflection loss ``-10 lg(1 - a)`` per
    reflection (``a`` = absorption ratio of the wall) and, through vegetation, the foliage rate of
    :func:`simulate_noise` times the length crossed. Paths and sources add up as energies, so two equal
    sources give +3 dB. Receivers inside obstacles are skipped. The levels are then grouped into bands of
    ``db_sim_step`` and returned as polygons, like :func:`simulate_noise`.

    The computation runs in parallel in ObjectNat's Rust extension.

    Args:
        source_points:
            Noise sources. Points are point sources. Lines (roads, railways) are line sources, sampled
            every ``source_spacing`` metres. Polygons (industrial sites, noisy buildings) radiate from
            their outline as line sources. Optional columns ``source_noise_db`` (level at 1 m from the
            point or line, dB) and ``geometric_mean_freq_hz`` override the arguments of the same name
            per source.
        obstacles:
            Buildings, walls and other obstacles (polygons or lines). ``None`` or empty for free field.
        source_noise_db:
            Level at 1 m for sources without a ``source_noise_db`` column, dB.
        geometric_mean_freq_hz:
            Characteristic frequency for sources without a ``geometric_mean_freq_hz`` column, Hz.
        absorb_ratio_column:
            Column of ``obstacles`` with the sound absorption ratio of each obstacle (0-1).
        standard_absorb_ratio:
            Absorption ratio for obstacles without a value. Default 0.05 (concrete).
        trees:
            Vegetation polygons; sound crossing them is attenuated.
        air_temperature:
            Air temperature, °C.
        target_noise_db:
            Lowest level to report, dB. Sources reach as far as their own level drops to it.
        db_sim_step:
            Width of the output level bands, dB.
        reflections:
            Include first-order reflections from obstacle walls.
        reflection_max_dist:
            Only walls closer than this to a source reflect it, metres (default 50, as the maximum
            reflection distance of engineering models). Larger values are slower.
        grid_step:
            Receiver spacing in metres; smaller is finer and slower.
        source_spacing:
            Spacing of the point samples along line and polygon sources, metres (default 5). Each
            sample gets ``source_noise_db - 10 lg(pi / s)``, so that a long line follows the line-source
            law ``source_noise_db - 10 lg d - k d`` (3 dB per doubling of distance). Closer to the line
            than about ``source_spacing`` the individual samples show.
        source_position_buffer_r:
            Radius in metres cut out of an obstacle or tree that contains a source, as in
            :func:`simulate_noise`.
        return_receivers:
            Also return the receiver points with their levels.
        max_workers:
            Maximum number of threads (``None`` uses all cores).

    Returns:
        GeoDataFrame with columns ``noise_level`` (upper bound of each ``db_sim_step`` band, dB, like
        :func:`simulate_noise`) and ``geometry`` (one polygon per band), in the CRS of ``source_points``.
        With ``return_receivers=True``, a tuple ``(bands, receivers)`` where ``receivers`` has the
        exact ``noise_level`` of every receiver that hears at least ``target_noise_db``.

    Notes:
        Not modelled: ground effect, diffraction over or around buildings, reflections of higher
        order. Behind an obstacle only reflected sound arrives.
    """
    if source_points is None or len(source_points) == 0:
        raise ValueError("source_points is empty.")
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0.")
    if db_sim_step <= 0:
        raise ValueError("db_sim_step must be > 0.")
    if source_position_buffer_r < 0:
        raise ValueError("source_position_buffer_r must be non-negative.")

    levels = _per_source_values(source_points, "source_noise_db", source_noise_db)
    freqs = _per_source_values(source_points, "geometric_mean_freq_hz", geometric_mean_freq_hz)
    if (levels > MAX_DB_VALUE).any():
        raise ValueError(f"source_noise_db exceeds the physical limit of {MAX_DB_VALUE} dB in air.")

    original_crs = source_points.crs
    local_crs = (obstacles if obstacles is not None and len(obstacles) else source_points).estimate_utm_crs()
    sources = source_points.to_crs(local_crs)
    spacing = float(source_spacing) if source_spacing is not None else 5.0
    if spacing <= 0:
        raise ValueError("source_spacing must be > 0.")
    samples = _sample_sources(sources.geometry.values, levels, freqs, spacing)
    if len(samples["xy"]) == 0:
        raise ValueError("source_points has no usable geometry.")
    xy, levels, freqs = samples["xy"], samples["level"], samples["freq"]
    source_geoms = shapely.points(xy[samples["is_point"]])

    air = np.array([get_air_resist_ratio(air_temperature, f, True) for f in freqs])
    veg = np.array([_vegetation_db_per_m(f) for f in freqs])
    reach = np.array(
        [
            _reach(full, target_noise_db, freq, air_temperature, k, is_point)
            for full, freq, k, is_point in zip(samples["full_level"], freqs, air, samples["is_point"])
        ]
    )
    # Reflections of a line sample are followed only while that sample alone stays within
    # FLOOR_MARGIN_DB of the target: far reflections of single samples are negligible next to the
    # direct field of the whole line, which is summed in full.
    reflection_reach = np.array(
        [
            r if is_point else _reach(level, target_noise_db - FLOOR_MARGIN_DB, freq, air_temperature, k, True)
            for r, level, freq, k, is_point in zip(reach, levels, freqs, air, samples["is_point"])
        ]
    )

    # receiver grid over the reach of the sources; obstacles and trees outside it cannot matter
    origin = xy.min(axis=0) - reach.max()
    x0, y0 = np.floor(origin / grid_step) * grid_step
    x1, y1 = xy.max(axis=0) + reach.max()
    nx = int(np.floor((x1 - x0) / grid_step)) + 1
    ny = int(np.floor((y1 - y0) / grid_step)) + 1
    gx = x0 + np.arange(nx) * grid_step
    gy = y0 + np.arange(ny) * grid_step
    receivers = shapely.points(np.repeat(gx[None, :], ny, axis=0).ravel(), np.repeat(gy, nx))
    area = shapely.box(x0 - grid_step, y0 - grid_step, x1 + grid_step, y1 + grid_step)

    # obstacles -> walls with reflection losses
    if obstacles is not None and len(obstacles):
        obstacles = obstacles.to_crs(local_crs)
        obstacles = obstacles.iloc[obstacles.sindex.query(area, predicate="intersects")]
    if obstacles is not None and len(obstacles):
        absorb = (
            obstacles[absorb_ratio_column].fillna(standard_absorb_ratio).to_numpy(dtype=float)
            if absorb_ratio_column is not None
            else np.full(len(obstacles), float(standard_absorb_ratio))
        )
        obstacle_geoms = _cut_holes_around_sources(obstacles.geometry.values, source_geoms, source_position_buffer_r)
        segments, one_sided, owner = geometries_to_segments(obstacle_geoms, return_owner=True)
        wall_loss = -10 * np.log10(np.clip(1 - absorb[owner], 1e-6, 1))
        obstacle_union = shapely.union_all(obstacle_geoms)
    else:
        obstacle_geoms = np.empty(0, dtype=object)
        segments, one_sided, wall_loss = np.empty((0, 4)), np.empty(0, dtype=bool), np.empty(0)
        obstacle_union = shapely.Polygon()

    # vegetation -> boundaries of non-overlapping polygons
    tree_segments = np.empty((0, 4))
    if trees is not None and len(trees):
        trees = trees.to_crs(local_crs)
        trees = trees.iloc[trees.sindex.query(area, predicate="intersects")]
    if trees is not None and len(trees):
        tree_geoms = _cut_holes_around_sources(trees.geometry.values, source_geoms, source_position_buffer_r)
        merged = shapely.get_parts(shapely.union_all(tree_geoms))
        merged = merged[shapely.get_type_id(merged) == 3]
        tree_segments, _ = geometries_to_segments(merged)

    active = np.ones(nx * ny, dtype=bool)
    if len(obstacle_geoms):
        inside, _ = shapely.STRtree(obstacle_geoms).query(receivers, predicate="within")
        active[inside] = False

    # local coordinates keep the kernel's arithmetic precise
    shift = np.array([x0, y0])
    shifted = segments.copy()
    if len(shifted):
        shifted[:, :2] -= shift
        shifted[:, 2:] -= shift
    shifted_trees = tree_segments.copy()
    if len(shifted_trees):
        shifted_trees[:, :2] -= shift
        shifted_trees[:, 2:] -= shift
    energy = noise_grid_energy(
        np.ascontiguousarray(xy - shift),
        np.ascontiguousarray(levels, dtype=float),
        np.ascontiguousarray(air, dtype=float),
        np.ascontiguousarray(veg, dtype=float),
        np.ascontiguousarray(reach, dtype=float),
        np.ascontiguousarray(shifted, dtype=float),
        np.ascontiguousarray(one_sided, dtype=bool),
        np.ascontiguousarray(wall_loss, dtype=float),
        np.ascontiguousarray(shifted_trees, dtype=float),
        (0.0, 0.0, float(grid_step), nx, ny),
        active,
        reflections=bool(reflections),
        reflection_max_dist=float(reflection_max_dist),
        floor_db=float(target_noise_db) - (FLOOR_MARGIN_DB if samples["is_point"].all() else LINE_FLOOR_MARGIN_DB),
        n_threads=max_workers,
        reflection_reach=np.ascontiguousarray(reflection_reach, dtype=float),
    ).ravel()

    with np.errstate(divide="ignore"):
        level = 10 * np.log10(energy)
    heard = level >= target_noise_db
    bands = _level_bands(receivers[heard], level[heard], target_noise_db, db_sim_step, grid_step, obstacle_union)
    bands = gpd.GeoDataFrame(bands, geometry="geometry", crs=local_crs).to_crs(original_crs)
    if not return_receivers:
        return bands
    points = gpd.GeoDataFrame({"noise_level": level[heard]}, geometry=receivers[heard], crs=local_crs)
    return bands, points.to_crs(original_crs)


def _line_level_offset(spacing: float) -> float:
    """
    dB to subtract from a line's level for each of its point samples ``spacing`` metres apart, so that
    the samples of a long line add up to the line law ``L - 10 lg d`` at distance ``d``: ``10 lg(pi / s)``
    (the energy sum of samples ``s`` apart seen from ``d`` tends to ``pi / (s d)``).
    """
    return float(10 * np.log10(np.pi / spacing))


def _sample_sources(geometries: np.ndarray, levels: np.ndarray, freqs: np.ndarray, spacing: float) -> dict:
    """
    Point sources as they are; lines sampled every ``spacing`` metres; polygons radiate from their outline
    (0.1 m outside, as in calculate_simplified_noise_frame). Line samples get calibrated levels.
    """
    parts, owner = shapely.get_parts(geometries, return_index=True)
    types = shapely.get_type_id(parts)
    point_mask = types == 0
    polygon_mask = types == 3
    lines = parts.copy()
    if polygon_mask.any():
        lines[polygon_mask] = shapely.boundary(shapely.buffer(parts[polygon_mask], 0.1, quad_segs=1))
    lines, line_inner = shapely.get_parts(lines[~point_mask], return_index=True)
    line_owner = owner[~point_mask][line_inner]
    keep = shapely.get_type_id(lines) == 1
    lines, line_owner = lines[keep], line_owner[keep]

    xy = [shapely.get_coordinates(parts[point_mask])]
    level = [levels[owner[point_mask]]]
    full_level = [levels[owner[point_mask]]]
    freq = [freqs[owner[point_mask]]]
    is_point = [np.ones(point_mask.sum(), dtype=bool)]
    lengths = shapely.length(lines)
    for line, length, src in zip(lines, lengths, line_owner):
        if length <= 0:
            continue
        n = max(1, int(np.ceil(length / spacing)))
        step = length / n
        pts = shapely.line_interpolate_point(line, (np.arange(n) + 0.5) * step)
        xy.append(shapely.get_coordinates(pts))
        level.append(np.full(n, levels[src] - _line_level_offset(step)))
        full_level.append(np.full(n, levels[src]))
        freq.append(np.full(n, freqs[src]))
        is_point.append(np.zeros(n, dtype=bool))
    return {
        "xy": np.vstack(xy),
        "level": np.concatenate(level),
        "full_level": np.concatenate(full_level),
        "freq": np.concatenate(freq),
        "is_point": np.concatenate(is_point),
    }


def _reach(level: float, target: float, freq: float, temperature: float, air: float, is_point: bool) -> float:
    """Distance at which a source drops to ``target``: point law, or line law ``L - 10 lg d - k d``."""
    if level <= target:
        return 0.0
    if is_point:
        return float(dist_to_target_db(level, target, freq, temperature))
    lo, hi = 1.0, 1e6
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if level - 10 * np.log10(mid) - air * mid > target:
            lo = mid
        else:
            hi = mid
    return lo


def _per_source_values(source_points: gpd.GeoDataFrame, column: str, default: float | None) -> np.ndarray:
    if column in source_points.columns:
        values = source_points[column]
        if values.isnull().any():
            raise ValueError(f"Column '{column}' contains missing (NaN) values")
        return values.to_numpy(dtype=float)
    if default is None:
        raise ValueError(f"Either `{column}` must be provided or `source_points` must contain a '{column}' column.")
    return np.full(len(source_points), float(default))


def _level_bands(
    points: np.ndarray, level: np.ndarray, target_db: float, step: float, cell: float, obstacles_union
) -> pd.DataFrame:
    """Merge receiver cells into one polygon per level band (labelled by the band's upper bound)."""
    if len(points) == 0:
        return pd.DataFrame({"noise_level": pd.Series(dtype=float), "geometry": pd.Series(dtype=object)})
    label = target_db + step * np.ceil(np.round((level - target_db) / step, 9))
    xy = shapely.get_coordinates(points)
    half = cell / 2
    cells = shapely.box(xy[:, 0] - half, xy[:, 1] - half, xy[:, 0] + half, xy[:, 1] + half)
    rows = []
    for value in np.unique(label)[::-1]:
        geom = shapely.coverage_union_all(cells[label == value])
        if not obstacles_union.is_empty:
            geom = geom.difference(obstacles_union)
        if not geom.is_empty:
            rows.append({"noise_level": float(value), "geometry": geom})
    return pd.DataFrame(rows)
