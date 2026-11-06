"""Receiver-grid noise model computed in ObjectNat's Rust extension."""

from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from objectnat._rust import noise_grid_energy  # pylint: disable=no-name-in-module
from objectnat.methods.noise.noise_reduce import (
    decay_distance,
    get_air_resist_ratio,
    source_values,
    vegetation_db_per_m,
)
from objectnat.methods.utils.raster import grid_attrs, grid_zones, resolve_output
from objectnat.methods.visibility._native import geometries_to_segments

FLOOR_MARGIN_DB = 10.0  # paths this far below the target can still add up above it
LINE_FLOOR_MARGIN_DB = 30.0  # many weak samples of a long line add up: keep more of them
DIFFRACTION_SIMPLIFY = 0.5  # m: outline details smaller than this are no edges of their own
CORNER_OFFSET = 0.05  # m: what a corner sees is computed from this far outside it
MIN_CORNER_SINE = 0.05  # an outline turning by less than about 3 degrees has no corner there


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
    diffraction: bool = True,
    grid_step: float = 2.0,
    source_spacing: float | None = None,
    source_position_buffer_r: float = 0,
    output: Literal["zones", "receivers", "both"] = "zones",
    max_workers: int | None = None,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Simulate noise from point, line and polygon sources on a regular grid of receivers.

    An alternative to :func:`simulate_noise` built on the receiver-grid approach used by engineering
    noise models. Every receiver collects sound energy from every source along:

    * the direct path, when no obstacle blocks it;
    * first-order reflections from obstacle walls (image-source method), when ``reflections`` is True;
    * diffraction around one corner of an obstacle (a vertical building edge) into its shadow, when
      ``diffraction`` is True.

    Each path loses ``20 lg r`` (geometric spreading, ``r`` in metres), air absorption ``k r`` (``k`` from
    temperature and frequency, as in :func:`simulate_noise`), a reflection loss ``-10 lg(1 - a)`` per
    reflection (``a`` = absorption ratio of the wall) and, through vegetation, the foliage rate of
    :func:`simulate_noise` times the length crossed. A diffracted path has the level at the direct distance
    minus Maekawa's barrier attenuation ``10 lg(3 + 20 N)``, with the Fresnel number ``N = 2 δ / λ`` (``δ``:
    how much longer the way through the corner is, ``λ``: wavelength), at most 20 dB as ISO 9613-2 allows
    for one edge. Paths and sources add up as energies, so two equal
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
        diffraction:
            Bend sound around the corners of obstacles: a receiver hidden from a source but seen from a
            corner that the source sees at the edge of an obstacle gets the diffracted sound. Corners are the
            convex vertices of the outlines simplified by 0.5 m, and the ends of lines.
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
        output:
            ``"zones"`` (default) for the level bands, ``"receivers"`` for the receiver points with their exact
            levels only (no band polygons are built), ``"both"`` for the tuple ``(bands, receivers)``.
        max_workers:
            Maximum number of threads (``None`` uses all cores).

    Returns:
        GeoDataFrame with columns ``noise_level`` (upper bound of each ``db_sim_step`` band, dB, like
        :func:`simulate_noise`) and ``geometry`` (one polygon per band), in the CRS of ``source_points``.
        With ``output="receivers"`` the receivers, with ``output="both"`` a tuple ``(bands, receivers)``; the
        receivers have the
        exact ``noise_level`` of every receiver that hears at least ``target_noise_db``; their ``attrs``
        record the grid, so :func:`rasterize` turns ``noise_level`` into a :class:`Raster`.

    Notes:
        Not modelled: ground effect, diffraction over roofs (obstacle heights are not used) and around
        more than one corner, reflections of higher order. Right behind a building, where the sound would
        have to bend around two corners, only reflected sound arrives.
    """
    output = resolve_output(output, "receivers")
    if source_points is None or len(source_points) == 0:
        raise ValueError("source_points is empty.")
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0.")
    if db_sim_step <= 0:
        raise ValueError("db_sim_step must be > 0.")
    if source_position_buffer_r < 0:
        raise ValueError("source_position_buffer_r must be non-negative.")

    levels = source_values(source_points, "source_noise_db", source_noise_db)
    freqs = source_values(source_points, "geometric_mean_freq_hz", geometric_mean_freq_hz)

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
    veg = np.array([vegetation_db_per_m(f) for f in freqs])
    reach = np.array(
        [
            _reach(full, target_noise_db, k, is_point)
            for full, k, is_point in zip(samples["full_level"], air, samples["is_point"])
        ]
    )
    # Reflections of a line sample are followed only while that sample alone stays within
    # FLOOR_MARGIN_DB of the target: far reflections of single samples are negligible next to the
    # direct field of the whole line, which is summed in full.
    reflection_reach = np.array(
        [
            r if is_point else _reach(level, target_noise_db - FLOOR_MARGIN_DB, k, True)
            for r, level, k, is_point in zip(reach, levels, air, samples["is_point"])
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
        # corners of the merged outlines: where buildings touch, their facades run on without an edge
        merged = shapely.get_parts(obstacle_union)
        corners, links = _diffraction_corners(merged) if diffraction else (np.empty((0, 8)), None)
    else:
        obstacle_geoms = np.empty(0, dtype=object)
        segments, one_sided, wall_loss = np.empty((0, 4)), np.empty(0, dtype=bool), np.empty(0)
        obstacle_union = shapely.Polygon()
        corners, links = np.empty((0, 8)), None

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
    speed_of_sound = 331.3 + 0.606 * air_temperature  # m/s
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
        corners=np.ascontiguousarray(corners - np.tile(shift, 4), dtype=float) if len(corners) else None,
        corner_links=np.ascontiguousarray(links, dtype=np.int64) if len(corners) else None,
        wavelength=np.ascontiguousarray(speed_of_sound / freqs, dtype=float),
    ).ravel()

    with np.errstate(divide="ignore"):
        level = 10 * np.log10(energy)
    heard = level >= target_noise_db
    if output != "receivers":
        bands = _level_bands(receivers[heard], level[heard], target_noise_db, db_sim_step, grid_step, obstacle_union)
        bands = gpd.GeoDataFrame(bands, geometry="geometry", crs=local_crs).to_crs(original_crs)
        if output == "zones":
            return bands
    points = gpd.GeoDataFrame({"noise_level": level[heard]}, geometry=receivers[heard], crs=local_crs)
    points = points.to_crs(original_crs)
    points.attrs.update(grid_attrs(local_crs, grid_step, (x0, y0)))
    return points if output == "receivers" else (bands, points)


def _diffraction_corners(geometries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertical edges that sound bends around, as rows ``x, y`` of the corner, of its previous and of its next
    vertex and of a point ``CORNER_OFFSET`` metres outside it: the convex corners of polygon outlines (outer
    and courtyard rings, simplified by ``DIFFRACTION_SIMPLIFY``) and the ends and bends of lines. A corner whose
    outside point lies in an obstacle is no free edge and is left out. Also returns, for the corners on the
    convex hull of an outline, the rows of the neighbouring hull corners on the previous and the next side
    (``-1`` otherwise): the way sound goes around an obstacle from corner to corner.
    """
    parts = shapely.get_parts(geometries)
    types = shapely.get_type_id(parts)
    rows, links = [np.empty((0, 8))], [np.empty((0, 2), dtype=np.int64)]
    count = 0
    polygons = parts[types == 3]
    if len(polygons):
        outlines = shapely.orient_polygons(shapely.simplify(polygons, DIFFRACTION_SIMPLIFY))
        outlines = outlines[~shapely.is_empty(outlines)]
        # oriented rings have the obstacle on their left
        rings, ring_outline = shapely.get_rings(outlines, return_index=True)
        exterior = np.r_[True, ring_outline[1:] != ring_outline[:-1]]
        for ring, is_exterior in zip(rings, exterior):
            xy = shapely.get_coordinates(ring)[:-1]
            corners, _ = _vertex_corners(xy, closed=True, two_sided=False)
            hull_links = np.full((len(corners), 2), -1, dtype=np.int64)
            if is_exterior and len(corners) > 2:
                # hull vertices are corners of the outline: link each to the hull corners beside it
                hull = shapely.get_coordinates(shapely.convex_hull(shapely.polygons(xy)))[:-1]
                rows_at = {tuple(p): count + k for k, p in enumerate(corners[:, :2])}
                on_hull = [rows_at[tuple(p)] for p in hull if tuple(p) in rows_at]
                if len(on_hull) > 2:
                    on_hull = np.asarray(on_hull)
                    # rows follow the ring, so hull order around the ring decides which side is "previous"
                    order = np.argsort(on_hull)
                    sequence = on_hull[order]
                    hull_links[sequence - count] = np.column_stack([np.roll(sequence, 1), np.roll(sequence, -1)])
            links.append(hull_links)
            rows.append(corners)
            count += len(corners)
    for line in parts[np.isin(types, (1, 2))]:
        xy = shapely.get_coordinates(line)
        closed = len(xy) > 3 and np.allclose(xy[0], xy[-1])
        corners, _ = _vertex_corners(xy[:-1] if closed else xy, closed=closed, two_sided=True)
        links.append(np.full((len(corners), 2), -1, dtype=np.int64))
        rows.append(corners)
        count += len(corners)
    corners, links = np.vstack(rows), np.vstack(links)
    if len(corners):
        inside, _ = shapely.STRtree(geometries).query(shapely.points(corners[:, 6:8]), predicate="intersects")
        free = np.ones(len(corners), dtype=bool)
        free[inside] = False
        renumber = np.where(free, np.cumsum(free) - 1, -1)
        links = np.where(links >= 0, renumber[np.maximum(links, 0)], -1)
        corners, links = corners[free], links[free]
    return corners, links


def _vertex_corners(xy: np.ndarray, *, closed: bool, two_sided: bool) -> tuple[np.ndarray, np.ndarray]:
    """Corner rows (see :func:`_diffraction_corners`) of the vertices of one ring or line, and which vertices
    they are."""
    if len(xy) < 2:
        return np.empty((0, 8)), np.zeros(len(xy), dtype=bool)
    if closed:
        prev, nxt = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
    else:  # the ends of a line have one neighbour, given as both
        prev, nxt = np.vstack([xy[1:2], xy[:-1]]), np.vstack([xy[1:], xy[-2:-1]])
    to_prev, to_next = prev - xy, nxt - xy
    length_prev, length_next = np.hypot(*to_prev.T), np.hypot(*to_next.T)
    ok = (length_prev > 1e-9) & (length_next > 1e-9)
    u_prev = to_prev / np.where(ok, length_prev, 1.0)[:, None]
    u_next = to_next / np.where(ok, length_next, 1.0)[:, None]
    sine = u_prev[:, 1] * u_next[:, 0] - u_prev[:, 0] * u_next[:, 1]  # > 0: a left turn
    convex = np.abs(sine) > MIN_CORNER_SINE if two_sided else sine > MIN_CORNER_SINE
    keep = ok & (convex | np.all(prev == nxt, axis=1))
    away = -(u_prev + u_next)[keep]  # from the corner away from the obstacle
    outside = xy[keep] + CORNER_OFFSET * away / np.hypot(*away.T)[:, None]
    return np.hstack([xy[keep], prev[keep], nxt[keep], outside]), keep


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


def _reach(level: float, target: float, air: float, is_point: bool) -> float:
    """Distance at which a point (or line) source drops to ``target``."""
    return decay_distance(level, target, air, line=not is_point) if level > target else 0.0


def _level_bands(
    points: np.ndarray, level: np.ndarray, target_db: float, step: float, cell: float, obstacles_union
) -> pd.DataFrame:
    """Merge receiver cells into one polygon per level band (labelled by the band's upper bound)."""
    if len(points) == 0:
        return pd.DataFrame({"noise_level": pd.Series(dtype=float), "geometry": pd.Series(dtype=object)})
    label = target_db + step * np.ceil(np.round((level - target_db) / step, 9))
    zones = grid_zones(shapely.get_coordinates(points), label, cell, obstacles_union)[::-1]
    return pd.DataFrame([{"noise_level": float(value), "geometry": geom} for value, geom in zones])
