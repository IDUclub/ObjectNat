"""OSM tree expansion and Beer–Lambert crown records for the insolation kernel."""

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from objectnat import config
from objectnat.methods.utils._building_details import osm_numbers

logger = config.logger


def _month_day(value):
    """Validate a recurring date against a leap year (29 February is allowed)."""
    try:
        if not isinstance(value, str) or len(value) != 5:
            raise ValueError
        day = dt.date.fromisoformat(f"2000-{value}")
    except (TypeError, ValueError) as exc:
        raise ValueError("leaf_on must contain two valid MM-DD dates") from exc
    return day.month, day.day


def leaf_state(days, leaf_on):
    """Inclusive deciduous leaf season, including seasons that cross New Year."""
    if not isinstance(leaf_on, (tuple, list)) or len(leaf_on) != 2:
        raise ValueError("leaf_on must contain two valid MM-DD dates")
    start, end = map(_month_day, leaf_on)
    dates = [(day.month, day.day) for day in days]
    return np.asarray(
        [start <= day <= end if start <= end else day >= start or day <= end for day in dates], dtype=np.uint8
    )


def validate_tree_parameters(height, diameter, base, transmittance, leaf_on, spacing, threshold):
    """Validate defaults even for an empty tree frame, before geometry expansion."""

    def finite_range(value, low, high, open_low=False, open_high=False):
        try:
            return (
                np.isscalar(value)
                and np.isfinite(value)
                and (value > low if open_low else value >= low)
                and (value < high if open_high else value <= high)
            )
        except TypeError:
            return False

    for name, value in (("tree_height", height), ("tree_crown_diameter", diameter), ("tree_spacing", spacing)):
        if value is not None or name != "tree_spacing":
            if not finite_range(value, 0, np.inf, open_low=True):
                raise ValueError(f"{name} must be finite and positive")
    if not finite_range(base, 0, 1, open_high=True):
        raise ValueError("tree_crown_base must be finite and in [0, 1)")
    try:
        transmission = np.asarray(transmittance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("tree_transmittance must be a pair in (0, 1]") from exc
    if transmission.shape != (2,) or not np.all(np.isfinite(transmission) & (transmission > 0) & (transmission <= 1)):
        raise ValueError("tree_transmittance must be a pair in (0, 1]")
    if not finite_range(threshold, 0, 1):
        raise ValueError("sunlit_threshold must be finite and in [0, 1]")
    leaf_state([], leaf_on)


def _line_points(lines, spacing):
    """Exact spacing, centred on each line; short lines receive a single tree."""
    lengths = shapely.length(lines)
    counts = np.maximum(np.floor(lengths / spacing).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(lines)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    along = (lengths[owner] - (counts[owner] - 1) * spacing[owner]) / 2
    along += (np.arange(len(owner)) - start) * spacing[owner]
    return shapely.get_coordinates(shapely.line_interpolate_point(lines[owner], along)), owner


def _polygon_points(polygons, spacing):
    """Hexagonal nearest-neighbour lattice, clipped in batches, including polygon holes."""
    coordinates, owners = [], []
    for owner, (polygon, step) in enumerate(zip(polygons, spacing)):
        x0, y0, x1, y1 = polygon.bounds
        dy = step * np.sqrt(3) / 2
        ys = np.arange(y0 + dy / 2, y1, dy)
        xs = np.arange(x0 + step / 2, x1, step)
        xx, yy = np.meshgrid(xs, ys)
        xx[1::2] += step / 2
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        xy = xy[shapely.covers(polygon, shapely.points(xy))]
        if len(xy) == 0:
            xy = shapely.get_coordinates(shapely.point_on_surface(polygon)).reshape(-1, 2)
        coordinates.append(xy)
        owners.append(np.full(len(xy), owner, dtype=np.int64))
    return np.concatenate(coordinates), np.concatenate(owners)


def _expand(geometries, spacing):
    """Expand multipart inputs; loops are over geometry types/polygons, never individual trees."""
    source = np.flatnonzero(np.isin(shapely.get_type_id(geometries), [0, 1, 3, 4, 5, 6]))
    skipped = len(geometries) - len(source)
    if skipped:
        logger.warning(f"{skipped} unsupported tree geometries are skipped")
    parts, parent = shapely.get_parts(np.array(geometries[source], dtype=object, copy=True), return_index=True)
    parent = source[parent]
    kind = shapely.get_type_id(parts)
    supported = np.isin(kind, [0, 1, 3]) & ~shapely.is_empty(parts)
    coordinates, owners = [], []
    for geometry_type in (0, 1, 3):
        selected = np.flatnonzero(supported & (kind == geometry_type))
        if len(selected) == 0:
            continue
        if geometry_type == 0:
            xy, owner = shapely.get_coordinates(parts[selected]), np.arange(len(selected))
        elif geometry_type == 1:
            xy, owner = _line_points(parts[selected], spacing[parent[selected]])
        else:
            xy, owner = _polygon_points(parts[selected], spacing[parent[selected]])
        coordinates.append(xy)
        owners.append(parent[selected[owner]])
    if not coordinates:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    return np.concatenate(coordinates), np.concatenate(owners)


def prepare_trees(scene, trees, footprints, height, diameter, base, transmittance, spacing):
    """Attach packed ``(x, y, z0/zb, r, c/h, mu_on, mu_off, shape)`` crown records to either scene mode."""
    if not isinstance(trees, gpd.GeoDataFrame) or trees.crs is None:
        raise ValueError("trees must be a GeoDataFrame with a CRS")
    rows = trees.to_crs(scene.local_crs)

    def numbers(column):
        return osm_numbers(rows[column]) if column in rows else np.full(len(rows), np.nan)

    heights = numbers("height")
    heights = np.where(np.isnan(heights), numbers("est_height"), heights)
    heights = np.where(np.isnan(heights), height, heights)
    diameters = numbers("diameter_crown")
    diameters = np.where(np.isnan(diameters), diameter, diameters)
    valid = (heights > 0) & (diameters > 0)
    skipped = int((~valid).sum())
    if skipped:
        logger.warning(f"{skipped} tree features with non-positive height or crown diameter are skipped")
    rows, heights, diameters = rows.iloc[np.flatnonzero(valid)], heights[valid], diameters[valid]
    steps = diameters if spacing is None else np.full(len(rows), spacing)
    xy, owner = _expand(rows.geometry.values, steps)
    if len(xy) and len(footprints):
        polygons = footprints.to_crs(scene.local_crs)
        polygons = polygons[polygons.geom_type.isin(["Polygon", "MultiPolygon"]) & ~polygons.is_empty]
        inside, _ = shapely.STRtree(polygons.geometry.values).query(shapely.points(xy), predicate="within")
        inside = np.unique(inside)
        if len(inside):
            logger.warning(f"{len(inside)} trees with trunks inside building footprints are skipped")
            keep = np.ones(len(xy), dtype=bool)
            keep[inside] = False
            xy, owner = xy[keep], owner[keep]
    leaf_type = rows.get("leaf_type", pd.Series("broadleaved", index=rows.index))
    conifer = leaf_type.fillna("").astype(str).str.strip().str.lower().to_numpy() == "needleleaved"
    cycle = rows.get("leaf_cycle", pd.Series("", index=rows.index))
    cycle = cycle.fillna("").astype(str).str.strip().str.lower().to_numpy()
    # semi_deciduous and semi_evergreen use the deciduous season; no intermediate optical tag is available.
    seasonal = (cycle == "deciduous") | np.char.startswith(cycle.astype(str), "semi_")
    evergreen = (cycle == "evergreen") | (~seasonal & conifer)
    shape = conifer[owner]
    crown_height = (1 - base) * heights[owner]
    ground = scene.ground_at(xy)
    bottom = ground + base * heights[owner]
    z = np.where(shape, bottom, bottom + crown_height / 2)
    mu_on = -np.log(transmittance[0]) / diameters[owner]
    mu_off = np.where(evergreen[owner], mu_on, -np.log(transmittance[1]) / diameters[owner])
    records = np.column_stack(
        [xy, z, diameters[owner] / 2, np.where(shape, crown_height, crown_height / 2), mu_on, mu_off, shape]
    )
    return scene._replace(trees=np.ascontiguousarray(records, dtype=float))
