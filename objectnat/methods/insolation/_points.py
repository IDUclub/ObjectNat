"""Conservative native point filters; every ambiguous predicate is resolved on the original GEOS geometry."""

import numpy as np
import shapely

from objectnat.methods.utils._building_details import polygon_arrays

# Kept private so tests can compare complete public results against the original predicates.
_USE_NATIVE_POINTS = True


def _polygon_parts(geometries):
    """Flatten polygon-only collections, preserving positional owners; unsupported inputs use GEOS."""
    geometries = np.asarray(geometries, dtype=object)
    if not shapely.is_valid(geometries).all():
        return None
    parts, owners = geometries, np.arange(len(geometries))
    while np.isin(shapely.get_type_id(parts), [6, 7]).any():
        parts, parent = shapely.get_parts(parts, return_index=True)
        owners = owners[parent]
    if (shapely.get_type_id(parts) != 3).any():
        return None
    nonempty = ~shapely.is_empty(parts)
    parts, owners = parts[nonempty], owners[nonempty]
    coordinates = shapely.get_coordinates(parts)
    # Extreme magnitudes cannot use the shared bbox grid safely; retain the original path.
    if not np.isfinite(coordinates).all() or (np.abs(coordinates) > 1e100).any():
        return None
    return parts, owners


def within_pairs(geometries, xy, *, gx=None, gy=None, max_workers=None):
    """Strict-interior (point, original row) pairs, with the legacy path available for reference tests."""
    geometries = np.asarray(geometries, dtype=object)
    parts = _polygon_parts(geometries) if _USE_NATIVE_POINTS else None
    if parts is None:
        return shapely.STRtree(geometries).query(shapely.points(xy), predicate="within")
    polygons, owners = parts
    # pylint: disable-next=import-outside-toplevel,no-name-in-module
    from objectnat._rust import classify_grid, classify_points

    arrays = polygon_arrays(polygons, np.zeros(2))
    if gx is None:
        inside, uncertain = classify_points(*arrays, np.ascontiguousarray(xy, dtype=float), n_threads=max_workers)
    else:
        _, (inside, uncertain) = classify_grid(*arrays, gx, gy, n_threads=max_workers)
    inside, uncertain = inside.copy(), uncertain.copy()
    inside[1] = owners[inside[1]]
    uncertain[1] = owners[uncertain[1]]
    if uncertain.shape[1]:
        # Construct each uncertain point once, even if several polygon parts reference it.
        point, inverse = np.unique(uncertain[0], return_inverse=True)
        points = shapely.points(xy[point])
        exact = shapely.within(points[inverse], geometries[uncertain[1]])
        inside = np.concatenate([inside, uncertain[:, exact]], axis=1)
    return inside


def grid_receivers(scene, area, gx, gy, receiver_height, max_workers=None):
    """Filter the original row-major coordinates without creating a GEOS point for every grid cell."""
    parts = _polygon_parts([area]) if _USE_NATIVE_POINTS else None
    if parts is None:
        return _legacy_grid_receivers(scene, area, gx, gy, receiver_height)
    # GEOS points store doubles, including when numpy inferred a wider grid-axis dtype.
    native_gx, native_gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
    if not np.isfinite(native_gx).all() or not np.isfinite(native_gy).all():
        return _legacy_grid_receivers(scene, area, gx, gy, receiver_height)
    gx, gy = native_gx, native_gy
    xy = np.column_stack([np.tile(gx, len(gy)), np.repeat(gy, len(gx))])
    # pylint: disable-next=import-outside-toplevel,no-name-in-module
    from objectnat._rust import classify_grid

    states, _ = classify_grid(*polygon_arrays(parts[0], np.zeros(2)), gx, gy, union=True, n_threads=max_workers)
    keep = states == 1
    uncertain = np.flatnonzero(states == 2)
    if len(uncertain):
        keep[uncertain] = shapely.covers(area, shapely.points(xy[uncertain]))
    if len(scene.blds):
        inside, volume = within_pairs(scene.blds.geometry.values, xy, gx=gx, gy=gy, max_workers=max_workers)
        if scene.volumes is not None:
            # Same coordinates, full ground sampling and bottom comparison as the legacy filter.
            heights = scene.ground_at(xy) + receiver_height
            inside = inside[(scene.base + scene.volumes["B"].to_numpy())[volume] <= heights[inside]]
        keep[inside] = False
    xy = xy[keep]
    return shapely.points(xy), xy


def _legacy_grid_receivers(scene, area, gx, gy, receiver_height):
    """Unchanged meshgrid/GEOS implementation, also used for non-polygonal or invalid areas."""
    xx, yy = np.meshgrid(gx, gy)
    receivers = shapely.points(xx.ravel(), yy.ravel())
    keep = shapely.covers(area, receivers)
    if len(scene.blds):
        inside, volume = shapely.STRtree(scene.blds.geometry.values).query(receivers, predicate="within")
        if scene.volumes is not None:
            heights = scene.ground_at(shapely.get_coordinates(receivers)) + receiver_height
            inside = inside[(scene.base + scene.volumes["B"].to_numpy())[volume] <= heights[inside]]
        keep[inside] = False
    receivers = receivers[keep]
    return receivers, shapely.get_coordinates(receivers)
