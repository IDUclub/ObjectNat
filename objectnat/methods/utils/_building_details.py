"""OSM Simple 3D Buildings volumes and the array interface to the roof mesher."""

import re

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from objectnat import config
from objectnat.methods.utils.scene import Scene, _reach_mask, sample_terrain

SHAPES = dict(
    enumerate(
        (
            "flat",
            "skillion",
            "gabled",
            "hipped",
            "pyramidal",
            "cone",
            "dome",
            "onion",
            "half-dome",
            "round",
            "half-hipped",
            "side_hipped",
            "quadruple_saltbox",
            "mansard",
            "gambrel",
            "saltbox",
            "many",
        )
    )
)
SHAPE_CODES = {name: code for code, name in SHAPES.items()}
CARDINALS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)")


def osm_number(value) -> float:
    """First numeric OSM value, accepting units, decimal commas and semicolon lists."""
    match = NUMBER.match(str(value))
    number = float(match[1].replace(",", ".")) if match else np.nan
    return number if np.isfinite(number) else np.nan


def osm_direction(value) -> float:
    """True downslope bearing in degrees, or NaN for missing/unrecognised values."""
    cardinal = str(value).strip().upper()
    return CARDINALS.index(cardinal) * 22.5 if cardinal in CARDINALS else osm_number(value)


def osm_numbers(values):
    """Parse a column with the scalar OSM number grammar and float conversion."""
    values = pd.Series(values, copy=False).reset_index(drop=True)
    result = np.full(len(values), np.nan)
    present = values.notna().to_numpy()
    # Series iteration yields Python floats; str/float round-trips preserve those values.
    # Object columns still stringify their original numpy scalars, just as osm_number does.
    if pd.api.types.is_float_dtype(values.dtype) or pd.api.types.is_integer_dtype(values.dtype):
        result = values.to_numpy(dtype=float, na_value=np.nan, copy=True)
    elif present.any():
        text = values[present].astype(str).str.extract(NUMBER, expand=False).str.replace(",", ".", regex=False)
        result[present] = text.astype(float).to_numpy()
    result[~np.isfinite(result)] = np.nan
    return result


def osm_directions(values):
    """Parse numeric and cardinal bearings with the scalar direction semantics."""
    values = pd.Series(values, copy=False).reset_index(drop=True)
    result = osm_numbers(values)
    cardinal = values.astype(str).str.strip().str.upper().map(dict(zip(CARDINALS, np.arange(16) * 22.5)))
    known = cardinal.notna().to_numpy()
    result[known] = cardinal.to_numpy()[known]
    return result


def polygon_arrays(polygons, shift):
    """Flatten polygon exteriors and holes to the Rust ring/polygon offset contract."""
    polygons = np.array(polygons, dtype=object, copy=True)
    ring_geometries, owners = shapely.get_rings(polygons, return_index=True)
    coords, ring_owner = shapely.get_coordinates(ring_geometries, return_index=True)
    # Each ring is closed. Remove only its final coordinate, including empty-ring handling.
    counts = np.bincount(ring_owner, minlength=len(ring_geometries))
    ends = np.cumsum(counts)
    keep = np.ones(len(coords), dtype=bool)
    keep[ends[counts > 0] - 1] = False
    coords = coords[keep] - shift
    # Preserve the empty exterior entry of an empty Polygon even if get_rings omits it.
    polys = np.r_[0, np.cumsum(1 + shapely.get_num_interior_rings(polygons))]
    actual_counts = np.bincount(owners, minlength=len(polygons))
    local_ring = np.arange(len(owners)) - np.repeat(np.cumsum(actual_counts) - actual_counts, actual_counts)
    ring_sizes = np.zeros(polys[-1], dtype=np.int64)
    ring_sizes[polys[owners] + local_ring] = np.maximum(counts - 1, 0)
    rings = np.r_[0, np.cumsum(ring_sizes)]
    return (
        np.asarray(coords, dtype=float).reshape(-1, 2),
        np.asarray(rings, dtype=np.int64),
        np.asarray(polys, dtype=np.int64),
    )


def _building_parents(blds):
    """Return positional group IDs and outlines replaced by parts, without changing row order."""
    is_part = blds["building:part"].notna().to_numpy() if "building:part" in blds else np.zeros(len(blds), bool)
    outlines, parts = np.flatnonzero(~is_part), np.flatnonzero(is_part)
    parent = np.arange(len(blds))
    dropped = np.zeros(len(blds), bool)
    if len(outlines) and len(parts):
        part_idx, outline_idx = shapely.STRtree(blds.geometry.values[outlines]).query(
            blds.geometry.values[parts].representative_point(), predicate="within"
        )
        dropped[outlines[outline_idx]] = True
        # Largest first, so the smallest containing outline wins; ties prefer the first input row.
        order = np.lexsort((-outlines[outline_idx], -blds.geometry.values[outlines[outline_idx]].area))
        # Select the last assignment per part explicitly; repeated advanced-index writes
        # alone do not guarantee the winner.
        _, reverse_first = np.unique(part_idx[order][::-1], return_index=True)
        winners = order[::-1][reverse_first]
        parent[parts[part_idx[winners]]] = outlines[outline_idx[winners]]
    return parent, dropped


def _volume_rows(blds, *, return_groups=False):
    """Replace outlines by parts plus substantial flat remainders in the projected scene frame."""
    parent, dropped = _building_parents(blds)
    geometries = blds.geometry.values
    assigned = np.flatnonzero(parent != np.arange(len(blds)))
    order = assigned[np.argsort(parent[assigned], kind="stable")]
    groups, starts = np.unique(parent[order], return_index=True)
    unions = {
        group: shapely.union_all(geometries[indices]) for group, indices in zip(groups, np.split(order, starts[1:]))
    }
    outlines = np.flatnonzero(dropped)
    covered = np.asarray([unions.get(outline, shapely.Polygon()) for outline in outlines], dtype=object)
    pieces, owners = shapely.get_parts(shapely.difference(geometries[outlines], covered), return_index=True)
    polygons = shapely.get_type_id(pieces) == 3
    pieces, owners = pieces[polygons], owners[polygons]
    # The scene already projected these geometries to metres. Test opened pieces,
    # but retain their original boundaries so substantial remainders fill the outline.
    opened = shapely.buffer(shapely.buffer(pieces, -0.5), 0.5)
    substantial = shapely.area(opened) >= 1.0
    remainder_pieces, remainder_owners = pieces[substantial], outlines[owners[substantial]]
    rows = blds.iloc[np.flatnonzero(~dropped)].copy()
    row_parents = parent[~dropped]
    if len(remainder_owners):
        extra = blds.iloc[remainder_owners].copy()
        extra.geometry = remainder_pieces
        for tag in ("shape", "height", "levels", "angle", "direction", "orientation"):
            if f"roof:{tag}" in extra:
                extra[f"roof:{tag}"] = None
        rows = pd.concat([rows, extra])
        row_parents = np.r_[row_parents, remainder_owners]
    labels = np.empty(len(blds), dtype=object)
    labels[:] = list(blds.index)
    rows["parent"] = labels[row_parents]
    rows["parent_geometry"] = list(geometries[row_parents])
    for tag in ("height", "building:levels"):
        values = blds[tag].to_numpy() if tag in blds else np.full(len(blds), np.nan)
        rows[f"_parent_{tag}"] = values[row_parents]
    if return_groups:
        group_rows = rows[[rows.geometry.name]].copy()
        group_rows.index = row_parents
        return rows.explode(index_parts=False), group_rows.explode(index_parts=False).index.to_numpy()
    return rows.explode(index_parts=False)


def _heights(rows, height_column, storeys_column, storey_height, default_height, default_roof_angle):
    def numbers(column):
        return osm_numbers(rows[column]) if column in rows else np.full(len(rows), np.nan)

    bottom = numbers("min_height")
    bottom = np.nan_to_num(np.where(np.isnan(bottom), numbers("building:min_level") * storey_height, bottom))
    shape_values = pd.Series(rows.get("roof:shape", [None] * len(rows)), copy=False)
    shapes = shape_values.astype(str).str.strip().str.lower().map(SHAPE_CODES).fillna(255)
    shapes = shapes.where(shape_values.notna(), 0).to_numpy(dtype=np.uint8)
    roof = numbers("roof:height")
    roof = np.where(np.isnan(roof), numbers("roof:levels") * storey_height, roof)
    rectangles = shapely.minimum_rotated_rectangle(rows.geometry.values)
    coords, rectangle_owner = shapely.get_coordinates(np.array(rectangles, dtype=object, copy=True), return_index=True)
    lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    same_rectangle = rectangle_owner[:-1] == rectangle_owner[1:]
    widths = np.full(len(rows), np.inf)
    np.minimum.at(widths, rectangle_owner[:-1][same_rectangle], lengths[same_rectangle])
    widths[(shapely.get_type_id(rectangles) != 3) | ~np.isfinite(widths)] = 0.0
    width = widths * np.where(shapes == 1, 1.0, 0.5)
    angle = numbers("roof:angle")
    angle = np.where(np.isfinite(angle) & (angle >= 0) & (angle < 90), angle, default_roof_angle)
    inferred = np.where((shapes >= 1) & (shapes <= 15), np.tan(np.radians(angle)) * width, 0.0)
    roof = np.maximum(np.where(np.isnan(roof), inferred, roof), 0)
    top = numbers("height")
    levels = numbers("building:levels")
    inherit = np.isnan(top) & np.isnan(levels)
    top = np.where(inherit, numbers("_parent_height"), top)
    levels = np.where(inherit, numbers("_parent_building:levels"), levels)
    eave = np.where(np.isnan(top) & np.isfinite(levels), levels * storey_height, top - roof)
    fallback = numbers(height_column)
    if storeys_column is not None:
        if storeys_column not in rows:
            raise ValueError(f"buildings have no column {storeys_column!r}")
        fallback = np.where(np.isnan(fallback), numbers(storeys_column) * storey_height, fallback)
    fallback = np.where(fallback > 0, fallback, np.nan)
    if default_height is not None:
        fallback = np.where(np.isnan(fallback), float(default_height), fallback)
    eave = np.where(np.isnan(eave), fallback - roof, eave)
    top = np.where(np.isnan(top), eave + roof, top)
    eave = np.minimum(np.maximum(eave, bottom), top)
    return bottom, eave, top, shapes


def _boundary_pieces(polygons, edges, owner):
    """Identify boundary subdivision edges and recover outward normals regardless of mesh winding."""
    a, b = edges[:, :2], edges[:, 3:5]
    vec = b - a
    length = np.linalg.norm(vec, axis=1)
    # Buffer once per footprint, then run all edge predicates in Shapely's vectorised loops.
    boundaries = shapely.buffer(shapely.boundary(polygons), 1e-7, quad_segs=16)
    boundary = (length > 1e-6) & shapely.covers(boundaries[owner], shapely.linestrings(np.stack([a, b], axis=1)))
    indices = np.flatnonzero(boundary)
    normals = vec[indices][:, [1, 0]] * [1, -1] / length[indices, None]
    probe = (a[indices] + b[indices]) / 2 + normals * 1e-5
    inward = shapely.contains_xy(polygons[owner[indices]], probe[:, 0], probe[:, 1])
    normals[inward] *= -1
    return indices, normals


def _facade_boundaries(scene, targets=None):
    # Detailed ground insolation never needs boundary pieces. Explicit arrays still support hand-built scenes.
    if scene.boundary_walls is not None:
        walls, normals = scene.boundary_walls, scene.boundary_normals
        if targets is not None:
            selected = np.isin(scene.owner[walls], targets)
            walls, normals = walls[selected], normals[selected]
        return walls, normals
    indices = np.arange(len(scene.owner)) if targets is None else np.flatnonzero(np.isin(scene.owner, targets))
    owners, local_owner = np.unique(scene.owner[indices], return_inverse=True)
    segments, tops = scene.segments[indices], scene.tops[indices]
    edges = np.column_stack([segments[:, :2], tops[:, 0], segments[:, 2:], tops[:, 1]])
    walls, normals = _boundary_pieces(scene.volumes.geometry.values[owners], edges, local_owner)
    return indices[walls], normals


def _grid_north_azimuths(local_crs, lon, lat):
    # Keep the scalar scene helper (and the legacy path) unchanged; one Transformer for the whole batch.
    to_local = Transformer.from_crs(4326, local_crs, always_xy=True)
    (xa, xb), (ya, yb) = to_local.transform(np.stack([lon, lon]), np.stack([lat, lat + 0.001]))
    return np.degrees(np.arctan2(xb - xa, yb - ya))


def _exposed_undersides(polygons, bottom, eave, raised):
    """Retain only raised bottoms not covered by supporting volumes, using absolute heights."""
    bottom, eave = np.asarray(bottom), np.asarray(eave)
    keep = raised.copy()
    indices = np.flatnonzero(raised)
    if indices.size == 0:
        return keep
    target, support = shapely.STRtree(polygons).query(polygons[indices], predicate="intersects")
    valid = (bottom[support] < bottom[indices[target]]) & (eave[support] >= bottom[indices[target]])
    target, support = target[valid], support[valid]
    if len(target) == 0:
        return keep
    # Even the sum of full support areas bounds the area covered inside a target. Most small
    # supports can be rejected here without constructing any pairwise intersection geometries.
    all_areas = shapely.area(polygons)
    areas = all_areas[indices]
    checked = np.unique(np.r_[indices, support])
    valid_area = np.zeros(len(polygons), dtype=bool)
    valid_area[checked] = shapely.is_valid(polygons[checked]) & np.isfinite(all_areas[checked])
    bound_valid = valid_area[indices].copy()
    np.logical_and.at(bound_valid, target, valid_area[support])
    support_area = np.zeros(len(indices))
    np.add.at(support_area, target, all_areas[support])
    safely_exposed = bound_valid & (support_area < (1 - 1e-6 - 1e-9) * areas)
    selected = ~safely_exposed[target]
    target, support = target[selected], support[selected]
    targets = polygons[indices[target]]
    covered_pair = shapely.covers(polygons[support], targets)
    covered_target = np.zeros(len(indices), dtype=bool)
    np.logical_or.at(covered_target, target, covered_pair)
    keep[indices[covered_target]] = False
    # Union area cannot exceed the sum of pairwise intersection areas. Leave a generous
    # rounding band around the threshold to the original union/difference decision.
    selected = ~(covered_target[target] & bound_valid[target])
    target, support, targets = target[selected], support[selected], targets[selected]
    overlap = np.zeros(len(indices))
    np.add.at(overlap, target, shapely.area(shapely.intersection(targets, polygons[support])))
    safely_exposed |= overlap < (1 - 1e-6 - 1e-9) * areas
    unresolved = ~(covered_target | safely_exposed)
    selected = unresolved[target]
    target, support = target[selected], support[selected]
    order = np.argsort(target, kind="stable")
    target, support = target[order], support[order]
    starts = np.r_[0, np.flatnonzero(np.diff(target)) + 1, len(target)]
    for start, end in zip(starts[:-1], starts[1:]):
        if start == end:
            continue
        i = indices[target[start]]
        covered = shapely.union_all(polygons[support[start:end]])
        # A ray reaching B from below inside a supported footprint already entered a lower volume.
        # Its walls/roof edges detect that entry; receivers are never inside volumes. E >= B is a
        # conservative roof bound. Strictly lower bottoms also prevent circular removal in stacks.
        if shapely.area(shapely.difference(polygons[i], covered)) <= 1e-6 * shapely.area(polygons[i]):
            keep[i] = False
    return keep


def detailed_scene(
    local_crs,
    blds,
    terrain_grid,
    height_column,
    storeys_column,
    storey_height,
    default_height,
    default_roof_angle,
    *,
    reach=None,
    facade_reach=False,
):
    """Build one batched mesh with endpoint tops, raised bottoms and flat undersides.

    Parameters
    ----------
    reach : tuple or None, keyword-only
        Local receiver region and minimum solar tangent; None retains all volumes.
    facade_reach : bool, keyword-only
        Expand the receiver region to complete target footprints and their window offset.
    """
    from objectnat._rust import roof_mesh  # pylint: disable=import-outside-toplevel,no-name-in-module

    if not np.isfinite(storey_height) or storey_height <= 0:
        raise ValueError("storey_height must be positive and finite")
    if not np.isfinite(default_roof_angle) or not 0 <= default_roof_angle < 90:
        raise ValueError("default_roof_angle must be finite and in [0, 90)")
    # Carry positional groups separately from labels (input labels need not be unique).
    rows, volume_groups = _volume_rows(blds, return_groups=True)
    bottom, eave, top, shapes = _heights(
        rows, height_column, storeys_column, storey_height, default_height, default_roof_angle
    )
    ok = np.isfinite(top) & (top > bottom)
    skipped = int((~ok).sum())
    if skipped:
        config.logger.warning(f"{skipped} building volumes without a height above their bottom are skipped")
    rows = rows.iloc[np.flatnonzero(ok)].copy()
    bottom, eave, top, shapes = bottom[ok], eave[ok], top[ok], shapes[ok]
    volume_groups = volume_groups[ok]
    points = shapely.get_coordinates(rows.representative_point().values)
    base = (
        np.nan_to_num(sample_terrain(terrain_grid, points), nan=0.0)
        if terrain_grid is not None
        else np.zeros(len(rows))
    )
    shift = points.min(axis=0) if len(points) else np.zeros(2)
    directions = osm_directions(rows.get("roof:direction", [None] * len(rows)))
    directed = np.isfinite(directions)
    if directed.any():
        lon, lat = Transformer.from_crs(local_crs, 4326, always_xy=True).transform(
            points[directed, 0], points[directed, 1]
        )
        directions[directed] -= _grid_north_azimuths(local_crs, lon, lat)
    across = (
        pd.Series(rows.get("roof:orientation", [None] * len(rows)), copy=False)
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("across")
        .to_numpy(dtype=bool)
    )
    keep = _reach_mask(
        rows,
        base + top,
        base,
        terrain_grid,
        reach,
        groups=volume_groups,
        parent_geometries=rows["parent_geometry"].to_numpy(),
        facade=facade_reach,
    )
    # Query the original support tree, in its original order, even for supports later dropped.
    exposed = _exposed_undersides(rows.geometry.values, base + bottom, base + eave, (bottom > 0) & keep)
    rows = rows.iloc[np.flatnonzero(keep)].copy()
    bottom, eave, top, shapes, base = bottom[keep], eave[keep], top[keep], shapes[keep], base[keep]
    directions, across, exposed = directions[keep], across[keep], exposed[keep]
    (xyz, offsets), face_owner, edges, owner, status, used = roof_mesh(
        *polygon_arrays(rows.geometry, shift),
        shapes,
        base + eave,
        base + top,
        direction=np.radians(directions),
        across=across,
    )
    edges[:, :2] += shift
    edges[:, 3:5] += shift
    xyz[:, :2] += shift
    fallbacks = int((status == 1).sum())
    if fallbacks:
        config.logger.warning(f"{fallbacks} building roofs use an approximation or fallback shape")
    rows["B"], rows["E"], rows["H"] = bottom, eave, top
    rows["base"], rows["used_shape"], rows["status"] = base, used, status
    raised = bottom > 0
    config.logger.debug(f"Building undersides: {int(exposed.sum())} retained of {int(raised.sum())} raised volumes")
    under = (*polygon_arrays(rows.geometry.values[exposed], np.zeros(2)), (base + bottom)[exposed])
    return Scene(
        local_crs,
        rows,
        top,
        base,
        terrain_grid,
        np.ascontiguousarray(edges[:, [0, 1, 3, 4]]),
        np.ascontiguousarray(edges[:, [2, 5]]),
        owner,
        volumes=rows,
        bottoms=np.where(bottom[owner] > 0, (base + bottom)[owner], -np.inf),
        undersides=under,
        roof_faces=(xyz, offsets, face_owner),
    )
