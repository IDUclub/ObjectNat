"""Straight skeleton of polygons: centre lines and roof faces."""

from typing import Literal

import geopandas as gpd
import numpy as np
import shapely

from objectnat import config
from objectnat._rust import straight_skeletons  # pylint: disable=no-name-in-module
from objectnat.methods.utils._building_details import polygon_arrays

logger = config.logger

_REJECTED = {
    1: "empty",
    2: "a ring with fewer than three distinct points",
    3: "non-finite coordinates",
    4: "a ring of zero area",
    5: "crossing or touching edges",
    6: "a hole outside the exterior",
    7: "a hole inside another hole",
    8: "a numerical failure",
}


def get_straight_skeleton(
    gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    output: Literal["arcs", "faces"] = "arcs",
    include_bisectors: bool = True,
    roof_angle: float | None = None,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Calculate the straight skeleton of polygons: their centre lines or the faces of a hipped roof.

    The straight skeleton is what the boundary of a polygon traces when every edge moves inwards at the same speed.
    Each edge sweeps one face, the faces meet along the arcs of the skeleton, and every skeleton vertex lies at a
    known distance from the boundary. The arcs between inner vertices follow the middle of the polygon: the centre
    line of a road, river or corridor polygon, with half its width as the distance. The faces are the slopes of a
    hipped roof over a building outline. Polygons are processed in parallel by the compiled extension.

    Args:
        gdf (gpd.GeoDataFrame | gpd.GeoSeries):
            Polygons or MultiPolygons, in any CRS. Every part of a MultiPolygon gets its own skeleton; rows of other
            geometry types and empty geometries are skipped.

        output:
            What to return:

            - ``"arcs"``: one LineString per skeleton arc, with ``dist_start`` and ``dist_end``, the distance from
              the polygon boundary at the two ends of the arc;
            - ``"faces"``: one Polygon per polygon edge, with ``ring`` (0 for the exterior, then the holes in their
              order) and ``edge`` (the edge number within the ring, from its first point).

        include_bisectors:
            For ``output="arcs"``: also return the arcs that start at the polygon corners. ``False`` keeps only the
            arcs between inner vertices, the centre lines.

        roof_angle (float, optional):
            For ``output="faces"``: slope of the roof in degrees (between 0 and 90). The faces are then returned in
            3D, with the height of a point equal to its distance from the boundary times ``tan(roof_angle)``.

        max_workers (int, optional):
            Number of threads. Defaults to all cores.

    Returns:
        gpd.GeoDataFrame:
            Skeleton arcs or faces with the column ``part`` (the part of a MultiPolygon, 0 for a Polygon) and the
            columns described in ``output``. The index repeats the index of the source row. Returned in the CRS of
            ``gdf``; distances are in the units of a projected CRS, and in metres for a geographic CRS, which is
            projected to its UTM zone for the calculation.

    Raises:
        ValueError: If ``output``, ``roof_angle`` or ``max_workers`` is invalid.

    Notes:
        - Polygons with crossing or touching edges, zero-area rings or holes outside the exterior have no straight
          skeleton; they are skipped with a warning that counts them by reason.
        - The centre line of a strip stops half its width before the strip ends, where the corner arcs meet.
    """
    if output not in {"arcs", "faces"}:
        raise ValueError(f"Unsupported output={output!r}")
    if roof_angle is not None and (output != "faces" or not 0 < roof_angle < 90):
        raise ValueError(f"roof_angle must be between 0 and 90 and requires output='faces', got {roof_angle!r}")
    if max_workers is not None and max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    geometry = gdf.geometry if isinstance(gdf, gpd.GeoDataFrame) else gdf
    original_crs = geometry.crs
    local_crs = geometry.estimate_utm_crs() if original_crs is not None and original_crs.is_geographic else original_crs
    if local_crs != original_crs:
        geometry = geometry.to_crs(local_crs)

    parts, source = shapely.get_parts(geometry.to_numpy(), return_index=True)
    part = np.arange(len(source)) - np.searchsorted(source, source)
    polygonal = (shapely.get_type_id(parts) == shapely.GeometryType.POLYGON) & ~shapely.is_empty(parts)
    skipped = int(np.count_nonzero(~polygonal))
    if skipped:
        logger.warning(f"{skipped} non-polygon or empty geometries are skipped")
    parts, source, part = parts[polygonal], source[polygonal], part[polygonal]

    columns = ["part", "dist_start", "dist_end"] if output == "arcs" else ["part", "ring", "edge"]
    if len(parts) == 0:
        return gpd.GeoDataFrame(columns=columns, geometry=[], crs=original_crs)

    bounds = shapely.total_bounds(parts)
    shift = np.array([(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2])
    coords, rings, polygons = polygon_arrays(parts, shift)
    (
        vertices,
        is_input,
        arcs,
        arc_polygon,
        face_vertices,
        face_offsets,
        face_polygon,
        face_ring,
        face_edge,
        errors,
    ) = straight_skeletons(coords, rings, polygons, max_workers)

    rejected = np.bincount(errors, minlength=len(_REJECTED) + 1)[1:]
    if rejected.any():
        reasons = ", ".join(f"{count} with {_REJECTED[code + 1]}" for code, count in enumerate(rejected) if count)
        logger.warning(f"{int(rejected.sum())} polygons have no straight skeleton and are skipped: {reasons}")

    if output == "arcs":
        if not include_bisectors:
            inner = ~is_input[arcs[:, 0]] & ~is_input[arcs[:, 1]]
            arcs, arc_polygon = arcs[inner], arc_polygon[inner]
        geometries = shapely.linestrings(vertices[arcs][:, :, :2] + shift)
        owner = arc_polygon
        data = {"part": part[owner], "dist_start": vertices[arcs[:, 0], 2], "dist_end": vertices[arcs[:, 1], 2]}
    else:
        sizes = np.diff(face_offsets.astype(np.int64))
        # a zero-length edge sweeps a two-vertex face, which is not a polygon
        faces = np.flatnonzero(sizes >= 3)
        face_of_vertex = np.repeat(np.arange(len(sizes)), sizes)
        kept = np.isin(face_of_vertex, faces)
        index = face_vertices[kept]
        xy = vertices[index, :2] + shift
        if roof_angle is not None:
            xy = np.column_stack([xy, vertices[index, 2] * np.tan(np.radians(roof_angle))])
        rings_of_faces = shapely.linearrings(xy, indices=np.searchsorted(faces, face_of_vertex[kept]))
        geometries = shapely.polygons(rings_of_faces)
        owner = face_polygon[faces]
        data = {"part": part[owner], "ring": face_ring[faces], "edge": face_edge[faces]}

    result = gpd.GeoDataFrame(data, geometry=geometries, index=geometry.index[source[owner]], crs=local_crs)
    return result if local_crs == original_crs else result.to_crs(original_crs)
