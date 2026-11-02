import math

import geopandas as gpd
import numpy as np
import shapely
from shapely import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from objectnat import config

logger = config.logger


POLYGON_TYPE_ID = 3


def clip_polygons(geometries, mask) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised ``GeoDataFrame.clip(mask, keep_geom_type=True).explode()`` for polygonal geometries.

    Returns the polygon parts of ``geometry ∩ mask`` (collections and multi-parts flattened, lines and
    points dropped) and, for each part, the position of the input geometry it comes from.
    """
    geometries = np.asarray(geometries, dtype=object)
    if geometries.size == 0 or mask is None or mask.is_empty:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)
    clipped = shapely.intersection(geometries, mask)
    parts, owner = shapely.get_parts(clipped, return_index=True)
    # collections may hold multi-parts: flatten once more
    parts, inner = shapely.get_parts(parts, return_index=True)
    owner = np.asarray(owner, dtype=np.int64)[inner]
    keep = (shapely.get_type_id(parts) == POLYGON_TYPE_ID) & ~shapely.is_empty(parts)
    return parts[keep], owner[keep]


def polygons_to_multilinestring(geom: Polygon | MultiPolygon) -> MultiLineString:

    def polygon_to_lines(polygon: Polygon) -> list[LineString]:
        return [LineString(polygon.exterior), *(LineString(interior) for interior in polygon.interiors)]

    if geom.geom_type == "Polygon":
        return MultiLineString(polygon_to_lines(geom))
    if geom.geom_type == "MultiPolygon":
        return MultiLineString([line for polygon in geom.geoms for line in polygon_to_lines(polygon)])
    raise TypeError(f"Expected Polygon or MultiPolygon, got {geom.geom_type}")


def get_point_from_a_thorough_b(a: Point, b: Point, dist):
    """
    Func to get Point from point a thorough point b on dist
    """
    direction = math.atan2(b.y - a.y, b.x - a.x)
    c_x = a.x + dist * math.cos(direction)
    c_y = a.y + dist * math.sin(direction)
    return Point(c_x, c_y)


def gdf_to_circle_zones_from_point(
    gdf: gpd.GeoDataFrame, point_from: Point, zone_radius, resolution=4, explode_multigeom=True
) -> gpd.GeoDataFrame:
    """n_segments = 4*resolution,e.g. if resolution = 4 that means there will be 16 segments"""
    crs = gdf.crs
    buffer = point_from.buffer(zone_radius, quad_segs=resolution)
    parts, _ = clip_polygons(gdf.geometry.values, buffer)
    gdf_unary = shapely.union_all(parts)
    if gdf_unary.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    boundaries = shapely.boundary(shapely.get_parts(gdf_unary))
    zones_lines = [LineString([Point(coords1), Point(point_from)]) for coords1 in buffer.exterior.coords[:-1]]
    zones = np.asarray(list(polygonize(unary_union(list(boundaries) + zones_lines))), dtype=object)
    if explode_multigeom:
        zone_parts, _ = clip_polygons(zones, gdf_unary)
        return gpd.GeoDataFrame(geometry=zone_parts, crs=crs)
    clipped = shapely.intersection(zones, gdf_unary)
    clipped = clipped[~shapely.is_empty(clipped)]
    return gpd.GeoDataFrame(geometry=clipped, crs=crs)


def remove_inner_geom(polygon: Polygon | MultiPolygon):
    """function to get rid of inner polygons"""
    if isinstance(polygon, Polygon):
        return Polygon(polygon.exterior.coords)
    if isinstance(polygon, MultiPolygon):
        polys = []
        for poly in polygon.geoms:
            polys.append(Polygon(poly.exterior.coords))
        return MultiPolygon(polys)
    return Polygon()


def combine_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Combine geometry of intersecting layers into a single GeoDataFrame.
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        A GeoPandas GeoDataFrame

    Returns
    -------
    gpd.GeoDataFrame
        The combined GeoDataFrame with aggregated in lists columns.

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> result = combine_geometry(gdf)
    """

    crs = gdf.crs

    enclosures = gpd.GeoDataFrame(
        geometry=list(polygonize(gdf["geometry"].apply(polygons_to_multilinestring).union_all())), crs=crs
    )
    enclosures_points = enclosures.copy()
    enclosures_points.geometry = enclosures.representative_point()
    joined = gpd.sjoin(enclosures_points, gdf, how="inner", predicate="within").reset_index()
    cols = joined.columns.tolist()
    cols.remove("geometry")
    joined = joined.groupby("index").agg({column: list for column in cols})
    joined["geometry"] = enclosures
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=crs)
    return joined


def distribute_points_on_linestrings(lines: gpd.GeoDataFrame, radius, lloyd_relax_n=2) -> gpd.GeoDataFrame:
    lines = lines.copy()
    lines = lines.explode(ignore_index=True)
    lines = lines[lines.geom_type == "LineString"]
    original_crs = lines.crs
    lines = lines.to_crs(crs=lines.estimate_utm_crs())
    lines = lines.reset_index(drop=True)
    lines = lines[["geometry"]]
    radius = radius * 1.1
    segmentized = lines.geometry.apply(lambda x: x.simplify(radius).segmentize(radius))
    points = [Point(pt) for line in segmentized for pt in line.coords]

    points = gpd.GeoDataFrame(geometry=points, crs=lines.crs)
    lines["lines"] = lines.geometry
    geom_concave = lines.buffer(5, resolution=1).union_all()

    for _ in range(lloyd_relax_n):
        points.geometry = points.voronoi_polygons().clip(geom_concave).centroid
        points = points.sjoin_nearest(lines, how="left")
        points = points[~points.index.duplicated(keep="first")]
        points["geometry"] = points["lines"].interpolate(points["lines"].project(points.geometry))
        points.drop(columns=["lines", "index_right"], inplace=True)

    return points.dropna().to_crs(original_crs)


def distribute_points_on_polygons(
    polygons: gpd.GeoDataFrame, radius, only_exterior=True, lloyd_relax_n=2
) -> gpd.GeoDataFrame:
    polygons = polygons.copy()
    polygons = polygons.explode(ignore_index=True)
    polygons = polygons[polygons.geom_type == "Polygon"]

    if only_exterior:
        polygons.geometry = polygons.geometry.apply(lambda x: LineString(x.exterior))
    else:
        polygons = gpd.GeoDataFrame(
            geometry=list(polygons.geometry.apply(polygons_to_multilinestring)), crs=polygons.crs
        )

    return distribute_points_on_linestrings(polygons, radius, lloyd_relax_n=lloyd_relax_n)
