import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely import Point, Polygon

from objectnat.methods.utils.raster import grid_attrs
from objectnat.methods.visibility._native import geometries_to_segments, native_visibility_polygons
from objectnat.methods.visibility.isovist import ISOVIST_METRICS, add_isovist_metrics

VISIBILITY_METHODS = ("accurate", "simple")
VIEW_DISC_QUAD_SEGS = 16  # matches Point.buffer(): the view polygon the kernel clips to


def dist_to_furthest_point(point_from: Point, view_polygon: Polygon) -> float:
    """
    Compute the maximum distance from an observer point to the boundary of a visibility polygon.

    The function measures the Euclidean distance from the observer point to all
    vertices of the polygon's exterior ring and returns the maximum distance,
    rounded to one decimal place.

    Args:
        point_from (Point):
            Observer location in the same CRS as ``view_polygon``.
        view_polygon (Polygon):
            Polygon representing the visibility area or any polygonal geometry
            around the observer.

    Returns:
        float:
            Maximum distance from ``point_from`` to the vertices of
            ``view_polygon``, rounded to one decimal place.
    """
    coords = np.asarray(view_polygon.exterior.coords, dtype="float64")
    dx = coords[:, 0] - point_from.x
    dy = coords[:, 1] - point_from.y
    res = float(np.sqrt(dx * dx + dy * dy).max())
    return round(res, 1)


def visibility_polygon(point_from: Point, obstacles: gpd.GeoDataFrame, view_distance: float) -> Polygon:
    """Exact visibility polygon of one observer among ``obstacles``, both in the same metric CRS."""
    point_buffer = point_from.buffer(view_distance)
    idx = obstacles.sindex.query(point_buffer, predicate="intersects")
    if len(idx) == 0:
        return point_buffer
    segments, one_sided = geometries_to_segments(obstacles.geometry.values[idx])
    xy = np.array([[point_from.x, point_from.y]])
    return native_visibility_polygons(xy, np.array([float(view_distance)]), segments, one_sided, n_threads=1)[0]


def _visibility_accurate_batch(
    points: gpd.GeoSeries, obstacles: gpd.GeoDataFrame, distances: np.ndarray, max_workers: int | None
) -> np.ndarray:
    """
    Visibility polygons for many observers at once (multi-threaded in Rust).

    Observers inside an obstacle get an empty polygon. ``max_workers`` limits the number of threads
    (``None`` uses all cores).
    """
    xy = shapely.get_coordinates(points.values)
    segments, one_sided = geometries_to_segments(obstacles.geometry.values)
    results = native_visibility_polygons(xy, distances, segments, one_sided, n_threads=max_workers)
    inside, _ = obstacles.sindex.query(points.values, predicate="within")
    if len(inside):
        results[np.unique(inside)] = Polygon()
    return results


def get_visibility(
    point_from: gpd.GeoDataFrame,
    obstacles: gpd.GeoDataFrame,
    view_distance: float | None = None,
    method: Literal["accurate", "simple"] = "accurate",
    *,
    resolution: int = 32,  # pylint: disable=unused-argument
    parallel: bool = False,  # pylint: disable=unused-argument
    max_workers: int | None = None,
    isovist_metrics: bool = False,
) -> gpd.GeoDataFrame:
    """
    Compute visibility polygons for one or many observer points.

    Each observer gets the exact area visible within its viewing radius: the view disc minus everything
    hidden behind obstacle walls. Points and obstacles are reprojected to a locally estimated projected
    CRS (typically UTM) for the calculation, and observers are processed in parallel by ObjectNat's Rust
    extension.

    Args:
        point_from (gpd.GeoDataFrame):
            GeoDataFrame with point geometries representing observer locations.
            Must have a valid CRS. Any additional attributes are preserved in
            the output.
        obstacles (gpd.GeoDataFrame):
            GeoDataFrame with obstacle geometries (Polygon, MultiPolygon, LineString,
            MultiLineString) in any CRS. If empty or ``None``, no occlusion is
            applied and visibility is limited only by the viewing radius.
        view_distance (float | None, optional):
            Global viewing radius for all points (in metres). Used when the
            ``"visibility_distance"`` column is not present in ``point_from``.
            If ``None`` and the column is also missing, a ``ValueError`` is raised.
        method (Literal["accurate", "simple"], optional):
            Deprecated and ignored: the result is always exact.
        resolution (int, optional):
            Deprecated and ignored.
        parallel (bool, optional):
            Deprecated and ignored: the calculation is always multi-threaded.
        max_workers (int | None, optional):
            Maximum number of threads (``None`` uses all cores).
        isovist_metrics (bool, optional):
            If ``True``, add isovist metric columns to the result, measured in metres in the local CRS:
            ``area``, ``perimeter``, ``compactness`` (4πA/P², 1 for the full disc), ``occlusivity``
            (length of the occluding radial edges), ``n_occlusions``, ``min_radial``, ``max_radial``,
            ``mean_radial``, ``std_radial``, ``skew_radial`` (moments of the distance to the boundary over
            all directions, integrated exactly), ``drift`` (distance from the observer to the centroid),
            ``view_share`` (area / area of the view disc) and ``open_share`` (share of the perimeter on the
            view-distance limit). Observers inside an obstacle get NaN.

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with the same index and attributes as ``point_from``,
            but with the geometry column replaced by visibility polygons. The
            result is returned in the original CRS of ``point_from``. Observers
            inside an obstacle get an empty polygon.

    Raises:
        TypeError:
            If ``point_from`` is not a GeoDataFrame.
        ValueError:
            If ``method`` is not ``"accurate"`` or ``"simple"``, if
            ``point_from`` is empty, has no CRS, or if neither
            ``view_distance`` nor the ``"visibility_distance"`` column is
            provided.

    Notes:
        If a column named ``"visibility_distance"`` is present in ``point_from``,
        its values are used as per-point view distances and the ``view_distance``
        argument is ignored.
    """
    if not isinstance(point_from, gpd.GeoDataFrame):
        raise TypeError("point_from must be a GeoDataFrame with a point geometry")

    if point_from.empty:
        raise ValueError("GeoDataFrame 'point_from' is empty.")

    if method not in VISIBILITY_METHODS:
        raise ValueError(f"method must be one of: {', '.join(repr(m) for m in VISIBILITY_METHODS)}; got {method!r}")
    if method == "simple":
        warnings.warn(
            "method='simple' is deprecated and computes the exact visibility; omit the argument.",
            DeprecationWarning,
            stacklevel=2,
        )

    original_crs = point_from.crs
    local_crs = point_from.estimate_utm_crs()
    points_local = point_from.to_crs(local_crs)

    if "visibility_distance" in points_local.columns:
        distances = points_local["visibility_distance"].to_numpy(dtype="float64")
    else:
        if view_distance is None:
            raise ValueError(
                "Either provide parameter view_distance or add column 'visibility_distance' to point_from GeoDataFrame."
            )
        distances = np.full(len(points_local), float(view_distance))

    obstacles_local = None
    if obstacles is not None and len(obstacles) > 0:
        obstacles_local = obstacles.to_crs(local_crs)
        allowed_geom_types = ["MultiPolygon", "Polygon", "LineString", "MultiLineString"]
        obstacles_local = obstacles_local[obstacles_local.geom_type.isin(allowed_geom_types)]

    if obstacles_local is None or obstacles_local.empty:
        results = shapely.buffer(points_local.geometry.values, distances, quad_segs=VIEW_DISC_QUAD_SEGS)
    else:
        results = _visibility_accurate_batch(points_local.geometry, obstacles_local, distances, max_workers)

    result_gdf = points_local.copy()
    result_gdf["geometry"] = results
    result_gdf.set_crs(local_crs, inplace=True)
    if isovist_metrics:
        add_isovist_metrics(result_gdf, points_local.geometry, distances)
    return result_gdf.to_crs(original_crs)


def get_isovist_grid(
    obstacles: gpd.GeoDataFrame | None,
    area: gpd.GeoDataFrame | gpd.GeoSeries | shapely.Geometry,
    *,
    view_distance: float = 200.0,
    cell_size: float = 5.0,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Isovist metrics on a regular grid of observers covering the open ground of an area (an isovist field).

    An observer is placed at the centre of every ``cell_size`` cell of ``area`` outside the obstacles, and its
    exact visibility polygon within ``view_distance`` is measured as with ``get_visibility(...,
    isovist_metrics=True)``. Obstacles outside the area but within ``view_distance`` of it still block the view.
    :func:`rasterize` turns any metric into a :class:`Raster`, which shows how openness, compactness or
    occlusion change across the area.

    Args:
        obstacles (gpd.GeoDataFrame | None):
            Obstacle geometries (Polygon, MultiPolygon, LineString, MultiLineString) in any CRS; ``None`` or
            empty for open ground.
        area (gpd.GeoDataFrame | gpd.GeoSeries | shapely.Geometry):
            Area to cover. A shapely geometry is taken in the CRS of ``obstacles``.
        view_distance (float, optional):
            Viewing radius of every observer in metres. Defaults to 200.
        cell_size (float, optional):
            Observer spacing in metres. Time grows with ``(area / cell_size²) * view_distance``. Defaults to 5.
        max_workers (int | None, optional):
            Number of threads (``None`` uses all cores).

    Returns:
        gpd.GeoDataFrame:
            One observer point per open cell, in the CRS of ``area`` (of ``obstacles`` for a shapely geometry),
            with the isovist metric columns of :func:`get_visibility`: ``area``, ``perimeter``,
            ``compactness``, ``occlusivity``, ``n_occlusions``, ``min_radial``, ``max_radial``,
            ``mean_radial``, ``std_radial``, ``skew_radial``, ``drift``, ``view_share`` and ``open_share``.
            ``attrs`` record the grid for :func:`rasterize`.

    Raises:
        ValueError: If ``area`` has no CRS or is empty, or ``view_distance`` or ``cell_size`` is not positive.
    """
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError(f"cell_size must be a positive number; got {cell_size!r}")
    if not np.isfinite(view_distance) or view_distance <= 0:
        raise ValueError(f"view_distance must be a positive number; got {view_distance!r}")
    has_obstacles = obstacles is not None and len(obstacles) > 0
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        area_series = gpd.GeoSeries(area.geometry.values, crs=area.crs)
    else:
        if not has_obstacles or obstacles.crs is None:
            raise ValueError("A shapely area needs obstacles with a CRS; pass area as a GeoDataFrame instead")
        area_series = gpd.GeoSeries([area], crs=obstacles.crs)
    if area_series.crs is None:
        raise ValueError("area must have a CRS")
    local_crs = area_series.estimate_utm_crs()
    area_geom = area_series.to_crs(local_crs).union_all()
    if area_geom.is_empty:
        raise ValueError("area is empty")

    x0, y0, x1, y1 = area_geom.bounds
    gx = np.arange(np.floor(x0 / cell_size) * cell_size + cell_size / 2, x1, cell_size)
    gy = np.arange(np.floor(y0 / cell_size) * cell_size + cell_size / 2, y1, cell_size)
    xx, yy = np.meshgrid(gx, gy)
    points = shapely.points(xx.ravel(), yy.ravel())
    keep = shapely.covers(area_geom, points)
    scene = None
    if has_obstacles:
        local_obstacles = obstacles.to_crs(local_crs)
        scene = local_obstacles.iloc[
            local_obstacles.sindex.query(area_geom.buffer(view_distance), predicate="intersects")
        ]
        inside, _ = shapely.STRtree(scene.geometry.values).query(points, predicate="intersects")
        keep[inside] = False
    observers = gpd.GeoDataFrame(geometry=points[keep], crs=local_crs)
    if observers.empty:
        result = gpd.GeoDataFrame({name: [] for name in ISOVIST_METRICS}, geometry=[], crs=area_series.crs)
    else:
        isovists = get_visibility(observers, scene, view_distance, max_workers=max_workers, isovist_metrics=True)
        result = gpd.GeoDataFrame(
            isovists[list(ISOVIST_METRICS)].to_numpy(), columns=list(ISOVIST_METRICS), geometry=observers.geometry
        ).to_crs(area_series.crs)
    result.attrs.update(grid_attrs(local_crs, cell_size, (gx[0], gy[0])))
    return result
