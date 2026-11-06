import geopandas as gpd
import numpy as np


def get_radius_coverage(gdf_from: gpd.GeoDataFrame, radius: float, resolution: int = 32):
    """
    Calculate radius-based coverage zones using Voronoi polygons.

    This is a graph-free coverage: zones are built purely from the service
    points, a fixed radius and a Voronoi partition, with no network involved.

    Args:
        gdf_from (gpd.GeoDataFrame):
            Service points for which coverage zones are calculated.

        radius (float):
            Maximum coverage radius in metres.

        resolution (int):
            Number of segments per quarter circle of the buffers (default 32).

    Returns:
        gpd.GeoDataFrame:
            Smoothed coverage zones, one polygon per point, in the CRS of ``gdf_from``.

    Notes:
        - The calculation runs in a local UTM zone, so ``radius`` is in metres.
        - The zones are smoothed by a slight inward and outward buffer, which rounds their corners.
    """
    original_crs = gdf_from.crs
    local_crs = gdf_from.estimate_utm_crs()
    gdf_from = gdf_from.to_crs(local_crs)
    bounds = gdf_from.buffer(radius).union_all()
    coverage_polys = gpd.GeoDataFrame(geometry=gdf_from.voronoi_polygons().clip(bounds, keep_geom_type=True))
    coverage_polys = coverage_polys.sjoin(gdf_from)
    coverage_polys["area"] = coverage_polys.area
    coverage_polys["buffer"] = np.pow(coverage_polys["area"], 1 / 3)
    coverage_polys.geometry = coverage_polys.buffer(-coverage_polys["buffer"], resolution=1, join_style="mitre").buffer(
        coverage_polys["buffer"] * 0.9, resolution=resolution
    )
    coverage_polys.drop(columns=["buffer", "area"], inplace=True)
    return coverage_polys.to_crs(original_crs)
