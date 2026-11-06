import geopandas as gpd
import numpy as np


def get_radius_coverage(gdf_from: gpd.GeoDataFrame, radius: float, resolution: int = 32):
    """
    Calculate radius-based coverage zones using Voronoi polygons.

    This is a graph-free coverage: zones are built purely from the source
    points, a fixed radius and a Voronoi partition, with no network involved.

    Args:
        gdf_from (gpd.GeoDataFrame):
            Source points for which coverage zones are calculated.

        radius (float):
            Maximum coverage radius in metres.

        resolution (int):
            Number of segments used to approximate quarter-circle in buffer (default=32).

    Returns:
        gpd.GeoDataFrame:
            GeoDataFrame with smoothed coverage zone polygons in the same CRS as original gdf_from.

    Notes:
        - Automatically converts to local UTM CRS for accurate distance measurements
        - Final zones are slightly contracted then expanded for smoothing effect
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
