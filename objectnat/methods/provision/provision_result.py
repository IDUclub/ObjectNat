from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from objectnat import config

from ._matrix import _lookup_distances

logger = config.logger


@dataclass(slots=True)
class ProvisionResult:
    """
    Result of a service provision calculation.

    Attributes:
        flow:
            Sparse building-service allocation matrix. Rows match building
            indices, columns match service indices, and non-zero values are
            allocated demand units.
        demand_rows:
            Per-building metrics indexed like ``flow.index``. Includes original
            demand, remaining demand, supplied demand within/beyond the
            threshold, the cost to the nearest service, the flow-weighted mean cost and
            provision value.
        capacity_rows:
            Per-service metrics indexed like ``flow.columns``. Includes
            original capacity, remaining capacity, carried capacity
            within/beyond the threshold, and total service load.
        distance_matrix:
            Aligned OD cost matrix used for the calculation (dense, or pandas
            sparse when a sparse matrix was passed). Its index and columns
            must match ``flow``.
        threshold:
            Normative distance or time threshold used to split flows into
            within-threshold and beyond-threshold metrics.
    """

    flow: pd.DataFrame
    demand_rows: pd.DataFrame
    capacity_rows: pd.DataFrame
    distance_matrix: pd.DataFrame
    threshold: float


def _validate_provision_result(provision_result: ProvisionResult) -> None:
    if not isinstance(provision_result, ProvisionResult):
        raise TypeError(f"provision_result must be ProvisionResult, got {type(provision_result).__name__}")
    if provision_result.flow.index.has_duplicates:
        raise ValueError("provision_result.flow.index must be unique")
    if provision_result.flow.columns.has_duplicates:
        raise ValueError("provision_result.flow.columns must be unique")
    if provision_result.demand_rows.index.has_duplicates:
        raise ValueError("provision_result.demand_rows.index must be unique")
    if provision_result.capacity_rows.index.has_duplicates:
        raise ValueError("provision_result.capacity_rows.index must be unique")
    if not provision_result.flow.index.equals(provision_result.distance_matrix.index):
        raise ValueError("provision_result.flow.index must match provision_result.distance_matrix.index")
    if not provision_result.flow.columns.equals(provision_result.distance_matrix.columns):
        raise ValueError("provision_result.flow.columns must match provision_result.distance_matrix.columns")


def _drop_overwritten_columns(df: gpd.GeoDataFrame | pd.DataFrame, columns_to_join: pd.Index, df_name: str):
    overwritten_columns = df.columns.intersection(columns_to_join)
    if overwritten_columns.empty:
        return df

    logger.warning(
        f"{df_name} already contains provision columns that will be overwritten: {overwritten_columns.tolist()}"
    )
    return df.drop(columns=overwritten_columns)


def get_provision_buildings(
    buildings_gdf: gpd.GeoDataFrame | pd.DataFrame,
    provision_result: ProvisionResult,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Join provision demand metrics to buildings.

    Existing columns with the same names as provision metrics are dropped and
    replaced. The returned object preserves the input type and index.
    """
    if not isinstance(buildings_gdf, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError(f"buildings_gdf must be DataFrame or GeoDataFrame, got {type(buildings_gdf).__name__}")
    _validate_provision_result(provision_result)
    if buildings_gdf.index.has_duplicates:
        raise ValueError("buildings_gdf.index must be unique")

    buildings_gdf = _drop_overwritten_columns(buildings_gdf, provision_result.demand_rows.columns, "buildings_gdf")
    return buildings_gdf.join(provision_result.demand_rows, how="left")


def get_provision_services(
    services_gdf: gpd.GeoDataFrame | pd.DataFrame,
    provision_result: ProvisionResult,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Join provision capacity metrics to services.

    Existing columns with the same names as provision metrics are dropped and
    replaced. The returned object preserves the input type and index.
    """
    if not isinstance(services_gdf, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError(f"services_gdf must be DataFrame or GeoDataFrame, got {type(services_gdf).__name__}")
    _validate_provision_result(provision_result)
    if services_gdf.index.has_duplicates:
        raise ValueError("services_gdf.index must be unique")

    services_gdf = _drop_overwritten_columns(services_gdf, provision_result.capacity_rows.columns, "services_gdf")
    return services_gdf.join(provision_result.capacity_rows, how="left")


def get_provision_links(
    buildings_gdf: gpd.GeoDataFrame,
    services_gdf: gpd.GeoDataFrame,
    provision_result: ProvisionResult,
) -> gpd.GeoDataFrame:
    """
    Build building-service link geometries from non-zero provision flows.

    Services are reprojected to the buildings CRS when both CRS values are set
    and differ. Empty flows return an empty ``GeoDataFrame``. Links whose
    building or service ids are absent from the supplied GeoDataFrames are
    filtered out.
    """
    if not isinstance(buildings_gdf, gpd.GeoDataFrame):
        raise TypeError(f"buildings_gdf must be GeoDataFrame, got {type(buildings_gdf).__name__}")
    if not isinstance(services_gdf, gpd.GeoDataFrame):
        raise TypeError(f"services_gdf must be GeoDataFrame, got {type(services_gdf).__name__}")
    _validate_provision_result(provision_result)

    if buildings_gdf.index.has_duplicates:
        raise ValueError("buildings_gdf.index must be unique")
    if services_gdf.index.has_duplicates:
        raise ValueError("services_gdf.index must be unique")

    flow_coo = provision_result.flow.sparse.to_coo()
    if flow_coo.nnz == 0:
        logger.warning(
            "Unable to create provision links - no demand could be matched with service locations. "
            "This is likely because either demand or capacity is zero, or no services are reachable."
        )
        return gpd.GeoDataFrame()

    services_for_links = services_gdf
    if buildings_gdf.crs is not None and services_gdf.crs is not None and buildings_gdf.crs != services_gdf.crs:
        services_for_links = services_gdf.to_crs(buildings_gdf.crs)

    building_points = buildings_gdf.representative_point()
    service_points = services_for_links.representative_point()
    building_labels = provision_result.flow.index.to_numpy()[flow_coo.row]
    service_labels = provision_result.flow.columns.to_numpy()[flow_coo.col]
    distance_values = _lookup_distances(provision_result.distance_matrix, flow_coo.row, flow_coo.col)

    links = pd.DataFrame(
        {
            "building_index": building_labels,
            "service_index": service_labels,
            "demand": flow_coo.data.astype(int),
            "distance": distance_values.round(2),
        }
    )
    links = links[
        (links["building_index"].isin(buildings_gdf.index)) & (links["service_index"].isin(services_gdf.index))
    ]
    if links.empty:
        return gpd.GeoDataFrame()

    start = shapely.get_coordinates(building_points.loc[links["building_index"]].values)
    end = shapely.get_coordinates(service_points.loc[links["service_index"]].values)
    links["geometry"] = shapely.linestrings(np.stack([start, end], axis=1))
    return gpd.GeoDataFrame(links, geometry="geometry", crs=buildings_gdf.crs)
