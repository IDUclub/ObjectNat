import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from objectnat import config
from objectnat._rust import provision_allocate  # pylint: disable=no-name-in-module

from ._matrix import _distance_entries, _is_sparse_matrix, _lookup_distances, _min_distances
from .provision_exceptions import CapacityKeyError, CapacityValueError, DemandKeyError, DemandValueError
from .provision_result import ProvisionResult, _validate_provision_result

logger = config.logger


def _validate_inputs(
    buildings: pd.DataFrame,
    services: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    threshold: float,
    buildings_demand_column: str,
    services_capacity_column: str,
) -> None:
    if not isinstance(buildings, pd.DataFrame):
        raise TypeError(f"buildings must be DataFrame, got {type(buildings).__name__}")
    if not isinstance(services, pd.DataFrame):
        raise TypeError(f"services must be DataFrame, got {type(services).__name__}")
    if not isinstance(distance_matrix, pd.DataFrame):
        raise TypeError(f"distance_matrix must be DataFrame, got {type(distance_matrix).__name__}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    if buildings_demand_column not in buildings.columns:
        raise DemandKeyError()
    if services_capacity_column not in services.columns:
        raise CapacityKeyError()

    if buildings.index.has_duplicates:
        raise ValueError("buildings index must be unique")
    if services.index.has_duplicates:
        raise ValueError("services index must be unique")
    if distance_matrix.index.has_duplicates:
        raise ValueError("distance_matrix index must be unique")
    if distance_matrix.columns.has_duplicates:
        raise ValueError("distance_matrix columns must be unique")

    missing_buildings = buildings.index.difference(distance_matrix.index)
    if not missing_buildings.empty:
        missing_values = missing_buildings[:10].tolist()
        raise ValueError(f"Some building indices are missing in distance_matrix.index: {missing_values}")
    missing_services = services.index.difference(distance_matrix.columns)
    if not missing_services.empty:
        missing_values = missing_services[:10].tolist()
        raise ValueError(f"Some service indices are missing in distance_matrix.columns: {missing_values}")

    for series, exc_type in (
        (buildings[buildings_demand_column], DemandValueError),
        (services[services_capacity_column], CapacityValueError),
    ):
        if series.isna().any():
            raise exc_type()
        try:
            values = series.astype(int)
        except (TypeError, ValueError) as exc:
            raise exc_type() from exc
        if (values < 0).any():
            raise exc_type()


def _empty_flow(building_index: pd.Index, service_index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame.sparse.from_spmatrix(
        sparse.coo_matrix((len(building_index), len(service_index))),
        index=building_index,
        columns=service_index,
    ).astype(pd.SparseDtype("int", 0))


def _flow_from_records(
    building_index: pd.Index,
    service_index: pd.Index,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    if len(values) == 0:
        return _empty_flow(building_index, service_index)

    flow_sparse = sparse.coo_matrix((values, (rows, columns)), shape=(len(building_index), len(service_index)))
    flow_sparse.sum_duplicates()
    return pd.DataFrame.sparse.from_spmatrix(flow_sparse, index=building_index, columns=service_index).astype(
        pd.SparseDtype("int", 0)
    )


def _summarize_flow(
    flow: pd.DataFrame,
    demand: pd.Series,
    capacity: pd.Series,
    distance_matrix: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = demand.astype(int)
    capacity = capacity.astype(int)
    flow_coo = flow.sparse.to_coo()

    building_count = len(flow.index)
    service_count = len(flow.columns)
    supplied_within = np.zeros(building_count, dtype=float)
    supplied_without = np.zeros(building_count, dtype=float)
    carried_within = np.zeros(service_count, dtype=float)
    carried_without = np.zeros(service_count, dtype=float)
    weighted_distance = np.zeros(building_count, dtype=float)

    if flow_coo.nnz > 0:
        flow_values = flow_coo.data.astype(float)
        distance_values = _lookup_distances(distance_matrix, flow_coo.row, flow_coo.col)
        within_mask = distance_values <= threshold
        without_mask = ~within_mask

        supplied_within = np.bincount(
            flow_coo.row[within_mask], weights=flow_values[within_mask], minlength=building_count
        )
        supplied_without = np.bincount(
            flow_coo.row[without_mask], weights=flow_values[without_mask], minlength=building_count
        )
        carried_within = np.bincount(
            flow_coo.col[within_mask], weights=flow_values[within_mask], minlength=service_count
        )
        carried_without = np.bincount(
            flow_coo.col[without_mask], weights=flow_values[without_mask], minlength=service_count
        )
        weighted_distance = np.bincount(flow_coo.row, weights=flow_values * distance_values, minlength=building_count)

    supplied_total = supplied_within + supplied_without
    carried_total = carried_within + carried_without
    demand_values = demand.reindex(flow.index).to_numpy(dtype=float)
    capacity_values = capacity.reindex(flow.columns).to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_dist = weighted_distance / supplied_total
        provision_value = supplied_within / demand_values

    avg_dist[supplied_total == 0] = np.nan
    provision_value[demand_values == 0] = np.nan

    demand_rows = pd.DataFrame(
        {
            "demand": demand.reindex(flow.index).astype(int),
            "demand_left": (demand_values - supplied_total).astype(int),
            "supplied_demands_within": supplied_within.astype(int),
            "supplied_demands_without": supplied_without.astype(int),
            "min_dist": _min_distances(distance_matrix),
            "avg_dist": np.round(avg_dist, 2),
            "provision_value": np.round(provision_value, 2),
        },
        index=flow.index,
    )
    capacity_rows = pd.DataFrame(
        {
            "capacity": capacity.reindex(flow.columns).astype(int),
            "capacity_left": (capacity_values - carried_total).astype(int),
            "carried_capacity_within": carried_within.astype(int),
            "carried_capacity_without": carried_without.astype(int),
            "service_load": carried_total.astype(int),
        },
        index=flow.columns,
    )
    return demand_rows, capacity_rows


def calculate_provision(
    *,
    buildings: gpd.GeoDataFrame | pd.DataFrame,
    services: gpd.GeoDataFrame | pd.DataFrame,
    distance_matrix: pd.DataFrame,
    threshold: float,
    buildings_demand_column: str = "demand",
    services_capacity_column: str = "capacity",
    seed: int = 0,
) -> ProvisionResult:
    """Calculate service provision with the gravity-probabilistic model (allocation runs in Rust)."""
    _validate_inputs(
        buildings=buildings,
        services=services,
        distance_matrix=distance_matrix,
        threshold=threshold,
        buildings_demand_column=buildings_demand_column,
        services_capacity_column=services_capacity_column,
    )

    building_index = buildings.index
    service_index = services.index
    if not (distance_matrix.index.equals(building_index) and distance_matrix.columns.equals(service_index)):
        distance_matrix = distance_matrix.loc[building_index, service_index]
    if not _is_sparse_matrix(distance_matrix) and not (distance_matrix.dtypes == np.float64).all():
        distance_matrix = distance_matrix.astype(float)

    demand = buildings[buildings_demand_column].astype(int)
    capacity = services[services_capacity_column].astype(int)

    # only pairs within 3 * threshold can ever be allocated
    rows, cols, costs = _distance_entries(distance_matrix, max_cost=3 * float(threshold))
    logger.debug(
        f"Calculating provision from {len(services)} services to {len(buildings)} buildings "
        f"over {len(costs)} reachable pairs."
    )
    flow_rows, flow_cols, flow_values = provision_allocate(
        rows,
        cols,
        costs,
        demand.to_numpy(dtype=np.int64),
        capacity.to_numpy(dtype=np.int64),
        float(threshold),
        seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
    )

    flow = _flow_from_records(building_index, service_index, flow_rows, flow_cols, flow_values)
    demand_rows, capacity_rows = _summarize_flow(flow, demand, capacity, distance_matrix, threshold)
    logger.debug("Done calculating provision.")
    return ProvisionResult(
        flow=flow,
        demand_rows=demand_rows,
        capacity_rows=capacity_rows,
        distance_matrix=distance_matrix,
        threshold=float(threshold),
    )


def recalculate_provision(provision_result: ProvisionResult, new_max_dist: float) -> ProvisionResult:
    """Filter provision flows by a stricter maximum distance without redistributing demand."""
    _validate_provision_result(provision_result)
    if new_max_dist < 0:
        raise ValueError(f"new_max_dist must be >= 0, got {new_max_dist}")

    flow_coo = provision_result.flow.sparse.to_coo()
    if flow_coo.nnz == 0:
        return provision_result

    distance_values = _lookup_distances(provision_result.distance_matrix, flow_coo.row, flow_coo.col)
    keep_mask = distance_values <= new_max_dist
    if keep_mask.all():
        logger.warning("To clip distance exceeds max links distance, returning full provision")
        return provision_result

    flow_sparse = sparse.coo_matrix(
        (flow_coo.data[keep_mask], (flow_coo.row[keep_mask], flow_coo.col[keep_mask])),
        shape=provision_result.flow.shape,
    )
    flow = pd.DataFrame.sparse.from_spmatrix(
        flow_sparse,
        index=provision_result.flow.index,
        columns=provision_result.flow.columns,
    ).astype(pd.SparseDtype("int", 0))
    demand_rows, capacity_rows = _summarize_flow(
        flow,
        provision_result.demand_rows["demand"],
        provision_result.capacity_rows["capacity"],
        provision_result.distance_matrix,
        provision_result.threshold,
    )
    return ProvisionResult(
        flow=flow,
        demand_rows=demand_rows,
        capacity_rows=capacity_rows,
        distance_matrix=provision_result.distance_matrix,
        threshold=provision_result.threshold,
    )
