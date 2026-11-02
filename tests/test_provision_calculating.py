"""Unit tests for the pure provision-calculation layer.

These tests build tiny in-memory buildings/services/matrix DataFrames and never touch
the network graph, parquet fixtures, or rendering. The gravity model runs in the Rust extension with a
seeded random stream, so results are deterministic.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from objectnat import ProvisionResult
from objectnat.methods.provision.provision_calculating import (
    _summarize_flow,
    _validate_inputs,
    calculate_provision,
    recalculate_provision,
)
from objectnat.methods.provision.provision_exceptions import (
    CapacityKeyError,
    CapacityValueError,
    DemandKeyError,
    DemandValueError,
)

DEMAND_ROW_COLUMNS = [
    "demand",
    "demand_left",
    "supplied_demands_within",
    "supplied_demands_without",
    "min_dist",
    "avg_dist",
    "provision_value",
]
CAPACITY_ROW_COLUMNS = [
    "capacity",
    "capacity_left",
    "carried_capacity_within",
    "carried_capacity_without",
    "service_load",
]


@pytest.fixture
def buildings():
    return pd.DataFrame({"demand": [10, 5, 8]}, index=[1, 2, 3])


@pytest.fixture
def services():
    return pd.DataFrame({"capacity": [12, 20]}, index=[10, 11])


@pytest.fixture
def matrix():
    # rows = buildings.index, columns = services.index; all distances well within threshold*3
    return pd.DataFrame(
        [
            [5.0, 15.0],
            [8.0, 4.0],
            [20.0, 6.0],
        ],
        index=[1, 2, 3],
        columns=[10, 11],
    )


# --------------------------------------------------------------------------- #
# _validate_inputs
# --------------------------------------------------------------------------- #
def _validate(buildings, services, matrix, threshold=10.0):
    _validate_inputs(
        buildings=buildings,
        services=services,
        distance_matrix=matrix,
        threshold=threshold,
        buildings_demand_column="demand",
        services_capacity_column="capacity",
    )


def test_validate_inputs_accepts_valid(buildings, services, matrix):
    # Should not raise.
    _validate(buildings, services, matrix)


def test_validate_inputs_rejects_non_dataframe(services, matrix):
    with pytest.raises(TypeError):
        _validate([1, 2, 3], services, matrix)


def test_validate_inputs_negative_threshold(buildings, services, matrix):
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        _validate(buildings, services, matrix, threshold=-1)


def test_validate_inputs_missing_demand_column(buildings, services, matrix):
    buildings = buildings.rename(columns={"demand": "population"})
    with pytest.raises(DemandKeyError):
        _validate(buildings, services, matrix)


def test_validate_inputs_missing_capacity_column(buildings, services, matrix):
    services = services.rename(columns={"capacity": "seats"})
    with pytest.raises(CapacityKeyError):
        _validate(buildings, services, matrix)


def test_validate_inputs_demand_nan(buildings, services, matrix):
    buildings = buildings.astype(float)
    buildings.loc[1, "demand"] = np.nan
    with pytest.raises(DemandValueError):
        _validate(buildings, services, matrix)


def test_validate_inputs_demand_negative(buildings, services, matrix):
    buildings.loc[1, "demand"] = -5
    with pytest.raises(DemandValueError):
        _validate(buildings, services, matrix)


def test_validate_inputs_demand_non_numeric(buildings, services, matrix):
    buildings["demand"] = ["a", "b", "c"]
    with pytest.raises(DemandValueError):
        _validate(buildings, services, matrix)


def test_validate_inputs_capacity_negative(buildings, services, matrix):
    services.loc[10, "capacity"] = -1
    with pytest.raises(CapacityValueError):
        _validate(buildings, services, matrix)


def test_validate_inputs_duplicate_building_index(services, matrix):
    buildings = pd.DataFrame({"demand": [10, 5]}, index=[1, 1])
    with pytest.raises(ValueError, match="buildings index must be unique"):
        _validate(buildings, services, matrix)


def test_validate_inputs_duplicate_service_index(buildings, matrix):
    services = pd.DataFrame({"capacity": [12, 20]}, index=[10, 10])
    with pytest.raises(ValueError, match="services index must be unique"):
        _validate(buildings, services, matrix)


def test_validate_inputs_building_missing_in_matrix(buildings, services, matrix):
    buildings = pd.DataFrame({"demand": [10, 5, 8]}, index=[1, 2, 99])
    with pytest.raises(ValueError, match="building indices are missing"):
        _validate(buildings, services, matrix)


def test_validate_inputs_service_missing_in_matrix(buildings, services, matrix):
    services = pd.DataFrame({"capacity": [12, 20]}, index=[10, 99])
    with pytest.raises(ValueError, match="service indices are missing"):
        _validate(buildings, services, matrix)


# --------------------------------------------------------------------------- #
# calculate_provision — invariants
# --------------------------------------------------------------------------- #
def _run(buildings, services, matrix, threshold=10.0, seed=0):
    return calculate_provision(
        buildings=buildings,
        services=services,
        distance_matrix=matrix,
        threshold=threshold,
        seed=seed,
    )


def test_calculate_provision_structure(buildings, services, matrix):
    result = _run(buildings, services, matrix)

    assert isinstance(result, ProvisionResult)
    assert result.threshold == 10.0
    assert isinstance(result.threshold, float)

    assert list(result.flow.index) == list(buildings.index)
    assert list(result.flow.columns) == list(services.index)
    assert result.flow.sparse.to_coo().nnz > 0

    assert list(result.demand_rows.columns) == DEMAND_ROW_COLUMNS
    assert list(result.capacity_rows.columns) == CAPACITY_ROW_COLUMNS
    assert result.distance_matrix.to_numpy().dtype == float


def test_calculate_provision_conservation(buildings, services, matrix):
    result = _run(buildings, services, matrix)
    demand_rows = result.demand_rows
    capacity_rows = result.capacity_rows

    supplied_total = demand_rows["supplied_demands_within"] + demand_rows["supplied_demands_without"]
    carried_total = capacity_rows["carried_capacity_within"] + capacity_rows["carried_capacity_without"]

    # Nobody is supplied more than they demand, nobody carries more than capacity.
    assert (supplied_total <= demand_rows["demand"]).all()
    assert (carried_total <= capacity_rows["capacity"]).all()

    # Leftovers are consistent and non-negative.
    assert (demand_rows["demand_left"] == demand_rows["demand"] - supplied_total).all()
    assert (demand_rows["demand_left"] >= 0).all()
    assert (capacity_rows["capacity_left"] >= 0).all()
    assert (capacity_rows["service_load"] == carried_total).all()

    # Total supplied demand equals total carried capacity (flow is symmetric).
    assert supplied_total.sum() == carried_total.sum()


def test_calculate_provision_value_bounds(buildings, services, matrix):
    result = _run(buildings, services, matrix)
    provision_value = result.demand_rows["provision_value"]
    assert ((provision_value >= 0) & (provision_value <= 1)).all()


def test_calculate_provision_within_without_split(buildings, services, matrix):
    threshold = 10.0
    result = _run(buildings, services, matrix, threshold=threshold)
    flow_coo = result.flow.sparse.to_coo()
    distances = result.distance_matrix.to_numpy(dtype=float)[flow_coo.row, flow_coo.col]

    demand_rows = result.demand_rows
    total_within = demand_rows["supplied_demands_within"].sum()
    total_without = demand_rows["supplied_demands_without"].sum()

    # The split of served demand matches the threshold applied to link distances.
    assert total_within == flow_coo.data[distances <= threshold].sum()
    assert total_without == flow_coo.data[distances > threshold].sum()


def test_calculate_provision_is_deterministic(buildings, services, matrix):
    first = _run(buildings, services, matrix)
    second = _run(buildings, services, matrix)
    pd.testing.assert_frame_equal(first.flow.sparse.to_dense(), second.flow.sparse.to_dense())
    pd.testing.assert_frame_equal(first.demand_rows, second.demand_rows)


def test_calculate_provision_does_not_mutate_inputs(buildings, services, matrix):
    buildings_before = buildings.copy()
    services_before = services.copy()
    matrix_before = matrix.copy()
    _run(buildings, services, matrix)
    pd.testing.assert_frame_equal(buildings, buildings_before)
    pd.testing.assert_frame_equal(services, services_before)
    pd.testing.assert_frame_equal(matrix, matrix_before)


# --------------------------------------------------------------------------- #
# calculate_provision — edge cases
# --------------------------------------------------------------------------- #
def test_calculate_provision_zero_demand(buildings, services, matrix):
    buildings["demand"] = 0
    result = _run(buildings, services, matrix)
    assert result.flow.sparse.to_coo().nnz == 0
    # provision_value is undefined (NaN) when demand is zero.
    assert result.demand_rows["provision_value"].isna().all()


def test_calculate_provision_zero_capacity(buildings, services, matrix):
    services["capacity"] = 0
    result = _run(buildings, services, matrix)
    assert result.flow.sparse.to_coo().nnz == 0
    assert (result.demand_rows["demand_left"] == buildings["demand"]).all()


def test_calculate_provision_all_distances_out_of_reach(buildings, services, matrix):
    # Every distance is beyond threshold*3, so nothing can be allocated.
    matrix = matrix + 1000
    result = _run(buildings, services, matrix, threshold=10.0)
    assert result.flow.sparse.to_coo().nnz == 0
    assert (result.demand_rows["supplied_demands_within"] == 0).all()


# --------------------------------------------------------------------------- #
# recalculate_provision
# --------------------------------------------------------------------------- #
def test_recalculate_provision_negative_distance(buildings, services, matrix):
    result = _run(buildings, services, matrix)
    with pytest.raises(ValueError, match="new_max_dist must be >= 0"):
        recalculate_provision(result, -1)


def test_recalculate_provision_empty_flow_shortcircuit(buildings, services, matrix):
    buildings["demand"] = 0
    result = _run(buildings, services, matrix)
    assert result.flow.sparse.to_coo().nnz == 0
    # Nothing to recalculate: the same object is returned.
    assert recalculate_provision(result, 5) is result


def test_recalculate_provision_large_dist_returns_original(buildings, services, matrix):
    result = _run(buildings, services, matrix)
    # A distance larger than any link keeps every flow -> original result returned.
    assert recalculate_provision(result, 10_000) is result


def test_recalculate_provision_tightens_links(buildings, services, matrix):
    result = _run(buildings, services, matrix, threshold=25.0)
    coo = result.flow.sparse.to_coo()
    original_nnz = coo.nnz
    link_distances = result.distance_matrix.to_numpy(dtype=float)[coo.row, coo.col]
    assert link_distances.max() > link_distances.min(), "test data must have a spread of link distances to cut"

    # Just below the farthest link so at least the longest flow is dropped.
    new_max_dist = float(link_distances.max()) - 0.5
    tightened = recalculate_provision(result, new_max_dist)

    tightened_coo = tightened.flow.sparse.to_coo()
    kept_distances = tightened.distance_matrix.to_numpy(dtype=float)[tightened_coo.row, tightened_coo.col]

    assert tightened_coo.nnz < original_nnz
    assert (kept_distances <= new_max_dist).all()
    # Aggregates are recomputed, not redistributed: served demand never grows.
    original_supplied = result.demand_rows["supplied_demands_within"] + result.demand_rows["supplied_demands_without"]
    tightened_supplied = (
        tightened.demand_rows["supplied_demands_within"] + tightened.demand_rows["supplied_demands_without"]
    )
    assert (tightened_supplied <= original_supplied).all()


# --------------------------------------------------------------------------- #
# Seeds and sparse input
# --------------------------------------------------------------------------- #
def test_calculate_provision_seed_is_reproducible(buildings, services, matrix):
    # Same seed -> identical flows; the seed threads through the whole allocation.
    first = _run(buildings, services, matrix, seed=7)
    second = _run(buildings, services, matrix, seed=7)
    pd.testing.assert_frame_equal(first.flow.sparse.to_dense(), second.flow.sparse.to_dense())


def test_sparse_distance_matrix_gives_the_same_result(buildings, services, matrix):
    # Unreachable pairs are simply absent from a sparse matrix (NaN fill value).
    sparse_matrix = matrix.where(matrix <= 15).astype(pd.SparseDtype(float, np.nan))
    dense_result = _run(buildings, services, matrix.where(matrix <= 15, np.inf))
    sparse_result = _run(buildings, services, sparse_matrix)

    pd.testing.assert_frame_equal(dense_result.flow.sparse.to_dense(), sparse_result.flow.sparse.to_dense())
    pd.testing.assert_frame_equal(dense_result.demand_rows, sparse_result.demand_rows)
    pd.testing.assert_frame_equal(dense_result.capacity_rows, sparse_result.capacity_rows)
    assert isinstance(sparse_result.distance_matrix.dtypes.iloc[0], pd.SparseDtype)


# --------------------------------------------------------------------------- #
# Provision exception classes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc_cls",
    [CapacityKeyError, CapacityValueError, DemandKeyError, DemandValueError],
)
def test_provision_exception_default_message(exc_cls):
    # The no-argument form renders a non-empty, informative default message.
    message = str(exc_cls())
    assert isinstance(message, str)
    assert message.strip()


@pytest.mark.parametrize(
    "exc_cls",
    [CapacityKeyError, CapacityValueError, DemandKeyError, DemandValueError],
)
def test_provision_exception_custom_message(exc_cls):
    # A custom message is surfaced in the string representation.
    assert "boom" in str(exc_cls("boom"))


def test_summarize_flow_hand_built():
    building_index = pd.Index([1, 2])
    service_index = pd.Index([10, 11])
    flow = pd.DataFrame.sparse.from_spmatrix(
        sparse.coo_matrix(np.array([[6, 0], [0, 4]])),
        index=building_index,
        columns=service_index,
    ).astype(pd.SparseDtype("int", 0))
    demand = pd.Series([10, 5], index=building_index)
    capacity = pd.Series([8, 6], index=service_index)
    # building 1 -> service 10 at dist 4 (within), building 2 -> service 11 at dist 12 (without)
    matrix = pd.DataFrame([[4.0, 20.0], [30.0, 12.0]], index=building_index, columns=service_index)

    demand_rows, capacity_rows = _summarize_flow(flow, demand, capacity, matrix, threshold=10.0)

    assert demand_rows.loc[1, "supplied_demands_within"] == 6
    assert demand_rows.loc[1, "supplied_demands_without"] == 0
    assert demand_rows.loc[2, "supplied_demands_within"] == 0
    assert demand_rows.loc[2, "supplied_demands_without"] == 4
    assert demand_rows.loc[1, "demand_left"] == 4
    assert demand_rows.loc[2, "demand_left"] == 1
    assert demand_rows.loc[1, "provision_value"] == pytest.approx(0.6)
    assert demand_rows.loc[1, "min_dist"] == 4.0
    assert demand_rows.loc[1, "avg_dist"] == 4.0

    assert capacity_rows.loc[10, "carried_capacity_within"] == 6
    assert capacity_rows.loc[11, "carried_capacity_without"] == 4
    assert capacity_rows.loc[10, "capacity_left"] == 2
    assert capacity_rows.loc[10, "service_load"] == 6
