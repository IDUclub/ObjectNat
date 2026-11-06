"""OD cost matrix access that works for dense and pandas-sparse DataFrames without densifying the latter."""

import numpy as np
import pandas as pd
from scipy import sparse


def _is_sparse_matrix(distance_matrix: pd.DataFrame) -> bool:
    """True for a pandas sparse DataFrame whose fill value means "unreachable" (NaN or infinity)."""
    dtypes = distance_matrix.dtypes
    if len(dtypes) == 0 or not all(isinstance(dtype, pd.SparseDtype) for dtype in dtypes):
        return False
    fill = dtypes.iloc[0].fill_value
    return all(dtype.fill_value is fill or (np.isnan(dtype.fill_value) and np.isnan(fill)) for dtype in dtypes) and (
        np.isnan(fill) or np.isinf(fill)
    )


def _distance_entries(
    distance_matrix: pd.DataFrame, max_cost: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions and values of the finite entries (row position, column position, cost), optionally capped."""
    if _is_sparse_matrix(distance_matrix):
        coo = distance_matrix.sparse.to_coo()
        rows, cols, values = coo.row, coo.col, coo.data.astype(float)
        keep = np.isfinite(values) if max_cost is None else values <= max_cost
        rows, cols, values = rows[keep], cols[keep], values[keep]
    else:
        dense = distance_matrix.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            mask = np.isfinite(dense) if max_cost is None else dense <= max_cost
        rows, cols = np.nonzero(mask)
        values = dense[rows, cols]
    return rows.astype(np.int64), cols.astype(np.int64), values


def _lookup_distances(distance_matrix: pd.DataFrame, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Costs at the given (row position, column position) pairs; unreachable pairs give ``inf``."""
    if len(rows) == 0:
        return np.empty(0, dtype=float)
    if _is_sparse_matrix(distance_matrix):
        coo = distance_matrix.sparse.to_coo()
        # shift costs by +1 so that a stored 0 differs from an absent (unreachable) pair
        shifted = sparse.csr_matrix((coo.data.astype(float) + 1.0, (coo.row, coo.col)), shape=coo.shape)
        values = np.asarray(shifted[rows, cols]).ravel() - 1.0
        values[values < 0] = np.inf
        return values
    return distance_matrix.to_numpy(dtype=float)[rows, cols]


def _min_distances(distance_matrix: pd.DataFrame) -> pd.Series:
    """Minimum finite cost per row (NaN when a row has none)."""
    if _is_sparse_matrix(distance_matrix):
        rows, _, values = _distance_entries(distance_matrix)
        minimum = np.full(len(distance_matrix.index), np.inf)
        np.minimum.at(minimum, rows, values)
    else:
        dense = distance_matrix.to_numpy(dtype=float)
        minimum = np.min(np.where(np.isfinite(dense), dense, np.inf), axis=1, initial=np.inf)
    minimum[np.isinf(minimum)] = np.nan
    return pd.Series(minimum, index=distance_matrix.index)
