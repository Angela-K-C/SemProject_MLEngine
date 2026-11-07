"""Lag and rolling features on a wide sales matrix: the one implementation used everywhere.

The maths is copied from ``computeRollingStats`` in ``LightGBM_BaseLocal.ipynb`` (cumulative sums
in float64, sample standard deviation, result cast to float32). When the matrix contains no NaN the
output is bit-for-bit what the notebook produced. NaN marks an unrecorded day: a lag that lands on
one is NaN, and so is any rolling statistic whose window touches one.
"""

from __future__ import annotations

import numpy as np

from mlengine import config


class InsufficientHistoryError(ValueError):
    pass


def window_feature_names() -> list[str]:
    names = [f"sales_lag_{lag}" for lag in config.LAG_DAYS]
    for window in config.ROLLING_WINDOWS:
        names += [f"rolling_mean_{window}", f"rolling_std_{window}"]
    return names


def window_features(sales_matrix: np.ndarray, target_columns: np.ndarray) -> dict[str, np.ndarray]:
    """Features for every (target column, series) pair, flattened target-major.

    ``target_columns`` are 0-based column positions of the target days in ``sales_matrix``.
    Output arrays have length ``len(target_columns) * n_series`` and the order the notebook used:
    all series for the first target day, then all series for the next day, and so on.
    Only columns strictly before each target are read, so the target day itself never leaks in.
    """
    matrix = np.asarray(sales_matrix, dtype=np.float32)
    targets = np.asarray(target_columns, dtype=np.int64)
    if targets.size and targets.min() - config.LOOKBACK_DAYS < 0:
        raise InsufficientHistoryError(f"Need {config.LOOKBACK_DAYS} days of history before the first target.")

    out: dict[str, np.ndarray] = {}
    for lag in config.LAG_DAYS:
        out[f"sales_lag_{lag}"] = matrix[:, targets - lag].T.reshape(-1)

    missing = np.isnan(matrix)
    has_missing = bool(missing.any())
    filled = np.where(missing, np.float32(0), matrix) if has_missing else matrix
    zeros = np.zeros((matrix.shape[0], 1))
    prefix_sum = np.concatenate([zeros, np.cumsum(filled, axis=1, dtype=np.float64)], axis=1)
    prefix_square = np.concatenate([zeros, np.cumsum(filled.astype(np.float64) ** 2, axis=1)], axis=1)
    prefix_missing = (
        np.concatenate([zeros.astype(np.int64), np.cumsum(missing, axis=1, dtype=np.int64)], axis=1)
        if has_missing
        else None
    )

    for window in config.ROLLING_WINDOWS:
        window_end = targets - config.ROLLING_DELAY + 1  # exclusive prefix index → last day is target - 1
        window_start = window_end - window
        window_sum = prefix_sum[:, window_end] - prefix_sum[:, window_start]
        window_square = prefix_square[:, window_end] - prefix_square[:, window_start]
        mean = window_sum / window
        variance = np.maximum((window_square - (window_sum**2) / window) / (window - 1), 0.0)
        mean, std = mean.astype("float32"), np.sqrt(variance).astype("float32")
        if prefix_missing is not None:
            gaps = (prefix_missing[:, window_end] - prefix_missing[:, window_start]) > 0
            mean[gaps] = np.nan
            std[gaps] = np.nan
        out[f"rolling_mean_{window}"] = mean.T.reshape(-1)
        out[f"rolling_std_{window}"] = std.T.reshape(-1)
    return out


def missing_in_lookback(sales_matrix: np.ndarray, as_of_column: int) -> np.ndarray:
    """Number of unrecorded days among the 21 days ending at ``as_of_column`` (inclusive)."""
    start = as_of_column - config.LOOKBACK_DAYS + 1
    if start < 0:
        raise InsufficientHistoryError(f"Need {config.LOOKBACK_DAYS} days of history.")
    return np.isnan(sales_matrix[:, start : as_of_column + 1]).sum(axis=1)
