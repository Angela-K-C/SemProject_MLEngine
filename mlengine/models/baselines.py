"""Traditional baselines. Both work on a sales matrix whose last column is the as-of day."""

from __future__ import annotations

import numpy as np

BASELINES = {
    "naive_last_value": "Same as last day",
    "seasonal_naive_7": "Same day last week",
}


def naive_last_value(history: np.ndarray, horizon: int) -> np.ndarray:
    """Every forecast day equals the last recorded day's sales (``sales_lag_1`` for day 1)."""
    history = np.asarray(history, dtype=np.float32)
    recorded = ~np.isnan(history)
    last_index = np.where(recorded.any(axis=1), history.shape[1] - 1 - np.argmax(recorded[:, ::-1], axis=1), -1)
    last = np.where(last_index >= 0, history[np.arange(len(history)), np.maximum(last_index, 0)], np.nan)
    return np.repeat(last[:, None], horizon, axis=1).astype(np.float32)


def seasonal_naive_7(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast for day t = sales on t − 7 (repeating the last observed week beyond 7 days)."""
    history = np.asarray(history, dtype=np.float32)
    if history.shape[1] < 7:
        return np.full((len(history), horizon), np.nan, dtype=np.float32)
    last_week = history[:, -7:]
    return np.stack([last_week[:, h % 7] for h in range(horizon)], axis=1)


def run(name: str, history: np.ndarray, horizon: int) -> np.ndarray:
    if name == "naive_last_value":
        return naive_last_value(history, horizon)
    if name == "seasonal_naive_7":
        return seasonal_naive_7(history, horizon)
    raise KeyError(f"Unknown baseline {name!r}. Known: {sorted(BASELINES)}")
