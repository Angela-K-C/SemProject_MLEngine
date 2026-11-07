"""MAE, RMSE and WAPE: the single implementation behind every number the app shows."""

from __future__ import annotations

import math

import numpy as np


def _pairs(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    keep = ~(np.isnan(actual) | np.isnan(forecast))
    return actual[keep], forecast[keep]


def mae(actual, forecast) -> float | None:
    a, f = _pairs(actual, forecast)
    return float(np.mean(np.abs(a - f))) if a.size else None


def rmse(actual, forecast) -> float | None:
    a, f = _pairs(actual, forecast)
    return float(math.sqrt(np.mean((a - f) ** 2))) if a.size else None


def wape(actual, forecast) -> float | None:
    """Σ|error| / Σ actual. ``None`` when there were no sales at all (undefined)."""
    a, f = _pairs(actual, forecast)
    total = a.sum()
    return float(np.abs(a - f).sum() / total) if a.size and total > 0 else None


def summary(actual, forecast) -> dict:
    a, _ = _pairs(actual, forecast)
    return {"n": int(a.size), "mae": mae(actual, forecast), "rmse": rmse(actual, forecast),
            "wape": wape(actual, forecast)}
