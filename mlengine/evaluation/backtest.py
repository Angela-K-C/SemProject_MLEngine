"""Recursive multi-day backtests: the check that decides whether a horizon > 1 is enabled."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from mlengine.evaluation import metrics
from mlengine.models import baselines
from mlengine.models.forecaster import Forecaster, SeriesBatch

BatchBuilder = Callable[[int], tuple[SeriesBatch, np.ndarray]]


def recursive_backtest(forecaster: Forecaster, build: BatchBuilder, origins: Iterable[int], horizon: int) -> dict:
    """Forecast ``horizon`` days recursively from each origin and score against actuals.

    ``build(origin)`` returns the batch whose history ends on that origin and the actuals for the
    following ``horizon`` days, shape (n_series, horizon). The horizon is "validated" when the AI
    forecast beats both baselines on aggregate RMSE and MAE.
    """
    ai, naive, seasonal, actual = [], [], [], []
    for origin in origins:
        batch, observed = build(origin)
        ai.append(forecaster.forecast(batch, horizon).ai)
        naive.append(baselines.naive_last_value(batch.sales, horizon))
        seasonal.append(baselines.seasonal_naive_7(batch.sales, horizon))
        actual.append(observed)
    ai, naive, seasonal, actual = (np.stack(x) for x in (ai, naive, seasonal, actual))
    keep = ~np.isnan(ai) & ~np.isnan(naive) & ~np.isnan(actual)
    methods = {"ai": ai, "naive_last_value": naive, "seasonal_naive_7": seasonal}
    by_day = []
    for h in range(horizon):
        day_keep = keep[:, :, h]
        by_day.append({"day": h + 1, **{name: metrics.summary(actual[:, :, h][day_keep], values[:, :, h][day_keep])
                                        for name, values in methods.items()}})
    overall = {name: metrics.summary(actual[keep], values[keep]) for name, values in methods.items()}
    validated = all(
        overall["ai"][m] is not None and overall["ai"][m] < overall[b][m]
        for b in ["naive_last_value", "seasonal_naive_7"] for m in ["rmse", "mae"]
    )
    bias = [float(np.nanmean(np.where(keep[:, :, h], ai[:, :, h] - actual[:, :, h], np.nan))) for h in range(horizon)]
    return {"horizon": horizon, "origins": int(len(ai)), "overall": overall, "byDay": by_day,
            "meanBiasByDay": bias, "validated": bool(validated),
            "rule": "AI beats naive_last_value and seasonal_naive_7 on aggregate RMSE and MAE"}
