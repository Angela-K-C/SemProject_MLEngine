"""LightGBM fitting, exactly as the notebooks call it."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import pandas as pd

from mlengine import config


def fit_with_early_stopping(
    x_train: pd.DataFrame, y_train, x_valid: pd.DataFrame, y_valid, params: dict | None = None
) -> lgb.LGBMRegressor:
    """Fit with early stopping on the validation window. ``best_iteration_`` holds the tree count."""
    model = lgb.LGBMRegressor(**(params or config.LGBM_PARAMS))
    with warnings.catch_warnings():
        # LightGBM 4.7 deprecates eval_set; kept because it is exactly the notebook's call.
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
        )
    return model


def fit_fixed(x: pd.DataFrame, y, n_estimators: int, params: dict | None = None) -> lgb.LGBMRegressor:
    """Refit with a fixed tree count (the early-stopped count) and no validation set."""
    model = lgb.LGBMRegressor(**{**(params or config.LGBM_PARAMS), "n_estimators": int(n_estimators)})
    model.fit(x, y)
    return model


def best_iteration(model: lgb.LGBMRegressor) -> int:
    return int(model.best_iteration_ or model.n_estimators)
