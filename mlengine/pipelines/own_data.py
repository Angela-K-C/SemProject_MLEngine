"""Train an own-data model for one workspace: the Stage 3 recipe on the ``portable-v1`` schema.

Splits are measured back from the last day of history (L), mirroring the M5 windows:

    train: up to L−56 · validation (early stopping): L−55 … L−28 · backtest: L−27 … L

Gate: the model is registered only if its backtest RMSE *and* MAE are both lower than the naive
"same as last day" baseline's on exactly the same rows. Otherwise nothing is registered and the
workspace keeps baseline forecasts.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlengine import config
from mlengine.evaluation import metrics
from mlengine.features import portable_v1
from mlengine.models import train
from mlengine.models.artifact import ModelRegistry, save_artifact

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["store_id", "product_id", "date", "units_sold"]


@dataclass
class WideHistory:
    series: pd.DataFrame  # store_id, product_id, category, first_price_date
    sales: np.ndarray  # (n, D) float32, NaN = unrecorded
    prices: np.ndarray  # (n, D) float32, forward-filled
    first_date: dt.date

    @property
    def last_date(self) -> dt.date:
        return self.first_date + dt.timedelta(days=self.sales.shape[1] - 1)


def model_id_for(workspace_id: str) -> str:
    return "own-" + re.sub(r"[^A-Za-z0-9_.-]", "-", str(workspace_id))


def to_wide(history: pd.DataFrame) -> WideHistory:
    """Long canonical rows → wide matrices. Days with no row are unrecorded (NaN)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(f"History is missing columns: {missing}")
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["store_id"] = frame["store_id"].astype(str)
    frame["product_id"] = frame["product_id"].astype(str)
    if "category" not in frame:
        frame["category"] = None
    if "unit_price" not in frame:
        frame["unit_price"] = np.nan
    if frame.duplicated(["store_id", "product_id", "date"]).any():
        raise ValueError("History has more than one row per store, product and date.")

    first, last = frame["date"].min(), frame["date"].max()
    n_days = (last - first).days + 1
    keys = frame[["store_id", "product_id"]].drop_duplicates().sort_values(["store_id", "product_id"])
    keys = keys.reset_index(drop=True)
    position = {k: i for i, k in enumerate(zip(keys["store_id"], keys["product_id"]))}
    rows = np.array([position[k] for k in zip(frame["store_id"], frame["product_id"])])
    cols = (frame["date"] - first).dt.days.to_numpy()
    sales = np.full((len(keys), n_days), np.nan, dtype=np.float32)
    prices = np.full((len(keys), n_days), np.nan, dtype=np.float32)
    sales[rows, cols] = pd.to_numeric(frame["units_sold"], errors="coerce").to_numpy(dtype=np.float32)
    raw_prices = pd.to_numeric(frame["unit_price"], errors="coerce").to_numpy(dtype=np.float32)
    raw_prices[raw_prices <= 0] = np.nan
    prices[rows, cols] = raw_prices

    priced = ~np.isnan(prices)
    first_price = [
        first.date() + dt.timedelta(days=int(np.argmax(p))) if p.any() else None for p in priced
    ]
    category = frame.groupby(["store_id", "product_id"])["category"].agg(
        lambda s: s.dropna().iloc[-1] if s.notna().any() else None
    )
    keys["category"] = [category.get(k) for k in zip(keys["store_id"], keys["product_id"])]
    keys["first_price_date"] = first_price
    return WideHistory(keys, sales, portable_v1.forward_fill_prices(prices), first.date())


def check_eligibility(wide: WideHistory, rules: dict | None = None) -> dict:
    rules = rules or config.OWN_DATA_ELIGIBILITY
    recorded = ~np.isnan(wide.sales)
    first = np.where(recorded.any(axis=1), np.argmax(recorded, axis=1), wide.sales.shape[1])
    per_product_days = wide.sales.shape[1] - first
    products_ok = int((per_product_days >= rules["minHistoryDaysPerProduct"]).sum())
    store_days = int(wide.sales.shape[1])
    observed = int(recorded.sum())
    checks = {
        "productsWithEnoughHistory": {"value": products_ok, "required": rules["minProducts"]},
        "storeHistoryDays": {"value": store_days, "required": rules["minStoreHistoryDays"]},
        "observedProductDays": {"value": observed, "required": rules["minObservedProductDays"]},
    }
    for check in checks.values():
        check["passed"] = check["value"] >= check["required"]
    return {"eligible": all(c["passed"] for c in checks.values()), "checks": checks}


def build_training_frame(wide: WideHistory, history_start: dt.date, country, subdivision) -> pd.DataFrame:
    """Feature rows for every day with a full lookback, dropping unrecorded targets."""
    targets = np.arange(config.LOOKBACK_DAYS, wide.sales.shape[1])
    frame = portable_v1.build_feature_frame(
        wide.series, wide.sales, wide.prices, wide.first_date, targets, history_start, country, subdivision
    )
    recorded = ~np.isnan(wide.sales)
    first = np.where(recorded.any(axis=1), np.argmax(recorded, axis=1), wide.sales.shape[1])
    target_col = np.repeat(targets, len(wide.series))
    series_first = np.tile(first, len(targets))
    keep = (target_col >= series_first + config.LOOKBACK_DAYS) & frame["sales"].notna().to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    frame["target_column"] = target_col[keep]
    return frame


RECURSIVE_HORIZON = 7


def _recursive_check(wide: WideHistory, booster, features, levels, backtest_start: int,
                     country, subdivision) -> dict:
    """Recursive 7-day backtest from every origin inside the backtest window (future prices unknown)."""
    from mlengine.evaluation.backtest import recursive_backtest
    from mlengine.models.artifact import ModelArtifact
    from mlengine.models.forecaster import Forecaster, SeriesBatch

    temporary = ModelArtifact(path=Path("."), metadata={
        "modelId": "candidate", "version": "candidate", "featureSchema": portable_v1.SCHEMA_NAME,
        "featureColumns": features, "categoricalLevels": levels,
        "training": {"historyStartDate": str(wide.first_date), "endDate": str(wide.last_date),
                     "country": country, "subdivision": subdivision},
        "capabilities": {}}, _booster=booster)
    last = wide.sales.shape[1] - 1

    def build(origin: int):
        batch = SeriesBatch(wide.series["product_id"].tolist(), wide.series["store_id"].tolist(),
                            wide.first_date + dt.timedelta(days=origin), wide.sales[:, : origin + 1],
                            wide.prices[:, : origin + 1], None, wide.series["category"].tolist(),
                            wide.series["first_price_date"].tolist())
        return batch, wide.sales[:, origin + 1 : origin + 1 + RECURSIVE_HORIZON]

    origins = range(backtest_start - 1, last - RECURSIVE_HORIZON + 1)
    return recursive_backtest(Forecaster(temporary), build, origins, RECURSIVE_HORIZON)


def train_own_data(
    history: pd.DataFrame,
    workspace_id: str,
    country: str | None = None,
    subdivision: str | None = None,
    registry_dir: Path | None = None,
    rules: dict | None = None,
    register: bool = True,
) -> dict:
    wide = to_wide(history)
    eligibility = check_eligibility(wide, rules)
    result: dict = {"workspaceId": workspace_id, "eligibility": eligibility, "registered": False,
                    "historyStartDate": str(wide.first_date), "historyEndDate": str(wide.last_date)}
    if not eligibility["eligible"]:
        result["status"] = "NOT_ELIGIBLE"
        return result

    frame = build_training_frame(wide, wide.first_date, country, subdivision)
    last = wide.sales.shape[1] - 1
    backtest_start = last - config.OWN_DATA_BACKTEST_DAYS + 1
    valid_start = backtest_start - config.OWN_DATA_VALID_DAYS
    column = frame["target_column"]
    train_rows, valid_rows = frame[column < valid_start], frame[column.between(valid_start, backtest_start - 1)]
    backtest_rows = frame[column >= backtest_start]
    if min(len(train_rows), len(valid_rows), len(backtest_rows)) == 0:
        result["status"] = "NOT_ELIGIBLE"
        result["message"] = "Not enough recorded days for training, validation and backtest windows."
        return result

    features = portable_v1.FEATURE_COLUMNS
    model = train.fit_with_early_stopping(train_rows[features], train_rows["sales"],
                                          valid_rows[features], valid_rows["sales"])
    n_trees = train.best_iteration(model)
    before_backtest = pd.concat([train_rows, valid_rows])
    refit = train.fit_fixed(before_backtest[features], before_backtest["sales"], n_trees)
    scored = backtest_rows[backtest_rows["sales_lag_1"].notna()]
    ai = refit.predict(scored[features])
    window = {"startDate": str(wide.first_date + dt.timedelta(days=int(backtest_start))),
              "endDate": str(wide.last_date)}
    backtest = {
        "window": window, "series": "all products in the workspace", "oneStepAhead": True,
        "ai": metrics.summary(scored["sales"], ai),
        "naive_last_value": metrics.summary(scored["sales"], scored["sales_lag_1"]),
        "seasonal_naive_7": metrics.summary(scored["sales"], scored["sales_lag_7"]),
    }
    gate_passed = bool(
        backtest["ai"]["rmse"] is not None
        and backtest["ai"]["rmse"] < backtest["naive_last_value"]["rmse"]
        and backtest["ai"]["mae"] < backtest["naive_last_value"]["mae"]
    )
    result.update(backtest=backtest, bestIteration=n_trees,
                  gate={"passed": gate_passed, "rule": "AI RMSE and MAE both below naive_last_value"})
    if not gate_passed:
        result["status"] = "GATE_FAILED"
        return result

    levels = {c: [str(v) for v in frame[c].cat.categories] for c in portable_v1.CATEGORICAL_COLUMNS}
    recursive = _recursive_check(wide, refit.booster_, features, levels, backtest_start, country, subdivision)
    horizons = [1, RECURSIVE_HORIZON] if recursive["validated"] else [1]
    result.update(recursive=recursive, horizons=horizons)
    if not register:
        result["status"] = "GATE_PASSED"
        return result

    final = train.fit_fixed(frame[features], frame["sales"], n_trees)
    registry = ModelRegistry(registry_dir)
    model_id = model_id_for(workspace_id)
    version = f"{len(registry.list(model_id)) + 1}.0.0"
    metadata = {
        "modelId": model_id, "version": version, "status": "active", "algorithm": "lightgbm",
        "featureSchema": portable_v1.SCHEMA_NAME, "featureColumns": features, "categoricalLevels": levels,
        "params": {**config.LGBM_PARAMS, "n_estimators": n_trees},
        "training": {
            "workspaceId": workspace_id, "country": country, "subdivision": subdivision,
            "historyStartDate": str(wide.first_date), "startDate": str(wide.first_date),
            "endDate": str(wide.last_date), "rows": int(len(frame)), "products": int(len(wide.series)),
        },
        "capabilities": {
            "horizons": horizons, "experimentalMaxHorizon": 28, "frequency": "DAILY",
            "minHistoryDays": config.LOOKBACK_DAYS,
            "maxMissingDaysInLookback": config.MAX_MISSING_DAYS_IN_LOOKBACK,
            "predictionIntervals": False, "baselines": ["naive_last_value", "seasonal_naive_7"],
        },
        "evaluation": {"backtest": backtest, "gate": result["gate"], "recursive": recursive},
    }
    card = {
        "title": "Demand model: LightGBM (Tweedie), trained on your sales",
        "model": f"{model_id}@{version}",
        "trainedOn": {"data": "This workspace's uploaded daily sales", "products": int(len(wide.series)),
                      "startDate": str(wide.first_date), "endDate": str(wide.last_date)},
        "predicts": "Expected units sold per product per day.",
        "horizonDays": horizons,
        "inputsPlainLanguage": [
            "Sales over the last 1, 7, 14 and 21 days",
            "Average and variability of sales over the last 3, 7 and 14 days",
            "Day of week, day of month, month and year",
            f"Public holidays ({country or 'none configured'})",
            "The product's price (the last known price is assumed for future days)",
        ],
        "backtest": backtest,
        "horizonAccuracy": {
            "description": f"Accuracy by days ahead for {RECURSIVE_HORIZON}-day forecasts made during the backtest window.",
            "validated": recursive["validated"],
            "byDay": [{"day": d["day"], "aiRmse": d["ai"]["rmse"], "aiMae": d["ai"]["mae"],
                       "naiveRmse": d["naive_last_value"]["rmse"], "naiveMae": d["naive_last_value"]["mae"]}
                      for d in recursive["byDay"]],
        },
        "limitations": [
            "Trained only on this workspace's history; new products have less to learn from.",
            "7-day forecasts are offered only if they beat both baselines on this workspace's backtest."
            if recursive["validated"] else "Validated for next-day forecasts only.",
            "Gives a single expected value, with no uncertainty range.",
        ],
    }
    artifact = save_artifact(registry.root / model_id / version, final.booster_, metadata, card)
    result.update(status="TRAINED", registered=True, model={"id": model_id, "version": artifact.version})
    return result
