"""Train, evaluate and register the Reference (M5 CA_1 FOODS) model versions.

Produces three versions of ``m5-ca1-foods``, each usable only for as-of dates on or after the
end of its training data (the registry's leakage rule):

* ``<v>-dev``: trained on d_22–d_1885, early-stopped on d_1886–d_1913 (the notebook's dev model).
* ``<v>-bt``: refit on d_22–d_1913 with the early-stopped tree count, scored on d_1914–d_1941.
* ``<v>``: refit on d_22–d_1941. Serves the true forecast from 2016-05-22 onward.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mlengine import config
from mlengine.data.m5 import daily_price_matrix, load_m5_scope, scope_to_batch
from mlengine.evaluation.backtest import recursive_backtest
from mlengine.evaluation import metrics
from mlengine.evaluation.demand_class import classify_matrix
from mlengine.features import m5_v1
from mlengine.models import train
from mlengine.models.artifact import load_artifact, save_artifact
from mlengine.models.forecaster import Forecaster

log = logging.getLogger(__name__)

MODEL_ID = "m5-ca1-foods"
CANDIDATE_HORIZON = 7  # enabled only if the recursive backtest validates it
EXPERIMENTAL_MAX_HORIZON = 28
RECURSIVE_ORIGINS = range(config.M5_BACKTEST[0] - 1, config.M5_BACKTEST[1] - CANDIDATE_HORIZON + 1)


def _metrics_block(frame: pd.DataFrame, ai: np.ndarray, window: tuple[int, int], model_ref: str) -> dict:
    actual = frame["sales"].to_numpy()
    return {
        "window": {"startDay": window[0], "endDay": window[1],
                   "startDate": str(config.m5_day_to_date(window[0])),
                   "endDate": str(config.m5_day_to_date(window[1]))},
        "series": "all CA_1 FOODS items", "modelVersion": model_ref, "oneStepAhead": True,
        "ai": metrics.summary(actual, ai),
        "naive_last_value": metrics.summary(actual, frame["sales_lag_1"].to_numpy()),
        "seasonal_naive_7": metrics.summary(actual, frame["sales_lag_7"].to_numpy()),
    }


def _per_product(frame: pd.DataFrame, ai: np.ndarray, classes: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame({
        "product_id": frame["item_id"].astype(str).to_numpy(), "date": frame["date"].dt.date.astype(str),
        "actual": frame["sales"].to_numpy(), "ai": ai,
        "naive_last_value": frame["sales_lag_1"].to_numpy(), "seasonal_naive_7": frame["sales_lag_7"].to_numpy(),
    })
    rows = []
    for product, group in work.groupby("product_id", sort=True):
        row = {"product_id": product, "n": len(group), "actual_total": float(group["actual"].sum())}
        for method in ["ai", "naive_last_value", "seasonal_naive_7"]:
            for name, value in metrics.summary(group["actual"], group[method]).items():
                if name != "n":
                    row[f"{method}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows).merge(classes, on="product_id", how="left"), work


def _classes(scope, up_to_day: int) -> pd.DataFrame:
    result = classify_matrix(scope.sales[:, :up_to_day])
    frame = pd.DataFrame(result)
    frame.insert(0, "product_id", scope.identity["item_id"].to_numpy())
    return frame.rename(columns={"class": "demand_class", "label": "demand_label"})


def _card(version_ref: str, training: dict, backtest: dict | None, horizons: list[int], note: str,
          recursive: dict | None = None) -> dict:
    horizon_accuracy = None
    if recursive is not None:
        horizon_accuracy = {
            "description": f"Accuracy by days ahead for {recursive['horizon']}-day forecasts "
                           f"({recursive['origins']} forecast dates, all products, 2016-04-25 to 2016-05-22).",
            "validated": recursive["validated"],
            "byDay": [{"day": d["day"], "aiRmse": d["ai"]["rmse"], "aiMae": d["ai"]["mae"],
                       "naiveRmse": d["naive_last_value"]["rmse"], "naiveMae": d["naive_last_value"]["mae"]}
                      for d in recursive["byDay"]],
        }
    return {
        "title": "Demand model: LightGBM (Tweedie)",
        "model": version_ref,
        "trainedOn": {
            "data": "M5 Forecasting – Accuracy (Walmart), store CA_1, FOODS category",
            "products": 1437, "startDate": training["startDate"], "endDate": training["endDate"],
            "source": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        },
        "predicts": "Expected units sold per product per day.",
        "horizonDays": horizons,
        "inputsPlainLanguage": [
            "Sales over the last 1, 7, 14 and 21 days",
            "Average and variability of sales over the last 3, 7 and 14 days",
            "Day of week, day of month, month and year",
            "US holidays and events, and SNAP food-assistance days in California",
            "The product's price and how long it has been on sale",
        ],
        "backtest": backtest,
        "horizonAccuracy": horizon_accuracy,
        "limitations": [
            "Validated on grocery (FOODS) products in one store only.",
            "Less precise for products that sell on only some days (intermittent demand).",
            "Accuracy falls slowly with distance: days 2–7 are forecast from the model's own earlier "
            "predictions and run about 4% high on average. Beyond 7 days forecasts are experimental.",
            "Gives a single expected value, with no uncertainty range.",
            "Only valid for products in the M5 CA_1 FOODS catalogue.",
        ],
        "note": note,
    }


def release(registry_dir: Path | None = None, version: str = "1.1.0", raw_dir: Path | None = None) -> dict:
    registry_dir = Path(registry_dir or config.model_registry_dir())
    out_dir = registry_dir / MODEL_ID
    log.info("Loading M5 scope…")
    scope = load_m5_scope(raw_dir)
    calendar_features = m5_v1.build_calendar_features(scope.calendar)
    context = m5_v1.M5Context(calendar_features, scope.prices,
                              m5_v1.build_release_weeks(scope.prices, calendar_features))
    target_days = np.arange(config.M5_FEATURE_START_DAY, config.M5_LAST_OBSERVED_DAY + 1)
    frame = m5_v1.build_feature_frame(scope.identity, scope.sales, 1, target_days, context)
    columns = m5_v1.feature_columns(frame)
    day = frame["d_num"]
    dev_train = frame.loc[day.between(config.M5_FEATURE_START_DAY, config.M5_DEV_TRAIN_END)].reset_index(drop=True)
    dev_valid = frame.loc[day.between(*config.M5_DEV_VALID)].reset_index(drop=True)
    backtest = frame.loc[day.between(*config.M5_BACKTEST)].reset_index(drop=True)
    levels = {c: [str(v) for v in frame[c].cat.categories] for c in m5_v1.CATEGORICAL_COLUMNS}

    def metadata(ver: str, end_day: int, n_trees: int, rows: int, extra: dict) -> dict:
        return {
            "modelId": MODEL_ID, "version": ver, "status": "active", "algorithm": "lightgbm",
            "featureSchema": m5_v1.SCHEMA_NAME, "featureColumns": columns, "categoricalLevels": levels,
            "params": {**config.LGBM_PARAMS, "n_estimators": n_trees},
            "training": {
                "scope": {"store": scope.store_id, "category": scope.category_id},
                "startDay": config.M5_FEATURE_START_DAY, "endDay": end_day,
                "startDate": str(config.m5_day_to_date(config.M5_FEATURE_START_DAY)),
                "endDate": str(config.m5_day_to_date(end_day)), "rows": rows,
            },
            "capabilities": {
                "horizons": horizons, "experimentalMaxHorizon": EXPERIMENTAL_MAX_HORIZON,
                "frequency": "DAILY", "minHistoryDays": config.LOOKBACK_DAYS,
                "maxMissingDaysInLookback": config.MAX_MISSING_DAYS_IN_LOOKBACK,
                "predictionIntervals": False, "baselines": ["naive_last_value", "seasonal_naive_7"],
                "calendarEndDate": str(config.m5_day_to_date(config.M5_CALENDAR_END_DAY)),
            },
            **extra,
        }

    calendar_frame = scope.calendar.copy()
    results: dict = {}
    horizons = [1]

    # 1. Development model (early stopping) ---------------------------------------------------
    log.info("Training development model…")
    dev_model = train.fit_with_early_stopping(dev_train[columns], dev_train["sales"],
                                              dev_valid[columns], dev_valid["sales"])
    n_trees = train.best_iteration(dev_model)
    dev_pred = dev_model.predict(dev_valid[columns], num_iteration=n_trees)
    dev_block = _metrics_block(dev_valid, dev_pred, config.M5_DEV_VALID, f"{MODEL_ID}@{version}-dev")
    results["dev"] = dev_block
    results["bestIteration"] = n_trees
    dev_training = metadata(f"{version}-dev", config.M5_DEV_TRAIN_END, n_trees, len(dev_train),
                            {"evaluation": {"development": dev_block}})
    save_artifact(out_dir / f"{version}-dev", dev_model.booster_, dev_training,
                  _card(f"{MODEL_ID}@{version}-dev", dev_training["training"], dev_block, [1],
                        "Development model: used for replays with as-of dates from 2016-03-27 to 2016-04-23."),
                  {"calendar": calendar_frame}, num_iteration=n_trees)

    # 2. Backtest model -------------------------------------------------------------------------
    log.info("Training backtest model (%s trees)…", n_trees)
    combined = pd.concat([dev_train, dev_valid], ignore_index=True)
    bt_model = train.fit_fixed(combined[columns], combined["sales"], n_trees)
    bt_pred = bt_model.predict(backtest[columns])
    bt_block = _metrics_block(backtest, bt_pred, config.M5_BACKTEST, f"{MODEL_ID}@{version}-bt")
    results["backtest"] = bt_block
    classes = _classes(scope, config.M5_BACKTEST[0] - 1)
    per_product, predictions = _per_product(backtest, bt_pred, classes)
    by_class = {}
    for name, group in predictions.merge(classes[["product_id", "demand_class"]], on="product_id").groupby("demand_class"):
        by_class[name] = {"products": int(group["product_id"].nunique()),
                          **{m: metrics.summary(group["actual"], group[m])
                             for m in ["ai", "naive_last_value", "seasonal_naive_7"]}}
    bt_block["byDemandClass"] = by_class
    del combined
    bt_meta = metadata(f"{version}-bt", config.M5_DEV_VALID[1], n_trees, len(dev_train) + len(dev_valid),
                       {"evaluation": {"development": dev_block, "backtest": bt_block}})
    frames = {"calendar": calendar_frame, "backtest/per_product": per_product,
              "backtest/predictions": predictions}
    save_artifact(out_dir / f"{version}-bt", bt_model.booster_, bt_meta,
                  _card(f"{MODEL_ID}@{version}-bt", bt_meta["training"], bt_block, [1],
                        "Backtest model: used for replays with as-of dates from 2016-04-24 to 2016-05-21."),
                  frames)
    (out_dir / f"{version}-bt" / "backtest" / "summary.json").write_text(json.dumps(bt_block, indent=2))

    log.info("Recursive %s-day backtest…", CANDIDATE_HORIZON)
    bt_artifact = load_artifact(out_dir / f"{version}-bt")
    prices = daily_price_matrix(scope)

    def build(origin: int):
        batch = scope_to_batch(scope, origin, 365, CANDIDATE_HORIZON, daily_prices=prices)
        return batch, scope.sales[:, origin : origin + CANDIDATE_HORIZON]

    recursive = recursive_backtest(Forecaster(bt_artifact), build, RECURSIVE_ORIGINS, CANDIDATE_HORIZON)
    results["recursive"] = recursive
    if recursive["validated"]:
        horizons = [1, CANDIDATE_HORIZON]
    bt_meta["capabilities"]["horizons"] = horizons
    dev_meta_file = out_dir / f"{version}-dev" / "metadata.json"
    dev_meta = json.loads(dev_meta_file.read_text())
    dev_meta["capabilities"]["horizons"] = horizons  # same recipe; validated by the recursive backtest
    dev_meta_file.write_text(json.dumps(dev_meta, indent=2, default=str))
    bt_meta["evaluation"]["recursive"] = recursive
    save_artifact(out_dir / f"{version}-bt", bt_model.booster_, bt_meta,
                  _card(f"{MODEL_ID}@{version}-bt", bt_meta["training"], bt_block, horizons,
                        "Backtest model: used for replays with as-of dates from 2016-04-24 to 2016-05-21.",
                        recursive),
                  frames)
    (out_dir / f"{version}-bt" / "backtest" / "summary.json").write_text(json.dumps(bt_block, indent=2))

    # 3. Production model -----------------------------------------------------------------------
    log.info("Training production model…")
    full = frame.loc[day.between(config.M5_FEATURE_START_DAY, config.M5_LAST_OBSERVED_DAY)]
    prod_model = train.fit_fixed(full[columns], full["sales"], n_trees)
    prod_meta = metadata(version, config.M5_LAST_OBSERVED_DAY, n_trees, len(full),
                         {"evaluation": {"development": dev_block, "backtest": bt_block, "recursive": recursive,
                                         "backtestNote": f"Measured with {MODEL_ID}@{version}-bt, which was "
                                                         "trained on the same recipe up to 2016-04-24."}})
    current_classes = _classes(scope, config.M5_LAST_OBSERVED_DAY)
    save_artifact(out_dir / version, prod_model.booster_, prod_meta,
                  _card(f"{MODEL_ID}@{version}", prod_meta["training"], bt_block, horizons,
                        "Active model: true forecasts from 2016-05-22 (the last day of observed sales).", recursive),
                  {"calendar": calendar_frame, "backtest/per_product": per_product,
                   "demand_classes": current_classes})
    (out_dir / version / "backtest" / "summary.json").write_text(json.dumps(bt_block, indent=2))
    results["createdAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return results
