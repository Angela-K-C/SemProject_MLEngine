"""Feature schema ``m5-v1``: the 31 features of the Stage 3 model.

Mirrors cells 37, 39 and 41 of ``LightGBM_BaseLocal.ipynb``: same columns, same order, same dtypes,
same joins and the same row order (target-day-major, series in sorted item order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlengine.data.m5 import EVENT_COLUMNS, IDENTITY_COLUMNS
from mlengine.features.core import window_features

SCHEMA_NAME = "m5-v1"

CALENDAR_FEATURE_COLUMNS = [
    "d_num", "date", "wm_yr_wk", "week_index", "weekday_num", "month_num", "year_num",
    "day_of_month", "is_weekend", *EVENT_COLUMNS, "snap_CA",
]
CATEGORICAL_COLUMNS = IDENTITY_COLUMNS + EVENT_COLUMNS
NON_FEATURE_COLUMNS = {"sales", "date"}


def build_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    features = calendar.copy()
    features["d_num"] = features["d"].astype(str).str.removeprefix("d_").astype("int32")
    features["weekday_num"] = features["date"].dt.dayofweek.astype("int8")
    features["month_num"] = features["date"].dt.month.astype("int8")
    features["year_num"] = features["date"].dt.year.astype("int16")
    features["day_of_month"] = features["date"].dt.day.astype("int8")
    features["is_weekend"] = features["weekday_num"].ge(5).astype("int8")
    for column in EVENT_COLUMNS:
        features[column] = features[column].astype("category")
    weeks = sorted(features["wm_yr_wk"].dropna().unique().tolist())
    week_index = {int(week): index for index, week in enumerate(weeks)}
    features["week_index"] = features["wm_yr_wk"].map(week_index).astype("int16")
    return features[CALENDAR_FEATURE_COLUMNS].rename(columns={"snap_CA": "snap"})


def build_release_weeks(prices: pd.DataFrame, calendar_features: pd.DataFrame) -> pd.DataFrame:
    """First priced week per item, as a week index (the notebook's ``releaseWeeks``)."""
    week_index = (
        calendar_features[["wm_yr_wk", "week_index"]].drop_duplicates().set_index("wm_yr_wk")["week_index"]
    )
    release = (
        prices.groupby(["store_id", "item_id"], as_index=False)["wm_yr_wk"].min()
        .rename(columns={"wm_yr_wk": "first_price_week"})
    )
    release["release_week_index"] = release["first_price_week"].map(week_index)
    return release


@dataclass
class M5Context:
    """Everything except sales that the m5-v1 features need."""

    calendar_features: pd.DataFrame
    prices: pd.DataFrame  # store_id, item_id, wm_yr_wk, sell_price
    release_weeks: pd.DataFrame  # store_id, item_id, release_week_index


def build_feature_frame(
    identity: pd.DataFrame,
    sales_matrix: np.ndarray,
    first_day: int,
    target_days: np.ndarray,
    context: M5Context,
) -> pd.DataFrame:
    """One row per (target day, series). Column j of ``sales_matrix`` is M5 day ``first_day + j``.

    The returned frame has the 31 feature columns plus ``sales`` (the target, NaN when unknown)
    and ``date``.
    """
    target_days = np.asarray(target_days, dtype=np.int32)
    target_columns = target_days.astype(np.int64) - first_day
    n_series = len(identity)

    data: dict[str, object] = {}
    for column in IDENTITY_COLUMNS:
        data[column] = np.tile(identity[column].astype(str).to_numpy(dtype=object), len(target_days))
    data["d_num"] = np.repeat(target_days, n_series).astype("int32")
    data.update(window_features(sales_matrix, target_columns))
    inside = target_columns < sales_matrix.shape[1]
    target_values = np.full((n_series, len(target_days)), np.nan, dtype=np.float32)
    target_values[:, inside] = sales_matrix[:, target_columns[inside]]
    data["sales"] = target_values.T.reshape(-1)
    frame = pd.DataFrame(data)

    frame = frame.merge(context.calendar_features, on="d_num", how="left", validate="many_to_one")
    if frame["wm_yr_wk"].isna().any():
        raise ValueError("Calendar join left unmatched days (target date outside the M5 calendar).")
    frame = frame.merge(
        context.prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one",
    )
    frame["price_missing"] = frame["sell_price"].isna().astype("int8")
    frame = frame.merge(
        context.release_weeks[["store_id", "item_id", "release_week_index"]],
        on=["store_id", "item_id"], how="left", validate="many_to_one",
    )
    frame["weeks_since_first_price"] = (
        frame["week_index"].astype("float32") - frame["release_week_index"].astype("float32")
    )
    frame = frame.drop(columns=["release_week_index", "week_index"])
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def identity_from_ids(item_ids, store_ids) -> pd.DataFrame:
    """Rebuild the six M5 identity columns from item and store IDs (e.g. FOODS_3_090, CA_1)."""
    items = pd.Series(list(item_ids), dtype=str)
    stores = pd.Series(list(store_ids), dtype=str)
    return pd.DataFrame({
        "id": items + "_" + stores + "_evaluation",
        "item_id": items,
        "dept_id": items.str.rsplit("_", n=1).str[0],
        "cat_id": items.str.split("_", n=1).str[0],
        "store_id": stores,
        "state_id": stores.str.split("_", n=1).str[0],
    })
