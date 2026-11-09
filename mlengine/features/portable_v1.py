"""Feature schema ``portable-v1``: the Stage 3 recipe without M5-specific inputs.

Used for models trained on a business's own sales history. Compared with ``m5-v1`` it:

* replaces the six M5 identity columns with ``product_id``, ``category`` and ``store_id``;
* replaces the M5 day/week numbers with ``days_since_history_start``;
* replaces M5 events and the California SNAP flag with the country's public holidays;
* uses a daily price, carried forward from the last known price.

Lags, rolling windows and calendar fields are identical to ``m5-v1``.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

from mlengine.features.core import window_feature_names, window_features

SCHEMA_NAME = "portable-v1"
CATEGORICAL_COLUMNS = ["product_id", "category", "store_id", "holiday_name"]
FEATURE_COLUMNS = [
    "product_id", "category", "store_id", "days_since_history_start",
    *window_feature_names(),
    "weekday_num", "month_num", "year_num", "day_of_month", "is_weekend",
    "holiday_name", "is_holiday",
    "sell_price", "price_missing", "weeks_since_first_price",
]
NO_HOLIDAY = "none"


@lru_cache(maxsize=32)
def _holiday_table(country: str, subdivision: str | None, first_year: int, last_year: int) -> dict:
    import holidays

    calendar = holidays.country_holidays(country, subdiv=subdivision, years=range(first_year, last_year + 1))
    return {day: name.split(";")[0].strip() for day, name in calendar.items()}


def holiday_names(dates: pd.DatetimeIndex, country: str | None, subdivision: str | None = None) -> np.ndarray:
    if not country or len(dates) == 0:
        return np.full(len(dates), NO_HOLIDAY, dtype=object)
    table = _holiday_table(country, subdivision, int(dates.year.min()), int(dates.year.max()))
    return np.array([table.get(day.date(), NO_HOLIDAY) for day in dates], dtype=object)


def forward_fill_prices(price_matrix: np.ndarray) -> np.ndarray:
    """Carry each series' last known price forward along the day axis."""
    prices = np.asarray(price_matrix, dtype=np.float32)
    valid = ~np.isnan(prices)
    index = np.where(valid, np.arange(prices.shape[1]), 0)
    np.maximum.accumulate(index, axis=1, out=index)
    filled = prices[np.arange(prices.shape[0])[:, None], index]
    filled[~np.maximum.accumulate(valid, axis=1)] = np.nan
    return filled


def build_feature_frame(
    series: pd.DataFrame,
    sales_matrix: np.ndarray,
    price_matrix: np.ndarray,
    first_date: dt.date,
    target_columns: np.ndarray,
    history_start: dt.date,
    country: str | None,
    subdivision: str | None = None,
) -> pd.DataFrame:
    """One row per (target day, series), target-day-major like ``m5-v1``.

    ``series`` has ``product_id``, ``category``, ``store_id`` and ``first_price_date`` per row of the
    matrices. Column j of both matrices is ``first_date + j`` days. ``price_matrix`` must already be
    forward-filled (see :func:`forward_fill_prices`).
    """
    target_columns = np.asarray(target_columns, dtype=np.int64)
    n_series, n_targets = len(series), len(target_columns)
    target_dates = pd.DatetimeIndex(pd.Timestamp(first_date) + pd.to_timedelta(target_columns, unit="D"))
    day_dates = target_dates.repeat(n_series)

    data: dict[str, object] = {}
    for column in ["product_id", "category", "store_id"]:
        data[column] = np.tile(series[column].astype(str).to_numpy(dtype=object), n_targets)
    data["days_since_history_start"] = np.repeat(
        (target_dates - pd.Timestamp(history_start)).days.to_numpy(), n_series
    ).astype("int32")
    data.update(window_features(sales_matrix, target_columns))
    data["weekday_num"] = day_dates.dayofweek.to_numpy().astype("int8")
    data["month_num"] = day_dates.month.to_numpy().astype("int8")
    data["year_num"] = day_dates.year.to_numpy().astype("int16")
    data["day_of_month"] = day_dates.day.to_numpy().astype("int8")
    data["is_weekend"] = (data["weekday_num"] >= 5).astype("int8")
    names = np.repeat(holiday_names(target_dates, country, subdivision), n_series)
    data["holiday_name"] = names
    data["is_holiday"] = (names != NO_HOLIDAY).astype("int8")
    prices = np.asarray(price_matrix, dtype=np.float32)
    data["sell_price"] = prices[:, target_columns].T.reshape(-1)
    data["price_missing"] = np.isnan(data["sell_price"]).astype("int8")
    first_price = pd.to_datetime(series["first_price_date"]).to_numpy()
    weeks = (day_dates.to_numpy().reshape(n_targets, n_series) - first_price[None, :]) / np.timedelta64(7, "D")
    data["weeks_since_first_price"] = np.floor(weeks).astype("float32").reshape(-1)

    inside = target_columns < sales_matrix.shape[1]
    target_values = np.full((n_series, n_targets), np.nan, dtype=np.float32)
    target_values[:, inside] = sales_matrix[:, target_columns[inside]]
    data["sales"] = target_values.T.reshape(-1)
    data["date"] = day_dates

    frame = pd.DataFrame(data)
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame
