"""Load the raw M5 files and scope them to one store and category.

This mirrors cells 5, 7 and 35 of ``LightGBM_BaseLocal.ipynb``: the same dtypes, the same
validation/evaluation overlap check and the same series order (sorted by item_id, store_id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlengine import config

IDENTITY_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


@dataclass
class M5Scope:
    """One store/category slice of M5, held as a wide matrix (series × days)."""

    store_id: str
    category_id: str
    identity: pd.DataFrame  # IDENTITY_COLUMNS as str, one row per series
    sales: np.ndarray  # float32, shape (n_series, n_days); column j is day j + 1
    calendar: pd.DataFrame  # raw calendar.csv with parsed dates
    prices: pd.DataFrame  # store_id, item_id, wm_yr_wk, sell_price for the scoped items

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]


def _day_columns(columns) -> list[str]:
    return [column for column in columns if re.fullmatch(r"d_\d+", str(column))]


def _read_sales(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {column: "category" for column in IDENTITY_COLUMNS if column in header}
    dtypes.update({column: "uint16" for column in _day_columns(header)})
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def load_calendar(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or config.raw_data_dir())
    category_columns = ["d", "weekday", *EVENT_COLUMNS]
    return pd.read_csv(
        raw_dir / "calendar.csv",
        dtype={column: "category" for column in category_columns},
        parse_dates=["date"],
        low_memory=False,
    )


def load_m5_scope(
    raw_dir: Path | None = None,
    store_id: str = config.M5_REFERENCE_STORE,
    category_id: str = config.M5_REFERENCE_CATEGORY,
    verify_overlap: bool = True,
) -> M5Scope:
    raw_dir = Path(raw_dir or config.raw_data_dir())
    evaluation_path = raw_dir / "sales_train_evaluation.csv"
    for required in [evaluation_path, raw_dir / "calendar.csv", raw_dir / "sell_prices.csv"]:
        if not required.is_file():
            raise FileNotFoundError(f"Missing M5 file: {required}. Download it from Kaggle into {raw_dir}.")

    evaluation = _read_sales(evaluation_path)
    day_columns = _day_columns(evaluation.columns)
    scoped = evaluation.loc[
        (evaluation["store_id"] == store_id) & (evaluation["cat_id"] == category_id)
    ].reset_index(drop=True)
    del evaluation
    if scoped.empty:
        raise ValueError(f"No series found for store_id={store_id!r}, cat_id={category_id!r}.")

    key = ["item_id", "store_id"]
    if scoped.duplicated(key).any():
        raise ValueError("Scoped sales contain duplicate item-store series.")
    scoped = scoped.sort_values(key, kind="stable").reset_index(drop=True)

    if verify_overlap:
        validation_path = raw_dir / "sales_train_validation.csv"
        if validation_path.is_file():
            validation = _read_sales(validation_path)
            validation_days = _day_columns(validation.columns)
            validation = validation.loc[
                (validation["store_id"] == store_id) & (validation["cat_id"] == category_id)
            ]
            validation = validation.sort_values(key, kind="stable").reset_index(drop=True)
            if not validation[key].astype("string").equals(scoped[key].astype("string")):
                raise ValueError("Validation and evaluation files contain different scoped series.")
            if not np.array_equal(validation[validation_days].to_numpy(), scoped[validation_days].to_numpy()):
                raise ValueError("Validation and evaluation sales differ over their overlapping days.")
            del validation

    identity = pd.DataFrame({column: scoped[column].astype(str).to_numpy() for column in IDENTITY_COLUMNS})
    sales = scoped[day_columns].to_numpy(dtype="float32", copy=True)

    calendar = load_calendar(raw_dir)

    price_header = pd.read_csv(raw_dir / "sell_prices.csv", nrows=0).columns
    price_dtypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"}
    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={k: v for k, v in price_dtypes.items() if k in price_header},
        low_memory=False,
    )
    prices = prices.loc[prices["store_id"].astype(str) == store_id].copy()
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)
    prices = prices.loc[prices["item_id"].isin(set(identity["item_id"]))].reset_index(drop=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Scoped price keys contain duplicates.")

    return M5Scope(store_id, category_id, identity, sales, calendar, prices)


def daily_price_matrix(scope: M5Scope, n_days: int | None = None) -> np.ndarray:
    """Weekly M5 prices spread onto days: float32 (n_series, n_days), NaN where unpriced."""
    n_days = n_days or config.M5_CALENDAR_END_DAY
    calendar = scope.calendar.iloc[:n_days]
    day_week = calendar["wm_yr_wk"].to_numpy()
    weeks = np.unique(day_week)
    week_pos = {int(week): i for i, week in enumerate(weeks)}
    item_pos = {item: i for i, item in enumerate(scope.identity["item_id"])}
    weekly = np.full((len(item_pos), len(weeks)), np.nan, dtype=np.float32)
    rows = scope.prices["item_id"].map(item_pos).to_numpy()
    cols = scope.prices["wm_yr_wk"].map(week_pos)
    valid = cols.notna().to_numpy()
    weekly[rows[valid], cols[valid].astype(int).to_numpy()] = scope.prices["sell_price"].to_numpy()[valid]
    return weekly[:, [week_pos[int(week)] for week in day_week]]


def first_price_dates(scope: M5Scope) -> pd.Series:
    """First day of each item's first priced Walmart week (the M5 'release' proxy)."""
    first_week = scope.prices.groupby("item_id")["wm_yr_wk"].min()
    week_start = scope.calendar.groupby("wm_yr_wk")["date"].min()
    dates = first_week.map(week_start)
    return scope.identity["item_id"].map(dates).dt.date


def scope_to_batch(scope: M5Scope, as_of_day: int, history_days: int = 365, horizon: int = 0,
                   rows=None, daily_prices: np.ndarray | None = None):
    """Arrange M5 data the way the backend would send it: history ending on ``as_of_day``.

    Known prices for the next ``horizon`` days are included as ``future_prices``.
    """
    from mlengine.models.forecaster import SeriesBatch

    rows = np.arange(len(scope.identity)) if rows is None else np.asarray(rows)
    start = max(1, as_of_day - history_days + 1)
    prices = daily_price_matrix(scope) if daily_prices is None else daily_prices
    first_prices = first_price_dates(scope)
    return SeriesBatch(
        product_ids=scope.identity["item_id"].iloc[rows].tolist(),
        store_ids=scope.identity["store_id"].iloc[rows].tolist(),
        as_of=config.m5_day_to_date(as_of_day),
        sales=scope.sales[rows, start - 1 : as_of_day],
        prices=prices[rows, start - 1 : as_of_day],
        future_prices=prices[rows, as_of_day : as_of_day + horizon] if horizon else None,
        first_price_dates=[first_prices.iloc[i] for i in rows],
    )
