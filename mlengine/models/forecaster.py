"""Inference: one-step and recursive multi-day forecasts for a batch of series.

Training and inference build features with the same functions (``features.m5_v1`` and
``features.portable_v1``). Inference only has to arrange the request's history into the same
wide-matrix shape: series × days, with NaN for unrecorded days and for the days being forecast.

Horizons longer than one day are produced recursively: the day-1 prediction is written into the
matrix as if it had been observed, features are rebuilt, and day 2 is predicted, and so on.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlengine import config
from mlengine.features import m5_v1, portable_v1
from mlengine.features.core import missing_in_lookback
from mlengine.models.artifact import ModelArtifact


class ForecastRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class SeriesBatch:
    """Daily history for several series, all ending on ``as_of`` (the last column)."""

    product_ids: list[str]
    store_ids: list[str]
    as_of: dt.date
    sales: np.ndarray  # float32 (n, D); NaN = unrecorded day
    prices: np.ndarray | None = None  # float32 (n, D) selling price per history day
    future_prices: np.ndarray | None = None  # float32 (n, H) known prices for forecast days
    categories: list[str | None] | None = None
    first_price_dates: list[dt.date | None] | None = None

    def __post_init__(self):
        self.sales = np.asarray(self.sales, dtype=np.float32)
        if self.sales.ndim != 2 or self.sales.shape[0] != len(self.product_ids):
            raise ValueError("sales must be a (n_series, n_days) matrix")

    @property
    def n_series(self) -> int:
        return len(self.product_ids)

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    @property
    def first_date(self) -> dt.date:
        return self.as_of - dt.timedelta(days=self.n_days - 1)


@dataclass
class ForecastOutput:
    ai: np.ndarray  # (n, H); NaN where the AI forecast is unavailable
    unavailable_reasons: list[str | None]
    price_assumptions: list[str]
    missing_in_lookback: np.ndarray
    warnings: list[list[str]] = field(default_factory=list)


def _first_valid_index(matrix: np.ndarray) -> np.ndarray:
    recorded = ~np.isnan(matrix)
    return np.where(recorded.any(axis=1), np.argmax(recorded, axis=1), matrix.shape[1])


class Forecaster:
    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        self.metadata = artifact.metadata
        self.feature_columns: list[str] = self.metadata["featureColumns"]
        self.levels: dict[str, list] = self.metadata["categoricalLevels"]
        if artifact.schema == m5_v1.SCHEMA_NAME:
            calendar = artifact.frame("calendar")
            if calendar is None:
                raise ValueError(f"{artifact.ref} has no calendar.parquet")
            self._calendar_features = m5_v1.build_calendar_features(calendar)
            self._calendar_end = dt.date.fromisoformat(str(calendar["date"].max())[:10])
            self._week_by_day = self._calendar_features.set_index("d_num")["wm_yr_wk"]
            self._week_index_by_day = self._calendar_features.set_index("d_num")
        elif artifact.schema != portable_v1.SCHEMA_NAME:
            raise ValueError(f"Unknown feature schema {artifact.schema!r}")

    # ------------------------------------------------------------------ public
    def forecast(self, batch: SeriesBatch, horizon: int) -> ForecastOutput:
        if horizon < 1:
            raise ForecastRequestError("HORIZON_NOT_SUPPORTED", "Horizon must be at least 1 day.")
        n, history_days = batch.n_series, batch.n_days
        pad = max(0, config.LOOKBACK_DAYS - history_days)
        width = pad + history_days + horizon
        sales = np.full((n, width), np.nan, dtype=np.float32)
        sales[:, pad : pad + history_days] = batch.sales
        as_of_column = pad + history_days - 1

        reasons: list[str | None] = [None] * n
        first_recorded = _first_valid_index(sales[:, : as_of_column + 1])
        gaps = missing_in_lookback(sales, as_of_column)
        for i in range(n):
            if as_of_column - first_recorded[i] + 1 < config.LOOKBACK_DAYS:
                reasons[i] = "INSUFFICIENT_HISTORY"
            elif gaps[i] > config.MAX_MISSING_DAYS_IN_LOOKBACK:
                reasons[i] = "TOO_MANY_GAPS"

        prices, assumptions = self._price_matrix(batch, pad, horizon)
        warnings: list[list[str]] = [[] for _ in range(n)]

        if self.artifact.schema == m5_v1.SCHEMA_NAME:
            self._forecast_m5(batch, sales, prices, pad, horizon, reasons, warnings)
        else:
            self._forecast_portable(batch, sales, prices, pad, horizon, reasons, warnings)

        ai = sales[:, pad + history_days :].copy()
        for i, reason in enumerate(reasons):
            if reason is not None:
                ai[i] = np.nan
        return ForecastOutput(ai, reasons, assumptions, gaps, warnings)

    # ------------------------------------------------------------------ helpers
    def _price_matrix(self, batch: SeriesBatch, pad: int, horizon: int) -> tuple[np.ndarray, list[str]]:
        n, history_days = batch.n_series, batch.n_days
        prices = np.full((n, pad + history_days + horizon), np.nan, dtype=np.float32)
        if batch.prices is not None:
            prices[:, pad : pad + history_days] = np.asarray(batch.prices, dtype=np.float32)
        known_future = np.zeros((n, horizon), dtype=bool)
        if batch.future_prices is not None:
            future = np.asarray(batch.future_prices, dtype=np.float32)[:, :horizon]
            prices[:, pad + history_days : pad + history_days + future.shape[1]] = future
            known_future[:, : future.shape[1]] = ~np.isnan(future)
        filled = portable_v1.forward_fill_prices(prices)
        assumptions = []
        for i in range(n):
            if known_future[i].all():
                assumptions.append("KNOWN")
            elif np.isnan(filled[i, -1]):
                assumptions.append("NONE")
            else:
                assumptions.append("LAST_KNOWN")
        return filled, assumptions

    def _first_price_dates(self, batch: SeriesBatch, prices: np.ndarray, first_date: dt.date,
                           warnings: list[list[str]]) -> list[dt.date | None]:
        given = batch.first_price_dates or [None] * batch.n_series
        first_priced = _first_valid_index(prices)
        out = []
        for i, value in enumerate(given):
            if value is None and first_priced[i] < prices.shape[1]:
                value = first_date + dt.timedelta(days=int(first_priced[i]))
                warnings[i].append("FIRST_PRICE_DATE_INFERRED")
            out.append(value)
        return out

    def _with_levels(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.feature_columns].copy()
        for column, levels in self.levels.items():
            if column in x:
                x[column] = pd.Categorical(x[column].astype(object), categories=levels)
        return x

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.artifact.booster.predict(self._with_levels(frame))
        return np.maximum(np.asarray(predictions, dtype=np.float32), 0)

    def _forecast_m5(self, batch, sales, prices, pad, horizon, reasons, warnings) -> None:
        as_of_day = config.date_to_m5_day(batch.as_of)
        if as_of_day + horizon > config.date_to_m5_day(self._calendar_end):
            raise ForecastRequestError(
                "DATE_OUTSIDE_CALENDAR", f"The M5 calendar ends on {self._calendar_end}; cannot forecast past it."
            )
        known_items, known_stores = set(self.levels["item_id"]), set(self.levels["store_id"])
        for i, (item, store) in enumerate(zip(batch.product_ids, batch.store_ids)):
            if item not in known_items or store not in known_stores:
                reasons[i] = "PRODUCT_NOT_IN_MODEL"
        rows = np.array([i for i, r in enumerate(reasons) if r != "PRODUCT_NOT_IN_MODEL"], dtype=int)
        if rows.size == 0:
            return
        first_day = as_of_day - (sales.shape[1] - horizon) + 1
        first_date = config.m5_day_to_date(first_day)
        identity = m5_v1.identity_from_ids(
            [batch.product_ids[i] for i in rows], [batch.store_ids[i] for i in rows]
        )
        release_dates = self._first_price_dates(batch, prices, first_date, warnings)
        week_lookup = self._week_index_by_day
        release_index = []
        for i in rows:
            value = release_dates[i]
            day = config.date_to_m5_day(value) if value is not None else None
            release_index.append(float(week_lookup.at[day, "week_index"]) if day in week_lookup.index else np.nan)
        release = pd.DataFrame({"store_id": identity["store_id"], "item_id": identity["item_id"],
                                "release_week_index": release_index})
        sub_sales, sub_prices = sales[rows], prices[rows]
        history_width = sales.shape[1] - horizon
        for h in range(horizon):
            target_day = as_of_day + 1 + h
            week = int(self._week_by_day.at[target_day])
            price_frame = pd.DataFrame({
                "store_id": identity["store_id"], "item_id": identity["item_id"],
                "wm_yr_wk": week, "sell_price": sub_prices[:, history_width + h],
            }).dropna(subset=["sell_price"])
            context = m5_v1.M5Context(self._calendar_features, price_frame, release)
            frame = m5_v1.build_feature_frame(identity, sub_sales, first_day, np.array([target_day]), context)
            sub_sales[:, history_width + h] = self._predict(frame)
        sales[rows] = sub_sales

    def _forecast_portable(self, batch, sales, prices, pad, horizon, reasons, warnings) -> None:
        training = self.metadata["training"]
        history_start = dt.date.fromisoformat(training["historyStartDate"])
        first_date = batch.first_date - dt.timedelta(days=pad)
        categories = batch.categories or [None] * batch.n_series
        known_products = set(self.levels.get("product_id", []))
        for i, product in enumerate(batch.product_ids):
            if product not in known_products:
                warnings[i].append("PRODUCT_NEW_SINCE_TRAINING")
        series = pd.DataFrame({
            "product_id": batch.product_ids,
            "category": [c or "UNCATEGORISED" for c in categories],
            "store_id": batch.store_ids,
            "first_price_date": self._first_price_dates(batch, prices, first_date, warnings),
        })
        history_width = sales.shape[1] - horizon
        for h in range(horizon):
            frame = portable_v1.build_feature_frame(
                series, sales, prices, first_date, np.array([history_width + h]), history_start,
                training.get("country"), training.get("subdivision"),
            )
            sales[:, history_width + h] = self._predict(frame)
