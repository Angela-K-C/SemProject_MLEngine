"""Export the Reference workspace (M5 CA_1 FOODS) in the canonical shape the backend stores.

Files written to ``data/processed/reference/`` (git-ignored):

* ``store.json``: store metadata.
* ``products.csv``: one row per product.
* ``sales_daily.csv``: one row per product per day, 2011-01-29 … 2016-05-22, with the day's price.
* ``prices_future.csv``: known prices for 2016-05-23 … 2016-06-19 (used for true forecasts).

Each CSV is also written as Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlengine import config
from mlengine.data.m5 import daily_price_matrix, first_price_dates, load_m5_scope


def export_reference(out_dir: Path | None = None, raw_dir: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir or config.project_root() / "data" / "processed" / "reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = load_m5_scope(raw_dir, verify_overlap=False)
    prices = daily_price_matrix(scope)
    n_series, n_days = scope.sales.shape
    dates = pd.date_range(config.M5_START_DATE, periods=config.M5_CALENDAR_END_DAY, freq="D")
    items = scope.identity["item_id"].to_numpy()

    sold = scope.sales > 0
    first_sale = np.where(sold.any(axis=1), np.argmax(sold, axis=1), -1)
    products = pd.DataFrame({
        "store_id": scope.store_id,
        "product_id": items,
        "name": items,
        "category": scope.identity["dept_id"].to_numpy(),
        "unit_of_measure": "units",
        "first_price_date": first_price_dates(scope).astype(str).to_numpy(),
        "first_sale_date": [str(dates[i].date()) if i >= 0 else None for i in first_sale],
    })

    sales_daily = pd.DataFrame({
        "store_id": scope.store_id,
        "product_id": np.repeat(items, n_days),
        "date": np.tile(dates[:n_days].strftime("%Y-%m-%d"), n_series),
        "units_sold": scope.sales.reshape(-1).astype("int32"),
        "unit_price": prices[:, :n_days].reshape(-1),
    })
    future_days = config.M5_CALENDAR_END_DAY - n_days
    prices_future = pd.DataFrame({
        "store_id": scope.store_id,
        "product_id": np.repeat(items, future_days),
        "date": np.tile(dates[n_days:].strftime("%Y-%m-%d"), n_series),
        "unit_price": prices[:, n_days:].reshape(-1),
    })

    store = {
        "storeId": scope.store_id, "name": "Demo Store (M5 CA_1)", "isReference": True,
        "country": "US", "subdivision": "CA", "timezone": "America/Los_Angeles", "currency": "USD",
        "historyStartDate": str(dates[0].date()), "historyEndDate": str(dates[n_days - 1].date()),
        "calendarEndDate": str(dates[-1].date()), "products": int(n_series),
        "modelId": "m5-ca1-foods",
        "source": "M5 Forecasting – Accuracy (Kaggle), used under the competition terms for research/education",
    }
    written = {}
    (out_dir / "store.json").write_text(json.dumps(store, indent=2))
    written["store"] = out_dir / "store.json"
    for name, frame in [("products", products), ("sales_daily", sales_daily), ("prices_future", prices_future)]:
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        written[name] = out_dir / f"{name}.csv"
    return written
