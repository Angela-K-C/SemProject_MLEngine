from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "m5_parity"
REFERENCE_MODEL_DIR = ROOT / "models" / "m5-ca1-foods"


def synthetic_history(n_products: int = 15, n_days: int = 200, start: str = "2025-01-01", seed: int = 0,
                      drop_share: float = 0.0) -> pd.DataFrame:
    """Daily Poisson demand with a weekly pattern, in the canonical long shape."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for p in range(n_products):
        base = rng.uniform(0.5, 8)
        weekly = 1 + 0.5 * np.sin(2 * np.pi * (dates.dayofweek.to_numpy() + p) / 7)
        units = rng.poisson(base * weekly)
        price = round(rng.uniform(20, 200), 0)
        for d, u in zip(dates, units):
            rows.append({"store_id": "S1", "product_id": f"P{p:03d}", "category": f"C{p % 3}",
                         "date": d.date(), "units_sold": float(u), "unit_price": price})
    frame = pd.DataFrame(rows)
    if drop_share:
        frame = frame.drop(index=frame.sample(frac=drop_share, random_state=seed).index)
    return frame.reset_index(drop=True)


SMALL_RULES = {"minProducts": 5, "minHistoryDaysPerProduct": 21, "minStoreHistoryDays": 120,
               "minObservedProductDays": 1000}


@pytest.fixture(scope="session")
def own_data_registry(tmp_path_factory):
    from mlengine.pipelines.own_data import train_own_data

    root = tmp_path_factory.mktemp("registry")
    result = train_own_data(synthetic_history(), "ws_test", "KE", None, root, rules=SMALL_RULES)
    assert result["status"] == "TRAINED", result
    return root, result


@pytest.fixture(scope="session")
def m5_fixture():
    if not FIXTURE_DIR.is_dir():
        pytest.skip("M5 parity fixture missing (run scripts/make_parity_fixture.py)")
    sales = pd.read_parquet(FIXTURE_DIR / "sales.parquet")
    prices = pd.read_parquet(FIXTURE_DIR / "prices.parquet")
    calendar = pd.read_parquet(FIXTURE_DIR / "calendar.parquet")
    expected = pd.read_parquet(FIXTURE_DIR / "expected_features.parquet")
    return sales, prices, calendar, expected


def as_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)
