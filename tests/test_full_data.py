"""Full-data parity (needs data/raw and a few minutes). Run with: MLENGINE_FULL_TESTS=1 pytest tests/test_full_data.py"""

import datetime as dt
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlengine import config
from mlengine.models.artifact import ModelRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("MLENGINE_FULL_TESTS") != "1" or not (config.raw_data_dir() / "calendar.csv").is_file(),
    reason="set MLENGINE_FULL_TESTS=1 with the M5 files in data/raw",
)


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    from mlengine.pipelines.m5_release import release

    root = tmp_path_factory.mktemp("registry")
    return root, release(root)


def test_training_reproduces_notebook(released):
    _, result = released
    assert result["bestIteration"] == 70
    assert result["dev"]["ai"]["rmse"] == pytest.approx(2.356, abs=0.005)
    assert result["dev"]["ai"]["mae"] == pytest.approx(1.308, abs=0.005)
    assert result["dev"]["naive_last_value"]["rmse"] == pytest.approx(3.194, abs=0.005)
    assert result["backtest"]["ai"]["rmse"] == pytest.approx(2.377, abs=0.005)
    assert result["backtest"]["ai"]["mae"] == pytest.approx(1.377, abs=0.005)


def test_service_backtest_reproduces_stage3(released, monkeypatch):
    """Acceptance criterion: the served model reproduces the Stage 3 backtest through the API."""
    from mlengine.data.m5 import daily_price_matrix, load_m5_scope
    from mlengine.service import app as service

    root, _ = released
    monkeypatch.setattr(service, "registry", ModelRegistry(root))
    service._forecasters.clear()
    scope = load_m5_scope(verify_overlap=False)
    prices = daily_price_matrix(scope)
    start_day = config.M5_BACKTEST[0] - 30
    series = []
    for i, item in enumerate(scope.identity["item_id"]):
        series.append({"productId": item, "storeId": "CA_1", "history": [
            {"date": str(config.m5_day_to_date(d)), "units": float(scope.sales[i, d - 1]),
             "price": None if np.isnan(prices[i, d - 1]) else float(prices[i, d - 1])}
            for d in range(start_day, config.M5_BACKTEST[1] + 1)]})
    response = TestClient(service.app).post("/internal/v1/backtest", json={
        "model": {"id": "m5-ca1-foods"}, "startDate": "2016-04-25", "endDate": "2016-05-22", "series": series})
    data = response.json()
    assert data["model"]["version"] == "1.1.0-bt"
    assert data["overall"]["ai"]["rmse"] == pytest.approx(2.377, abs=0.005)
    assert data["overall"]["naive_last_value"]["rmse"] == pytest.approx(3.219, abs=0.005)


def test_sample_shops(tmp_path):
    """The eligible sample trains and passes the gate; the small one isn't eligible; the messy one is messy."""
    import datetime as dt

    import pandas as pd

    from mlengine.pipelines.own_data import train_own_data
    from mlengine.pipelines.sample_shops import export_samples

    end = dt.date(2026, 11, 6)
    files = export_samples(tmp_path, end_date=end)
    eligible = pd.read_csv(files["sample_shop_eligible"])
    assert eligible["sales_date"].max() == "2026-11-06" and eligible["product_id"].nunique() == 100
    assert pd.Timestamp("2026-11-06").weekday() == pd.Timestamp("2016-05-20").weekday()

    def zero_filled(frame):
        frame = frame.assign(date=pd.to_datetime(frame["sales_date"]))
        last, parts = frame["date"].max(), []
        for pid, g in frame.groupby("product_id"):
            g = g.set_index("date").reindex(pd.date_range(g["date"].min(), last))
            g = g.assign(units_sold=g["units_sold"].fillna(0), unit_price=g["unit_price"].ffill(),
                         category=g["category"].ffill(), product_id=pid)
            parts.append(g.rename_axis("date").reset_index())
        return pd.concat(parts).assign(store_id="S")

    result = train_own_data(zero_filled(eligible), "sample", "US", "TX", registry_dir=tmp_path / "reg", register=False)
    assert result["gate"]["passed"] and result["horizons"] == [1, 7]
    small = pd.read_csv(files["sample_shop_small"]).rename(columns={"sales_date": "date"}).assign(store_id="S")
    assert train_own_data(small, "small", register=False)["status"] == "NOT_ELIGIBLE"
    assert files["sample_shop_messy"].read_text().startswith("Date;Item Name;Department;Qty;Price")
