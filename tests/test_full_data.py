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
