import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import FIXTURE_DIR, REFERENCE_MODEL_DIR, synthetic_history
from mlengine.data.m5 import IDENTITY_COLUMNS
from mlengine.models.artifact import ModelRegistry


@pytest.fixture()
def client(own_data_registry, monkeypatch):
    from mlengine.service import app as service

    root, _ = own_data_registry
    monkeypatch.setattr(service, "registry", ModelRegistry(root))
    service._forecasters.clear()
    return TestClient(service.app)


def _series(history, as_of, products=None):
    history = history[history["date"] <= as_of]
    out = []
    for product, group in history.groupby("product_id"):
        if products and product not in products:
            continue
        out.append({"productId": product, "storeId": "S1", "category": group["category"].iloc[0],
                    "history": [{"date": str(d), "units": u, "price": p}
                                for d, u, p in zip(group["date"], group["units_sold"], group["unit_price"])]})
    return out


def test_health_and_models(client):
    assert client.get("/internal/v1/health").json()["status"] == "ok"
    models = client.get("/internal/v1/models").json()["models"]
    assert models[0]["id"] == "own-ws_test"


def test_forecast_compare(client):
    as_of = dt.date(2025, 7, 19)
    body = {"model": {"id": "own-ws_test"}, "asOfDate": str(as_of), "horizonDays": 1,
            "baselines": ["naive_last_value", "seasonal_naive_7"],
            "series": _series(synthetic_history(), as_of, {"P000", "P001"})}
    response = client.post("/internal/v1/forecast", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model"]["version"] == "1.0.0" and data["horizon"]["validated"] is True
    assert data["series"][0]["points"][0]["ai"] is not None
    point = data["series"][0]["points"][0]
    assert point["date"] == "2025-07-20" and point["ai"] > 0
    assert set(point["baselines"]) == {"naive_last_value", "seasonal_naive_7"}
    assert data["series"][0]["demandClass"]["class"] in {"SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"}


def test_horizon_rules(client):
    as_of = dt.date(2025, 7, 19)
    body = {"model": {"id": "own-ws_test"}, "asOfDate": str(as_of), "horizonDays": 14,
            "series": _series(synthetic_history(), as_of, {"P000"})}
    error = client.post("/internal/v1/forecast", json=body).json()["error"]
    assert error["code"] == "HORIZON_NOT_SUPPORTED"
    data = client.post("/internal/v1/forecast", json={**body, "allowExperimentalHorizon": True}).json()
    assert data["horizon"]["validated"] is False and len(data["series"][0]["points"]) == 14


def test_leakage_and_errors(client):
    as_of = dt.date(2025, 6, 1)  # before the model's training end
    body = {"model": {"id": "own-ws_test"}, "asOfDate": str(as_of),
            "series": _series(synthetic_history(), as_of, {"P000"})}
    assert client.post("/internal/v1/forecast", json=body).json()["error"]["code"] == "NO_MODEL_FOR_AS_OF"
    baseline_only = client.post("/internal/v1/forecast", json={**body, "model": None, "includeAi": False}).json()
    assert baseline_only["series"][0]["aiUnavailableReason"] == "AI_NOT_REQUESTED"
    assert baseline_only["series"][0]["points"][0]["baselines"]["naive_last_value"] is not None
    bad = client.post("/internal/v1/forecast", json={**body, "model": {"id": "nope"}, "asOfDate": "2025-07-19"})
    assert bad.status_code == 404 and bad.json()["error"]["code"] == "MODEL_NOT_FOUND"
    invalid = client.post("/internal/v1/forecast", json={"asOfDate": "x"})
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"


def test_token(client, monkeypatch):
    monkeypatch.setenv("ML_SERVICE_TOKEN", "secret")
    assert client.get("/internal/v1/health").status_code == 401
    assert client.get("/internal/v1/health", headers={"X-Internal-Token": "secret"}).status_code == 200


def test_train_job_and_eligibility(client):
    history = synthetic_history(n_products=8, n_days=160, seed=5)
    rows = [{"storeId": r.store_id, "productId": r.product_id, "category": r.category, "date": str(r.date),
             "units": r.units_sold, "price": r.unit_price} for r in history.itertuples()]
    eligibility = client.post("/internal/v1/eligibility", json={"history": rows}).json()
    assert eligibility["eligible"] is False  # default rules need 10 products
    job = client.post("/internal/v1/train", json={"workspaceId": "ws_job", "country": "KE", "history": rows})
    assert job.status_code == 202
    for _ in range(100):
        status = client.get(job.json()["statusUrl"]).json()
        if status["status"] in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.1)
    assert status["status"] == "COMPLETED" and status["result"]["status"] == "NOT_ELIGIBLE"


# --- Reference model (committed artifacts + committed fixture; no raw data needed) ----------
@pytest.mark.skipif(not REFERENCE_MODEL_DIR.is_dir() or not FIXTURE_DIR.is_dir(), reason="reference model missing")
def test_reference_forecast_matches_stored_backtest(monkeypatch):
    from mlengine.service import app as service

    monkeypatch.setattr(service, "registry", ModelRegistry(REFERENCE_MODEL_DIR.parent))
    service._forecasters.clear()
    client = TestClient(service.app)
    sales = pd.read_parquet(FIXTURE_DIR / "sales.parquet")
    prices = pd.read_parquet(FIXTURE_DIR / "prices.parquet")
    calendar = pd.read_parquet(FIXTURE_DIR / "calendar.parquet")
    week_of = dict(zip(calendar["date"].dt.date, calendar["wm_yr_wk"]))
    as_of_day, as_of = 1920, dt.date(2016, 5, 1)
    series = []
    for _, row in sales.head(10).iterrows():
        item_prices = prices[prices["item_id"] == row["item_id"]].set_index("wm_yr_wk")["sell_price"]
        history = []
        for d in range(as_of_day - 59, as_of_day + 1):
            date = dt.date(2011, 1, 29) + dt.timedelta(days=d - 1)
            history.append({"date": str(date), "units": float(row[f"d_{d}"]),
                            "price": float(item_prices.get(week_of[date], np.nan))})
        target = as_of + dt.timedelta(days=1)
        series.append({"productId": row["item_id"], "storeId": "CA_1", "firstPriceDate": None,
                       "history": history,
                       "futurePrices": [{"date": str(target), "price": float(item_prices.get(week_of[target]))}]})
    data = client.post("/internal/v1/forecast", json={
        "model": {"id": "m5-ca1-foods"}, "asOfDate": str(as_of), "series": series}).json()
    assert data["model"]["version"] == "1.1.0-bt"
    stored = pd.read_parquet(REFERENCE_MODEL_DIR / "1.1.0-bt" / "backtest" / "predictions.parquet")
    stored = stored[stored["date"] == "2016-05-02"].set_index("product_id")["ai"]
    for s in data["series"]:
        assert s["points"][0]["ai"] == pytest.approx(stored[s["productId"]], abs=1e-3)
    unknown = client.post("/internal/v1/forecast", json={
        "model": {"id": "m5-ca1-foods"}, "asOfDate": str(as_of),
        "series": [{**series[0], "productId": "WATER-500ML"}]}).json()
    assert unknown["series"][0]["aiUnavailableReason"] == "PRODUCT_NOT_IN_MODEL"
    stored_bt = client.get("/internal/v1/models/m5-ca1-foods/1.1.0/backtest",
                           params={"productId": series[0]["productId"]}).json()
    assert stored_bt["product"]["n"] == 28
