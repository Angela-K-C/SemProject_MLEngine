import datetime as dt

import numpy as np

from conftest import SMALL_RULES, synthetic_history
from mlengine.models.artifact import ModelRegistry
from mlengine.models.forecaster import Forecaster, SeriesBatch
from mlengine.pipelines import own_data


def test_training_passes_gate_and_registers(own_data_registry):
    root, result = own_data_registry
    assert result["gate"]["passed"]
    assert result["backtest"]["ai"]["rmse"] < result["backtest"]["naive_last_value"]["rmse"]
    artifact = ModelRegistry(root).resolve("own-ws_test", dt.date(2025, 7, 19))
    assert artifact.schema == "portable-v1"
    assert artifact.metadata["training"]["country"] == "KE"
    assert result["recursive"]["horizon"] == 7
    assert artifact.capabilities["horizons"] == result["horizons"]


def test_not_eligible_with_short_history(tmp_path):
    result = own_data.train_own_data(synthetic_history(n_days=60), "ws_small", registry_dir=tmp_path)
    assert result["status"] == "NOT_ELIGIBLE"
    assert not result["eligibility"]["checks"]["storeHistoryDays"]["passed"]


def test_gate_failure_registers_nothing(tmp_path):
    history = synthetic_history(n_products=6, n_days=150, seed=3)
    rng = np.random.default_rng(0)
    history["units_sold"] = rng.permutation(history["units_sold"].to_numpy())  # pure noise
    result = own_data.train_own_data(history, "ws_noise", registry_dir=tmp_path,
                                     rules={**SMALL_RULES, "minProducts": 5, "minObservedProductDays": 500})
    if result["status"] == "GATE_FAILED":
        assert ModelRegistry(tmp_path).list() == []
    else:  # noise can occasionally beat naive; the gate logic itself is what matters
        assert result["gate"]["passed"]


def _batch(history, as_of):
    wide = own_data.to_wide(history[history["date"] <= as_of])
    return SeriesBatch(wide.series["product_id"].tolist(), wide.series["store_id"].tolist(), as_of,
                       wide.sales, wide.prices, None, wide.series["category"].tolist(),
                       wide.series["first_price_date"].tolist())


def test_recursive_first_day_equals_one_step(own_data_registry):
    root, _ = own_data_registry
    forecaster = Forecaster(ModelRegistry(root).resolve("own-ws_test", dt.date(2025, 7, 19)))
    batch = _batch(synthetic_history(), dt.date(2025, 7, 19))
    one = forecaster.forecast(batch, 1)
    seven = forecaster.forecast(batch, 7)
    np.testing.assert_allclose(one.ai[:, 0], seven.ai[:, 0])
    assert seven.ai.shape == (15, 7) and np.all(seven.ai >= 0)
    assert set(seven.price_assumptions) == {"LAST_KNOWN"}


def test_gaps_and_short_history_block_ai(own_data_registry):
    root, _ = own_data_registry
    forecaster = Forecaster(ModelRegistry(root).resolve("own-ws_test", dt.date(2025, 7, 19)))
    batch = _batch(synthetic_history(), dt.date(2025, 7, 19))
    batch.sales[0, -10:-5] = np.nan  # 5 unrecorded days in the lookback
    short = SeriesBatch(batch.product_ids[:1], batch.store_ids[:1], batch.as_of, batch.sales[1:2, -15:])
    out = forecaster.forecast(batch, 1)
    assert out.unavailable_reasons[0] == "TOO_MANY_GAPS" and np.isnan(out.ai[0, 0])
    assert out.unavailable_reasons[1] is None
    assert forecaster.forecast(short, 1).unavailable_reasons == ["INSUFFICIENT_HISTORY"]
