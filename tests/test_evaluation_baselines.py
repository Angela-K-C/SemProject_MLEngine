import numpy as np
import pytest

from mlengine.evaluation import metrics
from mlengine.evaluation.demand_class import classify
from mlengine.models import baselines


def test_metrics():
    actual, forecast = [1, 2, 3, np.nan], [2, 2, 1, 5]
    assert metrics.mae(actual, forecast) == pytest.approx(1.0)
    assert metrics.rmse(actual, forecast) == pytest.approx(np.sqrt(5 / 3))
    assert metrics.wape(actual, forecast) == pytest.approx(3 / 6)
    assert metrics.wape([0, 0], [1, 1]) is None


def test_naive_uses_last_recorded_value():
    history = np.array([[1, 2, 3], [4, 5, np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(baselines.naive_last_value(history, 2), [[3, 3], [5, 5]])


def test_seasonal_naive_repeats_last_week():
    history = np.arange(14, dtype=np.float32)[None, :]
    np.testing.assert_array_equal(baselines.seasonal_naive_7(history, 9)[0], [7, 8, 9, 10, 11, 12, 13, 7, 8])


def test_demand_classes():
    assert classify(np.full(60, 5.0))["class"] == "SMOOTH"
    lumpy = np.zeros(60); lumpy[::5] = [1, 30, 2, 40, 1, 25, 3, 50, 1, 35, 2, 45]
    assert classify(lumpy)["class"] == "LUMPY"
    intermittent = np.zeros(60); intermittent[::4] = 3
    assert classify(intermittent)["class"] == "INTERMITTENT"
    assert classify(np.full(10, 1.0))["class"] == "INSUFFICIENT_DATA"
