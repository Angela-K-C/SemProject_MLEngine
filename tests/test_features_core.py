import numpy as np
import pandas as pd
import pytest

from mlengine.features.core import InsufficientHistoryError, missing_in_lookback, window_features


def _matrix(seed=0, n=4, d=60):
    return np.random.default_rng(seed).poisson(3, size=(n, d)).astype(np.float32)


def test_lags_and_rolling_match_pandas():
    m = _matrix()
    targets = np.arange(21, 60)
    out = window_features(m, targets)
    for s in range(m.shape[0]):
        series = pd.Series(m[s].astype(np.float64))
        for k, t in enumerate(targets):
            row = k * m.shape[0] + s
            assert out["sales_lag_1"][row] == m[s, t - 1]
            assert out["sales_lag_21"][row] == m[s, t - 21]
            for w in (3, 7, 14):
                window = series.iloc[t - w : t]
                assert out[f"rolling_mean_{w}"][row] == pytest.approx(window.mean(), rel=1e-6)
                assert out[f"rolling_std_{w}"][row] == pytest.approx(window.std(ddof=1), rel=1e-5, abs=1e-6)


def test_no_leakage_from_target_or_future():
    m = _matrix(1)
    t = 40
    before = window_features(m, np.array([t]))
    changed = m.copy()
    changed[:, t:] = 999
    after = window_features(changed, np.array([t]))
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_missing_days_propagate_as_nan_only_where_touched():
    m = _matrix(2)
    m[0, 35] = np.nan
    out = window_features(m, np.array([40]))
    assert np.isnan(out["rolling_mean_7"][0])  # window 33..39 touches day 35
    assert not np.isnan(out["rolling_mean_3"][0])  # window 37..39 does not
    assert not np.isnan(out["rolling_mean_7"][1])  # other series unaffected
    assert missing_in_lookback(m, 39)[0] == 1


def test_insufficient_history_raises():
    with pytest.raises(InsufficientHistoryError):
        window_features(_matrix(), np.array([20]))
