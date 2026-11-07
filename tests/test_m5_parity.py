"""mlengine reproduces the notebook's m5-v1 features (fixture generated from the notebook code)."""

import numpy as np
import pandas as pd

from mlengine.data.m5 import IDENTITY_COLUMNS
from mlengine.features import m5_v1


def test_features_equal_notebook(m5_fixture):
    sales, prices, calendar, expected = m5_fixture
    identity = sales[IDENTITY_COLUMNS].astype(str)
    matrix = sales.drop(columns=IDENTITY_COLUMNS).to_numpy(dtype=np.float32)
    calendar_features = m5_v1.build_calendar_features(calendar)
    context = m5_v1.M5Context(calendar_features, prices, m5_v1.build_release_weeks(prices, calendar_features))
    frame = m5_v1.build_feature_frame(identity, matrix, 1, np.arange(22, 1942), context)
    frame = frame.reset_index(drop=True)
    assert list(frame.columns) == list(expected.columns)
    assert len(frame) == len(expected)
    for column in expected.columns:
        left, right = frame[column], expected[column]
        if column in m5_v1.CATEGORICAL_COLUMNS:
            np.testing.assert_array_equal(left.astype(object).where(left.notna(), None).to_numpy(),
                                          right.to_numpy(), err_msg=column)
        elif column == "date":
            assert (pd.to_datetime(left) == pd.to_datetime(right)).all()
        else:
            np.testing.assert_array_equal(left.to_numpy(), right.to_numpy(), err_msg=column)
    assert len(m5_v1.feature_columns(frame)) == 31


def test_identity_from_ids():
    identity = m5_v1.identity_from_ids(["FOODS_3_090"], ["CA_1"])
    assert identity.iloc[0].to_dict() == {
        "id": "FOODS_3_090_CA_1_evaluation", "item_id": "FOODS_3_090", "dept_id": "FOODS_3",
        "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA"}
