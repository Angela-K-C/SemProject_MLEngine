"""Run the original notebook code and check that mlengine builds identical features.

Executes the data-loading and feature cells of notebooks/LightGBM_BaseLocal.ipynb verbatim,
builds the same frame with mlengine, compares every column, and writes a small fixture
(tests/fixtures/m5_parity/) so the comparison can be repeated in CI without the raw M5 files.

    python scripts/make_parity_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlengine import config  # noqa: E402
from mlengine.data.m5 import load_m5_scope  # noqa: E402
from mlengine.features import m5_v1  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "LightGBM_BaseLocal.ipynb"
NOTEBOOK_CELLS = [3, 5, 7, 35, 37, 39, 41]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "m5_parity"
FIXTURE_ITEMS = 25


def run_notebook_cells() -> dict:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: dict = {"display": lambda *args, **kwargs: None}
    previous = Path.cwd()
    os.chdir(ROOT / "notebooks")
    try:
        for index in NOTEBOOK_CELLS:
            source = "".join(cells[index]["source"])
            source = source.replace("from IPython.display import display", "")
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous)
    return namespace


def compare(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    if list(expected.columns) != list(actual.columns):
        raise AssertionError(f"Column mismatch:\n{list(expected.columns)}\n{list(actual.columns)}")
    if len(expected) != len(actual):
        raise AssertionError(f"Row count mismatch: {len(expected)} vs {len(actual)}")
    for column in expected.columns:
        left, right = expected[column], actual[column]
        if isinstance(left.dtype, pd.CategoricalDtype):
            if list(left.cat.categories) != list(right.cat.categories):
                raise AssertionError(f"{column}: categorical levels differ")
            if not np.array_equal(left.cat.codes.to_numpy(), right.cat.codes.to_numpy()):
                raise AssertionError(f"{column}: categorical values differ")
        elif not left.equals(right):
            raise AssertionError(f"{column}: values differ (dtypes {left.dtype} vs {right.dtype})")
    print(f"All {len(expected.columns)} columns identical over {len(expected):,} rows.")


def main() -> None:
    print("Running notebook cells…")
    ns = run_notebook_cells()
    expected = ns["featureFrame"]

    print("Building the same frame with mlengine…")
    scope = load_m5_scope()
    calendar_features = m5_v1.build_calendar_features(scope.calendar)
    context = m5_v1.M5Context(
        calendar_features, scope.prices, m5_v1.build_release_weeks(scope.prices, calendar_features)
    )
    target_days = np.arange(config.M5_FEATURE_START_DAY, config.M5_LAST_OBSERVED_DAY + 1)
    actual = m5_v1.build_feature_frame(scope.identity, scope.sales, 1, target_days, context)
    compare(expected, actual)

    # Small fixture: the first FIXTURE_ITEMS series, raw inputs plus the notebook's own features.
    items = scope.identity["item_id"].iloc[:FIXTURE_ITEMS].tolist()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    sales = pd.DataFrame(scope.sales[:FIXTURE_ITEMS], columns=[f"d_{i}" for i in range(1, scope.n_days + 1)])
    pd.concat([scope.identity.iloc[:FIXTURE_ITEMS].reset_index(drop=True), sales], axis=1).to_parquet(
        FIXTURE_DIR / "sales.parquet", index=False
    )
    scope.prices.loc[scope.prices["item_id"].isin(items)].to_parquet(FIXTURE_DIR / "prices.parquet", index=False)
    scope.calendar.to_parquet(FIXTURE_DIR / "calendar.parquet", index=False)
    subset = expected.loc[expected["item_id"].astype(str).isin(items)].copy()
    for column in m5_v1.CATEGORICAL_COLUMNS:
        subset[column] = subset[column].astype(str).replace("nan", None)
    subset.reset_index(drop=True).to_parquet(FIXTURE_DIR / "expected_features.parquet", index=False)
    print(f"Fixture written to {FIXTURE_DIR} ({len(subset):,} expected rows).")


if __name__ == "__main__":
    main()
