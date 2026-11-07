"""Constants shared by training, inference and the ML service.

Every value here comes from the Stage 3 model in ``notebooks/LightGBM_BaseLocal.ipynb``.
Changing one of them changes the model, so it needs a new feature-schema or model version.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# --- Feature recipe (Stage 3) -------------------------------------------------------------
LAG_DAYS: tuple[int, ...] = (1, 7, 14, 21)
ROLLING_WINDOWS: tuple[int, ...] = (3, 7, 14)
ROLLING_DELAY = 1  # rolling windows end on the day before the target day
LOOKBACK_DAYS = max(LAG_DAYS)  # 21 days of history are needed to build one row
MAX_MISSING_DAYS_IN_LOOKBACK = 3  # more unrecorded days than this → no AI forecast for the series

# --- M5 reference data ---------------------------------------------------------------------
M5_START_DATE = dt.date(2011, 1, 29)  # d_1
M5_FEATURE_START_DAY = LOOKBACK_DAYS + 1  # d_22: first day with a full 21-day lag
M5_DEV_TRAIN_END = 1885
M5_DEV_VALID = (1886, 1913)
M5_BACKTEST = (1914, 1941)
M5_LAST_OBSERVED_DAY = 1941
M5_CALENDAR_END_DAY = 1969
M5_REFERENCE_STORE = "CA_1"
M5_REFERENCE_CATEGORY = "FOODS"

# --- LightGBM (Stage 3 hyperparameters) ----------------------------------------------------
RANDOM_SEED = 42
LGBM_PARAMS: dict = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 30

# --- Own-data workspaces (set from notebooks/PortableFeatures_CA1Foods.ipynb) --------------
OWN_DATA_VALID_DAYS = 28
OWN_DATA_BACKTEST_DAYS = 28
OWN_DATA_ELIGIBILITY = {
    "minProducts": 10,
    "minHistoryDaysPerProduct": LOOKBACK_DAYS,
    "minStoreHistoryDays": 120,
    "minObservedProductDays": 4500,
}


def m5_day_to_date(day_number: int) -> dt.date:
    return M5_START_DATE + dt.timedelta(days=int(day_number) - 1)


def date_to_m5_day(value: dt.date) -> int:
    return (value - M5_START_DATE).days + 1


def project_root() -> Path:
    """The repository root: the folder that contains ``mlengine/``."""
    return Path(__file__).resolve().parent.parent


def raw_data_dir() -> Path:
    return Path(os.environ.get("MLENGINE_RAW_DATA_DIR", project_root() / "data" / "raw"))


def model_registry_dir() -> Path:
    return Path(os.environ.get("MODEL_REGISTRY_PATH", project_root() / "models"))
