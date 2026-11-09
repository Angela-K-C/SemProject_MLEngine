# `mlengine` Architecture and Developer Guide

`mlengine` is the production form of the Stage 3 forecasting model: one Python package for training, inference, evaluation and serving. This guide explains how it fits together and how to change it safely. For the history behind it, see [`notebook-to-production.md`](notebook-to-production.md).

## 1. Data flow

```text
Training (release)                                 Serving (per request)
──────────────────                                 ─────────────────────
data/raw/*.csv                                     JSON from the backend
   │ data.m5.load_m5_scope()                          │ service.app._to_batch()
   ▼                                                  ▼
M5Scope (identity, sales matrix, calendar, prices)  SeriesBatch (ids, sales matrix ending at as-of, prices)
   │ features.m5_v1.build_feature_frame()  ◀── same function ──▶  models.forecaster.Forecaster
   ▼                                                  │ (recursive for H > 1)
feature frame (31 cols + target)                      ▼
   │ models.train.fit_*()                           predictions + baselines + demand class
   ▼                                                  │
models.artifact.save_artifact() ──▶ models/<id>/<version>/ ──▶ ModelRegistry.resolve()  (leakage rule)
```

## 2. Core conventions

| Convention | Meaning |
| --- | --- |
| **Wide matrix** | Sales are `float32` arrays shaped `(n_series, n_days)`. Column *j* is a fixed date: M5 day `first_day + j`, or `first_date + j` days. |
| **NaN = unrecorded** | Never zero-filled. Lags that land on NaN are NaN, and rolling statistics whose window touches a NaN are NaN. |
| **Target-major rows** | Feature frames list all series for the first target day, then all series for the next. This is the notebook's order, and LightGBM's bin sampling depends on it. |
| **Target columns** | Features for target column *t* read only columns `< t`. `tests/test_features_core.py` enforces this. |
| **As-of date** | The last day of history a forecast may use. Forecasts start the day after. |
| **Schemas are versions** | `m5-v1` and `portable-v1` define the columns, order and categoricals. Changing a schema means a new major model version. |

## 3. Modules

### `config.py`
This module holds every constant that defines the model: lags, windows, splits, hyperparameters, the early-stopping patience, the gap limit and the own-data eligibility thresholds. It also has date helpers (`m5_day_to_date`, `date_to_m5_day`) and path helpers (`raw_data_dir`, `model_registry_dir`), which can be overridden with the `MLENGINE_RAW_DATA_DIR` and `MODEL_REGISTRY_PATH` environment variables.

### `data/m5.py`
- `load_m5_scope(raw_dir, store_id, category_id, verify_overlap)` returns an `M5Scope`. It uses the notebook's dtypes, ordering and overlap check.
- `daily_price_matrix(scope)` spreads weekly prices onto days, giving `(n_series, 1969)` with NaN before an item's first priced week.
- `first_price_dates(scope)` gives the first day of each item's first priced week.
- `scope_to_batch(scope, as_of_day, history_days, horizon, rows, daily_prices)` returns a `SeriesBatch` shaped exactly like a backend request. It is used by the research notebooks and the release.

### `features/core.py`
- `window_features(sales_matrix, target_columns)` returns `sales_lag_{1,7,14,21}` and `rolling_{mean,std}_{3,7,14}`. It uses the notebook's cumulative-sum maths, with NaN handling added.
- `missing_in_lookback(sales_matrix, as_of_column)` counts unrecorded days in the 21-day lookback.

### `features/m5_v1.py`
- `build_calendar_features(calendar)` builds the calendar part of the features (notebook cell 37).
- `build_release_weeks(prices, calendar_features)` builds the release-week proxy.
- `M5Context(calendar_features, prices, release_weeks)` holds everything except sales.
- `build_feature_frame(identity, sales_matrix, first_day, target_days, context)` produces 31 features plus `sales` and `date` (notebook cells 39 and 41).
- `identity_from_ids(item_ids, store_ids)` rebuilds `id`/`dept_id`/`cat_id`/`state_id`.

### `features/portable_v1.py`
- `FEATURE_COLUMNS` and `CATEGORICAL_COLUMNS` define the own-data schema.
- `build_feature_frame(series, sales_matrix, price_matrix, first_date, target_columns, history_start, country, subdivision)` builds the own-data features.
- `forward_fill_prices(price_matrix)` carries the last known price forward.
- `holiday_names(dates, country, subdivision)` looks up public holidays through the `holidays` package; results are cached.

### `models/train.py`
- `fit_with_early_stopping(...)` makes the notebook's exact `LGBMRegressor.fit` call.
- `fit_fixed(x, y, n_estimators)` refits with a fixed tree count.
- `best_iteration(model)` returns the early-stopped tree count.

### `models/baselines.py`
`naive_last_value`, `seasonal_naive_7`, `run(name, history, horizon)`, and the display labels in `BASELINES`.

### `models/forecaster.py`
- `SeriesBatch` holds `product_ids`, `store_ids`, `as_of` and `sales`, plus optional `prices`, `future_prices`, `categories` and `first_price_dates`.
- `Forecaster(artifact).forecast(batch, horizon)` returns a `ForecastOutput`: `ai`, `unavailable_reasons`, `price_assumptions`, `missing_in_lookback` and `warnings`.
- `ForecastRequestError(code, message)` signals request-level problems (`HORIZON_NOT_SUPPORTED`, `DATE_OUTSIDE_CALENDAR`).

### `models/artifact.py`
- `ModelArtifact` exposes `metadata`, `card`, a lazily loaded `booster`, `frame(name)` for parquet side files, and the `model_id`, `version`, `ref`, `schema`, `training_end_date` and `capabilities` properties.
- `save_artifact(path, booster, metadata, card, frames, num_iteration)` writes an artifact. It adds `mlengineVersion`, `lightgbmVersion`, `createdAt` and `gitCommit` automatically.
- `ModelRegistry(root)` provides `list()`, `get(id, version)`, `resolve(id, as_of, version=None)` (the leakage rule) and `reload()`.

### `evaluation/`
- `metrics.py`: `mae`, `rmse`, `wape` and `summary`. NaN pairs are dropped. WAPE is `None` when there were no sales.
- `demand_class.py`: `classify(history)` computes ADI, CV² and the Syntetos–Boylan class with a plain-language label.
- `backtest.py`: `recursive_backtest(forecaster, build, origins, horizon)` computes per-day and overall metrics, bias, and the `validated` verdict.

### `pipelines/`
- `m5_release.release(registry_dir, version)` trains `-dev`, `-bt` (with the per-product backtest, demand classes and the recursive check) and prod, and writes the cards.
- `own_data.train_own_data(history, workspace_id, country, subdivision, registry_dir, rules, register)` runs eligibility, the gate and the recursive check, then registers the model. `to_wide()` and `check_eligibility()` are reused by the service.
- `reference_export.export_reference(out_dir)` writes the backend seed files.

### `service/`
- `schemas.py`: Pydantic models for every request and response.
- `app.py`: the FastAPI app, with error handling, token check, forecaster cache and training job executor. See [`ml-service-api.md`](ml-service-api.md).

## 4. How to…

### Add or change a feature
1. Prototype it in a notebook against the current model, on the same windows, as Stages 4–7 did.
2. If it is adopted, create a **new schema** (e.g. `m5-v2`) rather than editing `m5-v1`, because served artifacts depend on `m5-v1` staying fixed.
3. Teach `Forecaster` about the new schema, add parity tests, and release under a new **major** version.

### Release a new model version
```bash
python -m mlengine release-m5 --version 1.2.0
MLENGINE_FULL_TESTS=1 pytest
```
Leave older version folders in place or delete them. The registry serves whatever is under `models/`. Commit the new folder.

### Add a baseline
Add a function and a label in `models/baselines.py`, register it in `run()`, add it to `BaselineId` in `service/schemas.py`, and add it to `capabilities.baselines` in the release pipelines. Every baseline needs a test in `tests/test_evaluation_baselines.py`.

### Change own-data eligibility
Edit `config.OWN_DATA_ELIGIBILITY`, and justify the change with a run of `PortableFeatures_CA1Foods.ipynb`.

## 5. Tests

| File | Covers | Needs raw data? |
| --- | --- | --- |
| `test_features_core.py` | Lags/rolling vs pandas, no leakage, NaN propagation | No |
| `test_m5_parity.py` | `m5-v1` features equal the notebook's (25-item fixture) | No |
| `test_evaluation_baselines.py` | Metrics, baselines, demand classes | No |
| `test_registry.py` | Leakage rule and version choice | No |
| `test_own_data.py` | Training, gate, eligibility, recursion consistency, gap handling | No |
| `test_service.py` | API contract, errors, token, training job, and Reference predictions equal stored backtest predictions | No (uses committed artifacts) |
| `test_full_data.py` | Full retrain reproduces the notebook; HTTP backtest reproduces 2.377 | Yes (`MLENGINE_FULL_TESTS=1`) |

Regenerate the parity fixture with `python scripts/make_parity_fixture.py` whenever the raw data or the notebook changes.
