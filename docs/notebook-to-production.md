# From Notebook to Served Model: the Engineering Journey

This document records how the forecasting model went from research code in Jupyter notebooks to a versioned, tested model behind an HTTP service that the mobile app's backend can call. It covers what was built, and also why each decision was made, what was checked, and what changed along the way. The model-research stages (1–7) are summarised in [`README.md`](../README.md#modelling-journey). This document picks up where they end.

**Status at the time of writing (7 November 2026):** active model `m5-ca1-foods@1.1.0`, package `mlengine` 0.1.0, branch `model-implementation`.

---

## Contents

1. [Timeline](#1-timeline)
2. [Where the research left us](#2-where-the-research-left-us)
3. [Choosing what to ship](#3-choosing-what-to-ship)
4. [Making the app design fit the model](#4-making-the-app-design-fit-the-model)
5. [Extracting the pipeline into `mlengine`](#5-extracting-the-pipeline-into-mlengine)
6. [Proving the package equals the notebook](#6-proving-the-package-equals-the-notebook)
7. [Model versions, artifacts and the leakage rule](#7-model-versions-artifacts-and-the-leakage-rule)
8. [Inference: turning a request into a forecast](#8-inference-turning-a-request-into-a-forecast)
9. [Follow-up experiments and what they changed](#9-follow-up-experiments-and-what-they-changed)
10. [Own-data models: retraining the recipe per business](#10-own-data-models-retraining-the-recipe-per-business)
11. [The internal ML service](#11-the-internal-ml-service)
12. [Decision log](#12-decision-log)
13. [Known limitations and future work](#13-known-limitations-and-future-work)
14. [Reproducing everything](#14-reproducing-everything)

---

## 1. Timeline

| Date | Step | Outcome |
| --- | --- | --- |
| Sep–Oct 2026 | Dataset exploration and EDA | `reports/EDAInsights.md`; demand is heavily intermittent (68% of item-days are zero) |
| 26 Oct | Stage 1: LightGBM on a 500-series mixed sample | RMSE 1.832, about 28–31% better than naive lag-28 |
| 2 Nov | Stage 2: scope to CA_1 / FOODS, compare models (PR #1) | Model choice matters far less than features |
| 2 Nov | Stage 3: next-day lags + Tweedie objective (PR #2) | **Best model:** dev RMSE 2.356, backtest RMSE 2.377 |
| 3 Nov | Stages 4–5: hurdle and bucketed hybrids (PR #3) | Both worse; extra classification structure hurts |
| 3 Nov | Stage 6: pooled four-store model (PR #4) | Ties Stage 3 on CA_1 |
| 3 Nov | Stage 7: richer sales and price features (PR #5) | Within noise of Stage 3 |
| 7 Nov | App design rewritten around the real model (v2.0) | Ten mismatches found and fixed in the spec |
| 7 Nov | `mlengine` package + notebook parity | Features identical over 2,759,040 rows; metrics reproduced |
| 7 Nov | Model `1.0.0` released, ML service built | Service reproduces the backtest over HTTP |
| 7 Nov | Three follow-up experiments | 7-day horizon validated; portable features validated; error by demand class |
| 7 Nov | Model `1.1.0` released | Horizons `[1, 7]`; per-day accuracy on the model card |

---

## 2. Where the research left us

After Stage 7 the project had a well-understood model, but it existed only as notebook cells.

**What the notebooks had established:**

- **The winning recipe** (Stage 3, `notebooks/LightGBM_BaseLocal.ipynb`):
  - one LightGBM regressor with a Tweedie objective (`tweedie_variance_power=1.1`, `learning_rate=0.05`, `num_leaves=31`), early-stopped on a 28-day validation window;
  - inputs: sales lags at 1/7/14/21 days, rolling mean and standard deviation over 3/7/14 days, calendar, US events, the SNAP flag and price features, 31 features in all.
- **Chronological evaluation windows:** train on `d_22`–`d_1885`, validate on `d_1886`–`d_1913`, then refit on everything up to `d_1913` and backtest on `d_1914`–`d_1941`.
- **The two big lessons:**
  - Feature recency beat every change of model or architecture.
  - A well-matched objective (Tweedie) beat added structure (hurdle, buckets).

**What was missing for production:**

| Gap | Why it mattered |
| --- | --- |
| No saved model | Every notebook retrained from scratch; nothing could be served. |
| Code duplicated across notebooks | Each notebook repeated loading and feature building. There was no single implementation to trust. |
| `scripts/preprocess_m5.py` was stale | It still built the old 28-day-embargoed features, not the Stage 3 ones. |
| Features were built for training only | The code built a giant frame for 1,437 items × 1,920 days. Nothing could build *one row* from a short request history. |
| No inference path for multi-day horizons | The model predicts one day ahead; the app wanted a week. |
| M5-specific inputs | Item IDs, M5 day numbers, US events and California SNAP mean the trained model cannot score a business's own products. |
| No test of train/serve agreement | The biggest silent-failure risk in serving ML: features computed differently at serving time than during training. |

---

## 3. Choosing what to ship

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Stage 3: 31 features | Dev RMSE 2.356 / backtest 2.377 | **Shipped** |
| Stage 7: 47 features | Dev 2.345 / backtest 2.373; the notebook judged this within run-to-run noise | Not shipped: 16 more features to compute and maintain for no measurable gain |
| Stage 6: pooled CA_1–CA_4 | Ties Stage 3 on CA_1 | Kept as a candidate for a future multi-store demo |
| Stages 4–5: hurdle / bucketed | 3–10% worse | Rejected |

The simplest model that matched the best result won. Fewer features also means less to keep identical between training and serving.

---

## 4. Making the app design fit the model

Before writing any code, the draft mobile app design (v1.0) was checked against what the model actually does. Ten mismatches were found and fixed in the spec, which became v2.0 ([`retail_demand_forecasting_mobile_design.md`](../retail_demand_forecasting_mobile_design.md) §0). The ones that shaped the engineering:

1. **Horizon.** The draft offered 7/14/30 days, but the model is next-day only. Multi-day forecasts were therefore designed as recursive and gated behind a validation experiment (§9.1 below).
2. **Whose data.** The draft let any business forecast its own uploads with the M5-trained model, which is invalid. Two modes were introduced:
   - a **Reference workspace** where the trained model is valid;
   - **own-data workspaces**, where the same recipe is retrained per business using a portable feature set (§10).
3. **Datasets vs history.** A next-day model needs a continuous daily history per product, not "a dataset". Uploads therefore became import batches merged into one sales history.
4. **Missing vs zero.** Lags read the previous 21 days directly, so an unrecorded day must stay *missing* (NaN), never silently become zero.
5. **Leakage in replays.** Letting users "replay" past dates means a forecast could be served by a model that was trained on that date's outcome. This led to the leakage rule (§7.2).

---

## 5. Extracting the pipeline into `mlengine`

### 5.1 Principles

1. **One implementation, used everywhere.** Training, the service, backtests and the new notebooks all call the same feature functions. There is no separate "serving version" of a feature.
2. **Port, don't improve.** The first goal was to reproduce the notebook exactly. Improvements come later, behind a new version.
3. **Keep the notebook's data shape.** The notebook built features on a wide matrix (series × days) with cumulative sums, then reshaped once. That shape is kept, and generalised so the same function works on a 1,437 × 1,941 training matrix and on a 1 × 28 request.
4. **Prove it, don't assume it.** Every claim of equivalence has a test (§6).

### 5.2 Package layout

```text
mlengine/
├── config.py                # every constant from the notebook, in one place
├── data/m5.py               # raw M5 loading, scoping, daily prices, scope_to_batch()
├── features/core.py         # lag and rolling maths (shared by both schemas)
├── features/m5_v1.py        # the 31 Stage 3 features
├── features/portable_v1.py  # own-data features
├── models/train.py          # LightGBM calls, exactly as in the notebook
├── models/forecaster.py     # one-step and recursive inference
├── models/baselines.py      # naive and seasonal-naive
├── models/artifact.py       # artifacts, registry, leakage rule
├── evaluation/              # metrics, demand classes, recursive backtest
├── pipelines/               # m5_release, own_data, reference_export
├── service/                 # FastAPI internal ML service
└── cli.py                   # python -m mlengine …
```

[`docs/mlengine-architecture.md`](mlengine-architecture.md) describes each module in detail.

### 5.3 Step by step

**Step 1: constants (`config.py`).** Everything that defines the model was lifted out of the notebook cells:

| Constant | Value | Notebook origin |
| --- | --- | --- |
| `LAG_DAYS` | (1, 7, 14, 21) | cell 39 |
| `ROLLING_WINDOWS` | (3, 7, 14) | cell 39 |
| `ROLLING_DELAY` | 1 | cell 39 (`= HORIZON_DAYS`) |
| `M5_FEATURE_START_DAY` | 22 | cell 39 (`max(LAG_DAYS) + 1`) |
| Splits | dev train ≤ 1885, valid 1886–1913, backtest 1914–1941 | cell 39 |
| `LGBM_PARAMS` | tweedie, power 1.1, 1000 trees max, lr 0.05, 31 leaves, seed 42 | cell 44 |
| `EARLY_STOPPING_ROUNDS` | 30 | cell 44 |

Changing any of them changes the model, so the code treats them as part of the model version.

**Step 2: data loading (`data/m5.py`).** This mirrors cells 5, 7 and 35:
- the same dtypes (`category` for IDs, `uint16` for sales, `float32` for prices);
- the same scoping and the same validation/evaluation overlap check;
- the same series order (stable sort by `item_id`, `store_id`).

Two helpers were added for serving and research: `daily_price_matrix()` spreads weekly prices onto days, and `scope_to_batch()` arranges M5 data the way the backend will send it.

**Step 3: lag and rolling maths (`features/core.py`).** The notebook's `computeRollingStats` was copied with its numerics intact: float64 cumulative sums, sample variance clamped at zero, cast to float32. It was then generalised in two ways:
- **Positions instead of day numbers.** Targets are given as column positions in the matrix, so a short request matrix works the same as the full one.
- **NaN-aware.** The notebook never saw missing days (M5 is complete), but business data will. A lag that lands on an unrecorded day is NaN, and so is any rolling statistic whose window touches one. When the matrix has no NaN, the code path is bit-for-bit the notebook's.

**Step 4: the 31 features (`features/m5_v1.py`).** This mirrors cells 37, 39 and 41 operation by operation: the calendar features, the release-week proxy, the merges (calendar, then price, then release week), `price_missing`, `weeks_since_first_price`, and the categorical conversion. Several details had to be preserved exactly:

| Detail | Why it matters |
| --- | --- |
| Row order (all series for day 1, then day 2, …) | LightGBM samples rows (in order) to build its histogram bins, so row order can change the trained model. The notebook's order was kept. |
| Column order | LightGBM matches features by position as well as name. |
| Categorical levels sorted lexically | `read_csv(dtype="category")` and `.astype("category")` both sort levels; the category codes must match the training codes. |
| `id` suffix `_evaluation` | The scoped series come from `sales_train_evaluation.csv`. A `_validation` suffix would be an unseen category. |
| Event levels from the full calendar | Event categories keep every calendar level, not only those seen in training rows. |

For serving, `identity_from_ids()` rebuilds the six M5 identity columns from just `item_id` and `store_id`, so the backend never has to know about M5's ID conventions.

**Step 5: training (`models/train.py`).** These are the notebook's `LGBMRegressor(...).fit(...)` calls, unchanged, including `eval_set` and `eval_metric="rmse"`. LightGBM 4.7 deprecates `eval_set`, but it was kept for exactness and its warning is silenced. Early stopping picked **70 trees**, as in the notebook (the log shows its best iteration at 70, validation RMSE 2.35548).

**Step 6: portable features (`features/portable_v1.py`).** This schema was written for own-data models (§10). It shares `core.py`, so lags and rolling windows are identical by construction.

---

## 6. Proving the package equals the notebook

Four levels of evidence were collected, each stronger than the last.

### 6.1 Feature identity (whole dataset)

`scripts/make_parity_fixture.py` **executes the notebook's own cells** (3, 5, 7, 35, 37, 39, 41), read straight from the `.ipynb` file with no re-typing. It then builds the same frame with `mlengine` and compares every column: categorical levels and codes exactly, and numeric columns with `Series.equals` (exact, NaN-aware).

> **Result: all 33 columns identical over 2,759,040 rows.**

The script also writes a 25-item fixture (`tests/fixtures/m5_parity/`, 1.7 MB) holding the raw inputs and the notebook's own feature rows. `tests/test_m5_parity.py` repeats the comparison in seconds, without the raw M5 files.

### 6.2 Training reproduction

`python -m mlengine release-m5` retrains from scratch:

| Metric | Notebook | `mlengine` |
| --- | --- | --- |
| Early-stopped trees | 70 | 70 |
| Dev RMSE / MAE | 2.356 / 1.308 | 2.3555 / 1.3085 |
| Dev naive (lag-1) RMSE / MAE | 3.194 / 1.624 | 3.1939 / 1.6237 |
| Backtest RMSE / MAE | 2.377 / 1.377 | 2.3773 / 1.3766 |
| Backtest naive RMSE / MAE | 3.219 / 1.728 | 3.2186 / 1.7285 |

`tests/test_full_data.py::test_training_reproduces_notebook` asserts these within ±0.005.

### 6.3 Train/serve agreement

This test catches the classic serving bug. History for every item was arranged the way the backend will send it: a short window ending on the as-of date, with daily prices. It was passed through the serving code path (`Forecaster.forecast`), and the result was compared with the training-frame predictions for the same item and day.

> **Result: maximum difference 2.4e-6** (float32 rounding), with 21, 28 or 365 days of history, at three different as-of dates. A whole-store forecast (1,437 items) takes about 0.1 s.

### 6.4 End to end over HTTP

`tests/test_full_data.py::test_service_backtest_reproduces_stage3` posts all 1,437 items' history to the running service's `POST /internal/v1/backtest` for 2016-04-25 to 2016-05-22.

> **Result: RMSE 2.377 and naive RMSE 3.219, the README's numbers, through the API.**

### 6.5 Leakage tests

- `test_no_leakage_from_target_or_future`: overwriting every value on or after day *t* leaves day *t*'s features unchanged.
- `test_leakage_rule`: the registry refuses a model whose training ended after the as-of date (§7.2).

---

## 7. Model versions, artifacts and the leakage rule

### 7.1 Three versions of one model

The release pipeline (`pipelines/m5_release.py`) trains the same recipe three times, each on a different amount of history:

| Version | Trained on | Serves as-of dates | Purpose |
| --- | --- | --- | --- |
| `1.1.0-dev` | `d_22`–`d_1885` (to 2016-03-27) | 2016-03-27 … 2016-04-23 | The notebook's development model; research replays |
| `1.1.0-bt` | `d_22`–`d_1913` (to 2016-04-24) | 2016-04-24 … 2016-05-21 | Replays inside the backtest window, with known actuals |
| `1.1.0` | `d_22`–`d_1941` (to 2016-05-22) | 2016-05-22 onward | True forecasts beyond the data |

All three use the early-stopped 70 trees.

### 7.2 The leakage rule

> A model may serve a forecast made on as-of date *a* only if its training data ended on or before *a*.

`ModelRegistry.resolve()` enforces this. Without an explicit version it picks the qualifying model with the latest training end. With an explicit version that violates the rule, it raises `NoModelForAsOfError`, which the service returns as `NO_MODEL_FOR_AS_OF`. This is what makes the app's replay mode trustworthy for the research comparison: a replayed forecast can never come from a model that has seen the outcome.

### 7.3 Artifacts

Each version is a folder: `models/m5-ca1-foods/<version>/` holding `model.txt`, `metadata.json`, `card.json`, `calendar.parquet` and `backtest/`. `metadata.json` records:
- the feature columns and categorical levels;
- the hyperparameters, with the real tree count;
- the training window and row count;
- capabilities (horizons, minimum history, gap limit);
- the full evaluation results;
- the `mlengine` and LightGBM versions and the git commit.

The model card (`card.json`) is the plain-language version the app renders.

The artifacts total about 2.5 MB, so they are **committed to Git** (a change from the original plan of release assets). A fresh clone can serve forecasts without the 450 MB of raw M5 data.

### 7.4 Why 1.0.0 became 1.1.0

`1.0.0` shipped with horizons `[1]`. Once the recursive 7-day experiment passed (§9.1), the release pipeline was extended to run that check itself, and the model was re-released as `1.1.0` with horizons `[1, 7]` and a per-day accuracy curve on its card. The trees are identical (same data, same seed). Only capabilities and documentation changed, which counts as a minor version under the versioning rules.

---

## 8. Inference: turning a request into a forecast

`models/forecaster.py` takes a `SeriesBatch` (product IDs, store IDs, an as-of date, and a sales matrix whose last column is the as-of day) and returns forecasts.

1. **Pad and extend.** Short histories are padded on the left with NaN up to the 21-day lookback. *H* empty columns are appended for the forecast days.
2. **Per-series checks.**
   - Under 21 recorded days ending at the as-of date gives `INSUFFICIENT_HISTORY`.
   - More than 3 unrecorded days among the last 21 gives `TOO_MANY_GAPS`.
   - For the M5 model, a product or store outside the training categories gives `PRODUCT_NOT_IN_MODEL`.
   - Failing series get `ai = NaN` and a reason. The other series still run, and baselines are always computed.
3. **Prices.** Known future prices are used where provided; otherwise the last known price is carried forward. Each series reports `KNOWN`, `LAST_KNOWN` or `NONE`.
4. **Recursion.** For day *h* = 1 … *H*: build features for target day *a + h* with the same `build_feature_frame` used in training, predict, clip at zero, and write the prediction into the matrix, so day *h + 1*'s lags and rolling windows see it.
5. **Categorical alignment.** Before prediction, every categorical column is re-cast to the exact training levels stored in `metadata.json`.

**Baselines** (`models/baselines.py`) work on the same matrix:
- `naive_last_value` repeats the last *recorded* value. This equals the notebook's `sales_lag_1` baseline for day 1.
- `seasonal_naive_7` repeats the last observed week.

---

## 9. Follow-up experiments and what they changed

The v2.0 design listed three experiments as blockers. All three were built on `mlengine` (so they test the serving code itself), run, and their conclusions written into the notebooks.

### 9.1 Recursive 7-day horizon ([`RecursiveHorizon_CA1Foods.ipynb`](../notebooks/RecursiveHorizon_CA1Foods.ipynb))

**Question:** can the next-day model produce a useful week by feeding its own predictions back in?

**Setup:** the `-bt` model forecast 7 days ahead from each of 22 as-of dates (2016-04-24 … 05-15) for all 1,437 items. The decision rule was fixed before the run: enable 7 days only if the AI forecast beats both baselines on aggregate RMSE and MAE.

| Day ahead | AI RMSE | AI MAE | Naive RMSE | Seasonal naive RMSE | AI mean bias |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.337 | 1.374 | 3.194 | 3.115 | −0.01 |
| 2 | 2.396 | 1.438 | 3.507 | 3.132 | +0.09 |
| 3 | 2.469 | 1.454 | 3.698 | 3.170 | +0.09 |
| 4 | 2.482 | 1.464 | 3.738 | 3.168 | +0.08 |
| 5 | 2.504 | 1.478 | 3.614 | 3.178 | +0.09 |
| 6 | 2.529 | 1.493 | 3.376 | 3.207 | +0.09 |
| 7 | 2.580 | 1.505 | 3.257 | 3.257 | +0.08 |
| **All 7 days** | **2.472** | **1.458** | 3.489 | 3.176 | |

**Outcome:** the rule was met with a wide margin: 29% lower RMSE than naive and 22% lower than seasonal naive. Error grows gently, about 10% from day 1 to day 7. Recursion adds a small upward bias (about +4% from day 2), which is disclosed on the model card. Day 1 matched the stored one-step backtest to 3.7e-6.

**Changes made:** the check was built into the release pipeline (`evaluation/backtest.py::recursive_backtest`), and `1.1.0` was released with horizons `[1, 7]`.

### 9.2 Portable features and small stores ([`PortableFeatures_CA1Foods.ipynb`](../notebooks/PortableFeatures_CA1Foods.ipynb))

**Question 1:** what does it cost to drop the M5-only inputs (IDs, M5 day numbers, US events, SNAP)?

| Model | Backtest RMSE | MAE | WAPE |
| --- | --- | --- | --- |
| `m5-v1` (Stage 3) | 2.377 | 1.377 | 61.0% |
| `portable-v1` | 2.380 | 1.379 | 61.1% |
| Naive | 3.219 | 1.729 | 76.6% |

**Outcome:** about 0.1–0.2% worse, so practically nothing. The recipe transfers; the M5 weights don't need to.

**Question 2:** how much data does a business need? 48 simulated stores were built (10/25/50/100 products × 90/120/180/365 days × 3 seeds), each run through the real own-data training pipeline.

**Outcome:**
- Stores with 50+ products always passed the gate.
- Every store with 4,500+ product-days and 120+ days passed in at least 2 of 3 runs. Every store with 6,000+ product-days passed in all 3.
- Every failure was on MAE, never RMSE. Tweedie predicts the expected value, while MAE rewards the median, which is often zero for intermittent items.

**Changes made:** `OWN_DATA_ELIGIBILITY.minObservedProductDays` changed from the provisional 5,000 to 4,500.

### 9.3 Error by demand class ([`ErrorByDemandClass_CA1Foods.ipynb`](../notebooks/ErrorByDemandClass_CA1Foods.ipynb))

**Question:** does the AI advantage hold under volatile demand? This is the research question.

| Class (Syntetos–Boylan) | Items | AI RMSE | Naive RMSE | RMSE gain | MAE gain | Items where AI wins on MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Smooth | 221 | 3.73 | 5.22 | 29% | 26% | 96% |
| Erratic | 37 | 5.16 | 6.56 | 21% | 22% | 100% |
| Intermittent | 925 | 1.58 | 2.12 | 25% | 17% | 74% |
| Lumpy | 249 | 2.64 | 3.54 | 25% | 19% | 78% |

**Outcome:**
- AI wins in every class.
- The MAE advantage shrinks for intermittent and lumpy items, where naive still wins on MAE for about a quarter of the items.
- The per-class breakdown is now stored in every model's backtest summary (`byDemandClass`).

---

## 10. Own-data models: retraining the recipe per business

`pipelines/own_data.py` turns the portable experiment into a product feature:

1. **Canonical rows to wide matrices** (`to_wide`). Absent days stay NaN, prices are carried forward, and the first-price date is inferred.
2. **Eligibility.** At least 10 products with 21+ days, at least 120 store days, and at least 4,500 observed product-days.
3. **Windows measured back from the last history day *L*:**
   - train up to *L*−56;
   - validate (early stopping) on *L*−55 … *L*−28;
   - backtest on *L*−27 … *L*.
   On CA_1 data these are exactly the M5 windows.
4. **Gate.** The model is used only if its backtest RMSE **and** MAE both beat naive on the same rows. Otherwise the result is `GATE_FAILED` and nothing is registered.
5. **Recursive check.** A 7-day recursive backtest inside the backtest window decides whether the workspace model gets horizons `[1, 7]` or just `[1]`.
6. **Register.** Refit on all history and save the artifact as `own-<workspaceId>@<n>.0.0`, with its own model card.

---

## 11. The internal ML service

`mlengine/service/app.py` (FastAPI) is the only way the rest of the system uses the model. Its main features:

- **Stateless forecasting.** The backend sends the history needed for each request. The service keeps no business data, apart from own-data model artifacts.
- **Per-series reasons instead of request failures.** One bad product doesn't fail a store-wide batch.
- **The leakage rule at the boundary.** Clients cannot get a forecast from a model trained after the as-of date.
- **The service fills in M5 specifics itself.** It derives the identity columns and calendar, so the backend sends only `productId`, `storeId`, history and prices.
- **Shared-secret auth** (`X-Internal-Token`). It is meant for a private network only.

Endpoints are documented in [`docs/ml-service-api.md`](ml-service-api.md). Real request/response pairs are in [`docs/ml-service-examples/`](ml-service-examples/).

---

## 12. Decision log

| # | Decision | Reason | Alternatives rejected |
| --- | --- | --- | --- |
| 1 | Ship Stage 3, not Stage 7 | Same accuracy within noise, 16 fewer features | Stage 7; pooled Stage 6 for now |
| 2 | Port before improving | Any later change can be measured against an exact baseline | Re-implementing "cleanly" with new numerics |
| 3 | Keep the wide-matrix representation | It is the notebook's shape, it is fast (O(1) per window via cumsums), and it serves short requests and full training alike | Long-format `groupby().rolling()` (slower; numerics differ) |
| 4 | NaN means "unrecorded", never zero | Lags and rolling windows would silently treat gaps as no demand | Zero-filling |
| 5 | Three versions (`-dev`, `-bt`, prod) plus the leakage rule | Honest replays for the research comparison | One model for all dates |
| 6 | Recursive multi-day forecasts behind a pre-registered rule | No new model needed; enabled only with evidence | Direct per-horizon models (deferred) |
| 7 | Portable schema + per-workspace retraining | The M5 model is invalid for other catalogues; the recipe transfers (0.1% cost) | Applying the M5 model to unknown products |
| 8 | Gate on RMSE **and** MAE | A business should see AI only when it is better on both | RMSE only (would pass more stores) |
| 9 | Commit artifacts to Git | They are about 2.5 MB; the service runs from a fresh clone | Release assets (planned originally) |
| 10 | Minimum history 21 days (not 22) | The longest lag is 21; the spec's "22" was the first *target day*, not the history length | — |
| 11 | ML service separate from the backend | Keeps all forecasting logic in one Python codebase; the backend can be any stack | Forecasting code inside the backend |

---

## 13. Known limitations and future work

**Limitations:**
- **Scope.** The model is validated on one store and one dense category (FOODS). HOUSEHOLD and HOBBIES are more intermittent and untested.
- **Point forecasts only.** There are no prediction intervals. Quantile LightGBM is a candidate.
- **Recursion bias.** Days 2–7 run about 4% high. A direct multi-horizon model would avoid this.
- **Short backtests.** Per-product backtests are only 28 days, so they are noisy. The app labels them as small samples.
- **Future prices.** Own-data models assume the last known price.
- **Training jobs live in memory.** The service forgets unfinished jobs on restart. The backend should retry.
- **No uncertainty about the gate itself.** The gate's verdict comes from a single 28-day backtest.

**Untried levers from the research:**
- tuning the Tweedie variance power;
- hierarchical aggregate features (department, store);
- ensembling;
- blending regressors by predicted bucket probabilities;
- a pooled multi-store Reference model;
- a 28-day direct model (the Stage 2 design) for month-ahead views.

---

## 14. Reproducing everything

```bash
pip install -r requirements.txt
python scripts/make_parity_fixture.py        # notebook cells vs mlengine: identical features (~3 min)
python -m mlengine release-m5                # retrain and register 1.1.0-dev/-bt/1.1.0 (~2 min, ~2.3 GB RAM)
MLENGINE_FULL_TESTS=1 pytest                 # fast suite + full-data parity
python -m mlengine export-reference          # Reference workspace seed files
jupyter nbconvert --to notebook --execute --inplace notebooks/RecursiveHorizon_CA1Foods.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/PortableFeatures_CA1Foods.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/ErrorByDemandClass_CA1Foods.ipynb
```

Environment used: Python 3.12.3, pandas 2.3.3, NumPy 2.5.4, LightGBM 4.7.0, on a 12-thread machine with 7 GB RAM.
