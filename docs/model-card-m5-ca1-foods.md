# Model Card: `m5-ca1-foods` 1.1.0

The machine-readable version is [`models/m5-ca1-foods/1.1.0/card.json`](../models/m5-ca1-foods/1.1.0/card.json), with full details in `metadata.json`.

## Overview

| | |
| --- | --- |
| **What it predicts** | Expected units sold per product per day |
| **Horizons** | 1 day (validated, one step ahead), 7 days (validated, recursive); up to 28 days experimental |
| **Algorithm** | LightGBM gradient-boosted trees, Tweedie objective (variance power 1.1), 70 trees, 31 leaves, learning rate 0.05 |
| **Training data** | M5 Forecasting – Accuracy (Walmart), store CA_1, FOODS category: 1,437 items, 2011-02-19 to 2016-05-22, 2,759,040 item-days |
| **Feature schema** | `m5-v1`, 31 features (below) |
| **Versions** | `1.1.0` (trained to 2016-05-22), `1.1.0-bt` (to 2016-04-24), `1.1.0-dev` (to 2016-03-27). All share the recipe; each serves only as-of dates on or after its training end |
| **Software** | `mlengine` 0.1.0, LightGBM 4.7.0 |
| **Intended use** | The mobile app's Reference (demo) workspace, and the research comparison of AI vs traditional forecasting |
| **Not for** | Products outside the M5 CA_1 FOODS catalogue (the service returns `PRODUCT_NOT_IN_MODEL`). Businesses get their own retrained model instead (`portable-v1`) |

## Inputs

| Group | Features |
| --- | --- |
| Identity (categorical) | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| Time index | `d_num`, `wm_yr_wk`, `year_num` |
| Recent sales | `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_21` |
| Rolling statistics (window ends the day before the target) | `rolling_mean_3/7/14`, `rolling_std_3/7/14` |
| Calendar | `weekday_num` (Mon = 0), `month_num`, `day_of_month`, `is_weekend` |
| Events (categorical) | `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` |
| Assistance | `snap` (California SNAP day) |
| Price | `sell_price`, `price_missing`, `weeks_since_first_price` |

## Evaluation

All windows are chronological and out of sample. Each version is evaluated only on days after its training data ends.

**Next day (one step ahead).** Backtest 2016-04-25 to 2016-05-22, model `1.1.0-bt`, 40,236 item-days:

| Method | RMSE | MAE | WAPE |
| --- | --- | --- | --- |
| **AI** | **2.377** | **1.377** | **61.0%** |
| Naive (same as last day) | 3.219 | 1.729 | 76.6% |
| Seasonal naive (same day last week) | 3.179 | 1.748 | 77.4% |

The development window (2016-03-28 to 04-24, model `-dev`) gave AI 2.356 / 1.308 against naive 3.194 / 1.624.

**Seven days (recursive).** 22 forecast dates from 2016-04-24 to 05-15, 221,298 item-days:

| Days ahead | 1 | 2 | 3 | 4 | 5 | 6 | 7 | All |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI RMSE | 2.34 | 2.40 | 2.47 | 2.48 | 2.50 | 2.53 | 2.58 | **2.47** |
| Naive RMSE | 3.19 | 3.51 | 3.70 | 3.74 | 3.61 | 3.38 | 3.26 | 3.49 |
| Seasonal naive RMSE | 3.12 | 3.13 | 3.17 | 3.17 | 3.18 | 3.21 | 3.26 | 3.18 |

**By demand class** (next-day backtest):

| Class | Items | AI RMSE | Naive RMSE | AI wins on MAE (share of items) |
| --- | --- | --- | --- | --- |
| Smooth | 221 | 3.73 | 5.22 | 96% |
| Erratic | 37 | 5.16 | 6.56 | 100% |
| Intermittent | 925 | 1.58 | 2.12 | 74% |
| Lumpy | 249 | 2.64 | 3.54 | 78% |

## Limitations and caveats

- Validated on one store and one category (FOODS, the densest M5 category). HOUSEHOLD and HOBBIES are untested.
- Point forecasts only. There are no prediction intervals.
- 7-day forecasts feed the model's own predictions back in. From day 2 they average about 0.09 units (about 4%) above actual sales.
- For intermittent and lumpy items, the naive forecast has a lower MAE for about a quarter of products. On RMSE, AI wins for 94–97% of items in every class.
- Per-product backtests cover 28 days only and are noisy.
- Needs 21 days of history ending on the as-of date, with no more than 3 unrecorded days.
- Trained on 2011–2016 US retail data. It says nothing direct about other markets.

## Data and licence

The M5 data is from the Kaggle M5 Forecasting – Accuracy competition and is used under the competition terms for research and education. The raw files are not redistributed in this repository.
