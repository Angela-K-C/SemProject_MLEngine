# Handoff: Mobile App and Backend Build

**For:** the Claude agent building the Retail Demand Forecasting Android app (and the backend it talks to)
**From:** the ML engine work in `SemProject_MLEngine`, branch `model-implementation`
**Date:** 7 November 2026

Read this file first. Then read `retail_demand_forecasting_mobile_design.md` (v2.1), which is the full product and UI spec. Where the two differ, this file and the code are newer.

---

## 1. Where things stand

| Layer | State | Notes |
| --- | --- | --- |
| Model research (notebooks, Stages 1–7) | Done | `README.md` → "Modelling journey" |
| `mlengine` Python package | **Done, tested** | Features, training, inference, baselines, metrics, demand classes, model registry |
| Released model `m5-ca1-foods@1.1.0` (+ `1.1.0-bt`, `1.1.0-dev`) | **Done, committed** | `models/m5-ca1-foods/` (about 2.5 MB; no raw data needed to serve) |
| Internal ML service (FastAPI) | **Done, tested** | `mlengine/service/`, `Dockerfile` |
| Reference workspace seed data | **Done** (generated, not committed) | `python -m mlengine export-reference` |
| **Backend API** (app ↔ backend, design §12.1) | **Slice 1 done** (auth, demo workspace, forecast jobs); imports and own-data training not yet built | Separate repo `../SemProject_Backend`: start with its `docs/api.md` and `docs/openapi.json` |
| **Android app** | **Not started** | Your main job |

Proven on the model side:
- Features are identical to the notebook over all 2.76M rows.
- Training reproduces the Stage 3 numbers: 70 trees, dev RMSE 2.356, backtest RMSE 2.377.
- The HTTP service reproduces the backtest end to end.
- A recursive 7-day forecast beats both baselines, so horizons **1 and 7** are enabled.

---

## 2. The architecture you must keep

```text
Android app ──HTTPS──▶ Backend API ──private HTTP + X-Internal-Token──▶ ML service (mlengine)
 (Kotlin/Compose)      auth, permissions, sales history,                features, LightGBM,
                       imports, forecast runs/jobs, DB                  baselines, metrics, registry
```

Non-negotiables:

1. **The app never calls the ML service.** It calls only the backend. The ML service has no user auth and must stay off the internet.
2. **The app contains no forecasting maths.** No lags, baselines, metrics, demand classes or "accuracy". Every number on a forecast screen comes from the backend, which gets it from the ML service.
3. **The M5 model is used only for M5 products** (the Reference workspace). A business's own products get baselines until an own-data model has been trained *and* has passed its backtest gate.
4. **Offer only the horizons in `capabilities.horizons`.** Today that is `[1, 7]`. Never hardcode 14 or 30.
5. **Never fabricate data.** Show metrics only where actual sales were known. Label any mock fixtures clearly.
6. **Respect the leakage rule.** Replays must use the model version the ML service resolves. Don't let a client pick a version trained after the as-of date; the service rejects it anyway with `NO_MODEL_FOR_AS_OF`.

---

## 3. Run the ML service locally

From the repository root, with Python 3.10+:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                                                # 23 fast tests; should all pass
ML_SERVICE_TOKEN=dev-token python -m mlengine serve --host 127.0.0.1 --port 8100
# OpenAPI docs: http://127.0.0.1:8100/docs  (send header X-Internal-Token: dev-token)
```

Or with Docker: `docker build -t mlengine-service . && docker run -p 8100:8100 -e ML_SERVICE_TOKEN=dev-token mlengine-service`.

Quick check:

```bash
curl -s localhost:8100/internal/v1/health -H "X-Internal-Token: dev-token"
curl -s localhost:8100/internal/v1/forecast -H "X-Internal-Token: dev-token" \
     -H "Content-Type: application/json" -d @docs/ml-service-examples/forecast_request.json
```

Environment variables: `ML_SERVICE_TOKEN` (required in any shared environment) and `MODEL_REGISTRY_PATH` (default `./models`; own-data models are written here too, so it must be writable and persistent).

The Android emulator reaches the host machine at `10.0.2.2`. That address is for the **backend**, never the ML service.

---

## 4. Recommended build order

The backend is a separate repository, `../SemProject_Backend` (FastAPI + PostgreSQL, its own sign-in, Docker Compose + Caddy on one server). Slice 1 is built and tested end to end against this ML service: auth, workspaces/stores/products, demo seeding, history, readiness, capabilities, model card, forecast jobs, history and CSV export. **The app should call the backend as documented in `SemProject_Backend/docs/api.md`**, and generate its network client from `SemProject_Backend/docs/openapi.json`. Where that guide differs from design §12.1 (for example, IDs are UUIDs, and the SKU is a separate `sku` field), the guide is correct.

1. ~~Backend skeleton~~ and 2. ~~Forecast path for the Reference workspace~~: **done** (backend slice 1).
3. **Android foundation.** Shell, theme, Fragments + `ComposeView`, navigation, the three menus, state components (design §4–§7, §20 Phase D).
4. **Android forecast flow against the real backend.** Products → configure → progress → results → history → model info.
5. **Imports** (backend slice 2, CSV wizard, validation, readiness) for own-data workspaces. Baseline-only forecasts (`includeAi: false`).
6. **Own-data AI** (backend slice 3). Eligibility, the training job, and showing the gate result (§5.6).

If the backend must lag behind, build the app against a **fake repository** that returns the real example JSON from `docs/ml-service-examples/`, reshaped into the public API, and label it clearly as sample data in the UI.

---

## 5. ML service contract: what the backend needs to know

All endpoints are under `/internal/v1`. The full schemas are at `/docs` when the service is running, and in `mlengine/service/schemas.py`. Real examples are in `docs/ml-service-examples/`:

| File | What it shows |
| --- | --- |
| `forecast_request.json` / `forecast_response.json` | True 7-day forecast from 2016-05-22 for `FOODS_3_090` (resolved to `1.1.0`) |
| `replay_request.json` / `replay_response.json` | Replay from as-of 2016-05-08 (resolved to `1.1.0-bt`); the actual for 2016-05-09 was 72 |
| `error_no_model_for_as_of.json` | What happens when a replay asks for a version trained after the as-of date |

The examples send only 28 days of history, so their `demandClass` is `INSUFFICIENT_DATA`. Send **365 days** in production.

### 5.1 `POST /internal/v1/forecast`

Request essentials:
- `model: {id, version?}`. Omit the version: the service picks the newest version trained only on data up to `asOfDate`.
- `asOfDate`: the last day of history the forecast may use. The forecast starts the next day.
- `horizonDays`: must be in the model's `capabilities.horizons` (`[1, 7]`). `allowExperimentalHorizon: true` allows up to 28 for research users only; those responses carry `horizon.validated: false`.
- `baselines`: `["naive_last_value", "seasonal_naive_7"]`. The primary research baseline is `naive_last_value`.
- `includeAi: false` with no model: baselines and demand class only (own-data stores without a model).
- `series[]`, up to 5,000:
  - `productId`, `storeId`, optional `category` and `firstPriceDate`;
  - `history[{date, units, price}]`. Units of `null`, or an absent day, mean "not recorded", which is not zero. Don't send dates after `asOfDate`. At most 400 days are used;
  - `futurePrices[{date, price}]` for the forecast days, when known.
- For M5 products, send **only** `productId` (e.g. `FOODS_3_090`) and `storeId` (`CA_1`). The service derives the other M5 IDs, the calendar, events and SNAP itself.

Response essentials:
- Top level: `model{id, version, featureSchema, trainingEndDate}`, `horizon{startDate, endDate, days, validated}`, `mlengineVersion`, `baselineLabels` (display names, e.g. "Same as last day").
- Per series:
  - `points[{date, ai, baselines{naive_last_value, seasonal_naive_7}}]`;
  - `aiUnavailableReason`, `missingDaysInLookback`, `priceAssumption`;
  - `demandClass{class, label, adi, cv2, zeroShare, observedDays, saleDays}`;
  - `warnings`.
- **A per-series failure is not an HTTP error.** If AI can't run for one product, `ai` is `null`, `aiUnavailableReason` says why, and the baselines are still filled.

Values are expected units, rounded to 4 decimals, for example `48.1623`. Store them raw. The app shows one decimal place ("≈ 48.2 units") and sums raw values for totals.

### 5.2 Other endpoints

| Endpoint | Use it for |
| --- | --- |
| `GET /health`, `GET /models` | Startup check; capabilities of every version |
| `GET /models/{id}/resolve?asOfDate=` | Show which model a replay will use before running it |
| `GET /models/{id}/{version}/card` | `ModelInfoFragment` (design §8.13) |
| `GET /models/{id}/{version}/backtest?productId=` | "How the model did on this product" (design §8.10, item 2) |
| `POST /backtest` | Recompute a one-step backtest over a window (≤ 91 days) |
| `POST /demand-class` | Product list chips and filters (design §6.3, §8.5) |
| `POST /eligibility` | Own-data "progress toward AI", e.g. "64 of 120 days" |
| `POST /train` → `GET /train/{jobId}` | Own-data training. The job result `status` is `TRAINED`, `GATE_FAILED` or `NOT_ELIGIBLE` |

Training jobs live in the ML service's memory. If it restarts, the backend should treat unfinished jobs as failed and retry. Finished models persist in the registry.

### 5.3 Error codes

- **Request-level** (`{"error": {"code", "message", "details", "requestId"}}`):
  - `INVALID_REQUEST`, `MODEL_REQUIRED`, `NO_MODEL_FOR_AS_OF`, `HORIZON_NOT_SUPPORTED`, `HISTORY_AFTER_AS_OF`, `DATE_OUTSIDE_CALENDAR`, `INVALID_WINDOW`, `INVALID_HISTORY` (all 422);
  - `MODEL_NOT_FOUND`, `NO_STORED_BACKTEST`, `JOB_NOT_FOUND` (404);
  - `UNAUTHORIZED` (401).
- **Per-series** (`aiUnavailableReason`): `INSUFFICIENT_HISTORY` (< 21 days), `TOO_MANY_GAPS` (> 3 unrecorded days in the last 21), `PRODUCT_NOT_IN_MODEL`, `AI_NOT_REQUESTED`.

Suggested app copy (localise in string resources; always fall back to the backend's `message`):

| Code | Message |
| --- | --- |
| `INSUFFICIENT_HISTORY` | "This product needs at least 21 days of sales history for an AI forecast." |
| `TOO_MANY_GAPS` | "Several recent days have no sales recorded, so only the simple forecast is shown." |
| `STALE_HISTORY` (backend) | "Your latest sales are from {date}. Upload sales up to yesterday to forecast tomorrow." |
| `AI_NOT_AVAILABLE` (backend) | "AI forecasts aren't available for this store yet. 'Same as last day' forecasts are available now." |
| `PRODUCT_NOT_IN_MODEL` | "The demo model only knows the demo store's products." |
| `NO_MODEL_FOR_AS_OF` | "No model is available for that date without using information from the future." |
| `HORIZON_NOT_SUPPORTED` | Hide the option. This should never reach the user. |

---

## 6. Seeding the Reference workspace

Run `python -m mlengine export-reference` (needs the raw M5 CSVs in `data/raw/`). It writes `data/processed/reference/`:

| File | Rows | Columns |
| --- | --- | --- |
| `store.json` | 1 | `storeId: CA_1`, name "Demo Store (M5 CA_1)", `isReference: true`, `country: US`, `subdivision: CA`, `timezone: America/Los_Angeles`, `currency: USD`, history 2011-01-29 → 2016-05-22, `calendarEndDate: 2016-06-19`, `modelId: m5-ca1-foods` |
| `products.csv` / `.parquet` | 1,437 | `store_id, product_id, name, category (FOODS_1/2/3), unit_of_measure (units), first_price_date, first_sale_date` |
| `sales_daily.csv` / `.parquet` | 2,789,217 | `store_id, product_id, date, units_sold, unit_price` (price empty before an item's first priced week) |
| `prices_future.csv` / `.parquet` | 40,236 | `store_id, product_id, date, unit_price` for 2016-05-23 → 2016-06-19 |

Every M5 day is recorded (zeros included), so the Reference workspace has no gaps. Store `first_price_date` on the product and send it as `firstPriceDate`. Send `prices_future` rows as `futurePrices` for true forecasts. Mark the workspace read-only.

### 6.1 Reference workspace dates (put these in the capabilities response)

| As-of date | Model resolved | What the user gets |
| --- | --- | --- |
| 2016-05-22 (**default**) | `1.1.0` | True forecast for 2016-05-23 (→ 05-29 for 7 days). No actuals. |
| 2016-04-24 … 2016-05-21 | `1.1.0-bt` | **Replay**: the backend attaches actuals from `sales_daily`. For 7-day replays with all 7 actuals known, the latest as-of is 2016-05-15. |
| 2016-03-27 … 2016-04-23 | `1.1.0-dev` | Research-user replays only |
| before 2016-03-27 | none | `NO_MODEL_FOR_AS_OF` |

The M5 calendar ends 2016-06-19, so the service rejects forecasts past it with `DATE_OUTSIDE_CALENDAR`.

**Actuals for replays come from the backend's own `sales_daily`, not from the ML service.** The backend computes per-run errors with the same definitions (MAE = mean |actual − forecast|) or calls `POST /internal/v1/backtest`.

---

## 7. Public API → ML service mapping

| Public endpoint (design §12.1) | Backend does | ML service call |
| --- | --- | --- |
| `GET /v1/stores/{sid}/models/capabilities` | Merge the store's mode (reference/own) and AI availability with model capabilities; add the as-of ranges in §6.1 | `GET /models`, `GET /models/{id}/resolve` |
| `GET /v1/stores/{sid}/models/active` | Model card for the store's model | `GET /models/{id}/{version}/card` |
| `POST /v1/forecasts` (job worker) | Load ≤ 365 days of history + future prices; call the service; store run + points; attach actuals if replay; attach stored product backtest | `POST /forecast`, then `GET /models/{id}/{bt-version}/backtest?productId=` |
| `GET /v1/stores/{sid}/readiness` | Freshness (latest day = yesterday in store timezone) is the backend's own check; per-product history, gap and eligibility checks | `POST /eligibility` (own-data) |
| Product list demand-class chips | Cache per product; refresh after each import | `POST /demand-class` |
| `POST /v1/stores/{sid}/models/train` | Owner-only; send the store's full history | `POST /train`, poll `GET /train/{jobId}` |
| Own-data forecast with no active model | Baselines only | `POST /forecast` with `includeAi: false` |

For own-data workspaces the model ID is `own-<workspaceId>`. The service creates it on the first successful training.

---

## 8. What the screens should show (backed by real fields)

| Screen element (design section) | Source field |
| --- | --- |
| "Tomorrow's expected demand ≈ 48.2 units" (§8.10) | `points[0].ai` |
| AI vs baseline lines, table (§8.10) | `points[].ai`, `points[].baselines.naive_last_value` (label from `baselineLabels`) |
| "Based on sales up to 22 May 2016" banner | `asOfDate` |
| Demand class card | `demandClass.class` + `demandClass.label` |
| Model name/version in the header and run details | `model.id`, `model.version` |
| "Experimental" badge | `horizon.validated == false` |
| "Price assumed unchanged" note | `priceAssumption == "LAST_KNOWN"` |
| This product's backtest (28 days, 2016-04-25 → 05-22) | stored backtest `product.ai/naive_last_value.{mae, rmse, wape}` |
| Model card: training data, inputs, limitations | `card.trainedOn`, `card.inputsPlainLanguage`, `card.limitations` |
| Model card: accuracy by days ahead (a small line chart) | `card.horizonAccuracy.byDay[].aiRmse/naiveRmse` |
| Model card: overall backtest | `card.backtest.ai` vs `card.backtest.naive_last_value` |

For your understanding only (never hardcode these in the app), the current numbers are:
- **Store-level next-day backtest:** AI RMSE 2.377 / MAE 1.377 against naive 3.219 / 1.729.
- **7-day:** AI RMSE rises from 2.34 on day 1 to 2.58 on day 7, against naive 3.19–3.74.
- **Demand classes:** 925 of the 1,437 demo products are intermittent, so many demo products will show "Sells on some days only".

---

## 9. Pitfalls already found

- **Zero vs missing.** Absent days are "not recorded", not zero. The backend must apply the import wizard's answer ("no row means zero sales?") when it stores data, and send explicit `units: 0` for true zeros.
- **Store timezone defines "a day".** Aggregate transactions per store-local date before storing.
- **Send history ending exactly at the as-of date.** If the last recorded day is earlier, those days count as gaps and can trigger `TOO_MANY_GAPS`. That is the right behaviour, but the backend's readiness check should catch stale data first, with a clearer message.
- **Demand class needs history.** With under 30 observed days, or under 5 sale days, the class is `INSUFFICIENT_DATA`.
- **Decimals are intentional.** "0.4 units" and "1.4 units" are different forecasts. Don't round them away by default.
- **7-day forecasts run about 4% high** on days 2–7 (recursion bias). This is on the model card's limitations. Don't hide it.
- **On MAE, naive beats AI for about a quarter of intermittent products.** If a product's stored backtest shows that, say so plainly (design §8.10); it is a valid research outcome.
- **Own-data training can return `GATE_FAILED`.** That is a normal outcome, not an error. Show "the AI model didn't beat the simple forecast on your recent data", and keep baselines.

---

## 10. Files you will need

| Path | Why |
| --- | --- |
| `retail_demand_forecasting_mobile_design.md` | Full app spec (screens §8, menus §6, design system §7, public API §12.1, acceptance §19) |
| `docs/ml-service-api.md` | Field-by-field ML service reference (`schemas.py` and `app.py` are the source of truth) |
| `docs/ml-service-examples/*.json` | Real payloads for fakes and tests |
| `models/m5-ca1-foods/1.1.0/card.json` | The model card the app renders |
| `mlengine/config.py` | Thresholds (21-day lookback, 3 gaps, own-data eligibility) |
| `notebooks/RecursiveHorizon_CA1Foods.ipynb`, `ErrorByDemandClass_CA1Foods.ipynb`, `PortableFeatures_CA1Foods.ipynb` | Evidence behind the 7-day horizon, the demand-class claims and the own-data thresholds |

## 11. Do not

- Do not change anything under `mlengine/` or `models/` to make the app work. If the contract needs to change, raise it with the project owner. Model or feature changes need a new model version and the parity tests (`pytest`, `MLENGINE_FULL_TESTS=1 pytest`).
- Do not expose the ML service publicly or put `ML_SERVICE_TOKEN` in the APK.
- Do not create XML screen layouts. Screens are Compose inside Fragments (design §4).
- Do not add forecast horizons, intervals, accuracy claims or explanations that the API does not return.
