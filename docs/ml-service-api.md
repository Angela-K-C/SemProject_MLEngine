# ML Service API Reference (`/internal/v1`)

This is the internal HTTP API in front of `mlengine`. **Only the backend calls it.** It must not be reachable from the internet or from the mobile app.

- Base URL (local): `http://127.0.0.1:8100`
- Interactive docs: `/docs` (OpenAPI generated from `mlengine/service/schemas.py`)
- Auth: when the environment variable `ML_SERVICE_TOKEN` is set, every request must send `X-Internal-Token: <token>`. Otherwise the response is `401 UNAUTHORIZED`.
- Correlation: send `X-Request-Id`; it is echoed in error bodies.
- Dates are ISO `YYYY-MM-DD`. Quantities are units sold per day.

Real request/response examples: [`ml-service-examples/`](ml-service-examples/).

## Errors

Request-level errors always have this shape:

```json
{"error": {"code": "NO_MODEL_FOR_AS_OF", "message": "…", "details": {}, "requestId": "…"}}
```

| Code | HTTP | Meaning |
| --- | --- | --- |
| `INVALID_REQUEST` | 422 | The body doesn't match the schema; `details.errors` lists the fields |
| `UNAUTHORIZED` | 401 | Missing or wrong `X-Internal-Token` |
| `MODEL_REQUIRED` | 422 | `includeAi` is true but `model` was not given |
| `MODEL_NOT_FOUND` | 404 | Unknown model ID or version |
| `NO_MODEL_FOR_AS_OF` | 422 | The leakage rule: no (or the requested) version was trained only on data up to `asOfDate` |
| `HORIZON_NOT_SUPPORTED` | 422 | `horizonDays` is not in `capabilities.horizons` (and not allowed as experimental) |
| `HISTORY_AFTER_AS_OF` | 422 | A history date is later than `asOfDate` |
| `DATE_OUTSIDE_CALENDAR` | 422 | M5 model asked to forecast past 2016-06-19 |
| `INVALID_WINDOW` | 422 | Backtest window reversed or longer than 91 days |
| `INVALID_HISTORY` | 422 | Training/eligibility history is malformed (e.g. duplicate product-days) |
| `NO_STORED_BACKTEST` | 404 | No stored backtest for that product/version |
| `JOB_NOT_FOUND` | 404 | Unknown training job |

Per-series problems are **not** HTTP errors. See `aiUnavailableReason` below.

## Shared objects

**SeriesIn**

| Field | Type | Notes |
| --- | --- | --- |
| `productId` | string | M5 models: the M5 item ID, e.g. `FOODS_3_090` |
| `storeId` | string | M5 models: `CA_1` |
| `category` | string? | Used by own-data models; `UNCATEGORISED` if absent |
| `firstPriceDate` | date? | First day the product had a price. Inferred from history if absent (adds the warning `FIRST_PRICE_DATE_INFERRED`) |
| `history` | `[{date, units, price}]`, 1–1000 | `units ≥ 0`; `units: null` or an absent date = not recorded. At most the last 400 days before `asOfDate` are used. Send 365 days for a meaningful demand class |
| `futurePrices` | `[{date, price}]`? | Known prices for the forecast days. Missing ones fall back to the last known price |

**DemandClass**: `{class, label, adi, cv2, zeroShare, observedDays, saleDays}`. `class` is one of `SMOOTH`, `ERRATIC`, `INTERMITTENT`, `LUMPY`, `INSUFFICIENT_DATA` (fewer than 30 observed days or fewer than 5 sale days in the last 365).

## Endpoints

### `GET /internal/v1/health`
Returns `{"status": "ok", "mlengineVersion", "models": ["m5-ca1-foods@1.1.0", …], "registry"}`.

### `GET /internal/v1/models`
Lists every registered version: `{id, version, status, featureSchema, trainingEndDate, capabilities, createdAt}`. `capabilities` contains `horizons`, `experimentalMaxHorizon`, `frequency`, `minHistoryDays` (21), `maxMissingDaysInLookback` (3), `predictionIntervals` (false), `baselines`, and, for M5 models, `calendarEndDate`.

### `GET /internal/v1/models/{id}/resolve?asOfDate=YYYY-MM-DD`
Returns the version the leakage rule would use for that date: `{id, version, trainingEndDate, capabilities}`.

### `GET /internal/v1/models/{id}/{version}` and `…/card`
Return metadata (without the categorical levels) and the model card. The card fields are `title`, `model`, `trainedOn`, `predicts`, `horizonDays`, `inputsPlainLanguage`, `backtest`, `horizonAccuracy{description, validated, byDay[{day, aiRmse, aiMae, naiveRmse, naiveMae}]}`, `limitations` and `note`.

### `GET /internal/v1/models/{id}/{version}/backtest[?productId=]`
Returns the stored store-level backtest summary (including `byDemandClass`). With `productId`, it also returns `product{productId, n, window, demandClass, ai{mae, rmse, wape}, naive_last_value{…}, seasonal_naive_7{…}}`.

### `POST /internal/v1/forecast`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `requestId` | string? | | Echoed back |
| `model` | `{id, version?}`? | | Omit `version` to let the leakage rule choose |
| `includeAi` | bool | `true` | `false` = baselines + demand class only (no model needed) |
| `asOfDate` | date | | Last day of history the forecast may use |
| `horizonDays` | int 1–28 | 1 | Must be in `capabilities.horizons` unless experimental |
| `allowExperimentalHorizon` | bool | `false` | Research only: allows up to `experimentalMaxHorizon` |
| `baselines` | list | `["naive_last_value"]` | Also `seasonal_naive_7` |
| `series` | `SeriesIn[]`, 1–5000 | | |

The response is shown below. Its points come from the real example, but the `demandClass` values are illustrative; see `ml-service-examples/` for complete real output.

```json
{
  "requestId": "…", "asOfDate": "2016-05-22",
  "horizon": {"startDate": "2016-05-23", "endDate": "2016-05-29", "days": 7, "validated": true},
  "model": {"id": "m5-ca1-foods", "version": "1.1.0", "featureSchema": "m5-v1", "trainingEndDate": "2016-05-22"},
  "mlengineVersion": "0.1.0",
  "baselineLabels": {"naive_last_value": "Same as last day", "seasonal_naive_7": "Same day last week"},
  "series": [{
    "productId": "FOODS_3_090", "storeId": "CA_1",
    "points": [{"date": "2016-05-23", "ai": 48.1623, "baselines": {"naive_last_value": 64.0, "seasonal_naive_7": 38.0}}],
    "aiUnavailableReason": null, "missingDaysInLookback": 0, "priceAssumption": "KNOWN",
    "demandClass": {"class": "SMOOTH", "label": "Steady seller", "adi": 1.0, "cv2": 0.2, "zeroShare": 0.0,
                    "observedDays": 365, "saleDays": 365},
    "warnings": []
  }]
}
```

- `aiUnavailableReason`: `null`, `INSUFFICIENT_HISTORY` (< 21 recorded days ending at as-of), `TOO_MANY_GAPS` (> 3 unrecorded days in the last 21), `PRODUCT_NOT_IN_MODEL`, or `AI_NOT_REQUESTED`. When set, every `ai` value is `null`, but baselines are still returned.
- `priceAssumption`: `KNOWN` (all future prices supplied), `LAST_KNOWN`, `NONE` (no price at all), or `NOT_USED` (baselines only).
- `warnings`: `FIRST_PRICE_DATE_INFERRED`, `PRODUCT_NEW_SINCE_TRAINING` (own-data product added after training).
- `horizon.validated: false` means an experimental horizon was used.

### `POST /internal/v1/backtest`
Body: `{requestId?, model{id, version?}, startDate, endDate, series[]}`. `series[].history` must cover `startDate − 21` … `endDate`, including the actual sales. The model is resolved for as-of `startDate − 1`.

The service runs one-step-ahead forecasts for each day in the window and returns `{model, window, oneStepAhead: true, overall{ai, naive_last_value, seasonal_naive_7}, series[{productId, storeId, ai, naive_last_value, seasonal_naive_7, points[{date, actual, ai, naive_last_value}]}]}`. Each metrics block is `{n, mae, rmse, wape}`, computed on the rows where AI, naive and the actual are all present.

### `POST /internal/v1/demand-class`
Body: `{series[]}`. Returns `{series[{productId, storeId, class, label, adi, cv2, zeroShare, observedDays, saleDays}]}`, computed on history up to the latest date sent.

### `POST /internal/v1/eligibility`
Body: `{history: [{storeId, productId, category?, date, units, price?}]}`. Returns `{eligible, checks{productsWithEnoughHistory, storeHistoryDays, observedProductDays: {value, required, passed}}, rules}`.

### `POST /internal/v1/train` → `202`
Body: `{workspaceId, country?, subdivision?, history: [SalesRow]}`. `country` is an ISO code for the holiday calendar, e.g. `KE`. Returns `{jobId, status: "QUEUED", statusUrl}`. Jobs run one at a time in a background thread.

### `GET /internal/v1/train/{jobId}`
Returns `{jobId, workspaceId, status, createdAt, finishedAt?, result?, error?}`, where `status` is `QUEUED`, `RUNNING`, `COMPLETED` or `FAILED`.

`result.status` is one of:
- `TRAINED`: the model is registered as `own-<workspaceId>@<n>.0.0`; see `result.model` and `result.horizons`.
- `GATE_FAILED`: the AI did not beat naive on both RMSE and MAE. Nothing was registered. This is a normal outcome.
- `NOT_ELIGIBLE`: see `result.eligibility`.

`result` also carries `backtest`, `gate` and `recursive` when training ran.

**Jobs live in memory.** After a service restart, unknown job IDs return `JOB_NOT_FOUND`, so the backend should mark such jobs as failed and resubmit.
