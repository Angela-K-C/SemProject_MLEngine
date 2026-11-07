# Handoff: Android App Build

**For:** the Claude agent building the Retail Demand Forecasting Android app.
**From:** the model and backend work (finished 7 November 2026).
**Your job:** build the Android app described in [`retail_demand_forecasting_mobile_design.md`](retail_demand_forecasting_mobile_design.md) (the spec), on top of the **finished backend API**. Everything server-side exists and is tested. You don't need to build, mock or change the backend or the model.

Read this file first, then the spec, then [`SemProject_Backend/docs/api.md`](../SemProject_Backend/docs/api.md). Where they disagree, api.md is right about the API, and this file is right about everything else. Ignore any spec text about building `mlengine` or the ML service (for example in §21); that work is done.

---

## 1. What exists

| Part | Where | Status |
| --- | --- | --- |
| Model research (notebooks) | this repo (`SemProject_MLEngine`), `notebooks/` | Done |
| ML engine + internal ML service | this repo, `mlengine/`, `models/` | Done, tested. **Not for the app to call** |
| **Backend API** (what the app calls) | `../SemProject_Backend` ([GitHub](https://github.com/Karanei-Kimutai/SemProject_Backend)) | **Done**: auth, workspaces, stores, products, history, readiness, capabilities, model cards, forecast jobs, CSV imports, own-data training, team, password reset, research export. 56 tests; checked end to end in Docker |
| Demo data | Seeded into the backend from `python -m mlengine export-reference` | Done |
| Sample business data | `python -m mlengine export-samples` → `data/processed/samples/` | Done |
| **Android app** | `VolatileDemandForecastingMobileApplication` (the project is already created; you work in it) | **Not started: your job** |

## 2. Architecture and non-negotiables

```text
Android app ──HTTPS──▶ Backend API ──(private)──▶ ML service
 (this job)            ../SemProject_Backend       this repo; never called by the app
```

1. **The app calls only the backend** (`/v1/...`). It never calls the ML service (`/internal/v1/...`, port 8100) and never holds `ML_SERVICE_TOKEN`.
2. **The app contains no forecasting maths.** No lags, baselines, errors, metrics, demand classes or "accuracy". Every number on a forecast or model screen comes from the API. The only arithmetic allowed is display formatting and summing daily values for totals and a weekly view (spec §4.3).
3. **Build the UI from `capabilities`.** Horizons, methods, as-of ranges and AI availability come from `GET /v1/stores/{sid}/models/capabilities`. Never hardcode `[1, 7]`, dates or thresholds.
4. **Never fabricate data.** Show errors and metrics only where the API returned them (`runMetrics`, `productBacktest`, model card). No placeholder accuracy, totals or "insights".
5. **Be honest about results.** If the baseline beat the AI on a product, or a business's training ended `GATE_FAILED`, say so plainly. These are valid research outcomes (spec §8.10).
6. **Follow the spec's stack** (§4): Kotlin, Fragments that each host a `ComposeView`, all screens in Jetpack Compose (no XML screen layouts), Material 3, MVVM, Coroutines/Flow, Retrofit + OkHttp, Kotlin Serialization, Room, DataStore, WorkManager, Storage Access Framework. It also requires the **three menu types** (options, context, popup; §6).

## 3. Run the whole stack locally

You need Docker, plus the three repositories side by side in one folder:

```text
<parent>/
├── VolatileDemandForecastingMobileApplication/   ← you are here
├── SemProject_MLEngine/        (branch model-implementation until it is merged)
└── SemProject_Backend/
```

**The project owner normally runs `SemProject_Backend\scripts\setup-local.ps1` before you start.** It does every step in this section, so the stack should already be up and seeded. Check with `curl http://localhost:8000/healthz` before redoing anything below.

If you can't read `../SemProject_Backend` or `../SemProject_MLEngine`, ask the owner to start you with `claude --add-dir ../SemProject_MLEngine --add-dir ../SemProject_Backend`, or to run `/add-dir`.

**On Windows**, use Docker Desktop. Run the commands below in Git Bash, which handles `$PWD` and `( … )`. In PowerShell, replace `$PWD/..` with `${PWD}/..`, and run the `cd` steps separately. `export-reference` and `export-samples` need a Python environment in the ML engine (`py -m venv .venv`, `.venv\Scripts\pip install -e .`). If the owner copied `data/processed/` across, skip the exports.

```bash
cd ../SemProject_Backend
cp .env.example .env        # then set the local values below
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
curl http://localhost:8000/healthz                     # {"status":"ok",...}; API docs at http://localhost:8000/docs
```

Local `.env` values that work:

```env
API_DOMAIN=localhost
POSTGRES_PASSWORD=localdevpw
DATABASE_URL=postgresql+psycopg://forecast:localdevpw@db:5432/forecast
JWT_SECRET=local-docker-secret-0123456789abcdef0123456789
ML_SERVICE_TOKEN=local-docker-ml-token
ML_SERVICE_URL=http://ml:8100
MLENGINE_PATH=../SemProject_MLEngine
DB_PORT=55433
API_PORT=8000
```

**Load the demo workspace once.** It needs the raw M5 CSVs in `SemProject_MLEngine/data/raw/`; the project owner has them. Seeding takes about 7 minutes in Docker:

```bash
(cd ../SemProject_MLEngine && python -m mlengine export-reference && python -m mlengine export-samples)
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm \
  -v "$PWD/../SemProject_MLEngine/data/processed/reference:/seed:ro" api python -m app.cli seed-reference /seed
echo "a-long-password-1" | docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm -T api \
  python -m app.cli create-user research@example.com --research --password-stdin
```

- **Base URL:** from the Android emulator, use `http://10.0.2.2:8000`. On a phone on the same Wi-Fi, use your machine's LAN IP and allow cleartext for debug builds only.
- **Email** (password reset codes, invitations): locally, messages only appear in `docker compose logs api`.
- **Research mode:** the user created above has `isResearch: true`. That unlocks research dates, model-version choice and `/v1/research/*`.
- **Fresh sample data:** re-run `export-samples` before a demo, so the sample history ends yesterday.
- **Starting over:** `docker compose ... down` keeps the data; adding `-v` wipes it.

## 4. The contract

- **[`SemProject_Backend/docs/api.md`](../SemProject_Backend/docs/api.md)** is the guide to every endpoint, field, error code and flow (auth refresh, polling, imports, training, team).
- **[`SemProject_Backend/docs/openapi.json`](../SemProject_Backend/docs/openapi.json)** is the machine-readable contract. Generate the DTOs and Retrofit service from it, or check hand-written DTOs against it. A backend test keeps it current.
- **Where api.md and spec §12.1 differ, api.md is right.** The differences:
  - IDs are UUIDs, and a product has a separate `sku`.
  - It adds `/v1/auth/register`, password reset and change, job retry, forecast delete, the history summary, eligibility, training runs, members and invitations, and the research summary.
  - `points[]` use `baseline` and `seasonalNaive`.
  - The training run is its own resource (`/v1/training-runs/{id}`).
- **Errors** always have the shape `{"error": {"code", "message", "details", "requestId"}}`. Map `code` to string resources, and fall back to `message`, which is already plain language.

## 5. Build order

1. **Foundation:** project setup, theme and tokens (§7), app shell and navigation (§5), the three menus (§6), and state components (loading, empty, error, offline).
2. **Auth:** register, login, refresh in an OkHttp `Authenticator`, logout, and forgot password. Tokens go in Keystore-backed storage (§14).
3. **Demo journey:** workspace picker → dashboard → products → forecast configuration → job progress → results (chart, table, three kinds of evidence) → history → model info. This is the core of the research demo.
4. **Business journey:** create workspace and store → CSV import wizard (7 steps, §8.6) → readiness → baseline forecasts → eligibility progress → train → AI forecasts.
5. **Team and settings:** members, invitations (owner) and "My invitations" (everyone), plus profile and change password.
6. **Research:** summary screen and CSV export (research users only).
7. **Offline and cache:** Room for products, imports and completed forecasts (§13).
8. **Tests and acceptance:** spec §19.

Build and test after each step. Commit as you go.

## 6. Screens → endpoints and fields

| Screen (spec) | Endpoint(s) | Fields to use |
| --- | --- | --- |
| Workspace picker (§8.3) | `GET /v1/me`, `GET /v1/workspaces`, `GET /v1/workspaces/{wid}/stores` | `role` (`REFERENCE` = demo, read-only); `isReference` → "Demo data" badge |
| Dashboard (§8.4) | `…/history/summary`, `…/readiness`, `…/models/capabilities`, `GET /v1/forecasts?storeId=` | `historyEnd` ("Sales recorded up to …"); `readiness.stale` → upload notice; AI status from `capabilities.aiAvailable` / `aiUnavailableReason` (`TRAINING` / `NOT_ELIGIBLE` / `GATE_FAILED` / `NOT_TRAINED`) |
| Products (§8.5) | `GET …/products?q=&category=&demandClass=`, `…/products/{pid}`, `…/products/{pid}/history` | `demandClass.label` chip; in history, `units: null` = **not recorded** (draw a gap, not 0) |
| Forecast configuration (§8.8) | `GET …/models/capabilities` | `methods`, `horizons`, `asOf.default / editable / replayMin / replayMax / researchMin`; `baselines[].label` |
| Job progress (§8.9) | `POST /v1/forecasts` → `GET /v1/forecast-jobs/{id}` | `status`; `error.message`; offer Retry only if `retryable` |
| Results (§8.10) | `GET /v1/forecasts/{id}` | `points[].ai / baseline / seasonalNaive / actual`; `history[]` (the recent days for the chart); `asOfDate`; `isReplay`; `model.version`; `horizon.validated` (`false` → "Experimental" badge); `priceAssumption` (`LAST_KNOWN` → "price assumed unchanged"); `product.demandClass.label`; `aiUnavailableReason` |
| Results: evidence 1 (this run) | same | `runMetrics` (replays only): "Error on these days (actual sales known)" |
| Results: evidence 2 (this product) | same | `productBacktest.ai / naive_last_value` (`mae`, `rmse`, `wape`), `n`, `window` (demo only; add the small-sample note) |
| Results: evidence 3 (the model) | `GET …/models/active` | `card.backtest`, `card.horizonAccuracy.byDay[]` (`aiRmse` vs `naiveRmse`, a small line chart), `card.limitations` |
| History (§8.11) | `GET /v1/forecasts?storeId=&productId=&replay=&status=` | `isReplay` → "Replay" tag; `actualsAvailable` → "Actuals now available" |
| Export | `GET /v1/forecasts/{id}/export?format=csv` | Save through SAF |
| Data screen + import wizard (§8.6) | `POST …/imports` (multipart, `Idempotency-Key`), `PUT /v1/imports/{iid}/mapping`, `GET …/validation`, `GET …/errors`, `POST …/confirm`, `DELETE`, `GET …/imports` | `detected.suggestedMapping` (highlight `confidence: "CHECK"`), `dateFormatAmbiguous`; the required **"no row means zero sales?"** question → `missingDaysAreZero`; `summary.*`; `overlappingDays` → SKIP/REPLACE choice; `result.*` |
| Model info (§8.13) | `GET …/models/active`, `GET …/models/eligibility`, `POST …/models/train`, `GET /v1/training-runs/{id}`, `GET …/models/training-runs` | Eligibility `checks.*.value / required` ("64 of 120 days"); run `outcome` (`TRAINED` / `GATE_FAILED` / `NOT_ELIGIBLE`), `backtest`, `horizonAccuracy` |
| Team (§8.12) | `…/members`, `…/invitations`, `GET /v1/me/invitations`, `POST /v1/invitations/{id}/accept`, `…/decline` | Owners manage; check `/v1/me/invitations` after sign-in and badge it |
| Auth extras (§8.2) | `POST /v1/auth/password-reset/request`, `…/confirm`, `POST /v1/auth/change-password` | 8-digit code; on `429 RATE_LIMITED` show `Retry-After` |
| Research (research users) | `GET /v1/research/summary`, `GET /v1/research/export` | `overall`, `byDemandClass`, `byDayAhead` for AI vs both baselines; always show the `note` |

**For orientation only (never hardcode these):** the demo store's next-day backtest is AI RMSE 2.377 against naive 3.219. 925 of its 1,437 products are "intermittent", so many will show "Sells on some days only". The sample shop's own model scores AI RMSE 2.50 against 3.06.

## 7. Suggested copy for common codes

Put these in string resources, and fall back to the API's `message`.

| Code | Message |
| --- | --- |
| `INSUFFICIENT_HISTORY` | "This product needs at least 21 days of sales history for an AI forecast." |
| `TOO_MANY_GAPS` | "Several recent days have no sales recorded, so only the simple forecast is shown." |
| `STALE_HISTORY` / `readiness.stale` | "Your latest sales are from {date}. Upload sales up to yesterday to forecast tomorrow." |
| `AI_NOT_AVAILABLE` | "AI forecasts aren't available for this store yet. 'Same as last day' forecasts are available now." |
| `GATE_FAILED` (training outcome) | "The AI model didn't beat the simple forecast on your recent sales, so we'll keep using the simple forecast. Try again after adding more sales." |
| `NOT_ELIGIBLE` | "AI forecasting becomes available once your store has enough history." (show the checks) |
| `AS_OF_OUT_OF_RANGE` | "Choose a date between {min} and {max}." |
| `NO_MODEL_FOR_AS_OF` | "No model is available for that date without using information from the future." |
| `HORIZON_NOT_SUPPORTED` | Hide the option; this should never reach the user |
| `RATE_LIMITED` | "Too many attempts. Try again in {retryAfterSeconds} seconds." |
| `ML_SERVICE_UNAVAILABLE` | "Forecasting is temporarily unavailable. Try again shortly." |

## 8. Demo scripts (use these to test your screens)

1. **Demo replay:** demo store → product `FOODS_3_090` → as-of 2016-05-08, next day, Compare. Expect AI ≈ 57.1, same-as-last-day 77, actual 72, replay error panels and the product backtest.
2. **Demo true forecast:** the same product, as-of 2016-05-22 (the default), 7 days. Expect seven AI and baseline points and no actuals.
3. **Business with enough data:** create a workspace and a store (US / TX / USD / America/Chicago) and import `sample_shop_eligible.csv`.
   - Answer **"Yes"** to "no row means zero sales?".
   - The import shows Needs Attention because of real spikes (expected), then confirm.
   - Eligibility should pass; tap Train. In about 25 s the outcome is `TRAINED` with horizons `[1, 7]`, and the app now offers AI forecasts.
4. **Business not ready yet:** import `sample_shop_small.csv` (answer "No"). Expect "60 of 120 days", Train refused with `NOT_ELIGIBLE`, and baseline forecasts working.
5. **Messy import:** upload `sample_shop_messy.csv`. Expect a semicolon file with names only and an ambiguous date format, so choose `DD/MM/YYYY`. The report shows 5 rejected rows with line numbers, plus warnings.
6. **Team:** invite a second email as Viewer, register that user, accept from "My invitations". Expect read-only access: no Generate, no Upload.
7. **Password reset:** request a code, read it from `docker compose logs api`, reset.

`data/processed/samples/README.md` lists the wizard answers for each file.

## 9. Pitfalls

- **Not recorded is not zero.** `units: null` in history means unrecorded. Draw a gap, and never show it as 0.
- **Decimals are intentional.** The model predicts *expected* demand: show "≈ 2.4 units" (one decimal by default, with a popup option for whole units; spec §8.10). Compute totals from the raw values.
- **Replay vs true forecast.** `isReplay: true` has actuals and run errors; a true forecast has neither. Label them differently (spec §8.8).
- **Business stores can't pick an as-of date** (`asOf.editable: false`). Only the demo store and research users can.
- **Keep metrics paired.** AI and baseline metrics on one screen must come from the same block (`runMetrics`, `productBacktest` or the card), never mixed. Put "Lower error means it did better on that sample, not that it will do better tomorrow." under every metrics block.
- **7-day forecasts run about 4% high** on days 2–7. This is listed in the card's `limitations`; show it, don't hide it.
- **Refresh tokens rotate, and reuse signs the user out everywhere.** Serialise refreshes (one `Authenticator`, with a mutex), and replace both tokens every time.
- **Polling:** forecasts every 1 s for 10 s, then every 3 s, up to 2 min (§8.9); training every 2–3 s. The user can leave; jobs continue on the server.
- **Imports are synchronous.** Validation and confirm return when they're done (a few seconds for 10 MB), so show progress during the request. Send one `Idempotency-Key` per user action, and reuse it on retry.
- **The demo workspace is read-only.** Hide Upload, Train, product editing and Team there, and show "Try replay" instead of Upload (spec §8.4).

## 10. Files

| Path | Why |
| --- | --- |
| `retail_demand_forecasting_mobile_design.md` | The full app spec: screens §8, menus §6, design system §7, offline §13, security §14, accessibility §16, acceptance §19 |
| `../SemProject_Backend/docs/api.md` | The API guide (the source of truth for fields and flows) |
| `../SemProject_Backend/docs/openapi.json` | Generate the client from this |
| `data/processed/samples/` (after `export-samples`) | Business CSVs for demos and UI tests |
| `docs/model-card-m5-ca1-foods.md` | What the demo model is, and its limits, in plain language |
| `docs/notebook-to-production.md` | Background: how the model was built and validated |

## 11. Do not

- Do not call `/internal/v1/*` or the ML service, and do not put any server secret in the APK.
- Do not change `SemProject_Backend`, `mlengine/` or `models/` to make the app work. If the contract seems wrong or something is missing, raise it with the project owner. The backend has tests and a committed OpenAPI file that must stay in step.
- Do not create XML screen layouts. Screens are Compose inside Fragments (spec §4).
- Do not add horizons, intervals, accuracy claims, "AI insights" or explanations that the API doesn't return.
- Do not add a `Co-Authored-By` trailer or a "Generated with Claude Code" line to commits or PRs. The project owner asked for none.
