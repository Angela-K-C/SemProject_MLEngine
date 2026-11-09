# Retail Demand Forecasting Mobile — Design Specification

**Document version:** 2.1 (model side implemented)
**Date:** 7 November 2026
**Platform:** Android
**Working project name:** Retail Demand Forecasting Mobile
**Primary implementation:** Kotlin + Jetpack Compose (Material 3)
**Navigation constraint:** AndroidX Fragments
**ML engine:** This repository (`SemProject_MLEngine`), package `mlengine/`: Tweedie LightGBM, Stage 3 of the modelling journey in `README.md`. Active model `m5-ca1-foods@1.1.0`.
**Handoff for the Android app build:** `MOBILE_APP_HANDOFF.md`
**Audience:** Project owner, Android developer, backend/ML developers, and AI coding assistant (Claude)

---

## 0. What changed from version 1.0

Version 1.0 described a generic forecasting app. It did not match the model this project actually built. Version 2.0 aligns the app with that model. The main corrections are:

| Area | v1.0 assumption | What the model actually does | v2.0 decision |
|---|---|---|---|
| Forecast horizon | 7, 14 or 30 days | Predicts **one day ahead**. It needs yesterday's actual sales (`sales_lag_1`). All reported metrics are next-day errors. | Next-day is the validated horizon. A 7-day horizon is produced by recursive forecasting, and it stays switched off until a new backtest validates it (§10.5). 14/28-day horizons are deferred. |
| Whose data | Any business uploads any CSV and gets an AI forecast | Trained on **Walmart M5 data, store CA_1, FOODS (1,437 items)**. It uses M5 item/store IDs, M5 day numbers, US holidays and California SNAP days as features. It cannot score a product it has never seen in a meaningful way. | Two workspace modes (§9.2). The **Reference workspace** runs the trained model on M5 CA_1 FOODS data. This is the MVP, and the research results are valid there. **Own-data workspaces** get the same recipe retrained on their own history with a portable feature set, and AI is enabled only after that workspace passes its own backtest. |
| Traditional method | Unspecified | The notebooks compare against a **naive lag-1 baseline**: tomorrow equals today. | The primary baseline is naive persistence. Seasonal naive (same weekday last week) is a secondary baseline (§10.6). |
| Dataset selection | The user picks one dataset to forecast from | Features come from a continuous daily history per product. | An upload is an **import batch** that merges into the store's sales history. A forecast runs against a dated snapshot of that history, not against one file (§8.6, §8.8). |
| Prediction intervals | Shown if supplied | The model produces point predictions only. | No intervals in v1. The field is `null`. Quantile models are a later enhancement. |
| Volatility | CV, thresholds undefined | The EDA uses CV, zero-day share and average demand interval (ADI). | ADI/CV² demand classification (smooth / erratic / intermittent / lumpy) with documented thresholds (§10.8). |
| Metrics | MAE, RMSE | MAE and RMSE, plus WAPE in the README for scale comparison | MAE, RMSE and WAPE. Model-level and product-level backtests are labelled separately (§8.10). |
| Notebooks → production | Not covered | The code lives only in notebooks. `scripts/preprocess_m5.py` builds the old 28-day features. Nothing saves a model. | A new `mlengine/` Python package, a model artifact format, parity tests and an internal ML service (§11, §12). |
| Output values | Whole units | Tweedie output is a continuous non-negative number (for example 1.37 units). | Store raw values. Display to one decimal place. Sum raw values for totals (§8.10). |

Everything outside these areas (app shell, menus, design system, security, accessibility) carries over from v1.0 with small edits.

### 0.1 Implementation status (v2.1)

The whole model side of this document is built, tested and committed on the `model-implementation` branch:

| Item | Status | Where |
| --- | --- | --- |
| `mlengine` package (features, training, inference, baselines, metrics, demand classes) | Done | `mlengine/` |
| Notebook parity: features identical over all 2,759,040 rows; 70 trees; dev RMSE 2.356; backtest RMSE 2.377 | Verified | `scripts/make_parity_fixture.py`, `tests/` |
| Reference model `m5-ca1-foods` versions `1.1.0-dev`, `1.1.0-bt`, `1.1.0` | Released | `models/m5-ca1-foods/` |
| 7-day recursive horizon | **Validated and enabled** (§10.5) | `notebooks/RecursiveHorizon_CA1Foods.ipynb` |
| `portable-v1` schema and own-data training with backtest gate | Done; thresholds set from simulation (§9.4.2) | `notebooks/PortableFeatures_CA1Foods.ipynb` |
| Error by demand class | Done (§10.8) | `notebooks/ErrorByDemandClass_CA1Foods.ipynb` |
| Internal ML service (FastAPI) | Done; reproduces the backtest end to end over HTTP | `mlengine/service/`, `Dockerfile` |
| Reference workspace seed export | Done | `python -m mlengine export-reference` |
| Sample business CSVs (real M5 store TX_1, dates shifted) | Done | `python -m mlengine export-samples` |
| Backend API (§12.1) | **Done**: FastAPI + PostgreSQL, own JWT sign-in, Docker Compose + Caddy | `../SemProject_Backend` (`docs/api.md` is the API contract) |
| **Android app** | **Not started** | Next phase. See `MOBILE_APP_HANDOFF.md` |

Where this document and the code disagree, the code and `MOBILE_APP_HANDOFF.md` win; for the public API, `SemProject_Backend/docs/api.md` wins over §12.1. The sections below have been updated to match what was built.

---

## 1. Executive summary

The Retail Demand Forecasting mobile application is an Android app that helps small and medium-sized retail businesses plan inventory and replenishment using demand forecasts generated from their historical daily sales.

A business user will be able to:

1. Sign in and open a workspace: the built-in **Reference workspace** (M5 store CA_1), or their own business workspace.
2. Maintain a product catalogue, or import products along with sales data.
3. Upload historical daily sales data in CSV format, map the columns, review validation results, and merge the data into the store's sales history.
4. Select a product and request a **next-day** or **7-day** demand forecast. The forecast can come from the AI model, the traditional baseline, or both side by side.
5. In the Reference workspace, "replay" a past date and compare the forecast with what actually sold.
6. Review forecast results through a chart, summary cards, a date-by-date table, and clearly labelled backtest metrics.
7. Review previous forecast runs and export results.

The mobile app is the presentation and workflow layer. **It does not train or host the machine-learning model.** It talks only to the backend API. The backend stores sales history and forecast runs, enforces permissions, and calls the private **ML service**. The ML service is built from this repository: it computes features, runs the LightGBM model and the baselines, and returns predictions together with model metadata.

The research question guiding the project is:

> To what extent can AI-based forecasting improve demand prediction for retail businesses under volatile demand conditions, compared with traditional forecasting methods?

The app must therefore support a fair comparison. It must not claim that AI is automatically more accurate. The AI model and the baseline always run on the same series, the same as-of date, the same horizon and the same preprocessing. MAE, RMSE and WAPE are shown only when they were calculated against known actual sales.

### 1.1 The model in one paragraph

The production model is the Stage 3 model from the modelling journey: a single LightGBM regressor with a Tweedie objective (`tweedie_variance_power=1.1`, learning rate 0.05, 31 leaves, early-stopped). It was trained on the 1,437 FOODS items of Walmart store CA_1 from the M5 dataset. Each prediction is the expected unit sales of one item on one day. It uses 31 features: identity categoricals, sales lags at 1/7/14/21 days, rolling mean and standard deviation over 3/7/14 days, calendar and event fields, the SNAP flag, and price fields. On the later backtest (2016-04-25 to 2016-05-22) it scores RMSE 2.377 and MAE 1.377, against RMSE 3.194 for the naive "tomorrow equals today" baseline on the development window. These are **one-step-ahead** errors: every predicted day had the real previous day's sales available.

---

## 2. Product vision and objectives

### 2.1 Product vision

Make demand forecasting understandable and practical for a retail business owner without a data-science background. The user should be able to go from a sales-history file to an interpretable forecast without understanding the underlying algorithm, and should always be able to see how trustworthy that forecast is.

### 2.2 Product objectives

- Reduce the effort required to prepare and submit daily sales data.
- Make data quality problems visible before a forecast is generated.
- Present demand in the product's own unit of measure (units, bottles, packets, kilograms).
- Show the forecast's assumptions, data coverage, model version and as-of date.
- Support the research comparison between the AI model and traditional baselines.
- Keep a history of imports and forecast runs, with enough metadata to reproduce each one.
- Provide a mobile-first experience on common Android phones.

### 2.3 What the application is not

- It is not an accounting, point-of-sale, payment or inventory-management system.
- It does not know current stock unless stock data is supplied.
- It does not train or run the ML model on the phone.
- It does not present a forecast as a guarantee of future sales.
- It does not invent confidence intervals, accuracy scores or explanations that the backend has not supplied.
- It does not apply the M5-trained model to products outside the M5 catalogue (§9.2).

---

## 3. Target users and permissions

The app uses a business workspace. A workspace contains one or more stores. A user can belong to several workspaces and may have access to some or all stores in each, depending on backend permissions.

### 3.1 User roles

**Owner / Administrator**
Manages the workspace and stores, invites and removes users, manages products, uploads and confirms imports, requests own-data model training (§9.4), generates, views, exports and deletes forecasts, and manages settings.

**Manager / Analyst**
Views workspace data, manages products when permitted, uploads and validates data, generates and compares forecasts, uses replay mode, views history and exports permitted reports. Cannot transfer ownership or remove the owner.

**Staff / Viewer**
Views permitted products, dashboards and forecasts. Cannot delete data, manage users or change workspace configuration.

**Research user (flag, not a role)**
A per-user flag the backend grants to project members. It unlocks model-version selection, replay over any as-of date, and the research export (§10.9). It is never granted to business users by default.

The backend is the authority for permissions. Hiding a button is not security; every protected API operation enforces the role and workspace/store access.

### 3.2 The Reference workspace

Every signed-in user can open a read-only **Reference workspace** called "Demo Store (M5 CA_1)". It contains:

- Store `CA_1` (California, timezone `America/Los_Angeles`, currency `USD`).
- The 1,437 FOODS items, with M5 item IDs as SKUs and departments (`FOODS_1`, `FOODS_2`, `FOODS_3`) as categories.
- Daily sales from 2011-01-29 to 2016-05-22 (`d_1`–`d_1941`), and prices and calendar through 2016-06-19 (`d_1969`).

This workspace is where the trained model is valid. It is the default first-run experience and the place where the research comparison is demonstrated. Users cannot upload, edit or delete anything in it, but they can generate forecasts and replay dates.

### 3.3 Main user journeys

**First-time user (demo path):** Welcome → Sign in → "Explore the demo store" → Dashboard (Reference workspace) → Pick a product → Generate next-day forecast (AI + baseline) → View results → Try replay of a past date to see forecast vs. actual.

**First-time owner (own business):** Sign in → Create workspace → Add first store → Upload sales CSV → Map columns → Review validation → Confirm import → See data readiness → Baseline forecasts available immediately → Request AI model training when the store is eligible → AI forecasts unlocked once the workspace backtest passes.

**Returning manager:** Sign in → Dashboard → Upload yesterday's sales (incremental import) → Generate next-day forecasts → Compare methods → Export.

**Viewer:** Sign in → Dashboard → Open a forecast → Review chart and table → Share/export if permitted.

---

## 4. Platform and technology decisions

### 4.1 Android technology stack

- **Language:** Kotlin.
- **UI toolkit:** Jetpack Compose.
- **Design system:** Material 3 with a custom application theme.
- **Navigation:** AndroidX Fragments and AndroidX Navigation Component.
- **Screen implementation:** Each destination Fragment hosts a `ComposeView`; all screen UI is written in Compose.
- **Architecture:** MVVM with unidirectional UI state flow, repository pattern, and domain use cases where useful.
- **Asynchronous work:** Kotlin Coroutines and Flow.
- **Lifecycle-aware state:** `collectAsStateWithLifecycle()` for Flow collected by Compose UI.
- **Networking:** Retrofit with OkHttp.
- **JSON:** Kotlin Serialization (chosen; use it everywhere).
- **Local cache:** Room for cached products, import summaries and completed forecast results.
- **Preferences:** DataStore for non-secret preferences.
- **Background upload/retry:** WorkManager for confirmed uploads that need retry.
- **File selection:** Storage Access Framework via the Activity Result API.
- **Charts:** A maintained Compose-compatible chart library, or custom Canvas charts for the small, fixed chart set (history + forecast line, comparison overlay, forecast-vs-actual).
- **Splash:** AndroidX Core SplashScreen compatibility API.
- **Testing:** JUnit, coroutine/ViewModel tests, Compose UI tests, and navigation/integration tests.

**Implementation constraint:** Screens are built with Jetpack Compose. Do not create XML layout files for screen UI. Android XML is still allowed for the manifest, themes, splash attributes, drawables and an optional navigation graph.

### 4.2 Fragments and Compose together

The project keeps Fragments because they are a project requirement. `ComposeView` is the interoperability mechanism for placing Compose content inside a Fragment, with `DisposeOnViewTreeLifecycleDestroyed` as the composition strategy.

Implementation pattern:

- `MainActivity` owns the app-level Fragment host.
- Each major navigation destination is a Fragment.
- The Fragment returns a `ComposeView` from `onCreateView()`, sets `ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed`, and calls `setContent { ... }` with a screen composable.
- ViewModels expose immutable UI state and event methods; composables render state and send events back.
- No business logic, API calls, parsing or forecasting logic inside composables.

### 4.3 App architecture

**Presentation layer:** Fragments, screen composables, reusable components, ViewModels, UI state and UI events.

**Domain layer:** Entities and use cases such as `ImportSalesData`, `GetDataReadiness`, `GenerateForecast`, `ReplayForecast`, `GetForecastHistory` and `GetModelCapabilities`.

**Data layer:** Repositories, Retrofit services, Room cache, DataStore, file handling, and DTO-to-domain mapping.

The UI never calls Retrofit directly. A screen sends an event to its ViewModel; the ViewModel calls a use case; the use case calls a repository.

**The app contains no forecasting maths.** It does not compute lags, baselines, metrics or volatility classes. Every number on a forecast screen comes from the backend. The only arithmetic the app performs is display formatting and summing daily values for a weekly chart view.

### 4.4 Suggested project structure

```text
app/src/main/java/<package-name>/
├── MainActivity.kt
├── app/                 # Application class, DI/container, navigation
├── core/
│   ├── network/         # HTTP client, interceptors, serializers, error mapping
│   ├── database/        # Room database, DAOs, entities
│   ├── datastore/       # Non-secret preferences
│   ├── files/           # URI handling, CSV preview, upload preparation
│   ├── security/        # Secure token storage
│   └── ui/              # Theme, reusable composables
├── data/                # Remote/local data sources, DTO mappers, repositories
├── domain/              # Domain models, repository contracts, use cases
└── feature/
    ├── auth/ onboarding/ dashboard/ products/
    ├── data/             # Upload, mapping, validation, import history, data readiness
    ├── forecasts/        # Configuration, job progress, results, replay
    ├── models/           # Model info and own-data training status
    └── history/ team/ settings/
```

Within each feature, keep separate files for the Fragment, screen composable, feature components, ViewModel, UI state, UI events and navigation arguments.

---

## 5. Navigation and application shell

### 5.1 Main navigation

After sign-in, the app shows a bottom navigation bar with four destinations:

1. **Dashboard**: overview and shortcuts.
2. **Products**: catalogue and product-level demand information.
3. **Data**: sales history, imports, validation and data readiness.
4. **Forecasts**: forecast creation and saved results.

The top app bar shows the screen title, a **workspace/store chip** that always tells the user which workspace they are in (the Reference workspace is visibly marked "Demo data"), a contextual action, and the Options menu.

History is reached from Forecasts and from a dashboard shortcut. Settings, team management and model information are reached from the Options menu.

### 5.2 Fragment destination list

**Authentication:** `LoginFragment`, `RegisterFragment` (only if self-registration is enabled). Bottom navigation hidden.

**Workspace:** `WorkspacePickerFragment` lists the Reference workspace and the user's own workspaces and lets the user create one. Bottom navigation hidden.

**Primary destinations (bottom navigation visible):**
- `DashboardFragment`
- `ProductsFragment`
- `DataManagementFragment`
- `ForecastsFragment`

**Supporting destinations (bottom navigation hidden):**
- `ProductDetailFragment`: product details, history chart, demand class, latest forecast.
- `ImportWizardFragment`: file selection, mapping, preview, validation, confirmation.
- `ImportDetailFragment`: import status and validation report.
- `DataReadinessFragment`: per-store history coverage and AI eligibility (§8.7).
- `ForecastConfigurationFragment`: configure a forecast or a replay.
- `ForecastProgressFragment`: queued/running/failed job state.
- `ForecastDetailFragment`: result, comparison, table, metrics, export.
- `ForecastHistoryFragment`: search and filter previous runs.
- `ModelInfoFragment`: the active model's card (§8.13) and own-data training status.
- `TeamManagementFragment`: owners/administrators only.
- `SettingsFragment`: workspace, preferences, help, sign out.

The navigation graph must preserve back-stack behaviour, must not duplicate a destination when a bottom item is reselected, and must survive process recreation.

---

## 6. The three required menu types

The app visibly implements all three menu patterns.

### 6.1 Options menu (screen/application-level actions)

**Location:** Overflow icon in the top app bar.

**Compose implementation:** An `IconButton` toggles a `DropdownMenu` of `DropdownMenuItem`s. Contents can depend on the screen.

**Entries:**
- Switch workspace/store
- About the forecasting model (opens `ModelInfoFragment`)
- Manage team (authorised roles only)
- Settings
- Help and support
- About this application
- Sign out

Primary actions such as Generate Forecast stay visible on their screens, never only in this menu.

### 6.2 Context menu (actions on a specific item)

**Trigger:** Long-press on a product card, an import row, or a saved forecast.

**Compose implementation:** `combinedClickable` with `onLongClick` shows an anchored menu for that exact item.

**Actions:**
- Product: View details, Forecast next day, Edit, Archive (if permitted; not available in the Reference workspace).
- Import batch: View import report, Rename label, Delete import (if permitted and confirmed; removes that batch's rows from history and marks dependent forecasts as "based on deleted data").
- Forecast: Open result, Export, Re-run with same settings, Delete (if permitted and confirmed).

Destructive actions need a confirmation dialog and backend authorisation.

### 6.3 Popup menu (quick anchored choices)

**Trigger:** Tap a filter, sort, chart or table control.

**Compose implementation:** An anchored `DropdownMenu` triggered by a tap. It chooses a temporary view option; it is not an action menu for an object.

**Examples:**
- Sort products by name, SKU, category or demand class.
- Filter products by demand class (smooth / erratic / intermittent / lumpy).
- Filter imports by status or date.
- Chart range: last 28 days, last 90 days, all history.
- Chart series: AI, baseline, or both.
- Table display: raw decimals or rounded.

Use a bottom sheet for complex multi-filter forms.

---

## 7. Visual design system

### 7.1 Design direction

Trustworthy, calm, analytical and approachable: a practical business tool, not a research dashboard full of jargon. Technical terms (RMSE, Tweedie, lag) appear only on the model info screen and in expandable "How is this calculated?" sections.

### 7.2 Proposed visual tokens

Implement as theme tokens so the palette can change centrally.

- **Primary:** Deep navy, approximately `#17324D`.
- **Secondary:** Teal, approximately `#168C86`.
- **Background:** Very light neutral, approximately `#F6F8FA`.
- **Surface:** White in light theme.
- **Main text:** Dark slate, approximately `#17212B`.
- **Secondary text:** Muted slate, approximately `#64748B`.
- **Success / Warning / Error:** Accessible green, amber, red.
- **Demo marker:** A distinct neutral badge colour for "Demo data" so it is never confused with a status colour.
- **Shape:** Rounded cards and controls, moderate radius.

Support light and dark themes. Charts use labels, legends and different line styles or markers (for example solid for actual history, dashed for AI forecast, dotted for baseline), so meaning never depends on colour alone.

### 7.3 Typography and layout

- Clear hierarchy: screen title, section heading, card title, body, supporting label.
- Narrow phones first, scrollable content.
- Spacing tokens 4, 8, 12, 16, 24, 32 dp.
- Empty, loading, success, partial-success and error states on every data screen.

### 7.4 Reusable Compose components

- `AppScaffold`, `AppTopBar`, `WorkspaceChip`, `AppBottomNavigationBar`
- `PrimaryButton`, `SecondaryButton`, `DestructiveButton`
- `MetricCard`, `StatusChip`, `DemandClassChip`, `DemoDataBadge`
- `ProductListItem`, `ImportListItem`, `ForecastListItem`
- `EmptyState`, `LoadingState`, `ErrorState`
- `SearchField`, `FilterButton`
- `FileUploadCard`, `ValidationSummaryCard`, `ValidationIssueItem`
- `DataReadinessCard`
- `ForecastLineChart`, `ForecastDataTable`, `MetricsComparisonTable`
- `AsOfDateBanner` (shows "Forecast made as of <date>" and whether actuals are known)
- `ConfirmationDialog`

Keep components stateless where practical.

---

## 8. Screen-by-screen specification

### 8.1 Splash screen

Application mark and name on a theme background, using the AndroidX SplashScreen API. Keep it only while checking the local session. Route to Login, Workspace Picker or Dashboard. No fake progress percentage.

### 8.2 Authentication

**Login:** email and password, show/hide password, Sign In button, loading state, inline validation, useful errors, password reset if the backend supports it.

**Registration:** only if the backend allows self-registration; otherwise an invitation-acceptance path.

**Session:** No hardcoded or plain-text tokens. Store them in secure storage backed by Android Keystore. On refresh failure, return to login with a clear message.

### 8.3 Workspace picker and store setup

The picker lists the **Reference workspace** first (with the Demo data badge and a one-line explanation: "Real historical sales from a US grocery store, used to demonstrate and evaluate the forecasting model"), then the user's own workspaces, then "Create workspace".

Creating a workspace collects:
- Business/workspace name.
- Store/branch name.
- Country (used to choose the public-holiday calendar for own-data models; §9.4).
- Timezone (defines what "a day" of sales means).
- Currency code for price fields.
- Optional industry/category descriptor.

`Africa/Nairobi`, `KE` and `KES` are offered as defaults for Kenyan deployments and can be changed. Demand is always shown in the product's unit of measure, never in currency.

### 8.4 Dashboard

**Layout, top to bottom:**
1. Greeting and workspace/store chip.
2. Summary cards: active products; latest sales date in history ("Sales recorded up to 22 May 2016"); most recent forecast run; AI availability for this store (Available / Training / Not yet eligible / Baselines only).
3. Quick actions: **Generate Forecast** and **Upload Sales Data** (Upload is hidden in the Reference workspace; **Try replay** is shown instead).
4. Recent forecasts: product, as-of date, horizon, method, status.
5. Setup notice when there is no history, or when history is stale. Next-day forecasts need yesterday's sales, so the notice reads: "Your latest sales are from 3 Nov. Upload sales up to yesterday for an accurate next-day forecast."

No fabricated totals, accuracy or stockout predictions. With no data, show an onboarding empty state.

### 8.5 Products

**List:** search; sort/filter popup; product name; SKU; category; unit of measure; demand class chip (when computed); latest forecast indicator.

**Add/edit:** name (required), SKU (required and unique per workspace), category (optional but recommended, because own-data models use it as a feature), unit of measure (required), active status, optional default price.

**Detail:** identity, category, unit, sales-history coverage (first and last sale date, days observed), a 90-day history chart, demand class with a plain explanation, latest forecast summary, and a Generate Forecast action.

In the Reference workspace, products are read-only.

### 8.6 Data management

**How data is organised.** Each store has one **sales history**: one canonical record per product, store and day. Every CSV upload is an **import batch**. When confirmed, its rows merge into that history. Forecasts never point at an import; they point at the history as it stood at a given moment (the `historyVersion`, §11.3). This is what allows daily incremental uploads to feed the next-day model.

**Data screen:**
- Upload Sales Data primary button (hidden in the Reference workspace).
- Store history summary card: date range, products, days with data, last updated.
- Data Readiness link (§8.7).
- Import list: display name, upload date, rows, date coverage, status.
- Statuses: Uploaded, Validating, Needs Attention, Ready to Import, Importing, Imported, Failed.
- Filter by status/date via popup menu.

**Import wizard:**

1. **Choose file.** Pick a UTF-8 CSV through the Storage Access Framework. XLSX comes later.
2. **File details.** Filename, size, first rows preview. Nothing uploads without user action.
3. **Column mapping.** Map detected columns to canonical fields (§8.7.1). Suggest matches; require confirmation of ambiguous ones. The user also answers one required question: **"If a product has no row for a day, does that mean it sold zero that day?"** (Yes / No, those days were not recorded). This sets how gaps are treated (§8.7.3).
4. **Preview and validation.** Normalised sample rows and the backend's validation summary.
5. **Overlap handling.** If the batch contains days already in history for the same product, the user chooses: replace existing days with the new file, or skip overlapping rows. The default is "skip", and the counts are shown.
6. **Confirm import.** Accepted/rejected counts, date range, products added, and what happens next. Explicit confirmation is required.
7. **Import result.** Final status, counts, warnings, an updated readiness summary, and a route to forecasting.

### 8.7 Data format, validation and readiness

#### 8.7.1 Canonical fields

The model works on **daily unit sales per product per store**. Every upload is converted to this shape.

| Canonical field | Required? | Meaning | Used by the model as |
|---|---|---|---|
| `sales_date` | Yes | Business day of the sale (store timezone) | Calendar features; position in the daily series |
| `product_id` | Yes (or `product_name`, from which the backend creates a SKU) | Product identifier | Series key; categorical feature in own-data models |
| `units_sold` | Yes | Units sold that day | The target; source of all lag and rolling features |
| `store_id` | Required for multi-store files; otherwise the selected store | Store identifier | Series key |
| `unit_price` | Strongly recommended | Selling price that day | `sell_price`, `price_missing`, and the history behind `weeks_since_first_price` |
| `category` | Recommended | Product category / department | Categorical feature in own-data models |
| `unit_of_measure` | Optional | Display unit | Display only |
| `store_closed` | Optional | `true` if the store did not trade that day | Marks the day as not observed rather than zero |

Other columns (`revenue`, `promotion_flag`, `inventory_on_hand`, `returns_quantity`) may be **stored** but are **not used by the current model**. The mapping screen labels them "Saved, not used for forecasting yet". A column is only used as a feature after it has been added to a versioned feature schema and tested in a notebook.

If `units_sold` is missing but `revenue` and `unit_price` are present, the backend does **not** derive units. The row is rejected with an explanation.

Transaction-level files (several rows per product per day) are allowed. The backend sums `units_sold` per product, store and day and reports how many rows were aggregated.

**Example CSV:**

```csv
sales_date,product_id,product_name,category,units_sold,unit_price,store_id
2026-10-01,WATER-500ML,500 ml bottled water,Beverages,18,50,NAIROBI-01
2026-10-02,WATER-500ML,500 ml bottled water,Beverages,22,50,NAIROBI-01
2026-10-03,WATER-500ML,500 ml bottled water,Beverages,35,45,NAIROBI-01
```

#### 8.7.2 Validation rules (backend)

- Required columns are mapped.
- Dates parse; invalid dates are reported with row numbers.
- `units_sold` is numeric and ≥ 0. Negative values are rejected (returns are not part of the demand target).
- Missing product identifiers are flagged.
- Duplicate product/store/date rows are summed and reported (transaction files), never silently dropped.
- Future dates (after today in the store timezone) are rejected.
- `unit_price` ≤ 0 is treated as missing and flagged.
- Spikes greater than 10× the product's 28-day median (when that median > 0) are flagged for review, **not** removed or capped. The model was trained on uncapped data.
- Report accepted/rejected/attention counts, date range, products and stores found.
- Limits: 10 MB per file on the client, and the backend's documented limit is authoritative.
- Messages are plain language, with row and column where possible.

#### 8.7.3 Zero sales versus missing days

This matters more for this model than for most, because the lag and rolling features read the previous 21 days directly.

- A **recorded zero** is a real observation and stays 0.
- A **missing day** is treated according to the user's answer in the import wizard. If they answered "no row means zero sales", missing days become 0. Otherwise they stay missing.
- `store_closed = true` days are always missing, never zero.
- In the history the backend keeps an `is_observed` flag per day. Missing days produce `NaN` lag/rolling values, which LightGBM handles natively. The model was trained on M5's complete history, though, so heavy gaps are outside what it has seen. A product whose last 21 days contain more than 3 missing days is marked "Too many gaps for AI forecast", and only baselines are offered for it.

#### 8.7.4 Data readiness

The backend computes readiness per store and per product. Thresholds come from model metadata (`GET /v1/models/capabilities`) and are never hardcoded in the app.

| Check | Rule for the current model | What the user sees if it fails |
|---|---|---|
| Minimum history per product | ≥ 21 days ending on the as-of date (the longest lag is 21 days) | "This product needs at least 21 days of sales history." |
| Freshness | Latest observed day = yesterday (next-day horizon) | "Upload sales up to yesterday. The forecast would otherwise start from old data." |
| Gaps | ≤ 3 missing days in the last 21 | "Too many unrecorded days recently." |
| Own-data AI eligibility (per store) | See §9.4.2 | "AI forecasting becomes available once your store has enough history." |

In the Reference workspace every product passes these checks for any as-of date from 2016-03-27 onward, the earliest date with a leakage-safe model (§9.3).

### 8.8 Forecast configuration

**Fields:**
- **Store** (preselected).
- **Product.** One product per run in the MVP. The backend can score many products in one call; the UI limit keeps results readable.
- **As-of date.** The last day of sales history the forecast may use. Default: the latest day in history. The forecast starts the day after.
  - Business workspaces: read-only, always the latest history day.
  - Reference workspace: editable. Any date from 2016-04-24 (`d_1913`) to 2016-05-21 (`d_1940`) is a **replay** (actuals are known, so real errors can be shown). The default 2016-05-22 is a **true forecast** of 2016-05-23 (`d_1942`) onward, with no actuals. Research users can choose any date from 2016-03-27 (`d_1885`), served by the `-dev` model.
- **Horizon.** Options come from `capabilities.horizons`. For `m5-ca1-foods@1.1.0` that is **Next day (1)** and **Next 7 days (7)**. The 7-day option was enabled after the recursive backtest passed (§10.5). Own-data models get 7 only if their own recursive backtest passes. Unsupported horizons are not shown, not just disabled.
- **Method.** AI model, Traditional baseline, or Compare (default Compare, because that is what the research needs).
- **Model version.** Research users only. Everyone else gets the active version.

**Validation:** Generate stays disabled until the selection is valid. Readiness failures (§8.7.4) are shown inline, with the next step to take.

### 8.9 Forecast processing and job status

The model itself is fast. Scoring one product for 7 days takes milliseconds. A job can still take seconds when it has to load history, build features and compute a backtest, and own-data model training takes minutes. So forecast requests are asynchronous: `POST /v1/forecasts` returns `202` with a job ID.

The progress screen shows product, store, as-of date, horizon and method, and a status: Queued → Preparing Data → Running Model → Finalizing → Completed / Failed. It uses an indeterminate indicator unless the backend reports real progress. The user can leave the screen; the job stays visible in Forecasts and History. Retry is offered for retryable failures. Cancel is offered only if the backend supports it.

Polling: every 1 s for the first 10 s, then every 3 s, stopping after 2 minutes with a "Still working. We'll keep it in your history" message.

### 8.10 Forecast results

**Header:** product, SKU, store, `AsOfDateBanner` ("Based on sales up to 22 May 2016"), forecast dates, method, model name and version, and a Demo data badge in the Reference workspace.

**Summary cards:**
- **Tomorrow's expected demand**, for example "≈ 2.4 units".
- For 7-day horizons: total and daily average across the horizon.
- **Demand class** of the product (§10.8), with a plain explanation, such as "Intermittent: sells on fewer than 3 days in 4. Forecasts for products like this are less precise."

**Chart:** The last 28 days of actual sales (solid line), then forecast points (AI dashed, baseline dotted), with a vertical "as of" marker. In replay mode, actual sales for the forecast days are drawn as well, so the user sees forecast against reality. Tapping a point shows date, value and series. A text summary is provided for screen readers.

**Date-by-date table:** date, AI forecast, baseline forecast, and in replay mode the actual and the error. Values are shown to one decimal place by default; the popup menu can switch to rounded whole units. Totals are always computed from the raw values. Prediction-interval columns do not appear in v1.

**Why decimals:** the model predicts *expected* demand. "2.4 units" means "about 2 to 3, closer to 2", and the UI says so in a one-line hint. The app does not round to whole units by default, because that would hide the difference between 0.4 and 1.4.

**Accuracy section (three clearly separated kinds of evidence):**

1. **This run, against actuals.** Replay mode only. MAE and absolute error per day for AI and baseline, labelled "Error on these days (actual sales known)".
2. **This product, historical backtest.** Labelled "How the model did on this product over 25 Apr – 22 May 2016 (28 days)". MAE, RMSE and WAPE for AI and baseline, computed by the backend from a stored backtest. A note says that 28 days is a small sample.
3. **The model overall.** A link to the model card (§8.13) with the store-level backtest (RMSE 2.377 vs. baseline).

Rules:
- Metrics only appear when they were computed against actual sales.
- AI and baseline metrics on one screen always come from the same window and the same series.
- Lower error means it did better on that sample, not that it will do better tomorrow. This sentence appears under every metrics block.
- If the AI model did worse than the baseline on this product's backtest, the app says so plainly. That is a valid research outcome.

**Actions:** export CSV (PDF later), share a text summary, view run details (all metadata from §11.3), re-run, back to history.

### 8.11 Forecast history

A searchable, filterable list of runs: product, store, as-of date, horizon, method, model version, status, and a "Replay" tag where applicable. Completed runs reopen from cache when offline. Failed runs show the reason and a retry action. Once actual sales for a past forecast's dates have been imported, the run shows an **"Actuals now available"** tag and its detail screen shows forecast against actual, through the scheduled evaluation in §11.6.

### 8.12 Settings and team management

Settings: profile, workspace, selected store, timezone, currency, theme, notifications (later), help, privacy, app version, sign out. Team management is limited to owners/administrators: invite, assign role, revoke, and pending invitations.

### 8.13 Model information screen

`ModelInfoFragment` shows the **model card** returned by `GET /v1/models/active` for the current workspace:

- Name and version, for example "Demand model: LightGBM (Tweedie) · `m5-ca1-foods` v1.1.0".
- What it was trained on: data, store, product range, date range.
- What it predicts: expected units sold per product per day, next-day horizon.
- Inputs in plain language: recent sales (last 1, 7, 14 and 21 days and short averages), day of week and month, holidays and events, price.
- Backtest results: AI vs. baseline MAE, RMSE and WAPE on the stated window, with the window dates.
- Known limitations: validated on grocery (FOODS) products; less precise for intermittent products; next-day only; no uncertainty ranges.
- For own-data workspaces: training status, date trained, and the workspace's own backtest result.

---

## 9. Machine-learning integration: how the model serves the app

### 9.1 System boundary

```text
┌──────────────┐  HTTPS/JSON   ┌───────────────────────────┐  internal HTTP  ┌──────────────────────┐
│ Android app  │ ────────────▶ │ Backend API               │ ──────────────▶ │ ML service           │
│ (Kotlin)     │ ◀──────────── │ auth, permissions,        │ ◀────────────── │ (Python, this repo)  │
└──────────────┘               │ sales history, runs, jobs │                 │ features, LightGBM,  │
                               │ PostgreSQL                │                 │ baselines, backtests │
                               └───────────────────────────┘                 └──────────┬───────────┘
                                                                                        │ reads
                                                                              ┌─────────▼──────────┐
                                                                              │ Model registry     │
                                                                              │ (artifacts + cards)│
                                                                              └────────────────────┘
```

- The app talks **only** to the backend. The ML service is not reachable from the internet and has no user authentication of its own; it trusts only the backend (shared secret or private network).
- The backend owns all data. It sends the ML service exactly the history it needs for a request, so the ML service holds no business data between requests.
- The ML service owns all forecasting logic: feature computation, model inference, baselines, metrics and demand classification. Keeping it in one place stops the feature code used for training from drifting away from the code used for serving.

For the semester project, the backend and ML service may run as two processes on one machine, or as one FastAPI deployment with the ML code as a separate internal module. Either way the contract in §12 stays the same.

### 9.2 Workspace modes

The trained Stage 3 model uses the M5 item, department, store and state IDs as categorical features, M5 absolute time (`d_num`, `wm_yr_wk`, `year_num`), US events, and the California SNAP flag. A product it has never seen falls into LightGBM's "unseen category" branch at every ID split, and a Kenyan shop has no SNAP days or Super Bowl. Running it on a business's own products would therefore produce numbers with no validated meaning. The design handles this with two modes:

| | Reference workspace | Own-data workspace |
|---|---|---|
| Data | M5 CA_1 FOODS, seeded by the backend | The business's uploaded history |
| Model | Pre-trained `m5-ca1-foods` artifact (Stage 3) | `own-data` profile: same algorithm and hyperparameters, **portable feature schema** (§10.3), **trained per workspace** |
| AI forecasts available | Always | Only after training succeeds and the workspace backtest passes the gate (§9.4.3) |
| Baselines available | Always | As soon as a product has 1 recorded day (naive) or 7 days (seasonal naive) |
| Replay with actuals | Yes, `d_1913`–`d_1941` (and any date for research users) | Later, via scheduled evaluation (§11.6) |
| Research validity | This is where the study's results come from | Exploratory. Each workspace reports its own backtest |
| Release | **MVP** | **Phase 2** of the ML roadmap |

### 9.3 Reference workspace flow (MVP)

1. **Seeding (one-time).** A backend command (`seed_reference_workspace`) reads `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv` with the `mlengine` M5 loader, keeps CA_1 FOODS, converts the wide `d_*` columns to canonical daily rows (`d_1` = 2011-01-29), and writes products, sales history, prices and the M5 calendar into the database. The 1,437 × 1,941 ≈ 2.8 million rows fit comfortably in PostgreSQL.
2. **Model artifacts.** Three versions are registered (§11.3), all with 70 trees:
   - `m5-ca1-foods@1.1.0-dev`: trained on `d_22`–`d_1885` (the notebook's development model). Used for as-of dates 2016-03-27 to 2016-04-23.
   - `m5-ca1-foods@1.1.0-bt`: trained on `d_22`–`d_1913`. Used for any as-of date in `d_1913`–`d_1940`, so replayed forecasts never use a model that has seen the replay window.
   - `m5-ca1-foods@1.1.0`: refit on `d_22`–`d_1941`. Used for the true forecast from `d_1941` (2016-05-22).
   The ML service picks the newest version whose `training_end_date` is **on or before** the as-of date. If none qualifies, it refuses the request. This leakage rule is enforced in code, not left to convention.
3. **Forecast request.** The backend loads the product's last 365 days of history up to the as-of date (21 days are enough for the features; 365 make the demand class meaningful), the known prices for the forecast days, and the product's first-price date. It sends these to `POST /internal/v1/forecast` (§12.3). The ML service derives the M5 identity fields and the calendar itself; the backend sends neither.
4. **Response.** The ML service returns AI and baseline predictions per day, the model version, the feature-schema version and the demand class. The backend stores the run and its points, adds actuals when replaying, and attaches the product's stored backtest metrics.
5. **Stored backtests.** The release stores the `@1.1.0-bt` model's per-product AI and baseline metrics over `d_1914`–`d_1941` in the artifact. The backend reads them with `GET /internal/v1/models/m5-ca1-foods/1.1.0/backtest?productId=…` for §8.10 item 2.

### 9.4 Own-data workspace flow (Phase 2)

#### 9.4.1 Why retrain rather than reuse

What carries over from the research is the **recipe**: short lags, short rolling windows, a Tweedie objective, LightGBM and these hyperparameters. The M5 weights do not carry over. The notebooks showed that features and objective mattered far more than model choice, so the recipe is the valuable, transferable result. Each business gets the same recipe trained on its own history, and the model's categorical levels are that business's own products.

#### 9.4.2 Eligibility to train (set by `PortableFeatures_CA1Foods.ipynb`)

- ≥ 10 active products with ≥ 21 days of history each.
- ≥ 120 days of store history (enough for a training window plus a 28-day validation window and a 28-day backtest).
- ≥ 4,500 product-days of observed data in total.

These thresholds live in `mlengine.config.OWN_DATA_ELIGIBILITY`, and `POST /internal/v1/eligibility` returns each check with its value and requirement. In the simulation, every store with 4,500+ product-days and 120+ days passed the gate in at least 2 of 3 runs, and every one with 6,000+ passed in all 3. Until a store is eligible, only baselines are offered, and the dashboard shows progress toward eligibility, for example "64 of 120 days".

#### 9.4.3 Training and the backtest gate

1. The owner taps **Train AI model** (or the backend schedules a weekly retrain once the store is eligible).
2. The backend starts a training job and sends the ML service the store's full history (§12.4).
3. The ML service uses the same split pattern as the notebooks, measured back from the last history day: train → 28-day validation (early stopping) → 28-day backtest after refitting on everything before it.
4. **Gate:** the model is activated only if its backtest RMSE **and** MAE are both lower than the naive lag-1 baseline's on the same window. Otherwise the run is stored as "Not better than baseline". The workspace keeps baselines, and the model card says so honestly.
5. If the gate passes, the ML service refits on all history, stores the artifact under `own-data/<workspaceId>@<n>`, and the backend marks it active.
6. Weekly retrains follow the same gate. A new version that fails the gate does not replace a passing one.
7. Training also runs a recursive 7-day backtest inside the backtest window. If it beats both baselines, the model is registered with horizons `[1, 7]`; otherwise `[1]`.

### 9.5 Forecast request lifecycle (both modes)

1. The app posts a forecast request: store, product, as-of date, horizon, method.
2. The backend checks auth, role, store access and readiness (§8.7.4). Failures return `422` with a typed reason that the app maps to the messages in §8.7.4.
3. The backend creates a job and returns `202` with the job ID.
4. A worker resolves the model version (leakage rule, §9.3 step 2), assembles the input payload and calls the ML service.
5. The ML service validates the payload against the model's feature schema, builds features, predicts (recursively for multi-day horizons, §10.5), computes baselines and the demand class, and returns the result.
6. The backend stores the run, its points and metadata, attaches actuals (replay) and stored backtest metrics, and marks the job Completed.
7. The app polls the job, then fetches and caches the result.

---

## 10. The model: specification and research rules

### 10.1 Production model choice

| Candidate | Result | Decision |
|---|---|---|
| Stage 3: Tweedie LightGBM, 31 features | Dev RMSE 2.356 / backtest 2.377 | **Production model** |
| Stage 7: + 16 extra sales/price features | Dev 2.345 / backtest 2.373 (within noise) | Not adopted: 16 more features to compute and maintain for no measurable gain. Can be revisited. |
| Stage 6: pooled CA_1–CA_4 | Ties Stage 3 on CA_1 | Candidate for a later multi-store Reference workspace |
| Stages 4–5: hurdle and bucketed | Worse | Not used |

Hyperparameters (from `LightGBM_BaseLocal.ipynb`): `objective="tweedie"`, `tweedie_variance_power=1.1`, `learning_rate=0.05`, `num_leaves=31`, `n_estimators=1000` with early stopping (patience 30) on the validation window, `random_state=42`. The refit uses the early-stopped iteration count.

### 10.2 Feature schema `m5-v1` (Reference model)

This must match the notebook exactly. Column order matters, and so do the categorical levels.

| Group | Features | Computation for target day *t* |
|---|---|---|
| Identity (categorical) | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | M5 identifiers. Levels must equal the training levels (saved in the artifact). |
| Time index | `d_num`, `wm_yr_wk`, `year_num` | M5 day number of *t*; Walmart week; calendar year |
| Lags | `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_21` | Units sold on *t−1*, *t−7*, *t−14*, *t−21* |
| Rolling | `rolling_mean_{3,7,14}`, `rolling_std_{3,7,14}` | Mean and sample standard deviation (ddof = 1) of units over the *w* days ending *t−1* |
| Calendar | `weekday_num` (Mon = 0), `month_num`, `day_of_month`, `is_weekend` (Sat/Sun) | From the date of *t* |
| Events (categorical) | `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` | From M5 `calendar.csv` |
| SNAP | `snap` | `snap_CA` for *t* |
| Price | `sell_price`, `price_missing`, `weeks_since_first_price` | Price for *t*'s Walmart week; 1 if absent; weeks since the item's first priced week |

Total: 31 features. The target is units sold on *t*. `FEATURE_START_DAY = 22`.

### 10.3 Feature schema `portable-v1` (own-data models)

The portable schema keeps everything that transfers and replaces the M5-specific parts.

| `m5-v1` feature | `portable-v1` equivalent | Reason |
|---|---|---|
| `id`, `item_id` | `product_id` (categorical, workspace's own levels) | The model is trained per workspace |
| `dept_id`, `cat_id` | `category` (categorical; "UNCATEGORISED" if absent) | User-supplied grouping |
| `store_id`, `state_id` | `store_id` (categorical) | Multi-store workspaces |
| `d_num`, `wm_yr_wk`, `year_num` | `days_since_history_start`, `year_num` | No M5 day numbering |
| Lags, rolling, calendar | Unchanged | Same definitions |
| `event_*` | `holiday_name` (categorical), `is_holiday` | From the workspace country's public-holiday calendar (Python `holidays` package) |
| `snap` | *dropped* | US-specific |
| `sell_price` (weekly) | `sell_price` (daily, last known price carried forward) | User data is daily |
| `price_missing`, `weeks_since_first_price` | Unchanged in meaning (first price date from history) | Same |

**Future prices:** the price on a forecast day is unknown, so the ML service uses the last known price and records `priceAssumption: "LAST_KNOWN"` on the run. A planned-price input is a later enhancement.

### 10.4 Inference: what the ML service needs per product

For an as-of date *a* and horizon *H*, the backend sends:

- Daily units for *a−20 … a* at minimum (21 days, the longest lag). Sending 365 days is recommended so the demand class is meaningful. Unrecorded days are omitted or sent as `units: null`.
- The daily price for the history days, and known prices for *a+1 … a+H* (`futurePrices`). Missing future prices fall back to the last known price (`priceAssumption: "LAST_KNOWN"`).
- The product's first-price date (inferred from the history, with a warning, if absent).
- The category (own-data models).

The ML service supplies the rest itself: M5 identity fields, the M5 calendar and SNAP (`m5-v1`), or public holidays for the workspace country (`portable-v1`).

### 10.5 Horizons and recursive forecasting

The model is trained to predict day *t* from information up to *t−1*. The validated use is therefore **next day**: *H* = 1.

For *H* > 1, the ML service runs **recursively**: it predicts *a+1*, appends that prediction to the series as if it were observed, recomputes lags and rolling features, and predicts *a+2*, and so on. Errors compound, and the model never saw its own predictions as inputs during training. So:

- `RecursiveHorizon_CA1Foods.ipynb` runs the recursive 7-day forecast from every as-of date in `d_1913`–`d_1934` (22 origins × 1,437 items) and scores each horizon day against naive and seasonal naive.
- Rule: enable 7 only if the recursive AI forecast beats both baselines on aggregate 7-day RMSE and MAE.
- **Result: passed.** Aggregate 7-day RMSE / MAE: AI 2.472 / 1.458, seasonal naive 3.176 / 1.760, naive 3.489 / 1.860. AI RMSE rises from 2.34 on day 1 to 2.58 on day 7 and beats both baselines on every day. From day 2 the forecasts run about 0.09 units/day (about 4%) high.
- The release pipeline repeats this check itself (`mlengine.evaluation.backtest.recursive_backtest`). Since `m5-ca1-foods@1.1.0`, `capabilities.horizons` is `[1, 7]`, and the per-day curve is on the model card (`card.horizonAccuracy.byDay`).
- Horizons 8–28 can be requested with `allowExperimentalHorizon: true` (research only). They come back with `horizon.validated: false`.
- 14- and 28-day horizons would use a separately trained direct model with 28-day-embargoed lags (the Stage 2 design, which the README notes is right for month-ahead views). Deferred.

Baselines use the same recursion rules: the naive forecast for every day in the horizon is the last observed value.

### 10.6 Traditional baselines

| ID | Name in the app | Definition | Role |
|---|---|---|---|
| `naive_last_value` | "Same as last day" | Forecast for every horizon day = units on the as-of date | **Primary baseline.** Matches `sales_lag_1` in the notebooks. |
| `seasonal_naive_7` | "Same day last week" | Forecast for day *t* = units on *t−7* (recursive beyond 7 days) | Secondary baseline. Shown in research mode and on the model card. |

Statistical models (ETS/ARIMA via `statsmodels`) are a possible later addition. They would need their own backtest on the same windows before they appear in the app.

### 10.7 Metrics

All metrics are computed by the ML service with one shared implementation (`mlengine.evaluation.metrics`):

- **MAE** = mean |actual − forecast|, in product units.
- **RMSE** = √mean (actual − forecast)², in product units.
- **WAPE** = Σ|actual − forecast| / Σ actual. Undefined when Σ actual = 0; shown as "n/a".

Each metrics block records the evaluation window, the series it covers (product / store / all), the number of points, and the model version.

### 10.8 Demand classification (volatility)

Computed by the ML service from the last 365 observed days of a product (or all history if shorter), using the measures already in the EDA (`reports/EDAInsights.md`):

- **ADI** (average demand interval) = observed days / days with sales > 0.
- **CV²** = (std / mean)² of sales on days with sales > 0.
- **Zero-day share** = share of observed days with zero sales.

Classes (Syntetos–Boylan thresholds; to be confirmed in the research methodology):

| Class | Rule | Plain-language label |
|---|---|---|
| Smooth | ADI < 1.32 and CV² < 0.49 | "Steady seller" |
| Erratic | ADI < 1.32 and CV² ≥ 0.49 | "Sells often, amounts vary a lot" |
| Intermittent | ADI ≥ 1.32 and CV² < 0.49 | "Sells on some days only" |
| Lumpy | ADI ≥ 1.32 and CV² ≥ 0.49 | "Irregular and unpredictable" |
| Insufficient data | < 30 observed days or < 5 sale days | "Not enough history to classify" |

This classification is how the app supports the "under volatile demand conditions" part of the research question. The research export reports AI vs. baseline errors per class.

**Result on the CA_1 FOODS backtest** (`ErrorByDemandClass_CA1Foods.ipynb`): AI beats naive in every class, with an RMSE gain of 29% (smooth, 221 items), 21% (erratic, 37), 25% (intermittent, 925) and 25% (lumpy, 249). The MAE gain is smaller for intermittent and lumpy items (17–19%), and on MAE naive still wins for about a quarter of those items.

### 10.9 Research integrity requirements

- AI and baselines always use the same series, as-of date, horizon and preprocessing.
- Splits are chronological. The model-version leakage rule (§9.3) guarantees that no replayed forecast uses a model trained on the replayed days.
- Metrics are computed against actual sales, never against another model's prediction.
- Every run stores model version, feature-schema version, `mlengine` package version, history version, as-of date, horizon, baselines used, price assumption and any excluded/missing days (§11.3).
- **Research export** (research users): CSV of all replay runs with per-day actuals, AI and baseline forecasts, demand class and metadata, so the study's tables can be reproduced from the app's own data.
- Forecasts for future dates have no actuals at generation time. Their metrics are always labelled as historical backtests.

### 10.10 Notebook experiments

All three have been run. The outputs and conclusions are saved in the notebooks.

| Notebook | Purpose | Result |
|---|---|---|
| `RecursiveHorizon_CA1Foods.ipynb` | Validate the 7-day recursive forecast (§10.5) | Passed; horizon 7 enabled |
| `PortableFeatures_CA1Foods.ipynb` | Compare `portable-v1` with `m5-v1`; simulate small stores | `portable-v1` backtest RMSE 2.380 vs 2.377 (within 0.2%). Thresholds set (§9.4.2) |
| `ErrorByDemandClass_CA1Foods.ipynb` | Backtest error by demand class | AI beats naive in all classes (§10.8) |

---

## 11. From notebooks to production

### 11.1 The problem to solve

All the modelling code currently lives in notebooks. Each notebook repeats data loading and feature building. No model is saved, and `scripts/preprocess_m5.py` still builds the obsolete 28-day features. Serving needs one importable, tested implementation of the Stage 3 pipeline that reproduces the notebook's numbers.

### 11.2 Repository layout (as built)

```text
SemProject_MLEngine/
├── notebooks/                    # Research record. Stage 1–7 notebooks unchanged; 3 new ones import mlengine.
├── mlengine/                     # Installable package (pyproject.toml); `python -m mlengine …`
│   ├── config.py                 # Lags, windows, splits, hyperparameters, eligibility thresholds
│   ├── data/m5.py                # Load and scope raw M5; daily price matrix; scope_to_batch()
│   ├── features/core.py          # Lag/rolling maths, the single implementation (NaN-aware)
│   ├── features/m5_v1.py         # The 31 Stage 3 features (identical to the notebook)
│   ├── features/portable_v1.py   # Own-data features (holidays, daily prices)
│   ├── models/train.py           # LightGBM calls exactly as in the notebook
│   ├── models/forecaster.py      # One-step + recursive inference for both schemas
│   ├── models/baselines.py       # naive_last_value, seasonal_naive_7
│   ├── models/artifact.py        # Artifacts, ModelRegistry, leakage rule
│   ├── evaluation/               # metrics.py, demand_class.py, backtest.py (recursive)
│   ├── pipelines/m5_release.py   # Train/backtest/register the Reference versions
│   ├── pipelines/own_data.py     # Per-workspace training + gate + recursive check
│   ├── pipelines/reference_export.py  # Seed data for the backend
│   ├── service/                  # FastAPI internal ML service (app.py, schemas.py)
│   └── cli.py                    # release-m5 | export-reference | serve | models
├── models/m5-ca1-foods/          # Committed artifacts (about 2.5 MB): 1.1.0-dev, 1.1.0-bt, 1.1.0
├── tests/                        # 23 fast tests + 2 full-data tests (MLENGINE_FULL_TESTS=1)
├── docs/ml-service-examples/     # Real request/response JSON from the running service
├── scripts/make_parity_fixture.py  # Runs the notebook cells and checks feature identity
├── scripts/preprocess_m5.py      # Legacy (28-day features); not used by mlengine
└── Dockerfile                    # ML service image
```

### 11.3 Model artifacts and run metadata

Each registered version is a folder `models/<modelId>/<version>/` with `model.txt` (LightGBM booster), `metadata.json`, `card.json`, `calendar.parquet` (M5 models), and `backtest/` (`summary.json`, `per_product.parquet`, plus `predictions.parquet` for `-bt`). `metadata.json` records the feature columns, categorical levels, hyperparameters (including the real tree count, 70), training window, capabilities, evaluation results, `mlengineVersion`, `lightgbmVersion` and `gitCommit`.

**Decision change:** the artifacts are small (about 2.5 MB in total), so they are **committed to Git** instead of published as release assets. That way the ML service runs from a fresh clone without the raw M5 files. Own-data models are written to the same registry at runtime and are not committed.

Each **forecast run** stored by the backend records: `runId`, workspace/store/product, `asOfDate`, `horizonDays`, `method`, the resolved `model.id@version`, `featureSchema`, `mlengineVersion`, `historyVersion`, the baselines used, `priceAssumption`, `missingDaysInLookback`, `isReplay`, timestamps and the request ID. All of these except the backend's own fields come back in the ML service response.

### 11.4 Parity: proving the package equals the notebook (done)

1. **Feature parity:** `scripts/make_parity_fixture.py` executes the notebook's own cells and compares every column of its `featureFrame` with `mlengine`'s. **Identical over all 2,759,040 rows and 33 columns.** A 25-item fixture (`tests/fixtures/m5_parity/`) repeats the check in CI.
2. **Training parity:** 70 early-stopped trees; dev RMSE 2.3555 / MAE 1.3085; backtest RMSE 2.3773 / MAE 1.3766; naive 3.194 and 3.219. These match the notebook (`tests/test_full_data.py`).
3. **Inference parity:** the serving path (history arranged as the backend sends it) reproduces the training-frame predictions to within 4e-6, with 21, 28 or 365 days of history (`tests/test_service.py`).
4. **End to end:** `POST /internal/v1/backtest` over 2016-04-25 to 2016-05-22 for all 1,437 items returns RMSE 2.377.
5. **No leakage:** changing any sales value on or after day *t* leaves day *t*'s features unchanged (`tests/test_features_core.py`). The registry refuses models trained past the as-of date (`tests/test_registry.py`).

### 11.5 Release workflow for a model version

```bash
python -m mlengine release-m5 --version 1.1.0   # dev → bt → per-product backtest → recursive 7-day check → prod (~2 min)
python -m mlengine models                       # list registered versions
MLENGINE_FULL_TESTS=1 pytest                    # fast suite + full-data parity
```

Version numbers: **major** for a feature-schema change, **minor** for retraining or a capability change (1.0.0 → 1.1.0 added horizon 7), **patch** for metadata or card fixes. `-dev` and `-bt` mark the replay models.

### 11.6 Scheduled jobs (backend)

| Job | When | What |
|---|---|---|
| Actuals evaluation | Nightly | For completed runs whose forecast dates now have imported actuals, compute per-run errors and set "Actuals now available" |
| Demand class refresh | After each confirmed import | Recompute classes for the affected products |
| Own-data retrain | Weekly, eligible workspaces (Phase 2) | Train → gate → activate if passed |
| Daily store forecast | Optional, later | Score next-day demand for every product in a store after the day's import, for a dashboard "Tomorrow" list |

### 11.7 Performance expectations

- One-product next-day inference: under 50 ms in the ML service, dominated by feature assembly.
- A whole-store next-day forecast (1,437 products): one batched `predict` call, about 1 s.
- Model load: the `model.txt` file is a few MB; load once at startup and keep in memory.
- Own-data training on 120–1,000 days × 10–2,000 products: seconds to a few minutes on one CPU. Run in a worker, never inside a request.

---

## 12. API contracts

All paths are proposed contracts for the backend and ML service developers. Errors use one shape everywhere:

```json
{"error": {"code": "INSUFFICIENT_HISTORY", "message": "This product needs at least 21 days of sales history.",
           "details": {"availableDays": 12, "requiredDays": 21}, "requestId": "req_…"}}
```

The app maps `code` to its own localised message and falls back to `message`.

### 12.1 Public API (app ↔ backend)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/auth/login`, `/v1/auth/refresh` | Session |
| GET | `/v1/me` | User, roles, research flag, workspace access |
| GET/POST | `/v1/workspaces` | List (includes Reference) / create |
| GET/POST | `/v1/workspaces/{wid}/stores` | Stores |
| GET/POST/PATCH | `/v1/stores/{sid}/products[/{pid}]` | Products (read-only in Reference) |
| GET | `/v1/stores/{sid}/products/{pid}/history?from=&to=` | Daily history for charts |
| GET | `/v1/stores/{sid}/readiness` | Store and per-product readiness (§8.7.4) |
| POST | `/v1/stores/{sid}/imports` | Upload CSV (multipart, with `Idempotency-Key`) |
| PUT | `/v1/imports/{iid}/mapping` | Column mapping and the zero-vs-missing answer |
| GET | `/v1/imports/{iid}` / `/validation` / `/errors` | Status, summary, row errors |
| POST | `/v1/imports/{iid}/confirm` | Merge into history (with the overlap policy) |
| DELETE | `/v1/imports/{iid}` | Remove a batch's rows |
| GET | `/v1/stores/{sid}/models/capabilities` | Horizons, methods, thresholds, as-of range, AI availability |
| GET | `/v1/stores/{sid}/models/active` | Model card |
| POST | `/v1/stores/{sid}/models/train` | Phase 2: request own-data training |
| POST | `/v1/forecasts` | Create forecast job → `202` |
| GET | `/v1/forecast-jobs/{jobId}` | Job status |
| GET | `/v1/forecasts/{fid}` | Completed result |
| GET | `/v1/forecasts?storeId=&productId=&replay=&page=` | History |
| GET | `/v1/forecasts/{fid}/export?format=csv` | Export |
| GET | `/v1/research/export?storeId=` | Research export (research users) |

**Capabilities response (Reference workspace):**

```json
{
  "storeId": "ref_ca1",
  "isReference": true,
  "aiAvailable": true,
  "methods": ["AI", "BASELINE", "COMPARE"],
  "horizons": [1, 7],
  "frequency": "DAILY",
  "asOf": {"default": "2016-05-22", "editable": true, "replayMin": "2016-04-24", "replayMax": "2016-05-21",
           "researchMin": "2016-03-27", "calendarEnd": "2016-06-19"},
  "minHistoryDays": 21,
  "maxMissingDaysInLookback": 3,
  "predictionIntervals": false,
  "activeModel": {"id": "m5-ca1-foods", "version": "1.1.0"}
}
```

**Forecast request:**

```json
{"storeId": "ref_ca1", "productId": "FOODS_3_090", "asOfDate": "2016-05-15",
 "horizonDays": 1, "method": "COMPARE"}
```

**Completed forecast (replay example; numbers illustrative only):**

```json
{
  "forecastId": "fc_789",
  "status": "COMPLETED",
  "generatedAt": "2026-11-07T08:34:12Z",
  "storeId": "ref_ca1",
  "product": {"productId": "FOODS_3_090", "name": "FOODS_3_090", "unitOfMeasure": "units",
              "demandClass": {"class": "ERRATIC", "adi": 1.05, "cv2": 0.61, "zeroShare": 0.05}},
  "asOfDate": "2016-05-15",
  "isReplay": true,
  "horizon": {"startDate": "2016-05-16", "endDate": "2016-05-16", "days": 1, "frequency": "DAILY"},
  "model": {"id": "m5-ca1-foods", "version": "1.1.0-bt", "featureSchema": "m5-v1"},
  "baselines": [{"id": "naive_last_value", "label": "Same as last day"}],
  "priceAssumption": "KNOWN",
  "points": [{"date": "2016-05-16", "ai": 41.37, "baseline": 36.0, "actual": 44.0}],
  "history": [{"date": "2016-04-18", "actual": 39.0}],
  "predictionIntervals": null,
  "runMetrics": {"window": "2016-05-16/2016-05-16", "n": 1,
                 "ai": {"mae": 2.63}, "baseline": {"mae": 8.0}},
  "productBacktest": {"window": "2016-04-25/2016-05-22", "n": 28, "modelVersion": "1.1.0-bt",
                      "ai": {"mae": 5.1, "rmse": 6.8, "wape": 0.12},
                      "baseline": {"mae": 7.4, "rmse": 9.9, "wape": 0.18}},
  "historyVersion": 1,
  "mlRequestId": "ml_…"
}
```

The UI never hardcodes this sample as real data.

### 12.2 Error codes

**Backend → app** (request-level, `422` unless noted): `INSUFFICIENT_HISTORY`, `STALE_HISTORY`, `TOO_MANY_GAPS`, `AI_NOT_AVAILABLE`, `HORIZON_NOT_SUPPORTED`, `AS_OF_OUT_OF_RANGE`, `NO_MODEL_FOR_AS_OF`, `PRODUCT_NOT_IN_MODEL`, `FORBIDDEN` (`403`).

**ML service → backend:**
- Request-level errors (`{"error": {...}}`): `INVALID_REQUEST`, `MODEL_REQUIRED`, `MODEL_NOT_FOUND` (`404`), `NO_MODEL_FOR_AS_OF`, `HORIZON_NOT_SUPPORTED`, `HISTORY_AFTER_AS_OF`, `DATE_OUTSIDE_CALENDAR`, `INVALID_WINDOW`, `INVALID_HISTORY`, `NO_STORED_BACKTEST` (`404`), `JOB_NOT_FOUND` (`404`), `UNAUTHORIZED` (`401`).
- Per-series reasons (`series[i].aiUnavailableReason`, with baselines still returned): `INSUFFICIENT_HISTORY`, `TOO_MANY_GAPS`, `PRODUCT_NOT_IN_MODEL`, `AI_NOT_REQUESTED`.

### 12.3 Internal ML service: forecast (as built)

`POST /internal/v1/forecast`. Full real examples are in `docs/ml-service-examples/`.

```json
{
  "requestId": "req_123",
  "model": {"id": "m5-ca1-foods"},
  "asOfDate": "2016-05-22",
  "horizonDays": 7,
  "baselines": ["naive_last_value", "seasonal_naive_7"],
  "series": [{
    "productId": "FOODS_3_090", "storeId": "CA_1", "category": "FOODS_3", "firstPriceDate": "2011-01-29",
    "history": [{"date": "2016-05-22", "units": 64, "price": 1.25}],
    "futurePrices": [{"date": "2016-05-23", "price": 1.25}]
  }]
}
```

- `model.version` is optional. Without it, the registry picks the newest version trained only on data up to `asOfDate`.
- `includeAi: false` (with `model` omitted) returns baselines and demand class only. Use it for own-data stores without an active model.
- `history` must not contain dates after `asOfDate`. Missing days, or `units: null`, mean "not recorded". Send up to 365 days; at most 400 are used.
- For M5 products the service derives `id`, `dept_id`, `cat_id` and `state_id` from `productId`/`storeId`, and the calendar (events, SNAP, Walmart weeks) from the artifact. The backend sends neither.
- Response per series: `points[{date, ai, baselines{…}}]`, `aiUnavailableReason`, `missingDaysInLookback`, `priceAssumption` (`KNOWN` / `LAST_KNOWN` / `NONE` / `NOT_USED`), `demandClass{class, label, adi, cv2, zeroShare, observedDays, saleDays}`, and `warnings` (`FIRST_PRICE_DATE_INFERRED`, `PRODUCT_NEW_SINCE_TRAINING`). The top level carries `model{id, version, featureSchema, trainingEndDate}`, `horizon{startDate, endDate, days, validated}`, `mlengineVersion` and `baselineLabels`.

### 12.4 Internal ML service: other endpoints (as built)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/internal/v1/health` | Liveness, plus the model versions loaded |
| GET | `/internal/v1/models` | All versions with capabilities |
| GET | `/internal/v1/models/{id}/resolve?asOfDate=` | Which version the leakage rule picks |
| GET | `/internal/v1/models/{id}/{version}` | Metadata + card |
| GET | `/internal/v1/models/{id}/{version}/card` | Model card only (for §8.13) |
| GET | `/internal/v1/models/{id}/{version}/backtest[?productId=]` | Stored store-level and per-product backtest |
| POST | `/internal/v1/backtest` | One-step backtest over a window (≤ 91 days) for given series |
| POST | `/internal/v1/demand-class` | ADI, CV², class for series |
| POST | `/internal/v1/eligibility` | Own-data eligibility checks for a history |
| POST | `/internal/v1/train` | Own-data training job → `202 {jobId, statusUrl}` |
| GET | `/internal/v1/train/{jobId}` | `QUEUED` / `RUNNING` / `COMPLETED` (result `status`: `TRAINED` / `GATE_FAILED` / `NOT_ELIGIBLE`) / `FAILED` |

Authentication: when `ML_SERVICE_TOKEN` is set, every call needs `X-Internal-Token`. Training jobs are kept in memory; the resulting models persist in the registry.

## 13. Local persistence, uploads and offline behaviour

### 13.1 Local data

- Room caches permitted products, import summaries, and completed forecast results (immutable once completed, so safe to cache).
- DataStore holds non-sensitive preferences: last workspace, chart range, table display mode.
- Session secrets go in secure storage backed by Android Keystore.
- Upload files stay in app cache only until the upload succeeds.

### 13.2 Upload reliability

- Use the selected document URI; no broad storage permission.
- Validate type and size before upload.
- Send an `Idempotency-Key` per confirmed upload so retries do not create duplicate imports.
- Show real upload progress where measurable.
- Keep mapping and confirmation state across network loss and process death.
- Never upload without explicit confirmation.

### 13.3 Offline

Cached products, imports and completed forecasts can be viewed offline, with a "Last synced" time. Uploads, validation, readiness and forecast generation need a connection. A confirmed upload may retry automatically when the connection returns. Stale results are never shown as newly generated; every result shows its `generatedAt` and `asOfDate`.

---

## 14. Security, privacy and trust

- HTTPS for all public traffic. The ML service is reachable only from the backend.
- Server-side authorisation on every endpoint.
- No credentials or service secrets in the app or in source control.
- No logging of tokens, passwords, raw files or business sales data. Logs carry request IDs only.
- Data minimisation: forecasting needs aggregated daily sales, prices and categories. No customer names, phone numbers or payment data.
- The ML service keeps no business data after a request, apart from own-data model artifacts, which belong to the workspace and are deleted with it.
- Clear confirmation before upload, explaining what is sent.
- Sign out clears user-scoped local cache.
- A privacy notice and a data-deletion request path. Deleting a workspace deletes its history, runs and own-data models.
- The M5 data is used under the Kaggle competition terms for research/education. The Reference workspace credits the source on the model card.

---

## 15. Notifications and error handling

Notifications are a later enhancement: forecast completed, import finished, own-data model trained (or gate failed), "upload yesterday's sales" reminder.

Every network screen defines loading, content, empty, error-with-retry and partial-data states.

Example messages:
- "We couldn't read the date column. Check the column mapping and date format."
- "This product has only 12 days of sales history. Forecasts need at least 21."
- "Your latest sales are from 3 Nov. Upload sales up to yesterday to forecast tomorrow."
- "AI forecasts aren't available for this store yet. 'Same as last day' forecasts are available now."
- "The AI model didn't beat the simple baseline on your recent data, so we're showing the baseline. We'll retry after more sales are uploaded."
- "Your upload is saved and will retry when your connection returns."

Avoid "Something went wrong" when the backend returns an actionable code.

---

## 16. Accessibility and localisation

- System font scaling and screen-reader support, with text summaries for every chart.
- Touch targets of at least 48 dp.
- Never rely on colour alone: line styles, markers and labels on charts; text on status chips.
- Portrait phones first; tablets later.
- Locale-aware formatting for dates, numbers and currency.
- Business dates are interpreted in the store's timezone. In the Reference workspace that is `America/Los_Angeles`.
- English first, with Kiswahili later. All strings live in resources.

---

## 17. Non-functional requirements

- **Reliability:** recover from network errors and process recreation without corrupting an import or run.
- **Performance:** render from cache first. CSV preview parsing runs off the main thread.
- **Scalability:** paging for products (1,437 in the Reference workspace), imports and runs.
- **Maintainability:** feature-separated code. The ML logic exists only in `mlengine`.
- **Testability:** ViewModels, repositories, navigation and critical flows are unit/UI tested. The ML service has the parity suite (§11.4).
- **Reproducibility:** any stored run can be regenerated from its recorded model version, schema version, history version and as-of date.
- **Security:** no secrets in the APK; backend authorisation.
- **Observability:** structured logs with request IDs that follow a call from app to backend to ML service.
- **Compatibility:** document the minimum SDK in Gradle based on dependency support.

---

## 18. Scope and phasing

### 18.1 MVP

**ML engine (this repo):**
1. `mlengine` package with the M5 loader, the `m5-v1` features, training, one-step prediction, baselines, metrics and demand class.
2. Parity tests passing (§11.4).
3. Artifacts `m5-ca1-foods@1.1.0-dev`, `@1.1.0-bt` and `@1.1.0`, per-product backtests, and the model card.
4. Internal ML service: `health`, `models`, `forecast`, `backtest`, `demand-class`.

**Backend:**
5. Auth, workspaces, stores, products, Reference workspace seeding.
6. Sales history, CSV import with mapping/validation/confirm, readiness.
7. Forecast jobs, the leakage-safe model resolution, runs and history, replay with actuals.

**Android app:**
8. Splash, authentication, workspace picker (Reference + own).
9. Dashboard with honest empty/stale states.
10. Products list and detail with history chart and demand class.
11. Import wizard end to end. Own-data stores get baseline forecasts.
12. Forecast configuration (next day; as-of/replay in Reference), progress, results with the three evidence types, and history.
13. Model info screen.
14. Options, context and popup menus; settings; role-aware actions; error handling; secure session.

### 18.2 Phase 2

- ~~7-day recursive horizon~~: validated and moved into the MVP (§10.5).
- Own-data AI: `portable-v1`, per-workspace training, backtest gate, weekly retrain (§9.4).
- Actuals evaluation job and "Actuals now available".
- Research export.

### 18.3 Later

- 14/28-day direct model (Stage 2 design).
- Pooled multi-store Reference workspace (Stage 6 model, CA_1–CA_4).
- Quantile models for prediction intervals.
- Daily whole-store forecast and "Tomorrow" dashboard list.
- XLSX and Google Sheets import; POS integrations.
- Stock and replenishment suggestions, only once inventory data exists.
- Push notifications; Kiswahili.
- Feature attribution (for example SHAP from LightGBM) shown as "What drove this forecast", only after validating that the explanations are stable and understandable.

---

## 19. Acceptance criteria

The MVP is ready to demonstrate when:

1. The app launches and routes correctly after the splash screen.
2. A user signs in and sees only permitted workspaces, with the Reference workspace always available and visibly marked as demo data.
3. Screen UI is Kotlin + Jetpack Compose, with no XML screen layouts. Major destinations are Fragments hosting `ComposeView`s.
4. Options, context and popup menus are each implemented with their correct triggers.
5. In the Reference workspace, a user can generate a next-day forecast for any of the 1,437 products, with AI and baseline side by side.
6. A replay forecast for an as-of date in 2016-04-24 to 2016-05-21 shows forecast vs. actual. It used the `1.1.0-bt` model (visible in run details), and the ML service refuses a model trained past the as-of date.
7. ✅ Done: the `mlengine` parity tests pass, and the served model reproduces the Stage 3 backtest (RMSE 2.377 over `d_1914`–`d_1941`) through the ML service's backtest endpoint.
8. A user can upload a CSV to their own workspace, map columns (including the zero-vs-missing question), see validation, confirm, and see updated readiness.
9. Own-data products with fresh history get baseline forecasts. AI is clearly shown as not yet available, with the reason.
10. Forecast jobs show honest asynchronous states, and completed results reopen from history, including offline from cache.
11. Every number on a result screen comes from the backend. Metrics always state their window, the series covered and the model version, and only appear where actuals exist.
12. The model info screen shows the model card with training data, horizon, backtest results and limitations.
13. No secrets in the app or repo. The ML service is not publicly reachable.

---

## 20. Implementation plan

Build incrementally and keep everything runnable after each phase. Report files changed, test results, known gaps and the next step at the end of each phase.

**Phase A: ML engine package (this repo) ✅ done**
- Create `mlengine/` and `pyproject.toml`. Port the loader, the `m5-v1` feature builder, training, baselines and metrics from `LightGBM_BaseLocal.ipynb`, without changing their behaviour.
- Export a small parity fixture from the notebook. Write the tests in §11.4 and make them pass.
- Add `artifact.py` and the CLI. Produce `@1.1.0-dev`, `@1.1.0-bt`, the backtests and `@1.1.0`.
- Mark `scripts/preprocess_m5.py` as legacy in the README.

**Phase B: ML service ✅ done**
- FastAPI app with the §12.3–12.4 MVP endpoints, Pydantic schemas, the leakage rule, and contract tests.
- A Dockerfile and a `MODEL_REGISTRY_PATH` setting.

**Phase C: Backend**
- Data model: users, workspaces, stores, products, `sales_daily` (store, product, date, units, price, `is_observed`, `import_id`), imports, `history_versions`, forecast_jobs, forecast_runs, forecast_points, model_versions, product_backtests.
- Reference workspace seeding through the `mlengine` loader.
- Import pipeline, readiness, forecast jobs and the §12.1 endpoints.

**Phase D: Android foundation**
- Project setup, theme, `MainActivity`, Fragment host, navigation, a sample Compose destination.
- App shell, bottom navigation, workspace chip, three menus, reusable components, state screens.

**Phase E: Android data and forecast flows**
- Networking, auth, repositories, Room cache, error-code mapping.
- Products, the import wizard, readiness.
- Forecast configuration, jobs, results, replay, history and model info.
- During development, use fake repositories that return clearly labelled sample data, kept separate from production code.

**Phase F: Phase 2 ML features**
- ✅ `RecursiveHorizon_CA1Foods` → `H = 7` enabled.
- ✅ `PortableFeatures_CA1Foods` → `portable-v1`, own-data training and the gate (ML side done). Remaining: backend wiring and the training UI.

**Phase G: Quality and release**
- Tests, an accessibility pass, offline handling, permission checks, icon and splash polish.
- Document environment setup (backend URL, registry path) with no secrets committed.

---

## 21. Instructions to the AI coding assistant

Treat this document as the source of truth for scope and architecture.

- Build the app in **Kotlin**, with all screens in **Jetpack Compose + Material 3**, hosted by **AndroidX Fragments** through `ComposeView`. No XML screen layouts.
- Use the AndroidX SplashScreen API, MVVM, repositories, Coroutines and lifecycle-aware state collection.
- Implement the Options menu, the long-press Context menu and the tap-triggered Popup menu.
- **Never put forecasting logic in the app.** No lags, baselines, metrics or demand classes are computed on the device.
- **Never apply the M5-trained model to non-M5 products.** Own-data AI goes only through per-workspace training and the backtest gate.
- **Offer only the horizons listed in `capabilities`** (today 1 and 7). Do not add 14/30-day options that the backend has not enabled.
- **Port the notebook logic into `mlengine` without changing behaviour,** and prove it with the parity tests before serving.
- **Enforce the leakage rule** (model training end ≤ as-of date) in the ML service.
- Do not fabricate sales, forecasts, model details or accuracy. Label development fixtures clearly.
- Keep code readable: descriptive names, small functions, and comments only where a decision is not obvious.
- Build and test after each phase. Do not silently change requirements or add unrelated features.

---

## 22. Decisions

| # | Decision | Outcome |
|---|---|---|
| 1 | AI in the Reference workspace only for the MVP, own-data AI in Phase 2? | Adopted. The own-data ML side is also built (training, gate, service endpoints), so the backend can wire it up whenever it is ready. |
| 2 | Next-day-only horizon for the MVP? | Superseded: the 7-day recursive horizon was validated and is enabled. |
| 3 | Separate backend and ML service processes? | The ML service is built as its own FastAPI app (`mlengine/service`). The backend calls it over HTTP on a private network with `X-Internal-Token`. It can also be mounted inside a Python backend process if preferred. |
| 4 | Stage 3 or Stage 7 for production? | Stage 3 (31 features). |
| 5 | Syntetos–Boylan thresholds? | Adopted (ADI 1.32, CV² 0.49). Cite them in the research write-up. |
| 6 | Commit model artifacts? | **Changed:** committed to Git (about 2.5 MB). Own-data models are not committed. |
| 7 | Own-data eligibility thresholds? | Set from the simulation: 10 products, 120 days, 4,500 product-days (§9.4.2). |

Backend decisions (made 7 November 2026): FastAPI + PostgreSQL in a separate repository (`SemProject_Backend`), its own email-and-password sign-in with JWTs, hosted on one server with Docker Compose and Caddy.

---

## 23. Technical references

1. **Using Compose in Views / Fragments**: [Android Developers](https://developer.android.com/develop/ui/compose/migrate/interoperability-apis/compose-in-views)
2. **Migration strategy**: [Android Developers](https://developer.android.com/develop/ui/compose/migrate/strategy)
3. **Menus in Compose**: [Android Developers](https://developer.android.com/develop/ui/compose/components/menu)
4. **Splash screens**: [Android Developers](https://developer.android.com/develop/ui/views/launch/splash-screen)
5. **State and Jetpack Compose**: [Android Developers](https://developer.android.com/develop/ui/compose/state)
6. **LightGBM categorical features and model saving**: [LightGBM documentation](https://lightgbm.readthedocs.io/)
7. **M5 Forecasting – Accuracy dataset**: [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
8. **Demand classification**: Syntetos, A. A., Boylan, J. E. and Croston, J. D. (2005), "On the categorization of demand patterns", *Journal of the Operational Research Society* 56(5).
9. **This project's modelling record**: `README.md` (modelling journey), `notebooks/LightGBM_BaseLocal.ipynb` (Stage 3), `reports/EDAInsights.md` (demand variability).

---

**End of specification**
