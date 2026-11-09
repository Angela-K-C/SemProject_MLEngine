# Changelog

Model versions (`models/`), the `mlengine` package and the documents. Dates are 2026.

## 2026-11-07: Sample business data

- `python -m mlengine export-samples` writes three sample shop CSVs cut from a real M5 store that the Reference model never saw (TX_1 FOODS):
  - an eligible shop: 100 products × 365 days;
  - a small shop that isn't eligible yet: 30 × 60;
  - a messy export with deliberate validation problems.
- Dates are shifted forward by whole weeks so the history ends yesterday. Weekdays stay aligned; holidays don't.
- Fix: `scikit-learn` added to the package dependencies (`pyproject.toml`). Training uses LightGBM's scikit-learn interface; it was only in `requirements.txt`, so own-data training failed inside the Docker image. Found by the backend's end-to-end training run.
- Checked: the eligible shop passes the own-data gate (AI RMSE 2.50 vs naive 3.06; MAE 1.31 vs 1.57) with horizons 1 and 7. The small shop is `NOT_ELIGIBLE`. Covered by `tests/test_full_data.py::test_sample_shops`.

## 2026-11-07: `m5-ca1-foods` 1.1.0

- **Horizon 7 enabled.** The recursive 7-day forecast beat both baselines (aggregate RMSE 2.472 vs 3.176 seasonal naive and 3.489 naive; `notebooks/RecursiveHorizon_CA1Foods.ipynb`). The release pipeline now runs this check itself and publishes accuracy per horizon day on the model card (`card.horizonAccuracy`).
- Same trees as 1.0.0 (70 trees, same data and seed). Only capabilities, evaluation metadata and the card changed.
- The per-class backtest (`byDemandClass`) is stored in every summary.
- Own-data training also runs a per-workspace recursive 7-day check.
- Own-data eligibility: `minObservedProductDays` 5,000 → 4,500, from the small-store simulation (`notebooks/PortableFeatures_CA1Foods.ipynb`).
- Added the experiment notebooks `RecursiveHorizon_CA1Foods`, `PortableFeatures_CA1Foods` and `ErrorByDemandClass_CA1Foods`.

## 2026-11-07: `m5-ca1-foods` 1.0.0 and `mlengine` 0.1.0 (superseded by 1.1.0)

- First release of the `mlengine` package: the Stage 3 pipeline from `notebooks/LightGBM_BaseLocal.ipynb`, with one feature implementation shared by training and serving.
- Parity: features identical to the notebook over 2,759,040 rows; 70 trees; dev RMSE 2.356, backtest RMSE 2.377.
- Versions `-dev`, `-bt` and prod, with the leakage rule (a model serves only as-of dates on or after its training end).
- `portable-v1` feature schema and own-data training with a beat-the-baseline gate.
- Internal ML service (FastAPI), Dockerfile, and the Reference workspace seed export.

## 2026-11-07: Design and documentation

- Mobile app design spec v2.0, rewritten to fit the real model; v2.1 updated to match the implementation.
- `MOBILE_APP_HANDOFF.md` for the app and backend build.
- `docs/`: notebook-to-production journey, architecture guide, ML service API reference, model card, operations runbook.

## Earlier: model research (Stages 1–7)

See `README.md` → "Modelling journey" (PRs #1–#5, 2 to 3 November).
