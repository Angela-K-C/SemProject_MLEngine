# Operations Runbook: ML Engine and ML Service

How to set up, test, release, run and troubleshoot the model side.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                       # or: pip install -e ".[service,dev]"
pytest                                                # 23 tests in about 10 s; no raw data needed
```

The raw M5 CSVs (see the README) are needed only to retrain, regenerate the parity fixture, export the Reference seed data, or run the full-data tests. **Serving needs only the committed `models/` folder.**

## Everyday commands

| Task | Command | Time / memory |
| --- | --- | --- |
| Fast tests | `pytest` | ~10 s |
| Full-data parity tests | `MLENGINE_FULL_TESTS=1 pytest` | ~2 min, ~2.3 GB |
| List model versions | `python -m mlengine models` | instant |
| Run the ML service | `ML_SERVICE_TOKEN=… python -m mlengine serve --host 127.0.0.1 --port 8100` | loads models on first use |
| Retrain and register the Reference model | `python -m mlengine release-m5 --version X.Y.Z` | ~2 min, ~2.3 GB |
| Re-check features against the notebook | `python scripts/make_parity_fixture.py` | ~3 min |
| Export Reference seed data | `python -m mlengine export-reference [--out DIR]` | ~1 min; writes ~100 MB CSV + Parquet |
| Export sample business CSVs | `python -m mlengine export-samples [--store TX_1] [--end-date YYYY-MM-DD]` | ~20 s; writes `data/processed/samples/` (3 CSVs, manifest, README). Re-run so the history ends yesterday |
| Re-run an experiment notebook | `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` | 1–3 min each |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_REGISTRY_PATH` | `<repo>/models` | Where artifacts are read from, and where own-data models are written (must be persistent and writable) |
| `ML_SERVICE_TOKEN` | unset (no auth) | Shared secret for `X-Internal-Token`. **Always set it outside local development** |
| `MLENGINE_RAW_DATA_DIR` | `<repo>/data/raw` | Location of the M5 CSVs |
| `MLENGINE_FULL_TESTS` | unset | `1` enables the full-data tests |

## Deploying the service

```bash
docker build -t mlengine-service .
docker run -d --name mlengine -p 127.0.0.1:8100:8100 \
  -e ML_SERVICE_TOKEN="$(openssl rand -hex 24)" \
  -v mlengine-models:/app/models \
  mlengine-service
```

- Bind it to a private interface or an internal network only. The backend is its only client.
- If own-data training is enabled, `MODEL_REGISTRY_PATH` must point at persistent storage, or trained models disappear when the container is replaced. With a named volume mounted at `/app/models` (as above), Docker copies the image's committed `m5-ca1-foods` models into the empty volume on first start, and own-data models persist after that.
  Caveat: the copy happens only while the volume is empty, so after releasing a new Reference version, copy its folder into the volume yourself.
- Run one service process. Training jobs are kept in memory and run one at a time, and several processes would each have their own job table.
- Health check: `GET /internal/v1/health` (it lists the loaded model versions).

## Releasing a new Reference model version

1. Change the code or config on a branch. If features change, create a new schema (see `docs/mlengine-architecture.md`).
2. Run `python -m mlengine release-m5 --version X.Y.Z`. The pipeline writes `models/m5-ca1-foods/X.Y.Z{,-bt,-dev}`.
3. Run `MLENGINE_FULL_TESTS=1 pytest`. If metrics are *meant* to change, update the expected values in `tests/test_full_data.py` in the same commit and explain why.
4. Delete superseded version folders if they should no longer be served.
5. Commit `models/`, add an entry to `CHANGELOG.md`, and update `docs/model-card-m5-ca1-foods.md`.

Versioning: a feature-schema change is **major**; retraining or a capability change is **minor**; a metadata or card fix is **patch**.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError: Missing M5 file` | Put the CSVs in `data/raw/`, or set `MLENGINE_RAW_DATA_DIR` |
| `NO_MODEL_FOR_AS_OF` | Working as intended: no version was trained only on data up to that date. Reference models cover as-of dates from 2016-03-27 |
| `DATE_OUTSIDE_CALENDAR` | The M5 calendar ends 2016-06-19; shorten the horizon or move the as-of date earlier |
| Every series returns `TOO_MANY_GAPS` | The history doesn't reach the as-of date, or zeros were sent as `null`/omitted. Send explicit `units: 0` for real zero-sales days |
| `demandClass` is always `INSUFFICIENT_DATA` | Only a short history was sent. Send up to 365 days |
| Training job `JOB_NOT_FOUND` after a deploy | Jobs live in memory; resubmit |
| Full-data tests differ in the 4th decimal | Usually a different LightGBM version or thread count. Pin `lightgbm` to the version recorded in `metadata.json` |
| Out of memory during release | Needs about 2.3 GB free; close other programs |
| `pandas`/`pyarrow` import errors | `pip install -r requirements.txt` inside the project's virtual environment |
