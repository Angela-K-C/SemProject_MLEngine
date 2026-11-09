"""Internal ML service. Only the backend calls it; it must not be exposed to the internet.

Run with ``python -m mlengine serve`` (or ``uvicorn mlengine.service.app:app``).
Settings (environment variables):

* ``MODEL_REGISTRY_PATH``: folder of model artifacts (default ``<repo>/models``).
* ``ML_SERVICE_TOKEN``: if set, every request must send it in the ``X-Internal-Token`` header.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import mlengine
from mlengine import config
from mlengine.evaluation import metrics
from mlengine.evaluation.demand_class import classify
from mlengine.models import baselines as baseline_models
from mlengine.models.artifact import ModelArtifact, ModelNotFoundError, ModelRegistry, NoModelForAsOfError
from mlengine.models.forecaster import Forecaster, ForecastRequestError, SeriesBatch
from mlengine.pipelines import own_data
from mlengine.service import schemas

log = logging.getLogger("mlengine.service")
MAX_HISTORY_DAYS = 400


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details or {}


def _check_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = os.environ.get("ML_SERVICE_TOKEN")
    if expected and x_internal_token != expected:
        raise ServiceError("UNAUTHORIZED", "Missing or wrong X-Internal-Token.", 401)


app = FastAPI(title="mlengine internal ML service", version=mlengine.__version__,
              dependencies=[Depends(_check_token)])
registry = ModelRegistry()
_forecasters: dict[str, Forecaster] = {}
_forecaster_lock = threading.Lock()
_train_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)


@app.exception_handler(ServiceError)
async def _service_error(request: Request, error: ServiceError):
    return JSONResponse(status_code=error.status, content={"error": {
        "code": error.code, "message": error.message, "details": error.details,
        "requestId": request.headers.get("X-Request-Id")}})


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, error: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {
        "code": "INVALID_REQUEST", "message": "The request body did not match the schema.",
        "details": {"errors": json.loads(json.dumps(error.errors(), default=str))},
        "requestId": request.headers.get("X-Request-Id")}})


# ---------------------------------------------------------------------------- helpers
def _forecaster(artifact: ModelArtifact) -> Forecaster:
    with _forecaster_lock:
        if artifact.ref not in _forecasters:
            _forecasters[artifact.ref] = Forecaster(artifact)
        return _forecasters[artifact.ref]


def _resolve(ref: schemas.ModelRef, as_of: dt.date) -> ModelArtifact:
    try:
        return registry.resolve(ref.id, as_of, ref.version)
    except NoModelForAsOfError as error:
        raise ServiceError("NO_MODEL_FOR_AS_OF", str(error)) from None
    except ModelNotFoundError as error:
        raise ServiceError("MODEL_NOT_FOUND", str(error), 404) from None


def _to_batch(series: list[schemas.SeriesIn], as_of: dt.date, horizon: int) -> SeriesBatch:
    earliest = min(day.date for s in series for day in s.history)
    start = max(earliest, as_of - dt.timedelta(days=MAX_HISTORY_DAYS - 1))
    n_days = (as_of - start).days + 1
    n = len(series)
    sales = np.full((n, n_days), np.nan, dtype=np.float32)
    prices = np.full((n, n_days), np.nan, dtype=np.float32)
    future = np.full((n, horizon), np.nan, dtype=np.float32)
    for i, s in enumerate(series):
        for day in s.history:
            if day.date > as_of:
                raise ServiceError("HISTORY_AFTER_AS_OF", f"{s.productId}: history has {day.date}, after the "
                                   f"as-of date {as_of}.", details={"productId": s.productId})
            col = (day.date - start).days
            if col < 0:
                continue
            if day.units is not None:
                sales[i, col] = day.units
            if day.price is not None and day.price > 0:
                prices[i, col] = day.price
        for item in s.futurePrices or []:
            col = (item.date - as_of).days - 1
            if 0 <= col < horizon and item.price is not None and item.price > 0:
                future[i, col] = item.price
    return SeriesBatch(
        product_ids=[s.productId for s in series], store_ids=[s.storeId for s in series], as_of=as_of,
        sales=sales, prices=prices, future_prices=future, categories=[s.category for s in series],
        first_price_dates=[s.firstPriceDate for s in series],
    )


def _num(value) -> float | None:
    return None if value is None or np.isnan(value) else round(float(value), 4)


def _check_horizon(artifact: ModelArtifact | None, request: schemas.ForecastRequest) -> bool:
    if artifact is None:
        return True
    capabilities = artifact.capabilities
    if request.horizonDays in capabilities["horizons"]:
        return True
    if request.allowExperimentalHorizon and request.horizonDays <= capabilities.get("experimentalMaxHorizon", 1):
        return False
    raise ServiceError("HORIZON_NOT_SUPPORTED", f"{artifact.ref} supports horizons {capabilities['horizons']}.",
                       details={"supported": capabilities["horizons"]})


# ---------------------------------------------------------------------------- endpoints
@app.get("/internal/v1/health")
def health() -> dict:
    return {"status": "ok", "mlengineVersion": mlengine.__version__,
            "models": [a.ref for a in registry.list()], "registry": str(registry.root)}


@app.get("/internal/v1/models")
def list_models() -> dict:
    return {"models": [{
        "id": a.model_id, "version": a.version, "status": a.metadata.get("status", "active"),
        "featureSchema": a.schema, "trainingEndDate": str(a.training_end_date),
        "capabilities": a.capabilities, "createdAt": a.metadata.get("createdAt"),
    } for a in registry.list()]}


@app.get("/internal/v1/models/{model_id}/resolve")
def resolve_model(model_id: str, asOfDate: dt.date) -> dict:
    artifact = _resolve(schemas.ModelRef(id=model_id), asOfDate)
    return {"id": artifact.model_id, "version": artifact.version,
            "trainingEndDate": str(artifact.training_end_date), "capabilities": artifact.capabilities}


@app.get("/internal/v1/models/{model_id}/{version}")
def model_detail(model_id: str, version: str) -> dict:
    try:
        artifact = registry.get(model_id, version)
    except ModelNotFoundError as error:
        raise ServiceError("MODEL_NOT_FOUND", str(error), 404) from None
    metadata = {k: v for k, v in artifact.metadata.items() if k != "categoricalLevels"}
    return {"metadata": metadata, "card": artifact.card}


@app.get("/internal/v1/models/{model_id}/{version}/card")
def model_card(model_id: str, version: str) -> dict:
    return model_detail(model_id, version)["card"]


@app.get("/internal/v1/models/{model_id}/{version}/backtest")
def stored_backtest(model_id: str, version: str, productId: str | None = None) -> dict:
    try:
        artifact = registry.get(model_id, version)
    except ModelNotFoundError as error:
        raise ServiceError("MODEL_NOT_FOUND", str(error), 404) from None
    summary = artifact.metadata.get("evaluation", {}).get("backtest")
    per_product = artifact.frame("backtest/per_product")
    result: dict = {"model": artifact.ref, "summary": summary}
    if productId is not None:
        if per_product is None or productId not in set(per_product["product_id"]):
            raise ServiceError("NO_STORED_BACKTEST", f"No stored backtest for {productId} in {artifact.ref}.", 404)
        row = per_product.loc[per_product["product_id"] == productId].iloc[0].to_dict()
        result["product"] = {
            "productId": productId, "n": int(row["n"]), "window": summary["window"] if summary else None,
            "demandClass": row.get("demand_class"),
            **{m: {k: _num(row[f"{m}_{k}"]) for k in ["mae", "rmse", "wape"]}
               for m in ["ai", "naive_last_value", "seasonal_naive_7"]},
        }
    return result


@app.post("/internal/v1/forecast", response_model=schemas.ForecastResponse, response_model_by_alias=True)
def forecast(request: schemas.ForecastRequest) -> schemas.ForecastResponse:
    artifact = None
    if request.includeAi:
        if request.model is None:
            raise ServiceError("MODEL_REQUIRED", "Set model, or includeAi=false for baselines only.")
        artifact = _resolve(request.model, request.asOfDate)
    validated = _check_horizon(artifact, request)
    horizon = request.horizonDays
    batch = _to_batch(request.series, request.asOfDate, horizon)

    if artifact is not None:
        try:
            output = _forecaster(artifact).forecast(batch, horizon)
        except ForecastRequestError as error:
            raise ServiceError(error.code, str(error)) from None
        ai, reasons, assumptions = output.ai, output.unavailable_reasons, output.price_assumptions
        gaps, warnings = output.missing_in_lookback, output.warnings
    else:
        ai = np.full((batch.n_series, horizon), np.nan)
        reasons = ["AI_NOT_REQUESTED"] * batch.n_series
        assumptions = ["NOT_USED"] * batch.n_series
        lookback = batch.sales[:, -config.LOOKBACK_DAYS:]
        gaps = np.isnan(lookback).sum(axis=1) + max(0, config.LOOKBACK_DAYS - batch.n_days)
        warnings = [[] for _ in range(batch.n_series)]
    baseline_values = {name: baseline_models.run(name, batch.sales, horizon) for name in request.baselines}
    dates = [request.asOfDate + dt.timedelta(days=h + 1) for h in range(horizon)]

    series_out = []
    for i, s in enumerate(request.series):
        points = [schemas.ForecastPoint(date=d, ai=_num(ai[i, h]),
                                        baselines={k: _num(v[i, h]) for k, v in baseline_values.items()})
                  for h, d in enumerate(dates)]
        series_out.append(schemas.SeriesOut(
            productId=s.productId, storeId=s.storeId, points=points, aiUnavailableReason=reasons[i],
            missingDaysInLookback=int(gaps[i]), priceAssumption=assumptions[i],
            demandClass=schemas.DemandClass(**classify(batch.sales[i])), warnings=warnings[i],
        ))
    return schemas.ForecastResponse(
        requestId=request.requestId, asOfDate=request.asOfDate,
        horizon=schemas.Horizon(startDate=dates[0], endDate=dates[-1], days=horizon, validated=validated),
        model=None if artifact is None else schemas.ResolvedModel(
            id=artifact.model_id, version=artifact.version, featureSchema=artifact.schema,
            trainingEndDate=artifact.training_end_date),
        mlengineVersion=mlengine.__version__, series=series_out,
    )


@app.post("/internal/v1/backtest")
def backtest(request: schemas.BacktestRequest) -> dict:
    """One-step-ahead backtest over [startDate, endDate]; history must include those days' actuals."""
    if request.endDate < request.startDate:
        raise ServiceError("INVALID_WINDOW", "endDate is before startDate.")
    if (request.endDate - request.startDate).days > 90:
        raise ServiceError("INVALID_WINDOW", "Backtest windows are limited to 91 days.")
    as_of_first = request.startDate - dt.timedelta(days=1)
    artifact = _resolve(request.model, as_of_first)
    forecaster = _forecaster(artifact)
    full = _to_batch(request.series, request.endDate, 1)
    window_days = (request.endDate - request.startDate).days + 1
    n, total = full.n_series, full.n_days
    actual = full.sales[:, total - window_days:]
    ai = np.full((n, window_days), np.nan)
    naive = np.full((n, window_days), np.nan)
    seasonal = np.full((n, window_days), np.nan)
    for k in range(window_days):
        cut = total - window_days + k  # columns before the target day
        step = SeriesBatch(full.product_ids, full.store_ids, as_of_first + dt.timedelta(days=k),
                           full.sales[:, :cut], full.prices[:, :cut], full.prices[:, cut:cut + 1],
                           full.categories, full.first_price_dates)
        try:
            ai[:, k] = forecaster.forecast(step, 1).ai[:, 0]
        except ForecastRequestError as error:
            raise ServiceError(error.code, str(error)) from None
        naive[:, k] = baseline_models.naive_last_value(step.sales, 1)[:, 0]
        seasonal[:, k] = baseline_models.seasonal_naive_7(step.sales, 1)[:, 0]

    def block(i=None):
        sel = slice(None) if i is None else slice(i, i + 1)
        both = ~np.isnan(ai[sel]) & ~np.isnan(naive[sel]) & ~np.isnan(actual[sel])
        return {name: metrics.summary(actual[sel][both], values[sel][both])
                for name, values in [("ai", ai), ("naive_last_value", naive), ("seasonal_naive_7", seasonal)]}

    return {
        "model": {"id": artifact.model_id, "version": artifact.version},
        "window": {"startDate": str(request.startDate), "endDate": str(request.endDate)},
        "oneStepAhead": True, "overall": block(),
        "series": [{"productId": s.productId, "storeId": s.storeId, **block(i),
                    "points": [{"date": str(request.startDate + dt.timedelta(days=k)), "actual": _num(actual[i, k]),
                                "ai": _num(ai[i, k]), "naive_last_value": _num(naive[i, k])}
                               for k in range(window_days)]}
                   for i, s in enumerate(request.series)],
    }


@app.post("/internal/v1/demand-class")
def demand_class(request: schemas.DemandClassRequest) -> dict:
    as_of = max(day.date for s in request.series for day in s.history)
    batch = _to_batch(request.series, as_of, 1)
    return {"series": [{"productId": s.productId, "storeId": s.storeId, **classify(batch.sales[i])}
                       for i, s in enumerate(request.series)]}


def _history_frame(rows: list[schemas.SalesRow]) -> pd.DataFrame:
    return pd.DataFrame([{"store_id": r.storeId, "product_id": r.productId, "category": r.category,
                          "date": r.date, "units_sold": r.units, "unit_price": r.price} for r in rows])


@app.post("/internal/v1/eligibility")
def eligibility(request: schemas.EligibilityRequest) -> dict:
    try:
        wide = own_data.to_wide(_history_frame(request.history))
    except ValueError as error:
        raise ServiceError("INVALID_HISTORY", str(error)) from None
    return {**own_data.check_eligibility(wide), "rules": config.OWN_DATA_ELIGIBILITY}


def _run_training(job_id: str, request: schemas.TrainRequest) -> None:
    job = _train_jobs[job_id]
    job["status"] = "RUNNING"
    try:
        result = own_data.train_own_data(_history_frame(request.history), request.workspaceId,
                                         request.country, request.subdivision, registry.root)
        registry.reload()
        job.update(status="COMPLETED", result=json.loads(json.dumps(result, default=str)))
    except Exception as error:  # reported to the caller; the service keeps running
        log.exception("Training job %s failed", job_id)
        job.update(status="FAILED", error={"code": "TRAINING_FAILED", "message": str(error)})
    job["finishedAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@app.post("/internal/v1/train", status_code=202)
def start_training(request: schemas.TrainRequest) -> dict:
    job_id = f"train_{uuid.uuid4().hex[:12]}"
    _train_jobs[job_id] = {"jobId": job_id, "workspaceId": request.workspaceId, "status": "QUEUED",
                           "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    _executor.submit(_run_training, job_id, request)
    return {"jobId": job_id, "status": "QUEUED", "statusUrl": f"/internal/v1/train/{job_id}"}


@app.get("/internal/v1/train/{job_id}")
def training_status(job_id: str) -> dict:
    if job_id not in _train_jobs:
        raise ServiceError("JOB_NOT_FOUND", f"No training job {job_id}.", 404)
    return _train_jobs[job_id]
