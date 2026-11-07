"""Request and response models for the internal ML service (``/internal/v1``)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlengine.models.baselines import BASELINES

BaselineId = Literal["naive_last_value", "seasonal_naive_7"]


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class ModelRef(_Model):
    id: str
    version: str | None = Field(None, description="Omit to let the leakage rule pick the version.")


class HistoryDay(_Model):
    date: dt.date
    units: float | None = Field(None, ge=0, description="Units sold; null = day not recorded.")
    price: float | None = Field(None, description="Selling price that day, if known.")


class FuturePrice(_Model):
    date: dt.date
    price: float | None = None


class SeriesIn(_Model):
    productId: str
    storeId: str
    category: str | None = None
    firstPriceDate: dt.date | None = None
    history: list[HistoryDay] = Field(..., min_length=1, max_length=1000)
    futurePrices: list[FuturePrice] | None = None


class ForecastRequest(_Model):
    requestId: str | None = None
    model: ModelRef | None = Field(None, description="Required when includeAi is true.")
    includeAi: bool = True
    asOfDate: dt.date
    horizonDays: int = Field(1, ge=1, le=28)
    allowExperimentalHorizon: bool = False
    baselines: list[BaselineId] = ["naive_last_value"]
    series: list[SeriesIn] = Field(..., min_length=1, max_length=5000)


class DemandClass(_Model):
    class_: str = Field(..., alias="class")
    label: str
    adi: float | None
    cv2: float | None
    zeroShare: float | None
    observedDays: int
    saleDays: int


class ForecastPoint(_Model):
    date: dt.date
    ai: float | None
    baselines: dict[str, float | None]


class SeriesOut(_Model):
    productId: str
    storeId: str
    points: list[ForecastPoint]
    aiUnavailableReason: str | None
    missingDaysInLookback: int
    priceAssumption: str
    demandClass: DemandClass
    warnings: list[str]


class ResolvedModel(_Model):
    id: str
    version: str
    featureSchema: str
    trainingEndDate: dt.date


class Horizon(_Model):
    startDate: dt.date
    endDate: dt.date
    days: int
    validated: bool


class ForecastResponse(_Model):
    requestId: str | None
    asOfDate: dt.date
    horizon: Horizon
    model: ResolvedModel | None
    mlengineVersion: str
    baselineLabels: dict[str, str] = {k: v for k, v in BASELINES.items()}
    series: list[SeriesOut]


class BacktestRequest(_Model):
    requestId: str | None = None
    model: ModelRef
    startDate: dt.date
    endDate: dt.date
    series: list[SeriesIn] = Field(..., min_length=1, max_length=5000)


class DemandClassRequest(_Model):
    series: list[SeriesIn] = Field(..., min_length=1, max_length=5000)


class SalesRow(_Model):
    storeId: str
    productId: str
    category: str | None = None
    date: dt.date
    units: float | None = Field(None, ge=0)
    price: float | None = None


class TrainRequest(_Model):
    workspaceId: str
    country: str | None = Field(None, description="ISO country code for the holiday calendar, e.g. KE.")
    subdivision: str | None = None
    history: list[SalesRow] = Field(..., min_length=1)


class EligibilityRequest(_Model):
    history: list[SalesRow] = Field(..., min_length=1)
