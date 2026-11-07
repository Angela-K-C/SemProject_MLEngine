"""Demand classification (Syntetos–Boylan) from ADI and CV², as used in the EDA."""

from __future__ import annotations

import numpy as np

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
MIN_OBSERVED_DAYS = 30
MIN_SALE_DAYS = 5
CLASS_WINDOW_DAYS = 365

LABELS = {
    "SMOOTH": "Steady seller",
    "ERRATIC": "Sells often, amounts vary a lot",
    "INTERMITTENT": "Sells on some days only",
    "LUMPY": "Irregular and unpredictable",
    "INSUFFICIENT_DATA": "Not enough history to classify",
}


def classify(history) -> dict:
    """Classify one product from its recent daily units (NaN = unrecorded day, ignored)."""
    values = np.asarray(history, dtype=np.float64)[-CLASS_WINDOW_DAYS:]
    values = values[~np.isnan(values)]
    observed = int(values.size)
    positive = values[values > 0]
    sale_days = int(positive.size)
    result = {"observedDays": observed, "saleDays": sale_days,
              "zeroShare": float(1 - sale_days / observed) if observed else None,
              "adi": None, "cv2": None}
    if observed < MIN_OBSERVED_DAYS or sale_days < MIN_SALE_DAYS:
        result["class"] = "INSUFFICIENT_DATA"
    else:
        adi = observed / sale_days
        std = positive.std(ddof=1) if sale_days > 1 else 0.0
        cv2 = float((std / positive.mean()) ** 2)
        result.update(adi=float(adi), cv2=cv2)
        if adi < ADI_CUTOFF:
            result["class"] = "SMOOTH" if cv2 < CV2_CUTOFF else "ERRATIC"
        else:
            result["class"] = "INTERMITTENT" if cv2 < CV2_CUTOFF else "LUMPY"
    result["label"] = LABELS[result["class"]]
    return result


def classify_matrix(sales_matrix: np.ndarray) -> list[dict]:
    return [classify(row) for row in np.asarray(sales_matrix)]
