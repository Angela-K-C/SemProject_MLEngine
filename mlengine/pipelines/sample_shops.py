"""Sample business CSVs for demoing own-data workspaces, cut from a real M5 store.

The Reference model uses only CA_1. These samples come from a different store (TX_1 by default), so
an own-data model trained on them has never seen the Reference data. Each file looks like an
ordinary shop's sales export, and is uploaded through the app's import wizard.

Dates are shifted forward by a whole number of weeks, so the history ends on ``end_date`` (default:
yesterday) and weekdays stay aligned. Public holidays don't line up after the shift; the manifest
says so.

Files written to ``data/processed/samples/`` (git-ignored):

* ``sample_shop_eligible.csv``: 100 products × 365 days. Days with no sales are left out, so the
  wizard answer is "Yes, no row means zero sales". Big enough to train its own model.
* ``sample_shop_small.csv``: 30 products × 60 days, zeros written out. Not eligible yet
  (fewer than 120 days), so it shows the progress-to-eligibility screens and baseline-only forecasts.
* ``sample_shop_messy.csv``: 15 products × 45 days in a messy format (semicolons, DD/MM/YYYY, names
  only, split transactions), with deliberate problems, for demoing the validation report.
* ``manifest.json`` and ``README.md``: provenance, wizard answers and the expected validation outcome.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlengine import config
from mlengine.data.m5 import daily_price_matrix, load_m5_scope

SOURCE = "M5 Forecasting – Accuracy (Kaggle), Walmart store {store}, category {category}"


def _aligned_end(target_end: dt.date, last_m5: dt.date) -> dt.date:
    """Latest M5 day on or before last_m5 with the same weekday as target_end."""
    return last_m5 - dt.timedelta(days=(last_m5.weekday() - target_end.weekday()) % 7)


def _pick(sales: np.ndarray, prices: np.ndarray, n: int, rng: np.random.Generator, min_sale_days: int) -> np.ndarray:
    """Products that sold on enough days in the window and in its last 4 weeks (still listed)."""
    sale_days = (sales > 0).sum(axis=1)
    recent = (sales[:, -28:] > 0).sum(axis=1)
    priced_at_end = ~np.isnan(prices[:, -1])
    candidates = np.flatnonzero((sale_days >= min_sale_days) & (recent >= 3) & priced_at_end)
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} products qualify; asked for {n}.")
    return np.sort(rng.choice(candidates, size=n, replace=False))


def _rows(scope, prices, items, start_col, end_col, shift, include_zeros, sku_prefix):
    dates = [config.m5_day_to_date(c + 1) + shift for c in range(start_col, end_col + 1)]
    out = []
    for number, i in enumerate(items, start=1):
        item_id = scope.identity["item_id"].iat[i]
        dept = scope.identity["dept_id"].iat[i]
        for j, col in enumerate(range(start_col, end_col + 1)):
            price = prices[i, col]
            if np.isnan(price):
                continue  # not on sale yet (no price that week)
            units = int(scope.sales[i, col])
            if units == 0 and not include_zeros:
                continue
            out.append({"sales_date": dates[j].isoformat(), "product_id": f"{sku_prefix}-{number:04d}",
                        "product_name": f"Item {item_id}", "category": dept.replace("_", " ").title(),
                        "units_sold": units, "unit_price": round(float(price), 2)})
    return pd.DataFrame(out)


def _messy(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[str, dict]:
    """Semicolon CSV, DD/MM/YYYY dates, names only, some days split into two transactions, plus problems."""
    frame = frame.copy()
    frame["sales_date"] = pd.to_datetime(frame["sales_date"])
    lines, split = [], 0
    for row in frame.itertuples(index=False):
        date = row.sales_date.strftime("%d/%m/%Y")
        if row.units_sold >= 2 and rng.random() < 0.1:  # one sale recorded as two transactions
            first = int(row.units_sold) // 2
            lines.append([date, row.product_name, row.category, first, row.unit_price])
            lines.append([date, row.product_name, row.category, int(row.units_sold) - first, row.unit_price])
            split += 1
        else:
            lines.append([date, row.product_name, row.category, int(row.units_sold), row.unit_price])
    last = frame["sales_date"].max()
    name, category = frame["product_name"].iat[0], frame["category"].iat[0]
    problems = [
        ["31/02/" + str(last.year), name, category, 3, 2.5],                               # INVALID_DATE
        [last.strftime("%d/%m/%Y"), "Item EXTRA_1", category, -2, 1.0],                     # NEGATIVE_UNITS
        [last.strftime("%d/%m/%Y"), "Item EXTRA_2", category, "", 1.0],                     # MISSING_UNITS
        [(last + pd.Timedelta(days=30)).strftime("%d/%m/%Y"), "Item EXTRA_3", category, 1, 1.0],  # FUTURE_DATE
        [last.strftime("%d/%m/%Y"), "", category, 1, 1.0],                                  # MISSING_PRODUCT
        [last.strftime("%d/%m/%Y"), "Item EXTRA_4", category, 2, "free"],                   # INVALID_PRICE
    ]
    # A spike: one product's last day becomes 50x its typical sales (kept, but flagged).
    # The product that sells on most days, so its typical (median) daily sales are above zero.
    spike_name = frame["product_name"].value_counts().idxmax()
    spike_units = int(max(frame.loc[frame["product_name"] == spike_name, "units_sold"].median(), 1) * 50)
    spike_day = (last - pd.Timedelta(days=1)).strftime("%d/%m/%Y")
    spike_line = next((line for line in lines if line[1] == spike_name and line[0] == spike_day), None)
    if spike_line is None:
        price = frame.loc[frame["product_name"] == spike_name, "unit_price"].iat[-1]
        lines.append([spike_day, spike_name, category, spike_units, price])
    else:
        spike_line[3] = spike_units
    lines.extend(problems)
    text = "Date;Item Name;Department;Qty;Price\n" + "\n".join(";".join(str(v) for v in line) for line in lines)
    expected = {"errorsByCode": {"INVALID_DATE": 1, "NEGATIVE_UNITS": 1, "MISSING_UNITS": 1, "FUTURE_DATE": 1,
                                 "MISSING_PRODUCT": 1},
                "warningsIncluded": ["DUPLICATE_ROWS_SUMMED", "SPIKE", "INVALID_PRICE"], "splitDays": split}
    return text + "\n", expected


def export_samples(out_dir: Path | None = None, raw_dir: Path | None = None, store_id: str = "TX_1",
                   category_id: str = "FOODS", end_date: dt.date | None = None, seed: int = 7) -> dict[str, Path]:
    out_dir = Path(out_dir or config.project_root() / "data" / "processed" / "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    end_date = end_date or dt.date.today() - dt.timedelta(days=1)
    scope = load_m5_scope(raw_dir, store_id=store_id, category_id=category_id, verify_overlap=False)
    prices = daily_price_matrix(scope)[:, : scope.n_days]
    last_m5 = config.m5_day_to_date(scope.n_days)
    m5_end = _aligned_end(end_date, last_m5)
    shift = end_date - m5_end
    assert shift.days % 7 == 0
    end_col = config.date_to_m5_day(m5_end) - 1
    rng = np.random.default_rng(seed)

    specs = {
        "sample_shop_eligible": {"products": 100, "days": 365, "zeros": False, "prefix": "SKU",
                                 "missingDaysAreZero": True,
                                 "purpose": "Big enough for its own AI model (Train → gate → AI forecasts)."},
        "sample_shop_small": {"products": 30, "days": 60, "zeros": True, "prefix": "SM",
                              "missingDaysAreZero": False,
                              "purpose": "Too little history for its own model yet: progress screen and baselines."},
        "sample_shop_messy": {"products": 15, "days": 45, "zeros": False, "prefix": "MX",
                              "missingDaysAreZero": True,
                              "purpose": "Messy export with deliberate problems, for the validation report."},
    }
    written: dict[str, Path] = {}
    manifest = {"source": SOURCE.format(store=store_id, category=category_id), "storeId": store_id,
                "categoryId": category_id, "originalEndDate": str(m5_end), "endDate": str(end_date),
                "dateShiftDays": shift.days, "seed": seed,
                "note": ("Real Walmart daily sales with dates moved forward by whole weeks so the history ends "
                         "yesterday. Weekdays line up; public holidays do not."),
                "suggestedStore": {"country": "US", "subdivision": store_id.split("_")[0], "currency": "USD",
                                   "timezone": "America/Chicago" if store_id.startswith("TX") else "America/New_York"},
                "files": {}}
    for name, spec in specs.items():
        start_col = end_col - spec["days"] + 1
        items = _pick(scope.sales[:, start_col:end_col + 1], prices[:, start_col:end_col + 1], spec["products"], rng,
                      min_sale_days=max(10, spec["days"] // 6))
        frame = _rows(scope, prices, items, start_col, end_col, shift, spec["zeros"], spec["prefix"])
        path = out_dir / f"{name}.csv"
        info = {"products": int(frame["product_id"].nunique()), "days": spec["days"], "rows": len(frame),
                "dateRange": [frame["sales_date"].min(), frame["sales_date"].max()],
                "m5Items": sorted(scope.identity["item_id"].iloc[items].tolist()), "purpose": spec["purpose"],
                "wizard": {"missingDaysAreZero": spec["missingDaysAreZero"]}}
        if name == "sample_shop_messy":
            text, expected = _messy(frame, rng)
            path.write_text(text)
            info["wizard"].update({"dateFormat": "DD/MM/YYYY", "columns": {
                "salesDate": "Date", "productName": "Item Name", "category": "Department", "unitsSold": "Qty",
                "unitPrice": "Price"}})
            info["expectedValidation"] = expected
        else:
            frame.to_csv(path, index=False)
            info["wizard"]["columns"] = {"salesDate": "sales_date", "productId": "product_id",
                                         "productName": "product_name", "category": "category",
                                         "unitsSold": "units_sold", "unitPrice": "unit_price"}
            info["observedProductDays"] = _observed_product_days(frame, spec["zeros"])
        manifest["files"][path.name] = info
        written[name] = path
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "README.md").write_text(_readme(manifest))
    written["manifest"] = out_dir / "manifest.json"
    return written


def _observed_product_days(frame: pd.DataFrame, has_zeros: bool) -> int:
    """Product-days the backend will hold after import (zero-filled from each product's first row)."""
    if has_zeros:
        return len(frame)
    dates = pd.to_datetime(frame["sales_date"])
    last = dates.max()
    firsts = dates.groupby(frame["product_id"]).min()
    return int(((last - firsts).dt.days + 1).sum())


def _readme(manifest: dict) -> str:
    lines = ["# Sample shops", "", manifest["note"], "", f"Source: {manifest['source']}. "
             f"Dates shifted by {manifest['dateShiftDays']} days (original end {manifest['originalEndDate']}).", "",
             "Create a store with country `US`, subdivision `" + manifest["suggestedStore"]["subdivision"] + "`, "
             "currency `USD`, then upload a file through the import wizard with these answers:", "",
             "Real sales have real spikes, so the validation report flags some days as SPIKE and the import shows "
             "Needs Attention. That's expected: spikes are kept as they are, and you can import the file.", ""]
    for name, info in manifest["files"].items():
        answer = "Yes" if info["wizard"]["missingDaysAreZero"] else "No"
        lines += [f"## `{name}`", "", info["purpose"], "",
                  f"- {info['products']} products, {info['days']} days, {info['rows']:,} rows, "
                  f"{info['dateRange'][0]} to {info['dateRange'][1]}",
                  f"- \"No row means zero sales?\" → **{answer}**"]
        if "dateFormat" in info["wizard"]:
            lines.append(f"- Date format: **{info['wizard']['dateFormat']}**")
        if "expectedValidation" in info:
            lines.append(f"- Expected rejected rows: {info['expectedValidation']['errorsByCode']}")
        lines.append("")
    return "\n".join(lines)
