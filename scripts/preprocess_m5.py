"""Build model-ready M5 features in bounded batches and save Parquet outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IDENTITY_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
VALIDATION_HORIZON = 28
MAX_FEATURE_LAG = 56
FEATURE_START_DAY = MAX_FEATURE_LAG + 1
DEVELOPMENT_TRAIN_END = 1885
DEVELOPMENT_VALIDATION_START = 1886
DEVELOPMENT_VALIDATION_END = 1913
LATER_BACKTEST_TRAIN_END = 1913
LATER_BACKTEST_START = 1914
LATER_BACKTEST_END = 1941
COMPETITION_FORECAST_START = 1942
COMPETITION_FORECAST_END = 1969

LAG_DAYS = [28, 35, 42, 56]
ROLLING_WINDOWS = [7, 28]
ROLLING_DELAY = 28
CALENDAR_COLUMNS = [
    "date", "wm_yr_wk", "weekday_num", "month_num", "year_num",
    "day_of_month", "is_weekend", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
]
CATEGORICAL_COLUMNS = IDENTITY_COLUMNS + [
    "event_name_1", "event_type_1", "event_name_2", "event_type_2"
]
FEATURE_COLUMNS = IDENTITY_COLUMNS + [
    "d_num", "date", "wm_yr_wk", "weekday_num", "month_num", "year_num",
    "day_of_month", "is_weekend", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap", "weeks_since_first_price",
    "sell_price", "price_missing", "sales_lag_28", "sales_lag_35",
    "sales_lag_42", "sales_lag_56", "rolling_mean_7", "rolling_std_7",
    "rolling_mean_28", "rolling_std_28",
]


def read_sales(path: Path, day_limit: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    days = [name for name in header if re.fullmatch(r"d_\d+", name)]
    if day_limit is not None:
        days = [name for name in days if int(name[2:]) <= day_limit]
    numbers = [int(name[2:]) for name in days]
    if numbers != list(range(1, len(days) + 1)):
        raise ValueError(f"{path.name}: daily columns must be continuous from d_1.")
    identifiers = [name for name in IDENTITY_COLUMNS if name in header]
    dtypes = {name: "category" for name in identifiers}
    dtypes.update({name: "uint16" for name in days})
    return pd.read_csv(path, usecols=identifiers + days, dtype=dtypes, low_memory=False), days


def compare_validation_overlap(validation: pd.DataFrame, evaluation: pd.DataFrame, days: list[str]) -> None:
    key = ["item_id", "store_id"]
    if validation.duplicated(key).any() or evaluation.duplicated(key).any():
        raise ValueError("Sales data has duplicate item-store series.")
    validation = validation.sort_values(key, kind="stable").reset_index(drop=True)
    evaluation = evaluation.sort_values(key, kind="stable").reset_index(drop=True)
    if not validation[key].astype("string").equals(evaluation[key].astype("string")):
        raise ValueError("Validation and evaluation files do not contain the same item-store series.")
    for start in range(0, len(days), 128):
        columns = days[start:start + 128]
        if not np.array_equal(validation[columns].to_numpy(), evaluation[columns].to_numpy()):
            raise ValueError(f"Validation/evaluation sales differ in overlapping columns beginning {columns[0]}.")


def prepare_calendar(path: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    category_names = ["d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    dtypes = {name: "category" for name in category_names if name in header}
    calendar = pd.read_csv(path, dtype=dtypes, parse_dates=["date"], low_memory=False)
    if calendar["d"].isna().any() or calendar["d"].duplicated().any():
        raise ValueError("Calendar day IDs must be present and unique.")
    calendar["d_num"] = calendar["d"].astype(str).str.removeprefix("d_").astype("uint16")
    calendar["weekday_num"] = calendar["date"].dt.dayofweek.astype("int8")
    calendar["month_num"] = calendar["date"].dt.month.astype("int8")
    calendar["year_num"] = calendar["date"].dt.year.astype("int16")
    calendar["day_of_month"] = calendar["date"].dt.day.astype("int8")
    calendar["is_weekend"] = calendar["weekday_num"].ge(5).astype("int8")
    for name in ["snap_CA", "snap_TX", "snap_WI"]:
        if name not in calendar:
            raise ValueError(f"Calendar is missing required column {name}.")
        calendar[name] = calendar[name].astype("int8")
    for name in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if name not in calendar:
            calendar[name] = pd.Series(pd.Categorical([None] * len(calendar)))
        else:
            calendar[name] = calendar[name].astype("category")
    day_numbers = sorted(calendar["d_num"].astype(int).tolist())
    if day_numbers != list(range(1, max(day_numbers) + 1)):
        raise ValueError("Calendar day IDs must form a continuous sequence from d_1.")
    weeks = sorted(calendar["wm_yr_wk"].dropna().unique().tolist())
    week_index = {int(week): index for index, week in enumerate(weeks)}
    calendar["week_index"] = calendar["wm_yr_wk"].map(week_index).astype("int16")
    return calendar[CALENDAR_COLUMNS + ["d_num", "week_index"]], week_index


def rolling_features(sales_matrix: np.ndarray, target_days: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    # Rolling windows end 28 days before the target, never on the target day.
    prefix = np.concatenate(
        [np.zeros((sales_matrix.shape[0], 1), dtype=np.float64),
         np.cumsum(sales_matrix, axis=1, dtype=np.float64)], axis=1
    )
    square_prefix = np.concatenate(
        [np.zeros((sales_matrix.shape[0], 1), dtype=np.float64),
         np.cumsum(sales_matrix.astype(np.float64) ** 2, axis=1)], axis=1
    )
    target_indexes = target_days.astype(np.int64) - 1
    end = target_indexes - ROLLING_DELAY + 1
    start = end - window
    if start.min() < 0 or end.max() > sales_matrix.shape[1]:
        raise ValueError("Insufficient historical days for requested rolling window.")
    total = prefix[:, end] - prefix[:, start]
    square_total = square_prefix[:, end] - square_prefix[:, start]
    mean = total / window
    variance = np.maximum((square_total - (total * total / window)) / (window - 1), 0)
    return mean.astype("float32"), np.sqrt(variance).astype("float32")


def make_batch_frame(
    sales_batch: pd.DataFrame,
    sales_days: list[str],
    target_days: np.ndarray,
    calendar_features: pd.DataFrame,
    prices: pd.DataFrame,
    release_weeks: pd.DataFrame,
    week_index: dict[int, int],
    category_values: dict[str, list[str]],
) -> pd.DataFrame:
    series_count = len(sales_batch)
    sales_matrix = np.ascontiguousarray(sales_batch[sales_days].to_numpy(dtype="float32", copy=True))
    frame_data: dict[str, object] = {}
    for name in IDENTITY_COLUMNS:
        values = sales_batch[name].astype("string").to_numpy(dtype=object)
        frame_data[name] = pd.Categorical(np.tile(values, len(target_days)), categories=category_values[name])
    frame_data["d_num"] = np.repeat(target_days, series_count).astype("uint16")
    for lag in LAG_DAYS:
        source_indexes = target_days.astype(np.int64) - 1 - lag
        if source_indexes.min() < 0 or source_indexes.max() >= len(sales_days):
            raise ValueError(f"Not enough observed history for lag {lag}.")
        lag_values = sales_matrix[:, source_indexes]
        frame_data[f"sales_lag_{lag}"] = lag_values.T.reshape(-1)
    for window in ROLLING_WINDOWS:
        mean, std = rolling_features(sales_matrix, target_days, window)
        frame_data[f"rolling_mean_{window}"] = mean.T.reshape(-1)
        frame_data[f"rolling_std_{window}"] = std.T.reshape(-1)
    observed = target_days <= len(sales_days)
    labels = np.full((series_count, len(target_days)), np.nan, dtype="float32")
    labels[:, observed] = sales_matrix[:, target_days[observed] - 1]
    frame_data["sales"] = labels.T.reshape(-1)
    frame = pd.DataFrame(frame_data)

    before_calendar = len(frame)
    frame = frame.merge(calendar_features, on="d_num", how="left", validate="many_to_one", sort=False)
    if len(frame) != before_calendar or frame["wm_yr_wk"].isna().any():
        raise ValueError("Calendar join lost days or duplicated sales observations.")

    state_values = frame["state_id"].astype("string").to_numpy()
    snap = np.full(len(frame), -1, dtype=np.int8)
    for state, column in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
        mask = state_values == state
        snap[mask] = frame.loc[mask, column].to_numpy(dtype="int8")
    if (snap < 0).any():
        raise ValueError("Could not map a store state to its SNAP calendar column.")
    frame["snap"] = snap
    frame.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)

    price_key = ["store_id", "item_id", "wm_yr_wk"]
    before_price = len(frame)
    frame = frame.merge(prices[price_key + ["sell_price"]], on=price_key, how="left", validate="many_to_one", sort=False)
    if len(frame) != before_price:
        raise ValueError("Price join duplicated sales observations.")
    frame["price_missing"] = frame["sell_price"].isna().astype("int8")

    before_release = len(frame)
    frame = frame.merge(release_weeks, on=["store_id", "item_id"], how="left", validate="many_to_one", sort=False)
    if len(frame) != before_release:
        raise ValueError("Release-week join duplicated sales observations.")
    frame["release_week_index"] = frame["first_price_week"].map(week_index).astype("float32")
    frame["weeks_since_first_price"] = (frame["week_index"].astype("float32") - frame["release_week_index"]).astype("float32")
    frame.drop(columns=["first_price_week", "release_week_index", "week_index"], inplace=True)

    # Keep categorical features as stable dictionaries in Parquet across all batches.
    for name in CATEGORICAL_COLUMNS:
        if name in frame:
            categories = category_values.get(name)
            if categories is None:
                categories = calendar_features[name].cat.categories.astype(str).tolist()
            frame[name] = pd.Categorical(frame[name].astype("string"), categories=categories)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["sell_price"] = frame["sell_price"].astype("float32")
    frame["weeks_since_first_price"] = frame["weeks_since_first_price"].astype("float32")
    return frame


class ParquetOutputs:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.paths: dict[str, Path] = {}

    def append(self, name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        path = self.output_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(path, table.schema, compression="zstd", compression_level=5)
            self.paths[name] = path
        elif not table.schema.equals(self.writers[name].schema, check_metadata=False):
            raise ValueError(f"Parquet schema changed while writing {name}.")
        self.writers[name].write_table(table, row_group_size=65536)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def build_processed_data(project_root: Path, batch_size: int = 250, series_limit: int | None = None, output_name: str = "model_ready") -> Path:
    raw = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "labeled_history.parquet", "development_train_features.parquet",
        "development_validation_features.parquet", "development_validation_targets.parquet",
        "later_backtest_features.parquet", "later_backtest_targets.parquet",
        "competition_forecast_features.parquet",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    validation_path = raw / "sales_train_validation.csv"
    evaluation_path = raw / "sales_train_evaluation.csv"
    calendar_path = raw / "calendar.csv"
    prices_path = raw / "sell_prices.csv"
    for path in [validation_path, evaluation_path, calendar_path, prices_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    validation_sales, validation_days = read_sales(validation_path, day_limit=1913)
    evaluation_sales, evaluation_days = read_sales(evaluation_path)
    if len(validation_days) != 1913 or len(evaluation_days) != 1941:
        raise ValueError("Expected validation data through d_1913 and evaluation data through d_1941.")
    compare_validation_overlap(validation_sales, evaluation_sales, validation_days)
    validation_series_count = len(validation_sales)
    validation_zero_count = int((validation_sales[validation_days].to_numpy() == 0).sum())
    validation_duplicate_count = int(validation_sales.duplicated(["item_id", "store_id"]).sum())
    if series_limit is not None:
        grouped_series = list(evaluation_sales.groupby(["store_id", "cat_id"], observed=True, sort=True))
        base_count, remainder = divmod(series_limit, len(grouped_series))
        selected = [group.head(base_count + (index < remainder))
                    for index, (_, group) in enumerate(grouped_series)
                    if base_count + (index < remainder) > 0]
        evaluation_sales = pd.concat(selected, ignore_index=True)
        if len(evaluation_sales) < series_limit:
            raise ValueError("Pilot sample request exceeds the available stratified series count.")
        evaluation_sales = evaluation_sales.iloc[:series_limit].copy()

    calendar_features, week_index = prepare_calendar(calendar_path)
    if int(calendar_features["d_num"].max()) < COMPETITION_FORECAST_END:
        raise ValueError("Calendar does not cover the final 28-day forecast horizon through d_1969.")

    price_header = pd.read_csv(prices_path, nrows=0).columns.tolist()
    price_dtypes = {name: "category" for name in ["store_id", "item_id"] if name in price_header}
    price_dtypes["wm_yr_wk"] = "int32"
    price_dtypes["sell_price"] = "float32"
    prices = pd.read_csv(prices_path, dtype=price_dtypes, low_memory=False)
    price_key = ["store_id", "item_id", "wm_yr_wk"]
    if prices[price_key].isna().any().any() or prices.duplicated(price_key).any():
        raise ValueError("Price keys contain missing or duplicate store-item-week values.")
    release_weeks = (
        prices.groupby(["store_id", "item_id"], observed=True, as_index=False)["wm_yr_wk"]
        .min().rename(columns={"wm_yr_wk": "first_price_week"})
    )
    for name in ["store_id", "item_id"]:
        release_weeks[name] = release_weeks[name].astype("category")

    category_values = {
        name: evaluation_sales[name].cat.categories.astype(str).tolist()
        for name in IDENTITY_COLUMNS
    }
    for name in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        category_values[name] = calendar_features[name].cat.categories.astype(str).tolist()
    evaluation_days = [f"d_{number}" for number in range(1, 1942)]
    target_days = np.arange(FEATURE_START_DAY, COMPETITION_FORECAST_END + 1, dtype=np.int16)

    outputs = ParquetOutputs(output_dir)
    price_missing = {"dev_train": 0, "dev_validation": 0, "later_backtest": 0, "competition_forecast": 0}
    row_counts = {"dev_train": 0, "dev_validation": 0, "later_backtest": 0, "competition_forecast": 0}
    processed = 0
    try:
        for start in range(0, len(evaluation_sales), batch_size):
            batch = evaluation_sales.iloc[start:start + batch_size]
            features = make_batch_frame(
                batch, evaluation_days, target_days, calendar_features, prices,
                release_weeks, week_index, category_values
            )
            day = features["d_num"]
            train_mask = day.between(FEATURE_START_DAY, DEVELOPMENT_TRAIN_END)
            dev_train = features.loc[train_mask, FEATURE_COLUMNS + ["sales"]]
            outputs.append("development_train_features", dev_train)
            row_counts["dev_train"] += len(dev_train)
            price_missing["dev_train"] += int(dev_train["price_missing"].sum())
            dev_mask = day.between(DEVELOPMENT_VALIDATION_START, DEVELOPMENT_VALIDATION_END)
            dev_features = features.loc[dev_mask, FEATURE_COLUMNS]
            dev_labels = features.loc[dev_mask, IDENTITY_COLUMNS + ["d_num", "sales"]].rename(columns={"sales": "target_sales"})
            outputs.append("development_validation_features", dev_features)
            outputs.append("development_validation_targets", dev_labels)
            row_counts["dev_validation"] += len(dev_features)
            price_missing["dev_validation"] += int(dev_features["price_missing"].sum())

            late_mask = day.between(LATER_BACKTEST_START, LATER_BACKTEST_END)
            late_features = features.loc[late_mask, FEATURE_COLUMNS]
            late_labels = features.loc[late_mask, IDENTITY_COLUMNS + ["d_num", "sales"]].rename(columns={"sales": "target_sales"})
            outputs.append("later_backtest_features", late_features)
            outputs.append("later_backtest_targets", late_labels)
            row_counts["later_backtest"] += len(late_features)
            price_missing["later_backtest"] += int(late_features["price_missing"].sum())

            forecast_mask = day.between(COMPETITION_FORECAST_START, COMPETITION_FORECAST_END)
            forecast_features = features.loc[forecast_mask, FEATURE_COLUMNS]
            outputs.append("competition_forecast_features", forecast_features)
            row_counts["competition_forecast"] += len(forecast_features)
            price_missing["competition_forecast"] += int(forecast_features["price_missing"].sum())


            processed += len(batch)
            print(f"Processed series {processed:,}/{len(evaluation_sales):,}", flush=True)
    finally:
        outputs.close()

    expected = {
        "dev_train": len(evaluation_sales) * (DEVELOPMENT_TRAIN_END - FEATURE_START_DAY + 1),
        "dev_validation": len(evaluation_sales) * (DEVELOPMENT_VALIDATION_END - DEVELOPMENT_VALIDATION_START + 1),
        "later_backtest": len(evaluation_sales) * (LATER_BACKTEST_END - LATER_BACKTEST_START + 1),
        "competition_forecast": len(evaluation_sales) * (COMPETITION_FORECAST_END - COMPETITION_FORECAST_START + 1),
    }
    if row_counts != expected:
        raise ValueError(f"Unexpected row counts after processing: actual={row_counts}, expected={expected}")

    quality_rows = [
        ("validation series", validation_series_count, "Expected unique item-store histories."),
        ("evaluation series processed", len(evaluation_sales), "Pilot runs use a stratified subset; final run uses all series."),
        ("zero sales cells in validation history", validation_zero_count, "Valid observations; retained."),
        ("duplicate item-store keys in validation", validation_duplicate_count, "Expected zero."),
        ("overlapping sales days matched", len(validation_days), "Validation and evaluation sales match day by day."),
        ("calendar days", len(calendar_features), "Must cover d_1 through d_1969."),
        ("duplicate price keys", int(prices.duplicated(price_key).sum()), "Expected zero."),
        ("missing listed sell_price values", int(prices["sell_price"].isna().sum()), "Investigate; values are not filled."),
        ("nonpositive listed sell_price values", int(prices["sell_price"].le(0).sum()), "Investigate; values are not changed."),
        ("missing price joins in development validation", price_missing["dev_validation"], "Reported; not replaced."),
        ("missing price joins in later backtest", price_missing["later_backtest"], "Reported; not replaced."),
        ("missing price joins in competition forecast", price_missing["competition_forecast"], "Reported; not replaced."),
        ("development training rows written", row_counts["dev_train"], "Targets d_57 through d_1885; d_1–d_56 are feature warm-up."),
        ("development validation rows written", row_counts["dev_validation"], "Features and labels saved separately for d_1886–d_1913."),
        ("later backtest rows written", row_counts["later_backtest"], "Features and labels saved separately for d_1914–d_1941."),
        ("competition forecast rows written", row_counts["competition_forecast"], "Inputs only for d_1942–d_1969."),
    ]
    pd.DataFrame(quality_rows, columns=["check", "result", "interpretation"]).to_csv(output_dir / "data_quality_report.csv", index=False)

    definitions = [
        ("sales", "Target", "Observed units sold on d_num; kept separate from validation and forecast inputs."),
        ("sales_lag_28/35/42/56", "Sales history", "Observed sales 28, 35, 42, and 56 days before the target day."),
        ("rolling_mean_7/28", "Sales history", "Mean sales over 7/28 days ending 28 days before the target."),
        ("rolling_std_7/28", "Sales history", "Sample standard deviation over 7/28 days ending 28 days before the target."),
        ("weekday_num, month_num, year_num, day_of_month, is_weekend", "Calendar", "Calendar values for the target date."),
        ("event_name_1/event_type_1/event_name_2/event_type_2", "Calendar", "Event fields for the target date; blank means no listed event."),
        ("snap", "Calendar", "SNAP indicator selected for the state of the target store."),
        ("sell_price", "Price", "Listed price for the target item's store and calendar week; missing values stay missing."),
        ("price_missing", "Price", "1 when no sell_price matched the target store-item-week, otherwise 0."),
        ("weeks_since_first_price", "Release proxy", "Calendar week index minus the first week with a listed store-item price; may be negative before first listing."),
        ("id, item_id, dept_id, cat_id, store_id, state_id", "Identifiers", "Product and store labels; convert to categorical features for model training."),
        ("d_num, date, wm_yr_wk", "Time keys", "Integer day ID, target date, and week ID used for ordering and joins."),
    ]
    pd.DataFrame(definitions, columns=["feature", "group", "definition"]).to_csv(output_dir / "feature_dictionary.csv", index=False)

    manifest = {
        "sales_source": "sales_train_evaluation.csv (overlap checked against sales_train_validation.csv)",
        "series_count": len(evaluation_sales),
        "feature_start_day": FEATURE_START_DAY,
        "validation_horizon_days": VALIDATION_HORIZON,
        "forecast_strategy": "One direct model with 28-day-shifted sales history; each target's lag and rolling inputs end no later than target day minus 28. No validation target values are used as features.",
        "development": {"train_file": "development_train_features.parquet", "train_days": [FEATURE_START_DAY, DEVELOPMENT_TRAIN_END], "validation_features": "development_validation_features.parquet", "validation_targets": "development_validation_targets.parquet", "validation_days": [1886, 1913]},
        "later_backtest": {"train_with": "development_train_features.parquet plus development_validation_features.parquet joined to development_validation_targets.parquet", "train_days": [FEATURE_START_DAY, LATER_BACKTEST_TRAIN_END], "features": "later_backtest_features.parquet", "targets": "later_backtest_targets.parquet", "days": [1914, 1941]},
        "final_fit": {"train_with": "all labeled rows through d_1941, joining separate target files", "forecast_features": "competition_forecast_features.parquet", "forecast_days": [1942, 1969]},
        "train_file": "development_train_features.parquet",
        "feature_columns": FEATURE_COLUMNS,
        "label_column": "sales",
        "missing_price_policy": "Preserve missing prices and include price_missing; do not replace with zero or an average.",
        "zero_sales_policy": "Retain zero sales as valid observations.",
        "files": {name: path.name for name, path in outputs.paths.items()},
        "batch_size_series": batch_size,
        "pilot_only": series_limit is not None,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed: {output_dir}")
    for path in sorted(output_dir.glob("*.parquet")):
        print(f"{path.name}: {path.stat().st_size / (1024 ** 3):.2f} GiB")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--series-limit", type=int, default=None, help="Build a pilot subset only; never use it as the final dataset.")
    parser.add_argument("--output-name", default="model_ready")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.series_limit is not None and args.series_limit < 1:
        parser.error("--series-limit must be positive")
    build_processed_data(args.project_root.resolve(), args.batch_size, args.series_limit, args.output_name)


if __name__ == "__main__":
    main()






