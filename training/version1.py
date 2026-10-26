import gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# 1. Configuration
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SALES_FILE = RAW_DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"

MAX_SERIES = 300
VALIDATION_START, VALIDATION_END = 1914, 1941

# 2. Loading and Sampling
calendar = pd.read_csv(CALENDAR_FILE)
sales = pd.read_csv(SALES_FILE)

if MAX_SERIES and MAX_SERIES < len(sales):
    sales = sales.sample(n=MAX_SERIES, random_state=42).reset_index(drop=True)

# 3. Reshape (Melt) Wide to Long
day_cols = [col for col in sales.columns if col.startswith("d_")]
df = pd.melt(
    sales,
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_vars=day_cols, var_name="d", value_name="sales"
)
del sales; gc.collect()

# 4. Merges & Downcasting
df = df.merge(calendar.drop(columns=["date"], errors="ignore"), on="d", how="left")

prices = pd.read_csv(PRICES_FILE)
prices = prices[prices["item_id"].isin(df["item_id"].unique()) & prices["store_id"].isin(df["store_id"].unique())]
df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
del calendar, prices; gc.collect()

# 5. Extract Time & Convert Categoricals
df["day_num"] = df["d"].str.extract(r"(\d+)")[0].astype(np.int32)
df = df.sort_values(["id", "day_num"]).reset_index(drop=True)

cat_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", 
            "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# 6. Feature Engineering (Lags and Rollings)
df["sales_lag_28"] = df.groupby("id", observed=True)["sales"].shift(28)
df["rolling_mean_7"] = df.groupby("id", observed=True)["sales"].transform(
    lambda s: s.shift(28).rolling(7, min_periods=7).mean()
)

# 7. Chronological Split
train_df = df[df["day_num"] < VALIDATION_START].dropna(subset=["sales_lag_28", "rolling_mean_7"])
val_df = df[df["day_num"].between(VALIDATION_START, VALIDATION_END)].dropna(subset=["sales_lag_28", "rolling_mean_7"])

feature_cols = [c for c in df.columns if c not in {"sales", "d", "day_num"}]
X_train, y_train = train_df[feature_cols], train_df["sales"]
X_val, y_val = val_df[feature_cols], val_df["sales"]

# 8. Train LightGBM
model = lgb.LGBMRegressor(
    objective="regression", n_estimators=1000, learning_rate=0.05, 
    num_leaves=31, n_jobs=-1, random_state=42, verbosity=-1
)
# Fixed the warning by using eval_X and eval_y instead of eval_set
model.fit(
    X_train, y_train,
    eval_X=X_val, eval_y=y_val,
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(25)]
)

# 9. Evaluate
preds = model.predict(X_val, num_iteration=model.best_iteration_)
baseline_preds = X_val["sales_lag_28"].to_numpy()

print(f"\nLightGBM RMSE: {root_mean_squared_error(y_val, preds):.4f}")
print(f"Lag-28 RMSE:   {root_mean_squared_error(y_val, baseline_preds):.4f}")