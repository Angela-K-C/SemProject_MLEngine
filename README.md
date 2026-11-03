# M5 Retail Demand Forecasting

This project explores the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) retail sales dataset and builds next-day LightGBM demand forecasts for California FOODS items. It started as a 500-series mixed-category sample and has since been narrowed to CA_1/FOODS, moved to short-horizon lags and a Tweedie objective, and used to test several modelling variants (hurdle, bucketed, pooled). The results are summarised below.

## Modelling journey

The model went through six stages. Each stage changed one thing and was scored against the stage before it, so the gains and the dead ends are both on record. **Stage 3 is the best model so far**, and it is marked below.

All CA_1/FOODS numbers use the same chronological windows: a development validation window (`d_1886`–`d_1913`) and a later backtest (`d_1914`–`d_1941`, refit on all labelled history before it). Lower is better.

### Stage 1: LightGBM baseline on a mixed 500-series sample

[`LightGBM_BaseColab`](notebooks/LightGBM_BaseColab.ipynb) / [`LightGBM_BaseLocal`](notebooks/LightGBM_BaseLocal.ipynb) began with a 500-series sample across all stores and categories. It used 28-day-embargoed lag and rolling features, which match the M5 competition's 28-day-ahead setting, and a standard L2 objective, scored against a naive lag-28 baseline. LightGBM reached RMSE 1.832 and beat the baseline by about 28–31%. The 500-series cap was needed to keep the `pd.melt` reshaping within memory.

### Stage 2: Scope down to CA_1/FOODS and compare models

Narrowing to one store and one dense category (1,437 series) gave a cleaner, more realistic target, and building the lags on the wide matrix instead of using `pd.melt` let every series run in one pass without the cap.

| Model (28-day lags, L2) | Dev RMSE | Dev MAE | Backtest RMSE | Backtest MAE |
| --- | --- | --- | --- | --- |
| Naive lag-28 baseline | 3.386 | | | |
| Ridge | 2.689 | | | |
| Random Forest | 2.592 | | | |
| **LightGBM** | **2.591** | **1.454** | **2.751** | **1.506** |

- The raw error went *up* against Stage 1, but that is a scale effect: CA_1/FOODS sells 2.26 units/day against 1.23 for the old sample. Scale-adjusted, it is better (WAPE 66.7% vs. 93.8%).
- **Model choice mattered far less than features.** LightGBM, Random Forest and Ridge landed within about 4% of each other, so the features carry most of the signal.

### Stage 3: Short-horizon lags and a Tweedie objective (best model so far)

Two changes: next-day lags (1/7/14/21 days) replaced the 28-day-embargoed ones, and the objective switched from L2 to Tweedie, which suits zero-heavy, right-skewed unit sales. The baseline became "tomorrow looks like today" (`sales_lag_1`, RMSE 3.194).

| Metric | Stage 2 (28-day lags, L2) | **Stage 3 (1-day lags, Tweedie)** | Change |
| --- | --- | --- | --- |
| Dev RMSE | 2.591 | **2.356** | −9.1% |
| Dev MAE | 1.454 | **1.308** | −10.0% |
| Backtest RMSE | 2.751 | **2.377** | −13.6% |
| Backtest MAE | 1.506 | **1.377** | −8.6% |

- **Recency of the lag features was the biggest lever tried.** It gave a larger gain than any change of model or architecture, before or since.
- The gain over the naive baseline widened to about 26% RMSE even though the lag-1 baseline is stronger than lag-28.
- Trade-off: this only makes sense if the app needs next-day forecasts. The 28-day-ahead design in Stage 2 remains the right one for a month-ahead view.

### Stage 4: Hurdle model, classify first and then regress (no gain)

[`HurdleModel_CA1Foods`](notebooks/HurdleModel_CA1Foods.ipynb) split the problem in two. A classifier predicts whether an item sells at all, and a regressor, trained only on non-zero days, predicts how many.

| Metric | Stage 3 single model | Stage 4 hurdle | Change |
| --- | --- | --- | --- |
| Dev RMSE | **2.356** | 2.441 | +3.6% (worse) |
| Backtest RMSE | **2.377** | 2.456 | +3.3% (worse) |

The classifier was only 73.8% accurate (ROC-AUC 0.83), so roughly one day in four got a hard wrong zero/non-zero call. Removing the zeros did not make the regressor's job easier either, because Tweedie was already handling the zero spike.

### Stage 5: Bucketed range model (worse again)

[`BucketedDemandModel_CA1Foods`](notebooks/BucketedDemandModel_CA1Foods.ipynb) extended the idea to a 4-class demand-range classifier with one regressor per non-zero bucket.

| Metric | Stage 3 single model | Stage 4 hurdle | Stage 5 bucketed |
| --- | --- | --- | --- |
| Dev RMSE | **2.356** | 2.441 | 2.584 |
| Backtest RMSE | **2.377** | 2.456 | 2.598 |

Classification accuracy fell to 59%, and the "medium" bucket was nearly unrecoverable (F1 0.09). Each extra class boundary misroutes more rows to the wrong specialist, so the ranking is single model > hurdle > bucketed > naive baseline. **Adding classification structure made things steadily worse.** One untried idea is to blend the regressors by the full predicted bucket probabilities instead of hard-routing each row.

### Stage 6: Pool all four California stores (matches Stage 3, no gain on CA_1)

[`PooledModel_CA_FOODS`](notebooks/PooledModel_CA_FOODS.ipynb) trained one Tweedie LightGBM on CA_1–CA_4 FOODS (5,748 series), with the same features as Stage 3.

| Evaluated on CA_1 rows | Dev RMSE / MAE | Backtest RMSE / MAE |
| --- | --- | --- |
| **Stage 3, CA_1-only model** | **2.356** / 1.308 | **2.377** / 1.377 |
| Stage 6, pooled model | 2.370 / **1.305** | **2.377** / **1.371** |

RMSE is flat or slightly worse and MAE is better by only 0.003–0.006, which is within noise, while the pooled model trains on about four times the rows. Across all four stores it scores RMSE 2.283 (dev) and 2.285 (backtest) against 3.116 and 3.088 for the naive baseline. Error tracks store volume: CA_3 is highest (RMSE 2.831) and CA_4 lowest (1.549).

### Where we landed

| Stage | Change | Result |
| --- | --- | --- |
| 1 | Mixed 500-series sample, lag-28, L2 | Starting point |
| 2 | Scope to CA_1/FOODS, compare models | Better once scale-adjusted; features matter more than model |
| **3** | **1-day lags + Tweedie objective** | **Best: dev RMSE 2.356, backtest RMSE 2.377** |
| 4 | Hurdle (classify, then regress) | Worse by about 3–4% |
| 5 | Bucketed range model | Worse by about 10% |
| 6 | Pool four California stores | Ties Stage 3 on CA_1 |

- **The best model is Stage 3**: a single Tweedie LightGBM with short-horizon lags. Use the CA_1-only version if CA_1 is the target. Use the Stage 6 pooled model if you want one model for all four stores, since it is competitive on each store and cheaper to maintain than four separate ones.
- The two big lessons were that **feature recency beat every change of model or architecture**, and that **a well-matched objective (Tweedie) beat added structure** (hurdle and buckets).
- These conclusions are for FOODS, the densest category. Household and Hobbies are far more intermittent and need separate validation.

Each notebook ends with its own detailed conclusions section.

## Project contents

| File or folder | Purpose |
| --- | --- |
| [`notebooks/LightGBM_BaseLocal.ipynb`](notebooks/LightGBM_BaseLocal.ipynb) | Local EDA plus the main CA_1/FOODS model: short-horizon lags, Tweedie LightGBM, naive lag-1 baseline, and optional Ridge and Random Forest comparisons. Reads data from `data/raw/`. |
| [`notebooks/LightGBM_BaseColab.ipynb`](notebooks/LightGBM_BaseColab.ipynb) | Colab version of the same workflow. Mounts Google Drive for the data. |
| [`notebooks/HurdleModel_CA1Foods.ipynb`](notebooks/HurdleModel_CA1Foods.ipynb) | Two-stage hurdle model (classify sale vs. no sale, regress the non-zero magnitude), compared against the single model. |
| [`notebooks/BucketedDemandModel_CA1Foods.ipynb`](notebooks/BucketedDemandModel_CA1Foods.ipynb) | Multi-class demand-range classifier with one regressor per non-zero bucket. |
| [`notebooks/PooledModel_CA_FOODS.ipynb`](notebooks/PooledModel_CA_FOODS.ipynb) | One Tweedie LightGBM trained on all four California stores' FOODS items (5,748 series), with per-store and CA_1 comparisons. |
| [`notebooks/Dataset_Exploration.ipynb`](notebooks/Dataset_Exploration.ipynb), [`notebooks/Dataset_Exploration_Local.ipynb`](notebooks/Dataset_Exploration_Local.ipynb) | Short dataset-exploration notebooks (Colab and local) with no modelling. |
| [`scripts/preprocess_m5.py`](scripts/preprocess_m5.py) | Batch feature preparation for the full dataset (see the note below). |
| [`requirements.txt`](requirements.txt) | Python dependencies for the notebooks and preprocessing script. |
| [`reports/`](reports/) | EDA figures, tables, and the written [`EDAInsights.md`](reports/EDAInsights.md), generated by the notebooks. |
| `data/raw/` | Original M5 CSV files. Not included in Git. |
| `data/processed/` | Generated Parquet feature sets and reports. Not included in Git. |

The modelling notebooks share the same scope, features, and chronological splits, so their results are directly comparable. They are self-contained: each repeats the data loading and EDA setup, then adds its own model.

## Dataset

Download the files from the [M5 competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Kaggle may require you to sign in and accept the competition rules before downloading.

The project expects these files:

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
```

`sample_submission.csv` is optional. Keep the CSV filenames unchanged.

The validation sales file contains history through `d_1913`. The evaluation file includes the additional observed days through `d_1941`. The notebooks use the validation file for the EDA and the evaluation file for the later backtest window, and check that the overlapping history in the two files agrees.

## Run locally with VS Code

### 1. Install Python and VS Code

Install Python 3.10 or newer and VS Code. In VS Code, install the **Python** and **Jupyter** extensions from Microsoft.

### 2. Put the data in the project

Create `data/raw/` in the project root and copy the CSV files there. The CSVs must be in `data/raw/`, not directly in `data/`:

```text
SemProject_MLEngine/
└── data/
    └── raw/
        ├── calendar.csv
        ├── sales_train_evaluation.csv
        ├── sales_train_validation.csv
        └── sell_prices.csv
```

### 3. Create an environment and install dependencies

From a terminal opened at the project root, run one of the following sets of commands.

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name semproject-mlengine --display-name "Python (SemProject_MLEngine)"
```

If PowerShell blocks environment activation, use the environment’s Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name semproject-mlengine --display-name "Python (SemProject_MLEngine)"
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name semproject-mlengine --display-name "Python (SemProject_MLEngine)"
```

### 4. Open and run a notebook

Open the project folder in VS Code, then open a notebook from `notebooks/` (start with [`LightGBM_BaseLocal.ipynb`](notebooks/LightGBM_BaseLocal.ipynb)). Select the `Python (SemProject_MLEngine)` kernel when prompted and run the notebook from the top so its data checks, EDA setup, and model variables are initialized in order.

The notebooks locate `data/raw/` by searching the current directory and its parent directories, so they work when Jupyter starts from either the repository root or `notebooks/`.

Memory: the CA_1/FOODS notebooks load the full sales file, then keep only the scoped series. The pooled notebook builds about 11 million series-day rows and needs noticeably more memory and time (several minutes) than the single-store notebooks. To run a notebook headlessly:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute PooledModel_CA_FOODS.ipynb --output-dir ../outputs --ExecutePreprocessor.timeout=3000
```

Running a notebook also regenerates the EDA figures and tables in `reports/`, so those files may show up as modified in Git afterwards.

### 5. Optional: build the batch-prepared feature data

`scripts/preprocess_m5.py` prepares a full-dataset feature set in bounded batches and writes it to `data/processed/model_ready/`:

```bash
python scripts/preprocess_m5.py
```

To generate a smaller pilot dataset, use a different output folder name:

```bash
python scripts/preprocess_m5.py --series-limit 500 --output-name model_ready_pilot
```

`--batch-size` controls how many series are processed per batch (default 250).

The script checks the sales-file overlap and writes a data-quality report, feature dictionary, split manifest, and Parquet features/targets. The splits are:

- Development training targets: `d_57`–`d_1885`.
- Development validation: `d_1886`–`d_1913`.
- Later backtest: `d_1914`–`d_1941`.
- Competition forecast inputs: `d_1942`–`d_1969` (no target values).

**Note:** the script still builds the older 28-day-embargoed lag and rolling features (lags 28/35/42/56) for the 28-day-ahead competition setting. The notebooks have since moved to next-day lags (1/7/14/21), so the script's output is not the feature set behind the results above. The script prepares data only; it does not fit or save a model.

## Google Colab setup

Use [`notebooks/LightGBM_BaseColab.ipynb`](notebooks/LightGBM_BaseColab.ipynb) for the hosted Colab workflow. It mounts Google Drive, runs the EDA, and trains the CA_1/FOODS LightGBM model. A GPU is not required.

### 1. Prepare Google Drive

Download the four required CSVs and place them in this folder in your Google Drive:

```text
MyDrive/M5_Dataset/
├── calendar.csv
├── sales_train_evaluation.csv
├── sales_train_validation.csv
└── sell_prices.csv
```

Keep the filenames unchanged. The notebook’s default path is `/content/drive/MyDrive/M5_Dataset`. If you use a different folder, edit `RAW_DATA_DIR` in the path-configuration cell.

### 2. Open the notebook in Colab

In Google Colab, choose **File → Upload notebook** and upload `LightGBM_BaseColab.ipynb` from the `notebooks/` folder. Alternatively, open it from the project’s GitHub page if the repository is available to your Colab account. Choose a standard hosted Python runtime.

### 3. Connect Drive and install packages

Run the first notebook cell and complete Google’s Drive authorization prompt. The package cell installs LightGBM, scikit-learn, Matplotlib, and Seaborn. Pandas and NumPy are provided by the Colab runtime.

### 4. Run the notebook

Use **Runtime → Run all**. EDA tables and figures are saved under `MyDrive/reports/`. If you change the configuration, rerun from the beginning so the data and features are rebuilt consistently.

The hurdle, bucketed, and pooled notebooks are written for the local layout (`data/raw/`) and are not set up for Colab.

## EDA outputs

The EDA examines data completeness and key uniqueness, total sales over time, store and product differences, zero-sales frequency, demand variability, calendar patterns, events, SNAP indicators, and selling prices. It saves charts and summary tables in:

```text
reports/figures/   # local notebooks
reports/tables/    # local notebooks
MyDrive/reports/   # Colab notebook output
```

See [`reports/EDAInsights.md`](reports/EDAInsights.md) for the written EDA interpretation.

## Reproducibility and notes

- The notebooks use a fixed random seed (`42`).
- Splits are chronological, never random: train through `d_1885`, validate on `d_1886`–`d_1913`, and backtest on `d_1914`–`d_1941` after refitting on all labelled data before it.
- Zero sales and missing prices are preserved rather than replaced with fabricated values.
- Raw data and processed Parquet files are excluded from version control. Download the raw data separately and regenerate processed files when needed.
- `requirements.txt` pins compatible version ranges for the main libraries. Select the project’s virtual environment as the Jupyter kernel before running notebook cells.

## Troubleshooting

**The notebook cannot find the CSV files**

Confirm the files are under `data/raw/` (not just `data/`) and that the notebook kernel is using this project’s environment. For Colab, confirm Drive is mounted and the files are under `MyDrive/M5_Dataset/`.

**A Python package is missing**

Activate the project environment and run `pip install -r requirements.txt` again. In Colab, rerun the package-install cell and restart the runtime if imports still fail.

**The notebook runs out of memory**

The pooled notebook is the heaviest. Close other memory-hungry programs, restart the kernel, and run it from the top. For complete-dataset processing, use `scripts/preprocess_m5.py`, which works in batches; reduce `--batch-size` to lower its peak memory.

**Parquet support is unavailable**

Install the project requirements so `pyarrow` is present in the active Python environment.
