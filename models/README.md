# Model registry

The ML service serves every `models/<modelId>/<version>/` folder that holds a `metadata.json`.

| Version | Trained to | Serves as-of dates | Notes |
| --- | --- | --- | --- |
| `m5-ca1-foods/1.1.0` | 2016-05-22 | from 2016-05-22 | Active Reference model; horizons 1 and 7 |
| `m5-ca1-foods/1.1.0-bt` | 2016-04-24 | 2016-04-24 … 05-21 | Replays in the backtest window; holds `backtest/predictions.parquet` |
| `m5-ca1-foods/1.1.0-dev` | 2016-03-27 | 2016-03-27 … 04-23 | The notebook's development model; research replays |

Each folder holds `model.txt` (LightGBM), `metadata.json`, `card.json`, `calendar.parquet` and `backtest/`. Regenerate them with `python -m mlengine release-m5` (see `docs/operations.md`). Own-data models (`own-<workspaceId>/`) are created at runtime and are not committed.
