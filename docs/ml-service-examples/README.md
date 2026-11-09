# ML service examples

These request/response pairs were captured from the running service (`m5-ca1-foods@1.1.0`) on 2026-11-07.

| Files | Scenario |
| --- | --- |
| `forecast_request.json` → `forecast_response.json` | True 7-day forecast for `FOODS_3_090` from as-of 2016-05-22 (resolved to `1.1.0`), with known future prices |
| `replay_request.json` → `replay_response.json` | Next-day replay from as-of 2016-05-08 (resolved to `1.1.0-bt`). AI 57.1 vs naive 77; the actual sale that day was 72 (from `sales_daily`, which the backend attaches, not the service) |
| `error_no_model_for_as_of.json` | The replay request with `version: "1.1.0"` forced: refused by the leakage rule |

The requests send only 28 days of history to keep the files small, so `demandClass` comes back as `INSUFFICIENT_DATA`. In production, send 365 days.
