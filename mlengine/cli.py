"""Command line: ``python -m mlengine <command>``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlengine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    release = sub.add_parser("release-m5", help="Train, backtest and register the Reference model versions")
    release.add_argument("--version", default="1.1.0")
    release.add_argument("--registry", type=Path, default=None)
    release.add_argument("--raw-dir", type=Path, default=None)

    export = sub.add_parser("export-reference", help="Write the Reference workspace seed data for the backend")
    export.add_argument("--out", type=Path, default=None)
    export.add_argument("--raw-dir", type=Path, default=None)

    samples = sub.add_parser("export-samples", help="Write sample business CSVs (real M5 store, shifted dates)")
    samples.add_argument("--out", type=Path, default=None)
    samples.add_argument("--raw-dir", type=Path, default=None)
    samples.add_argument("--store", default="TX_1")
    samples.add_argument("--end-date", type=dt.date.fromisoformat, default=None,
                         help="Last day of history (default: yesterday)")

    serve = sub.add_parser("serve", help="Run the internal ML service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)

    sub.add_parser("models", help="List registered model versions")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "release-m5":
        from mlengine.pipelines.m5_release import release as run_release

        result = run_release(args.registry, args.version, args.raw_dir)
        print(json.dumps({k: result[k] for k in ["bestIteration", "dev", "backtest"]}, indent=2, default=str))
    elif args.command == "export-reference":
        from mlengine.pipelines.reference_export import export_reference

        for name, path in export_reference(args.out, args.raw_dir).items():
            print(f"{name}: {path}")
    elif args.command == "export-samples":
        from mlengine.pipelines.sample_shops import export_samples

        for name, path in export_samples(args.out, args.raw_dir, args.store, end_date=args.end_date).items():
            print(f"{name}: {path}")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("mlengine.service.app:app", host=args.host, port=args.port)
    elif args.command == "models":
        from mlengine.models.artifact import ModelRegistry

        for artifact in ModelRegistry().list():
            print(f"{artifact.ref:32} schema={artifact.schema:12} trainingEnd={artifact.training_end_date}")


if __name__ == "__main__":
    main()
