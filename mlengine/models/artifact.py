"""Model artifacts on disk and the registry that serves them.

Layout: ``<registry>/<model_id>/<version>/`` containing ``model.txt`` (LightGBM booster),
``metadata.json``, ``card.json`` and optional parquet side files (calendar, backtests).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from mlengine import config


class NoModelForAsOfError(LookupError):
    pass


class ModelNotFoundError(LookupError):
    pass


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.project_root(), capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class ModelArtifact:
    path: Path
    metadata: dict
    card: dict = field(default_factory=dict)
    _booster: lgb.Booster | None = None
    _frames: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.metadata["modelId"]

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def ref(self) -> str:
        return f"{self.model_id}@{self.version}"

    @property
    def schema(self) -> str:
        return self.metadata["featureSchema"]

    @property
    def training_end_date(self) -> dt.date:
        return dt.date.fromisoformat(self.metadata["training"]["endDate"])

    @property
    def capabilities(self) -> dict:
        return self.metadata["capabilities"]

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            self._booster = lgb.Booster(model_file=str(self.path / "model.txt"))
        return self._booster

    def frame(self, name: str) -> pd.DataFrame | None:
        if name not in self._frames:
            file = self.path / f"{name}.parquet"
            self._frames[name] = pd.read_parquet(file) if file.is_file() else None
        return self._frames[name]


def save_artifact(
    path: Path, booster: lgb.Booster, metadata: dict, card: dict | None = None,
    frames: dict[str, pd.DataFrame] | None = None, num_iteration: int | None = None,
) -> ModelArtifact:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    booster.save_model(str(path / "model.txt"), num_iteration=num_iteration)
    metadata = {
        **metadata,
        "mlengineVersion": __import__("mlengine").__version__,
        "lightgbmVersion": lgb.__version__,
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gitCommit": git_commit(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (path / "card.json").write_text(json.dumps(card or {}, indent=2, default=str))
    for name, frame in (frames or {}).items():
        target = path / f"{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    return load_artifact(path)


def load_artifact(path: Path) -> ModelArtifact:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    card_file = path / "card.json"
    card = json.loads(card_file.read_text()) if card_file.is_file() else {}
    return ModelArtifact(path=path, metadata=metadata, card=card)


class ModelRegistry:
    """All artifacts under one folder, with the leakage-safe version lookup."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.model_registry_dir())
        self.reload()

    def reload(self) -> None:
        self._artifacts: dict[tuple[str, str], ModelArtifact] = {}
        if self.root.is_dir():
            for metadata_file in sorted(self.root.glob("*/*/metadata.json")):
                artifact = load_artifact(metadata_file.parent)
                self._artifacts[(artifact.model_id, artifact.version)] = artifact

    def list(self, model_id: str | None = None) -> list[ModelArtifact]:
        return [a for (mid, _), a in sorted(self._artifacts.items()) if model_id in (None, mid)]

    def get(self, model_id: str, version: str) -> ModelArtifact:
        try:
            return self._artifacts[(model_id, version)]
        except KeyError:
            raise ModelNotFoundError(f"No model {model_id}@{version} in {self.root}") from None

    def resolve(self, model_id: str, as_of: dt.date, version: str | None = None) -> ModelArtifact:
        """The model to use for a forecast made on ``as_of``.

        Leakage rule: a model may only be used when its training data ended on or before the
        as-of date. Without an explicit version, the qualifying model trained most recently wins;
        among equal end dates, active (non-suffixed) versions win over ``-bt``/``-dev``.
        """
        if version is not None:
            artifact = self.get(model_id, version)
            if artifact.training_end_date > as_of:
                raise NoModelForAsOfError(
                    f"{artifact.ref} was trained on data up to {artifact.training_end_date}, "
                    f"after the as-of date {as_of}."
                )
            return artifact
        candidates = [
            a for a in self.list(model_id)
            if a.training_end_date <= as_of and a.metadata.get("status", "active") == "active"
        ]
        if not self.list(model_id):
            raise ModelNotFoundError(f"No versions of model {model_id!r} in {self.root}")
        if not candidates:
            raise NoModelForAsOfError(f"No version of {model_id!r} was trained only on data up to {as_of}.")
        return max(candidates, key=lambda a: (a.training_end_date, "-" not in a.version, a.metadata["createdAt"]))
