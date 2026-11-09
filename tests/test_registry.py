import datetime as dt
import json

import pytest

from mlengine.models.artifact import ModelRegistry, NoModelForAsOfError


def _fake(root, version, end, status="active", created="2026-01-01T00:00:00+00:00"):
    path = root / "m" / version
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({
        "modelId": "m", "version": version, "status": status, "featureSchema": "portable-v1",
        "training": {"endDate": end}, "capabilities": {}, "createdAt": created}))


def test_leakage_rule(tmp_path):
    _fake(tmp_path, "1.0.0-dev", "2016-03-27")
    _fake(tmp_path, "1.0.0-bt", "2016-04-24")
    _fake(tmp_path, "1.0.0", "2016-05-22")
    registry = ModelRegistry(tmp_path)
    assert registry.resolve("m", dt.date(2016, 4, 1)).version == "1.0.0-dev"
    assert registry.resolve("m", dt.date(2016, 4, 24)).version == "1.0.0-bt"
    assert registry.resolve("m", dt.date(2016, 5, 21)).version == "1.0.0-bt"
    assert registry.resolve("m", dt.date(2016, 5, 22)).version == "1.0.0"
    with pytest.raises(NoModelForAsOfError):
        registry.resolve("m", dt.date(2016, 3, 1))
    with pytest.raises(NoModelForAsOfError):
        registry.resolve("m", dt.date(2016, 5, 1), version="1.0.0")
