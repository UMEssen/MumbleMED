import json
from pathlib import Path

import pytest

from mumblemed.config import load_config


def test_load_json_config(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"num_docs": 3, "verbose": True}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg["num_docs"] == 3
    assert cfg["verbose"] is True


def test_load_yaml_config(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("dataset_path: /tmp/out\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["dataset_path"] == "/tmp/out"


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.json")

