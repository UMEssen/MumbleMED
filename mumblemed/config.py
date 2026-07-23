import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in runtime environments
    yaml = None


def load_env() -> None:
    load_dotenv()


def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def resolve_path(value: str | None, default: Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("Expected a path but got empty.")
        return default
    return Path(value).expanduser()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml.")
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}

    raise ValueError("Config file must be .yaml, .yml, or .json")
