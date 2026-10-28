"""Load config.yaml into an attribute-accessible dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v


def load_config(path: str | Path | None = None) -> Config:
    with open(path or ROOT / "config.yaml") as f:
        return Config(yaml.safe_load(f))
