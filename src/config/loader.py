"""YAML config loader.

Loads a YAML file into a nested attribute-access namespace so callers can write
``cfg.data.patch_size`` instead of dictionary indexing. CLI overrides are
applied with dotted keys (e.g. ``--data.patch_size 96``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict subclass that exposes keys as attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value


def load_yaml(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config._wrap(raw)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``key.sub=value`` style overrides parsed from CLI."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key_path, raw_value = item.split("=", 1)
        # YAML-parse the value so ints/floats/bools work without quoting.
        value = yaml.safe_load(raw_value)
        node = cfg
        keys = key_path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, Config())
            if not isinstance(node, Config):
                raise ValueError(f"Override path collides with non-dict value at {key}")
        node[keys[-1]] = Config._wrap(value)
    return cfg


def parse_config_args(description: str = "") -> tuple[Config, argparse.Namespace]:
    """Standard CLI: ``--config <yaml>`` + ``--set key=value`` overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set", nargs="*", default=[], help="Dotted-key overrides, e.g. data.fold=2"
    )
    parser.add_argument("--name", default=None, help="Experiment name override")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.set)
    if args.name is not None:
        cfg.experiment.name = args.name
    return cfg, args
