from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_raw_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config(raw: dict[str, Any], dataset_name: str | None = None) -> dict[str, Any]:
    name = dataset_name or raw.get("active_dataset")
    if not name:
        raise ValueError("Set active_dataset in config.yaml or pass --dataset")

    if "datasets" not in raw:
        return _legacy_flat_config(raw, name)

    if name not in raw["datasets"]:
        available = ", ".join(sorted(raw["datasets"]))
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}")

    profile = deepcopy(raw["datasets"][name])
    config = {
        "dataset_name": name,
        "data": profile["data"],
        "features": profile["features"],
        "model": deepcopy(raw["model"]),
        "training": deepcopy(raw["training"]),
        "evaluation": deepcopy(raw["evaluation"]),
        "extraction": deepcopy(raw.get("extraction", {})),
    }

    outputs = profile.get("outputs", {})
    if "checkpoint_path" in outputs:
        config["training"]["checkpoint_path"] = outputs["checkpoint_path"]
    if "embeddings_path" in outputs:
        config["data"]["embeddings_path"] = outputs["embeddings_path"]
        config["evaluation"]["embeddings_path"] = outputs["embeddings_path"]

    for section in ("training", "evaluation", "extraction"):
        if section in profile:
            config[section].update(profile[section])

    if "split_column" in config["data"]:
        config["evaluation"]["split_column"] = config["data"]["split_column"]
    if config["data"].get("use_split_column"):
        if "train_splits" in config["data"] and "val_splits" in config["data"]:
            config["evaluation"].setdefault(
                "train_splits",
                config["data"]["train_splits"] + config["data"]["val_splits"],
            )
        if "test_splits" in config["data"]:
            config["evaluation"].setdefault("test_splits", config["data"]["test_splits"])

    return config


def load_config(path: str | Path, dataset_name: str | None = None) -> dict[str, Any]:
    return resolve_config(load_raw_config(path), dataset_name=dataset_name)


def _legacy_flat_config(raw: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "dataset_name": name or "default",
        "data": raw["data"],
        "features": raw["features"],
        "model": raw["model"],
        "training": raw["training"],
        "evaluation": raw["evaluation"],
        "extraction": raw.get("extraction", {}),
    }
