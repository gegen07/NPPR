from pathlib import Path

import torch

from model import NPPRModel


def load_checkpoint(checkpoint_path: Path, map_location=None) -> dict:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def resolve_model_and_features(config: dict, checkpoint: dict) -> tuple[dict, dict]:
    """Use architecture from the checkpoint so weights load correctly."""
    saved = checkpoint.get("config")
    if saved:
        if "features" in saved:
            return saved["model"], saved["features"]
        if "datasets" in saved and "dataset_name" in saved:
            profile = saved["datasets"][saved["dataset_name"]]
            return saved["model"], profile["features"]
    return config["model"], config["features"]


def build_model_from_checkpoint(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[NPPRModel, dict, dict, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg, feat_cfg = resolve_model_and_features(config, checkpoint)

    model = NPPRModel(
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        categorical_features=feat_cfg["categorical"],
        num_numeric=len(feat_cfg["numeric_columns"]),
        feature_embed_dim=model_cfg["feature_embed_dim"],
        pr_weight=model_cfg["pr_weight"],
        decay_length=model_cfg["decay_length"],
        gru_num_layers=model_cfg["gru_num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, model_cfg, feat_cfg
