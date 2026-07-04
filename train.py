import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TransactionDataset, build_sequence_groups, collate_fn
from model import NPPRModel
from preprocess import TransactionPreprocessor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch, device):
    cat_inputs, num_inputs, delta_matrix, lengths = batch
    cat_inputs = {feat: t.to(device) for feat, t in cat_inputs.items()}
    return (
        cat_inputs,
        num_inputs.to(device),
        delta_matrix.to(device),
        lengths.to(device),
    )


def run_epoch(model, dataloader, device, optimizer=None, grad_clip=0.5, desc=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    batch_iter = tqdm(dataloader, desc=desc, leave=False) if desc else dataloader
    for batch in batch_iter:
        cat_inputs, num_inputs, delta_matrix, lengths = move_batch_to_device(batch, device)

        outputs = model(cat_inputs, num_inputs, delta_matrix)
        loss = model.loss(outputs, cat_inputs, num_inputs, delta_matrix, lengths)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if desc:
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def main(config_path: str):
    config = load_config(config_path)

    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(train_cfg["seed"])

    data_path = Path(data_cfg["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    entity_col = data_cfg["entity_column"]
    categorical_features = feat_cfg["categorical"]

    preprocessor = TransactionPreprocessor(
        categorical_features=categorical_features,
        numeric_mappings=feat_cfg.get("numeric_mappings"),
        time_gap_source=feat_cfg["time_gap_source"],
        time_features_zero_indexed=feat_cfg.get("time_features_zero_indexed", []),
        preprocessed=feat_cfg.get("preprocessed", False),
    )

    entity_ids = df[entity_col].unique()
    rng = random.Random(train_cfg["seed"])
    rng.shuffle(entity_ids)
    split = int(len(entity_ids) * train_cfg["train_frac"])
    train_entities = set(entity_ids[:split])

    train_df = preprocessor.fit_transform(df[df[entity_col].isin(train_entities)])
    val_df = preprocessor.transform(df[~df[entity_col].isin(train_entities)])

    train_sequences = build_sequence_groups(train_df, entity_col)
    val_sequences = build_sequence_groups(val_df, entity_col)

    cat_cols = list(categorical_features.keys())
    num_cols = feat_cfg["numeric_columns"]
    k_past = model_cfg["k_past"]

    train_dataset = TransactionDataset(
        train_sequences, cat_cols, num_cols, k_past=k_past
    )
    val_dataset = TransactionDataset(
        val_sequences, cat_cols, num_cols, k_past=k_past
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NPPRModel(
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        categorical_features=categorical_features,
        num_numeric=len(num_cols),
        feature_embed_dim=model_cfg["feature_embed_dim"],
        pr_weight=model_cfg["pr_weight"],
        decay_length=model_cfg["decay_length"],
        gru_num_layers=model_cfg["gru_num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint_path = Path(train_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(train_cfg["num_epochs"]), desc="Training")
    for epoch in epoch_bar:
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            grad_clip=train_cfg["grad_clip"],
            desc=f"Epoch {epoch + 1} train",
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device,
            desc=f"Epoch {epoch + 1} val",
        )

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain NPPR model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
