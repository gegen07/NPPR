import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nppr_config import load_config


class DownstreamMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_embeddings_parquet(
    parquet_path: Path,
    label_column: str,
    split_column: str,
    entity_column: str,
    level: str,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(parquet_path)
    embed_cols = [c for c in df.columns if c.startswith("emb_")]
    if not embed_cols:
        raise ValueError(f"No emb_* columns found in {parquet_path}")

    required = {label_column, split_column}
    if level == "entity":
        required.add(entity_column)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {sorted(missing)}")

    if level == "entity":
        df = (
            df.groupby(entity_column, as_index=False)
            .agg(
                **{
                    label_column: (label_column, "first"),
                    split_column: (split_column, "first"),
                    **{col: (col, "mean") for col in embed_cols},
                }
            )
        )

    return df, embed_cols


def split_embeddings_df(
    df: pd.DataFrame,
    embed_cols: list[str],
    label_column: str,
    split_column: str,
    train_splits: list[str],
    test_splits: list[str],
    val_splits: list[str] | None = None,
):
    train_mask = df[split_column].isin(train_splits)
    test_mask = df[split_column].isin(test_splits)
    val_mask = df[split_column].isin(val_splits) if val_splits else pd.Series(False, index=df.index)

    splits = {
        "train": (df.loc[train_mask, embed_cols].values, df.loc[train_mask, label_column].values),
        "test": (df.loc[test_mask, embed_cols].values, df.loc[test_mask, label_column].values),
    }
    if val_splits:
        splits["val"] = (
            df.loc[val_mask, embed_cols].values,
            df.loc[val_mask, label_column].values,
        )
    return splits


def train_downstream_classifier(X_train, y_train, X_test, y_test, eval_cfg, device):
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    clf = DownstreamMLP(
        input_dim=input_dim,
        hidden_dim=eval_cfg["downstream_hidden_dim"],
        num_layers=eval_cfg["downstream_num_layers"],
        num_classes=num_classes,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=eval_cfg.get("downstream_batch_size", 256),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(clf.parameters(), lr=eval_cfg["downstream_lr"])
    criterion = nn.CrossEntropyLoss()

    clf.train()
    epoch_bar = tqdm(range(eval_cfg["downstream_epochs"]), desc="Training head", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_bar.set_postfix(
            epoch=epoch + 1,
            avg_loss=f"{epoch_loss / max(n_batches, 1):.4f}",
            refresh=False,
        )

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.tensor(X_test, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    metrics = {"accuracy": accuracy_score(y_test, preds)}
    if num_classes == 2:
        metrics["auc"] = roc_auc_score(y_test, probs[:, 1])
        metrics["ap"] = average_precision_score(y_test, probs[:, 1])
        metrics["gini"] = 2 * metrics["auc"] - 1
    return metrics


def main(config_path: str, embeddings_path: str | None, dataset_name: str | None):
    config = load_config(config_path, dataset_name=dataset_name)
    print(f"Dataset profile: {config['dataset_name']}")
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    label_column = data_cfg.get("label_column")
    if not label_column:
        print("No label_column configured — skipping downstream evaluation.")
        print("Set data.label_column in config.yaml to enable evaluation.")
        return

    parquet_path = Path(
        embeddings_path
        or eval_cfg.get("embeddings_path")
        or data_cfg.get("embeddings_path")
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {parquet_path}")

    entity_col = data_cfg["entity_column"]
    split_column = eval_cfg.get("split_column", "split")
    train_splits = eval_cfg.get("train_splits", ["train", "val"])
    test_splits = eval_cfg.get("test_splits", ["test"])
    val_splits = eval_cfg.get("val_splits")
    level = eval_cfg.get("level", "transaction")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, embed_cols = load_embeddings_parquet(
        parquet_path,
        label_column=label_column,
        split_column=split_column,
        entity_column=entity_col,
        level=level,
    )
    splits = split_embeddings_df(
        df,
        embed_cols,
        label_column,
        split_column,
        train_splits,
        test_splits,
        val_splits,
    )

    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty train or test split. "
            f"train={len(X_train)}, test={len(X_test)}. "
            f"Check split column {split_column!r} and split values."
        )

    print(f"Loaded {len(df):,} rows from {parquet_path} ({level} level)")
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Embed dim: {len(embed_cols)}")

    metrics = train_downstream_classifier(
        X_train,
        y_train.astype(int),
        X_test,
        y_test.astype(int),
        eval_cfg,
        device,
    )

    print("Downstream evaluation (frozen embeddings from parquet):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frozen embeddings from parquet")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Path to embeddings parquet (default: dataset embeddings_path in config)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset profile name (default: active_dataset in config)",
    )
    args = parser.parse_args()
    main(args.config, args.embeddings, args.dataset)
