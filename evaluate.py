import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dataset import TransactionDataset, build_sequence_groups, collate_fn
from model import NPPRModel
from preprocess import TransactionPreprocessor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


@torch.no_grad()
def extract_entity_embeddings(model, dataloader, device, pool: str):
    model.eval()
    embeddings = []
    entity_lengths = []

    for batch in dataloader:
        cat_inputs, num_inputs, delta_matrix, lengths = batch
        cat_inputs = {feat: t.to(device) for feat, t in cat_inputs.items()}
        num_inputs = num_inputs.to(device)
        lengths = lengths.to(device)

        emb = model.embed(cat_inputs, num_inputs, pool=pool, lengths=lengths)
        embeddings.append(emb.cpu())
        entity_lengths.append(lengths.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(entity_lengths, dim=0)


def entity_labels_from_sequences(sequences, entity_column: str, label_column: str):
    labels = []
    for seq in sequences:
        label_values = seq[label_column].dropna().unique()
        if len(label_values) != 1:
            raise ValueError(
                f"Entity {seq[entity_column].iloc[0]} has inconsistent labels: {label_values}"
            )
        labels.append(label_values[0])
    return np.array(labels)


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
        batch_size=256,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(clf.parameters(), lr=eval_cfg["downstream_lr"])
    criterion = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(eval_cfg["downstream_epochs"]):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            optimizer.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.tensor(X_test, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    metrics = {"accuracy": accuracy_score(y_test, preds)}
    if num_classes == 2:
        metrics["auc"] = roc_auc_score(y_test, probs[:, 1])
    return metrics


def main(config_path: str, checkpoint_path: str | None):
    config = load_config(config_path)
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    label_column = data_cfg.get("label_column")
    if not label_column:
        print("No label_column configured — skipping downstream evaluation.")
        print("Set data.label_column in config.yaml to enable evaluation.")
        return

    checkpoint = Path(checkpoint_path or train_cfg["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_cfg["path"])
    entity_col = data_cfg["entity_column"]
    categorical_features = feat_cfg["categorical"]

    entity_ids = df[entity_col].unique()
    train_entities, test_entities = train_test_split(
        entity_ids,
        test_size=1 - train_cfg["train_frac"],
        random_state=train_cfg["seed"],
    )

    preprocessor = TransactionPreprocessor(
        categorical_features=categorical_features,
        numeric_mappings=feat_cfg["numeric_mappings"],
        time_gap_source=feat_cfg["time_gap_source"],
        time_features_zero_indexed=feat_cfg.get("time_features_zero_indexed", []),
    )
    train_df = preprocessor.fit_transform(df[df[entity_col].isin(train_entities)])
    test_df = preprocessor.transform(df[df[entity_col].isin(test_entities)])

    train_sequences = build_sequence_groups(train_df, entity_col)
    test_sequences = build_sequence_groups(test_df, entity_col)

    cat_cols = list(categorical_features.keys())
    num_cols = feat_cfg["numeric_columns"]

    train_dataset = TransactionDataset(
        train_sequences, cat_cols, num_cols, k_past=model_cfg["k_past"]
    )
    test_dataset = TransactionDataset(
        test_sequences, cat_cols, num_cols, k_past=model_cfg["k_past"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

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

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    pool = eval_cfg["embedding_pool"]
    X_train, _ = extract_entity_embeddings(model, train_loader, device, pool)
    X_test, _ = extract_entity_embeddings(model, test_loader, device, pool)

    y_train = entity_labels_from_sequences(train_sequences, entity_col, label_column)
    y_test = entity_labels_from_sequences(test_sequences, entity_col, label_column)

    metrics = train_downstream_classifier(
        X_train.numpy(),
        y_train,
        X_test.numpy(),
        y_test,
        eval_cfg,
        device,
    )

    print("Downstream evaluation (frozen NPPR embeddings):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NPPR embeddings on a downstream task")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
