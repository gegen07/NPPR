import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TransactionDataset, build_sequence_groups, collate_fn
from evaluate import extract_entity_embeddings, extract_transaction_embeddings
from model import NPPRModel
from preprocess import TransactionPreprocessor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_data(config, checkpoint_path: str | None):
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    checkpoint = Path(checkpoint_path or train_cfg["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_path = Path(data_cfg["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_path)
    entity_col = data_cfg["entity_column"]
    categorical_features = feat_cfg["categorical"]

    entity_ids_all = df[entity_col].unique()
    rng = random.Random(train_cfg["seed"])
    entity_ids_shuffled = list(entity_ids_all)
    rng.shuffle(entity_ids_shuffled)
    split = int(len(entity_ids_shuffled) * train_cfg["train_frac"])
    train_entities = set(entity_ids_shuffled[:split])

    preprocessor = TransactionPreprocessor(
        categorical_features=categorical_features,
        numeric_mappings=feat_cfg.get("numeric_mappings"),
        time_gap_source=feat_cfg["time_gap_source"],
        time_features_zero_indexed=feat_cfg.get("time_features_zero_indexed", []),
        preprocessed=feat_cfg.get("preprocessed", False),
    )
    preprocessor.fit(df[df[entity_col].isin(train_entities)])
    df = preprocessor.transform(df)

    sequences = build_sequence_groups(df, entity_col)

    cat_cols = list(categorical_features.keys())
    num_cols = feat_cfg["numeric_columns"]

    dataset = TransactionDataset(
        sequences,
        cat_cols,
        num_cols,
        k_past=model_cfg["k_past"],
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
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

    return {
        "device": device,
        "checkpoint": checkpoint,
        "entity_col": entity_col,
        "sequences": sequences,
        "loader": loader,
        "model": model,
    }


def main(
    config_path: str,
    checkpoint_path: str | None,
    output_path: str,
    level: str,
    pool: str | None,
):
    config = load_config(config_path)
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    if level not in ("transaction", "entity"):
        raise ValueError("level must be 'transaction' or 'entity'")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ctx = load_model_and_data(config, checkpoint_path)
    device = ctx["device"]
    checkpoint = ctx["checkpoint"]
    entity_col = ctx["entity_col"]
    sequences = ctx["sequences"]
    loader = ctx["loader"]
    model = ctx["model"]

    if level == "transaction":
        transaction_id_col = data_cfg.get("transaction_id_column", "transaction.id")
        if transaction_id_col not in sequences[0].columns:
            raise ValueError(
                f"Column {transaction_id_col!r} not found in dataset. "
                "Set data.transaction_id_column in config.yaml."
            )

        print(f"Extracting transaction embeddings for {len(sequences)} customer sequences...")
        embeddings, transaction_ids, customer_ids = extract_transaction_embeddings(
            model,
            tqdm(loader, desc="Extracting"),
            sequences,
            device,
            entity_column=entity_col,
            transaction_id_column=transaction_id_col,
        )

        torch.save(
            {
                "level": "transaction",
                "transaction_ids": transaction_ids,
                "customer_ids": customer_ids,
                "embeddings": embeddings,
                "checkpoint": str(checkpoint),
            },
            output,
        )
        print(
            f"Saved {embeddings.shape[0]} transaction embeddings "
            f"of dim {embeddings.shape[1]} to {output}"
        )
        return

    pool = pool or eval_cfg["embedding_pool"]
    if pool not in ("mean", "last"):
        raise ValueError("pool must be 'mean' or 'last'")

    entity_ids = [seq[entity_col].iloc[0] for seq in sequences]

    print(f"Extracting entity embeddings for {len(sequences)} customers (pool={pool})...")
    embeddings, lengths = extract_entity_embeddings(
        model, tqdm(loader, desc="Extracting"), device, pool=pool
    )

    torch.save(
        {
            "level": "entity",
            "entity_ids": entity_ids,
            "embeddings": embeddings,
            "lengths": lengths,
            "pool": pool,
            "checkpoint": str(checkpoint),
        },
        output,
    )
    print(f"Saved {embeddings.shape[0]} entity embeddings of dim {embeddings.shape[1]} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NPPR embeddings")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint (default: training.checkpoint_path from config)",
    )
    parser.add_argument(
        "--output",
        default="embeddings/transaction_embeddings.pt",
        help="Output path for saved embeddings",
    )
    parser.add_argument(
        "--level",
        default="transaction",
        choices=("transaction", "entity"),
        help="Extract one embedding per transaction or per customer",
    )
    parser.add_argument(
        "--pool",
        default=None,
        choices=("mean", "last"),
        help="Pooling for entity-level embeddings (default: evaluation.embedding_pool)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.level, args.pool)
