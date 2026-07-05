import argparse
import gc
import random
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TransactionDataset, build_sequence_groups, collate_fn_embed
from evaluate import (
    entity_embeddings_to_dataframe,
    extract_entity_embeddings,
    extract_transaction_embeddings_to_dataframe,
)
from model import NPPRModel
from preprocess import TransactionPreprocessor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_data(
    config,
    checkpoint_path: str | None,
    *,
    for_transactions: bool,
    batch_size: int,
):
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    checkpoint = Path(checkpoint_path or train_cfg["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_path = Path(data_cfg["emb_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity_col = data_cfg["entity_column"]
    transaction_id_col = data_cfg.get("transaction_id_column", "transaction.id")
    categorical_features = feat_cfg["categorical"]

    df = pd.read_csv(data_path)
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
    del df
    gc.collect()

    cat_cols = list(categorical_features.keys())
    num_cols = feat_cfg["numeric_columns"]

    dataset = TransactionDataset(
        sequences,
        cat_cols,
        num_cols,
        k_past=model_cfg["k_past"],
        transaction_id_column=transaction_id_col if for_transactions else None,
        entity_column=entity_col,
        include_delta=False,
    )
    del sequences
    gc.collect()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_embed,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
    model.eval()

    return {
        "device": device,
        "checkpoint": checkpoint,
        "entity_col": entity_col,
        "transaction_id_col": transaction_id_col,
        "dataset": dataset,
        "loader": loader,
        "model": model,
        "embed_dim": model_cfg["output_dim"],
    }


def main(
    config_path: str,
    checkpoint_path: str | None,
    output_path: str,
    level: str,
    pool: str | None,
    batch_size: int | None,
):
    config = load_config(config_path)
    eval_cfg = config["evaluation"]
    extract_cfg = config.get("extraction", {})

    if level not in ("transaction", "entity"):
        raise ValueError("level must be 'transaction' or 'entity'")

    output = Path(output_path)
    if level == "transaction":
        effective_batch_size = batch_size or extract_cfg.get("batch_size", 1)
        if output.suffix != ".parquet":
            output = output.with_suffix(".parquet")
    else:
        effective_batch_size = batch_size or extract_cfg.get("entity_batch_size", 64)
        if output.suffix != ".parquet":
            output = output.with_suffix(".parquet")

    ctx = load_model_and_data(
        config,
        checkpoint_path,
        for_transactions=level == "transaction",
        batch_size=effective_batch_size,
    )
    device = ctx["device"]
    checkpoint = ctx["checkpoint"]
    entity_col = ctx["entity_col"]
    transaction_id_col = ctx["transaction_id_col"]
    dataset = ctx["dataset"]
    loader = ctx["loader"]
    model = ctx["model"]
    embed_dim = ctx["embed_dim"]

    if level == "transaction":
        if "transaction_ids" not in dataset.samples[0]:
            raise ValueError(f"Column {transaction_id_col!r} not found in dataset.")

        print(
            f"Extracting {dataset.total_transactions:,} transaction embeddings "
            f"(batch_size={effective_batch_size})..."
        )
        meta = extract_transaction_embeddings_to_dataframe(
            model,
            tqdm(loader, desc="Extracting"),
            dataset,
            device,
            output,
            embed_dim=embed_dim,
            transaction_id_column=transaction_id_col,
            entity_column=entity_col,
        )
        print(
            f"Saved {meta['total_transactions']:,} rows x {embed_dim} embedding columns "
            f"to {output}"
        )
        print(f"Load with: pd.read_parquet('{output}')")
        return

    pool = pool or eval_cfg["embedding_pool"]
    if pool not in ("mean", "last"):
        raise ValueError("pool must be 'mean' or 'last'")

    entity_ids = [sample["customer_id"] for sample in dataset.samples]
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Extracting entity embeddings for {len(dataset):,} customers "
        f"(pool={pool}, batch_size={effective_batch_size})..."
    )
    embeddings, lengths = extract_entity_embeddings(
        model, tqdm(loader, desc="Extracting"), device, pool=pool
    )

    entity_df = entity_embeddings_to_dataframe(
        embeddings,
        entity_ids,
        entity_col,
        embed_dim,
        lengths=lengths,
        pool=pool,
    )
    entity_df.to_parquet(output, index=False)
    print(f"Saved {len(entity_df):,} rows x {embed_dim} embedding columns to {output}")
    print(f"Load with: pd.read_parquet('{output}')")


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
        default="embeddings/transaction_embeddings.parquet",
        help="Output parquet path",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: 1 for transactions, 64 for entities)",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.level, args.pool, args.batch_size)
