import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import build_model_from_checkpoint
from data_pipeline import load_dataframe, prepare_all_sequences
from dataset import TransactionDataset, collate_fn_embed
from embedding_extraction import (
    entity_embeddings_to_dataframe,
    extract_entity_embeddings,
    extract_transaction_embeddings_to_dataframe,
)
from nppr_config import load_config


def load_model_and_data(config, checkpoint_path: str | None, *, for_transactions: bool, batch_size: int):
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(checkpoint_path or train_cfg["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, _, model_cfg, feat_cfg = build_model_from_checkpoint(config, checkpoint, device)

    df = load_dataframe(data_cfg)
    prepared = prepare_all_sequences(df, data_cfg, feat_cfg, train_cfg, model_cfg["k_past"])
    del df
    gc.collect()

    transaction_id_col = data_cfg.get("transaction_id_column", "transaction.id")
    entity_col = prepared["entity_col"]

    dataset = TransactionDataset(
        prepared["sequences"],
        prepared["cat_cols"],
        prepared["num_cols"],
        k_past=prepared["k_past"],
        transaction_id_column=transaction_id_col if for_transactions else None,
        entity_column=entity_col,
        metadata_columns=data_cfg.get("metadata_columns"),
        include_delta=False,
    )
    del prepared["sequences"]
    gc.collect()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_embed,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

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
    output_path: str | None,
    level: str,
    pool: str | None,
    batch_size: int | None,
    dataset_name: str | None,
):
    config = load_config(config_path, dataset_name=dataset_name)
    print(f"Dataset profile: {config['dataset_name']}")

    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    extract_cfg = config.get("extraction", {})

    if level not in ("transaction", "entity"):
        raise ValueError("level must be 'transaction' or 'entity'")

    default_output = data_cfg.get("embeddings_path", "embeddings/transaction_embeddings.parquet")
    output = Path(output_path or default_output)
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
        default=None,
        help="Output parquet path (default: dataset embeddings_path in config)",
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
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset profile name (default: active_dataset in config)",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.checkpoint,
        args.output,
        args.level,
        args.pool,
        args.batch_size,
        args.dataset,
    )
