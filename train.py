import argparse
import gc
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import load_dataframe, prepare_datasets
from dataset import TransactionDataset, collate_fn
from model import NPPRModel
from nppr_config import load_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    cat_inputs, num_inputs, delta_matrix, lengths = batch
    non_blocking = device.type == "cuda"
    cat_inputs = {
        feat: tensor.to(device, non_blocking=non_blocking)
        for feat, tensor in cat_inputs.items()
    }
    return (
        cat_inputs,
        num_inputs.to(device, non_blocking=non_blocking),
        delta_matrix.to(device, non_blocking=non_blocking),
        lengths.to(device, non_blocking=non_blocking),
    )


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, prefetch_factor):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def run_epoch(model, dataloader, device, optimizer=None, grad_clip=0.5, use_amp=False, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in dataloader:
        cat_inputs, num_inputs, delta_matrix, lengths = move_batch_to_device(batch, device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(cat_inputs, num_inputs, delta_matrix)
            loss = model.loss(outputs, cat_inputs, num_inputs, delta_matrix, lengths)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main(config_path: str, dataset_name: str | None):
    config = load_config(config_path, dataset_name=dataset_name)
    tqdm.write(f"Dataset profile: {config['dataset_name']}")

    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(train_cfg["seed"])

    df = load_dataframe(data_cfg)
    prepared = prepare_datasets(df, data_cfg, feat_cfg, train_cfg, model_cfg["k_past"])

    train_dataset = TransactionDataset(
        prepared["train_sequences"],
        prepared["cat_cols"],
        prepared["num_cols"],
        k_past=prepared["k_past"],
    )
    val_dataset = TransactionDataset(
        prepared["val_sequences"],
        prepared["cat_cols"],
        prepared["num_cols"],
        k_past=prepared["k_past"],
    )
    del df, prepared
    gc.collect()

    train_loader = build_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_cfg.get("use_amp", False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        tqdm.write(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        use_amp = False

    categorical_features = feat_cfg["categorical"]
    model = NPPRModel(
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        categorical_features=categorical_features,
        num_numeric=len(feat_cfg["numeric_columns"]),
        feature_embed_dim=model_cfg["feature_embed_dim"],
        pr_weight=model_cfg["pr_weight"],
        decay_length=model_cfg["decay_length"],
        gru_num_layers=model_cfg["gru_num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_path = Path(train_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(train_cfg["num_epochs"]), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            grad_clip=train_cfg["grad_clip"],
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device,
            use_amp=use_amp,
        )

        epoch_bar.set_postfix(
            epoch=epoch + 1,
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            refresh=False,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "dataset_name": config["dataset_name"],
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["patience"]:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

    tqdm.write(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain NPPR model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset profile name (default: active_dataset in config)",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
