from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from dataset import build_sequence_groups
from preprocess import TransactionPreprocessor


def load_dataframe(data_cfg: dict) -> pd.DataFrame:
    data_path = Path(data_cfg.get("path") or data_cfg.get("emb_path"))
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def ensure_time_gap_column(df: pd.DataFrame, data_cfg: dict, feat_cfg: dict) -> pd.DataFrame:
    df = df.copy()
    if "Time_Gap" in df.columns or "timestamp_diff" in df.columns:
        return df

    entity_col = data_cfg["entity_column"]
    time_gap_source = feat_cfg.get("time_gap_source", "timestamp_diff")
    datetime_col = data_cfg.get("datetime_column", "datetime")

    if time_gap_source == "datetime":
        if datetime_col not in df.columns:
            raise ValueError(
                f"time_gap_source is 'datetime' but column {datetime_col!r} is missing."
            )
        if not pd.api.types.is_numeric_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col]).astype("int64") // 10**9

        sort_cols = [entity_col, datetime_col]
        df = df.sort_values(sort_cols)
        df["timestamp_diff"] = (
            df.groupby(entity_col, sort=False)[datetime_col].diff().fillna(0).clip(lower=0)
        )
        return df

    raise ValueError(
        "Dataset must include Time_Gap, timestamp_diff, or datetime-based time_gap_source."
    )


def get_train_val_entities(df: pd.DataFrame, data_cfg: dict, train_cfg: dict) -> tuple[set, set]:
    entity_col = data_cfg["entity_column"]

    if data_cfg.get("use_split_column"):
        split_col = data_cfg.get("split_column", "split")
        if split_col not in df.columns:
            raise ValueError(f"use_split_column is true but {split_col!r} is missing.")

        entity_split = df.groupby(entity_col)[split_col].first()
        train_splits = set(data_cfg.get("train_splits", ["train"]))
        val_splits = set(data_cfg.get("val_splits", ["val"]))

        train_entities = set(entity_split[entity_split.isin(train_splits)].index)
        val_entities = set(entity_split[entity_split.isin(val_splits)].index)
        overlap = train_entities & val_entities
        if overlap:
            raise ValueError(f"Entities appear in both train and val splits: {overlap}")
        return train_entities, val_entities

    entity_ids = df[entity_col].unique()
    rng = random.Random(train_cfg["seed"])
    entity_ids_shuffled = list(entity_ids)
    rng.shuffle(entity_ids_shuffled)
    split = int(len(entity_ids_shuffled) * train_cfg["train_frac"])
    train_entities = set(entity_ids_shuffled[:split])
    val_entities = set(entity_ids_shuffled[split:])
    return train_entities, val_entities


def build_preprocessor(feat_cfg: dict) -> TransactionPreprocessor:
    return TransactionPreprocessor(
        categorical_features=feat_cfg["categorical"],
        numeric_mappings=feat_cfg.get("numeric_mappings"),
        time_gap_source=feat_cfg.get("time_gap_source", "timestamp_diff"),
        time_features_zero_indexed=feat_cfg.get("time_features_zero_indexed", []),
        preprocessed=feat_cfg.get("preprocessed", False),
    )


def prepare_datasets(df: pd.DataFrame, data_cfg: dict, feat_cfg: dict, train_cfg: dict, k_past: int):
    entity_col = data_cfg["entity_column"]
    df = ensure_time_gap_column(df, data_cfg, feat_cfg)

    train_entities, val_entities = get_train_val_entities(df, data_cfg, train_cfg)
    preprocessor = build_preprocessor(feat_cfg)
    preprocessor.fit(df[df[entity_col].isin(train_entities)])

    train_df = preprocessor.transform(df[df[entity_col].isin(train_entities)])
    val_df = preprocessor.transform(df[df[entity_col].isin(val_entities)])

    cat_cols = list(feat_cfg["categorical"].keys())
    num_cols = feat_cfg["numeric_columns"]

    train_sequences = build_sequence_groups(train_df, entity_col)
    val_sequences = build_sequence_groups(val_df, entity_col)

    return {
        "preprocessor": preprocessor,
        "train_entities": train_entities,
        "val_entities": val_entities,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "entity_col": entity_col,
        "k_past": k_past,
    }


def prepare_all_sequences(df: pd.DataFrame, data_cfg: dict, feat_cfg: dict, train_cfg: dict, k_past: int):
    entity_col = data_cfg["entity_column"]
    df = ensure_time_gap_column(df, data_cfg, feat_cfg)

    train_entities, _ = get_train_val_entities(df, data_cfg, train_cfg)
    preprocessor = build_preprocessor(feat_cfg)
    preprocessor.fit(df[df[entity_col].isin(train_entities)])
    df = preprocessor.transform(df)

    cat_cols = list(feat_cfg["categorical"].keys())
    num_cols = feat_cfg["numeric_columns"]
    sequences = build_sequence_groups(df, entity_col)

    return {
        "df": df,
        "sequences": sequences,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "entity_col": entity_col,
        "k_past": k_past,
    }
