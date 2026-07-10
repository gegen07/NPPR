from pathlib import Path

import pandas as pd
import torch


@torch.inference_mode()
def extract_entity_embeddings(model, dataloader, device, pool: str):
    model.eval()
    embeddings = []
    entity_lengths = []

    for batch in dataloader:
        cat_inputs, num_inputs, lengths = batch
        cat_inputs = {feat: t.to(device, non_blocking=True) for feat, t in cat_inputs.items()}
        num_inputs = num_inputs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        emb = model.embed(cat_inputs, num_inputs, pool=pool, lengths=lengths)
        embeddings.append(emb.cpu())
        entity_lengths.append(lengths.cpu())
        del cat_inputs, num_inputs, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0), torch.cat(entity_lengths, dim=0)


@torch.inference_mode()
def extract_transaction_embeddings_to_dataframe(
    model,
    dataloader,
    dataset,
    device,
    output_path: Path,
    embed_dim: int,
    transaction_id_column: str,
    entity_column: str,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embed_columns = [f"emb_{i}" for i in range(embed_dim)]
    writer = None
    total_rows = 0
    seq_idx = 0

    for batch in dataloader:
        cat_inputs, num_inputs, lengths = batch
        cat_inputs = {feat: t.to(device, non_blocking=True) for feat, t in cat_inputs.items()}
        num_inputs = num_inputs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        step_embeddings = model.embed(
            cat_inputs, num_inputs, pool=None, lengths=lengths
        ).float().cpu().numpy()
        lengths = lengths.cpu().numpy()

        for i in range(step_embeddings.shape[0]):
            sample = dataset.samples[seq_idx]
            seq_len = int(lengths[i])
            chunk_df = pd.DataFrame(
                step_embeddings[i, :seq_len],
                columns=embed_columns,
            )
            chunk_df.insert(0, entity_column, sample["customer_id"])
            chunk_df.insert(0, transaction_id_column, sample["transaction_ids"][:seq_len])

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += seq_len
            seq_idx += 1

        del cat_inputs, num_inputs, step_embeddings, lengths
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()

    return {
        "total_transactions": total_rows,
        "embed_dim": embed_dim,
        "output_path": str(output_path),
        "transaction_id_column": transaction_id_column,
        "entity_column": entity_column,
        "embedding_columns": embed_columns,
    }


def entity_embeddings_to_dataframe(
    embeddings,
    entity_ids,
    entity_column: str,
    embed_dim: int,
    lengths=None,
    pool: str | None = None,
) -> pd.DataFrame:
    embed_columns = [f"emb_{i}" for i in range(embed_dim)]
    df = pd.DataFrame(embeddings.numpy(), columns=embed_columns)
    df.insert(0, entity_column, entity_ids)
    if lengths is not None:
        df["sequence_length"] = lengths.numpy()
    if pool is not None:
        df["pool"] = pool
    return df
