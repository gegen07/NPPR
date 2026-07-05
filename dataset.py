import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def compute_accumulated_gaps(time_gaps, k_past: int) -> np.ndarray:
    """Compute delta_{t, t-k} = sum of inter-event gaps from t-k+1 through t."""
    time_gaps = np.asarray(time_gaps, dtype=np.float32)
    seq_len = len(time_gaps)
    delta_matrix = np.zeros((seq_len, k_past), dtype=np.float32)
    if seq_len == 0:
        return delta_matrix

    cumsum = np.concatenate(([0.0], np.cumsum(time_gaps)))
    for k in range(1, k_past + 1):
        t_indices = np.arange(k, seq_len)
        delta_matrix[t_indices, k - 1] = cumsum[t_indices + 1] - cumsum[t_indices - k + 1]

    return delta_matrix


def _sequence_to_tensors(seq, categorical_columns, numeric_columns, time_gap_column, k_past):
    cat_inputs = {
        col: torch.tensor(seq[col].values, dtype=torch.long)
        for col in categorical_columns
    }
    num_inputs = torch.tensor(
        seq[numeric_columns].values,
        dtype=torch.float,
    )
    delta_matrix = torch.tensor(
        compute_accumulated_gaps(seq[time_gap_column].values, k_past),
        dtype=torch.float,
    )
    return cat_inputs, num_inputs, delta_matrix, len(seq)


class TransactionDataset(Dataset):
    def __init__(
        self,
        sequences,
        categorical_columns: list[str],
        numeric_columns: list[str],
        time_gap_column: str = "Time_Gap",
        k_past: int = 10,
    ):
        self.sequences = list(sequences)
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.time_gap_column = time_gap_column
        self.k_past = k_past
        self.items = [
            _sequence_to_tensors(
                seq,
                categorical_columns,
                numeric_columns,
                time_gap_column,
                k_past,
            )
            for seq in self.sequences
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    cat_inputs_batch = [item[0] for item in batch]
    num_inputs_batch = [item[1] for item in batch]
    delta_matrix_batch = [item[2] for item in batch]
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)

    padded_cat_inputs = {}
    for key in cat_inputs_batch[0].keys():
        sequences = [item[key] for item in cat_inputs_batch]
        padded_cat_inputs[key] = pad_sequence(sequences, batch_first=True, padding_value=0)

    padded_num_inputs = pad_sequence(num_inputs_batch, batch_first=True, padding_value=0.0)
    padded_delta_matrix = pad_sequence(delta_matrix_batch, batch_first=True, padding_value=0.0)

    return padded_cat_inputs, padded_num_inputs, padded_delta_matrix, lengths


def build_sequence_groups(df, entity_column: str):
    sort_cols = [
        col
        for col in (entity_column, "datetime", "Timestamp", "timestamp")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return [group for _, group in df.groupby(entity_column, sort=False)]
