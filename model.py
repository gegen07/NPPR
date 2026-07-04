import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_loss_like(tensor):
    return tensor.sum() * 0.0


class Preprocess(nn.Module):
    def __init__(self, categorical_features: dict, num_numeric: int, embed_dim: int = 16):
        super().__init__()
        self.categorical_features = list(categorical_features.keys())
        self.cardinalities = dict(categorical_features)
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
            for feat, num_categories in categorical_features.items()
        })
        self.numeric_proj = nn.Linear(num_numeric, embed_dim) if num_numeric > 0 else None

        self.output_dim = len(categorical_features) * embed_dim + (embed_dim if num_numeric > 0 else 0)

    def forward(self, cat_inputs: dict, num_inputs=None):
        parts = []
        for feat, emb in self.embeddings.items():
            values = cat_inputs[feat].long()
            max_id = self.cardinalities[feat]
            if values.numel() and (values.min() < 0 or values.max() > max_id):
                raise ValueError(
                    f"{feat} contains category ids outside [0, {max_id}]. "
                    "Use 0 only for padding and 1..num_categories for observed values."
                )
            parts.append(emb(values))
        if num_inputs is not None and self.numeric_proj is not None:
            parts.append(self.numeric_proj(num_inputs))
        if not parts:
            raise ValueError("At least one categorical or numeric feature is required.")
        return torch.cat(parts, dim=-1)


def sequence_mask(cat_targets: dict, num_targets=None, lengths=None):
    """Infer valid timesteps from padded categorical features or explicit lengths."""
    if lengths is not None:
        max_len = next(iter(cat_targets.values())).size(1) if cat_targets else num_targets.size(1)
        steps = torch.arange(max_len, device=lengths.device)
        return steps.unsqueeze(0) < lengths.unsqueeze(1)

    masks = [target.ne(0) for target in cat_targets.values()]
    if masks:
        return torch.stack(masks, dim=0).any(dim=0)
    if num_targets is None:
        raise ValueError("Cannot infer a sequence mask without categorical or numeric targets.")
    raise ValueError("Numeric-only batches require explicit sequence lengths.")


class InputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class OutputMLP(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__()
        self.mlp = InputMLP(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.mlp(x)
        hidden_states, _ = self.gru(x)
        return self.projection(hidden_states)


class DecoderNP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, categorical_features: dict, num_numeric: int):
        super().__init__()

        self.prediction_mlp = OutputMLP(input_dim, hidden_dim)

        self.cat_heads = nn.ModuleDict({
            feat: nn.Linear(hidden_dim, num_categories)
            for feat, num_categories in categorical_features.items()
        })

        self.num_head = nn.Linear(hidden_dim, num_numeric) if num_numeric > 0 else None

        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, e_t, return_probs: bool = False):
        x = self.prediction_mlp(e_t)

        cat_pred = {}
        for feat, head in self.cat_heads.items():
            logits = head(x)
            cat_pred[feat] = F.softmax(logits, dim=-1) if return_probs else logits

        num_pred = self.num_head(x) if self.num_head is not None else None
        return cat_pred, num_pred

    def loss(self, cat_pred, num_pred, cat_true, num_true=None, valid_mask=None):
        if valid_mask is None:
            valid_mask = sequence_mask(cat_true, num_true)

        transition_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
        n_transitions = transition_valid.sum().clamp_min(1)

        reference = next(iter(cat_pred.values())) if cat_pred else num_pred
        per_step = reference.new_zeros(reference.size(0), valid_mask.size(1) - 1)

        for feat, pred in cat_pred.items():
            true_next = cat_true[feat][:, 1:]
            pred_next = pred[:, :-1, :]
            adjusted_true = torch.where(true_next == 0, -1, true_next - 1)
            loss = self.ce_loss(
                pred_next.reshape(-1, pred_next.size(-1)),
                adjusted_true.reshape(-1),
            ).reshape(pred_next.size(0), -1)
            per_step = per_step + loss

        if num_pred is not None and num_true is not None:
            num_loss = self.mse_loss(num_pred[:, :-1, :], num_true[:, 1:, :])
            per_step = per_step + num_loss.sum(dim=-1)

        return (per_step * transition_valid).sum() / n_transitions


class DecoderPR(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, categorical_features: dict, num_numeric: int):
        super().__init__()

        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.cat_heads = nn.ModuleDict({
            feat: nn.Linear(hidden_dim, num_categories)
            for feat, num_categories in categorical_features.items()
        })

        self.num_head = nn.Linear(hidden_dim, num_numeric) if num_numeric > 0 else None

        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, e_t, delta_matrix, return_probs: bool = False):
        k_past = delta_matrix.shape[-1]

        all_cat_preds = {feat: [] for feat in self.cat_heads}
        all_num_preds = []

        for k in range(k_past):
            delta_k = delta_matrix[:, :, k].unsqueeze(-1)
            x = torch.cat([e_t, delta_k], dim=-1)
            x = self.reconstruction_mlp(x)

            for feat, head in self.cat_heads.items():
                logits = head(x)
                all_cat_preds[feat].append(
                    F.softmax(logits, dim=-1) if return_probs else logits
                )

            if self.num_head is not None:
                all_num_preds.append(self.num_head(x))

        return all_cat_preds, all_num_preds

    def loss(self, all_cat_preds, all_num_preds, cat_targets, num_targets, delta_matrix, lam=5_184_000, valid_mask=None):
        k_past = delta_matrix.shape[-1]
        seq_len = delta_matrix.shape[1]
        valid_mask = sequence_mask(cat_targets, num_targets) if valid_mask is None else valid_mask

        total_loss = delta_matrix.new_tensor(0.0)
        total_weight = delta_matrix.new_tensor(0.0)

        for k in range(k_past):
            offset = k + 1
            if seq_len <= offset:
                continue

            delta_k = delta_matrix[:, :, k]
            omega = torch.exp(-delta_k / lam)
            pair_mask = (
                valid_mask[:, offset:]
                & valid_mask[:, :-offset]
                & (delta_matrix[:, offset:, k] > 0)
            )
            if not pair_mask.any():
                continue

            omega_weights = omega[:, offset:] * pair_mask.float()
            feat_sum = omega_weights.new_zeros(omega_weights.shape)

            for feat in all_cat_preds:
                pred = all_cat_preds[feat][k][:, offset:]
                true = cat_targets[feat][:, :-offset]
                adjusted_true = torch.where(true == 0, -1, true - 1)
                loss_k = self.ce_loss(
                    pred.reshape(-1, pred.size(-1)),
                    adjusted_true.reshape(-1),
                ).reshape(pred.size(0), -1)
                feat_sum = feat_sum + loss_k

            if all_num_preds:
                pred = all_num_preds[k][:, offset:]
                target = num_targets[:, :-offset]
                loss_k = self.mse_loss(pred, target).sum(dim=-1)
                feat_sum = feat_sum + loss_k

            total_loss = total_loss + (omega_weights * feat_sum).sum()
            total_weight = total_weight + omega_weights.sum()

        if total_weight.item() == 0:
            source = next(iter(next(iter(all_cat_preds.values()))), None) if all_cat_preds else None
            if source is None and all_num_preds:
                source = all_num_preds[0]
            return _zero_loss_like(source if source is not None else delta_matrix)

        return total_loss / total_weight.clamp_min(1)


class NPPRModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        categorical_features: dict,
        num_numeric: int,
        feature_embed_dim: int = 16,
        pr_weight: float = 0.001,
        decay_length: float = 5_184_000,
        gru_num_layers: int = 1,
    ):
        super().__init__()
        if not 0 < pr_weight < 1:
            raise ValueError("pr_weight is the paper's alpha and must be in (0, 1).")
        self.pr_weight = pr_weight
        self.decay_length = decay_length
        self.preprocessor = Preprocess(categorical_features, num_numeric, embed_dim=feature_embed_dim)
        self.encoder = Encoder(
            input_dim=self.preprocessor.output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=gru_num_layers,
        )
        self.decoder_np = DecoderNP(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            categorical_features=categorical_features,
            num_numeric=num_numeric,
        )
        self.decoder_pr = DecoderPR(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            categorical_features=categorical_features,
            num_numeric=num_numeric,
        )

    def forward(self, cat_inputs, num_inputs, delta_matrix, return_probs: bool = False):
        x = self.preprocessor(cat_inputs, num_inputs)
        e_t = self.encoder(x)

        cat_preds_np, num_preds_np = self.decoder_np(e_t, return_probs=return_probs)
        cat_preds_pr, num_preds_pr = self.decoder_pr(e_t, delta_matrix, return_probs=return_probs)

        return cat_preds_np, num_preds_np, cat_preds_pr, num_preds_pr, e_t

    def loss(self, outputs, cat_targets, num_targets, delta_matrix, lengths=None):
        cat_preds_np, num_preds_np, cat_preds_pr, num_preds_pr, _ = outputs
        valid_mask = sequence_mask(cat_targets, num_targets, lengths)
        np_loss = self.decoder_np.loss(cat_preds_np, num_preds_np, cat_targets, num_targets, valid_mask)
        pr_loss = self.decoder_pr.loss(
            cat_preds_pr,
            num_preds_pr,
            cat_targets,
            num_targets,
            delta_matrix,
            lam=self.decay_length,
            valid_mask=valid_mask,
        )
        return (1.0 - self.pr_weight) * np_loss + self.pr_weight * pr_loss

    def embed(self, cat_inputs, num_inputs=None, pool=None, lengths=None):
        e_t = self.encoder(self.preprocessor(cat_inputs, num_inputs))
        if pool is None:
            return e_t
        valid_mask = sequence_mask(cat_inputs, num_inputs, lengths).unsqueeze(-1)
        if pool == "mean":
            return (e_t * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
        if pool == "last":
            seq_lengths = valid_mask.squeeze(-1).sum(dim=1).clamp_min(1).long()
            return e_t[torch.arange(e_t.size(0), device=e_t.device), seq_lengths - 1]
        raise ValueError("pool must be one of None, 'mean', or 'last'.")
