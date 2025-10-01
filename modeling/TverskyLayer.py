import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int, use_cached_forward=True):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, input_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(1))  # General scale
        self.use_cached_forward = use_cached_forward

        if use_cached_forward:
            self.register_buffer('cached_matrix', None)
            self.register_buffer('cached_proto_sum', None)

        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

    # Shifted indicator, [~0, ~1] since we need a differntiable signal > 0 but binary mask is not such a thing.
    # Should probably note, the paper just glosses over what actual indicator they used, but it's probably similar.
    def indicator(self, x, k=10.0):
        return (torch.tanh(k * (x - 0.5)) + 1) / 2

    # For reference (unused)
    def fast_tversky_similarity(self, a, b):
        a_features = a @ self.features.T
        b_features = b @ self.features.T

        diff = a_features - b_features  # [B, F]

        both_pos = self.indicator(a_features) * self.indicator(b_features)

        common = (a_features * b_features * both_pos).sum(dim=1)
        a_distinctive = (diff * both_pos * self.indicator(diff)).sum(dim=1)
        b_distinctive = ((-diff) * both_pos * self.indicator(-diff)).sum(dim=1)

        return self.theta * common - self.alpha * a_distinctive - self.beta * b_distinctive

    def forward(self, x):
        B = x.size(0)
        P = self.prototypes.size(0)
        F = self.features.size(0)

        a_features = x @ self.features.T
        p_features = self.prototypes @ self.features.T

        a_ind = self.indicator(a_features)
        p_ind = self.indicator(p_features)

        weighted_a = a_features * a_ind
        weighted_p = p_features * p_ind
        common = weighted_a @ weighted_p.T

        a_exp = a_features.unsqueeze(1)
        p_exp = p_features.unsqueeze(0)
        a_ind_exp = a_ind.unsqueeze(1)
        p_ind_exp = p_ind.unsqueeze(0)

        diff = a_exp - p_exp
        both_pos = a_ind_exp * p_ind_exp

        a_distinctive = (diff * both_pos * self.indicator(diff)).sum(dim=2)
        b_distinctive = ((-diff) * both_pos * self.indicator(-diff)).sum(dim=2)

        return self.theta * common - self.alpha * a_distinctive - self.beta * b_distinctive

    def forward_chunk(self, x, chunk_size=4):
        B = x.size(0)
        P = self.prototypes.size(0)
        F = self.features.size(0)

        if chunk_size is None:
            bytes_per_element = 4
            target_bytes = 200 * 1024 * 1024
            chunk_size = max(1, target_bytes // (B * F * bytes_per_element))
            chunk_size = min(chunk_size, P)

        a_features = x @ self.features.T  # [B, F]
        a_ind = self.indicator(a_features)  # [B, F]

        similarities = torch.empty(B, P, device=x.device, dtype=x.dtype)

        for i in range(0, P, chunk_size):
            chunk_end = min(i + chunk_size, P)

            p_chunk = self.prototypes[i:chunk_end]
            p_features = p_chunk @ self.features.T  # [C, F]
            p_ind = self.indicator(p_features)  # [C, F]

            weighted_a = a_features * a_ind  # [B, F]
            weighted_p = p_features * p_ind  # [C, F]
            common = weighted_a @ weighted_p.T  # [B, C]

            # Because gating depends on per-pair differences, distinctive terms still require element-wise ops
            a_exp = a_features.unsqueeze(1)  # [B, 1, F]
            p_exp = p_features.unsqueeze(0)  # [1, C, F]
            a_ind_exp = a_ind.unsqueeze(1)   # [B, 1, F]
            p_ind_exp = p_ind.unsqueeze(0)   # [1, C, F]

            diff = a_exp - p_exp  # [B, C, F]
            both_pos = a_ind_exp * p_ind_exp  # [B, C, F]

            a_dist = (diff * both_pos * self.indicator(diff)).sum(dim=2)
            b_dist = ((-diff) * both_pos * self.indicator(-diff)).sum(dim=2)

            similarities[:, i:chunk_end] = (
                self.theta * common - self.alpha * a_dist - self.beta * b_dist
            )

        return similarities
