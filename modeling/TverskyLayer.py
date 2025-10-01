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

    # Shifted indicator since we need a differntiable signal > 0 but binary mask is not such a thing.
    def indicator(self, x, k=10.0):
        return (torch.tanh(k * x) + 1) / 2

    def forward(self, x):
        B = x.size(0)

        # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
        A = x @ self.features.T          # [B, F]
        Pi = self.prototypes @ self.features.T  # [P, F]

        sigma_A = self.indicator(A)      # [B, F]
        sigma_Pi = self.indicator(Pi)    # [P, F]

        weighted_A = A * sigma_A         # [B, F]
        weighted_Pi = Pi * sigma_Pi      # [P, F]

        common = weighted_A @ weighted_Pi.T            # [B,F] @ [F,P] = [B,P]
        distinctive_A = weighted_A @ (1 - sigma_Pi).T  # [B,F] @ [F,P] = [B,P]
        distinctive_B = (1 - sigma_A) @ weighted_Pi.T  # [B,F] @ [F,P] = [B,P]

        #  S = θ·C - α·D_A - β·D_B
        return self.theta * common - self.alpha * distinctive_A - self.beta * distinctive_B # [B, P]
