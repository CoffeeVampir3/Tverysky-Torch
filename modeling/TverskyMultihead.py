import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def tversky_multihead_similarity(x, features, prototypes, theta, alpha, beta, n_heads):
    batch_size, total_dim = x.shape
    total_prototypes = prototypes.shape[0]
    prototypes_per_head = total_prototypes // n_heads

    x_features = x @ features.T  # [batch, num_features]
    x_present = F.relu(x_features)  # [batch, num_features]
    x_weighted = x_features * x_present  # [batch, num_features]
    x_weighted_sum = x_weighted.sum(dim=1, keepdim=True)  # [batch, 1]

    features_T = features.T  # [total_dim, num_features]

    result = torch.zeros(batch_size, total_prototypes, device=x.device, dtype=x.dtype)

    for head in range(n_heads):
        start_idx = head * prototypes_per_head
        end_idx = start_idx + prototypes_per_head

        head_prototypes = prototypes[start_idx:end_idx]  # [prototypes_per_head, total_dim]

        p_features = head_prototypes @ features_T
        p_present = F.relu(p_features)
        p_weighted = p_features * p_present

        combined_p = theta[head] * p_weighted + alpha[head] * p_present
        p_weighted_sum = p_weighted.sum(dim=1)  # [prototypes_per_head]

        term1 = x_weighted @ combined_p.T  # [batch, prototypes_per_head]
        term2 = x_present @ p_weighted.T   # [batch, prototypes_per_head]

        head_result = (term1 + beta[head] * term2 -
                      alpha[head] * x_weighted_sum -
                      beta[head] * p_weighted_sum.unsqueeze(0))

        result[:, start_idx:end_idx] = head_result

    return result # [batch_size, total_prototypes]

class TverskyMultihead(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, num_prototypes: int, num_features: int):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, hidden_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(n_heads, 1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(n_heads, 1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(n_heads, 1))  # General scale
        self.n_heads = n_heads

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

    def forward(self, x):
        batch_size, features = x.shape
        return tversky_multihead_similarity(x, self.features, self.prototypes, self.theta, self.alpha, self.beta, self.n_heads)
