import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int):
        super().__init__()

        self.features = nn.Parameter(torch.randn(num_features, input_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.ones(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.ones(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.ones(1))  # General scale

    # Use the vectorized method in forward, this is here for example only.
    def tversky_similarity(self, a, b):
        # Feature activations: how much each feature is present in each object
        a_features = a @ self.features.T  # [batch, num_features]
        b_features = b @ self.features.T  # [batch, num_features]

        # Binary presence: feature is "present" if activation > 0
        a_present = torch.sigmoid(a_features)
        b_present = torch.sigmoid(b_features)

        # Set operations
        both_present = a_present * b_present  # A ∩ B
        a_only = a_present * (1 - b_present)  # A - B
        b_only = b_present * (1 - a_present)  # B - A

        # Feature measures (Paper talks about using muliple possible methods of doing this probably should investigate.)
        common = (a_features * b_features * both_present).sum(dim=1)
        a_distinctive = (a_features * a_only).sum(dim=1)
        b_distinctive = (b_features * b_only).sum(dim=1)

        # Tversky contrast model
        return self.theta * common - self.alpha * a_distinctive - self.beta * b_distinctive

    # This is here for a reference modal on a literal implementation
    def forward_bad_slow(self, x):
        batch_size = x.size(0)
        similarities = []

        for proto in self.prototypes:
            proto_batch = proto.unsqueeze(0).expand(batch_size, -1)
            sim = self.tversky_similarity(x, proto_batch)
            similarities.append(sim)

        return torch.stack(similarities, dim=1)  # [batch, num_prototypes]

    def forward(self, x):
        batch_size, input_dim = x.shape
        num_prototypes = self.prototypes.shape[0]

        # Expand for all pairwise comparisons
        x_expanded = x.unsqueeze(1).expand(-1, num_prototypes, -1)  # [batch, num_prototypes, input_dim]
        proto_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_prototypes, input_dim]

        # Feature activations for inputs and prototypes
        x_features = torch.einsum('bpi,fi->bpf', x_expanded, self.features)  # [batch, num_prototypes, num_features]
        p_features = torch.einsum('bpi,fi->bpf', proto_expanded, self.features)  # [batch, num_prototypes, num_features]

        # Differentiable presence masks, not sure if this is what the paper intended.
        x_present = torch.sigmoid(x_features)
        p_present = torch.sigmoid(p_features)

        # Set operations
        both_present = x_present * p_present  # A ∩ B
        x_only = x_present * (1 - p_present)  # A - B
        p_only = p_present * (1 - x_present)  # B - A

        # Feature measures
        common = (x_features * p_features * both_present).sum(dim=2)  # [batch, num_prototypes]
        x_distinctive = (x_features * x_only).sum(dim=2)  # [batch, num_prototypes]
        p_distinctive = (p_features * p_only).sum(dim=2)  # [batch, num_prototypes]

        # Tversky similarity for all pairs
        return self.theta * common - self.alpha * x_distinctive - self.beta * p_distinctive
