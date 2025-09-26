import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int, initialize: False):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, input_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(1))  # General scale

        if initialize:
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

        # Recommended by paper
        #torch.nn.init.uniform_(self.alpha, 0, 2)
        #torch.nn.init.uniform_(self.beta, 0, 2)
        #torch.nn.init.uniform_(self.theta, 0, 2)

    # Use the vectorized method in forward, this is here for example only.
    def tversky_similarity(self, a, b):
        # Feature activations: how much each feature is present in each object
        a_features = a @ self.features.T  # [batch, num_features]
        b_features = b @ self.features.T  # [batch, num_features]

        # Binary presence: feature is "present" if activation > 0
        a_present = torch.nn.functional.relu(a_features)
        b_present = torch.nn.functional.relu(b_features)

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

    # Original fast forward, has intermediates that are too large to be practical
    # def forward(self, x):
    #     batch_size, input_dim = x.shape
    #     num_prototypes = self.prototypes.shape[0]

    #     # Expand for all pairwise comparisons
    #     x_expanded = x.unsqueeze(1).expand(-1, num_prototypes, -1)  # [batch, num_prototypes, input_dim]
    #     proto_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_prototypes, input_dim]

    #     # Feature activations for inputs and prototypes
    #     x_features = torch.einsum('bpi,fi->bpf', x_expanded, self.features)  # [batch, num_prototypes, num_features]
    #     p_features = torch.einsum('bpi,fi->bpf', proto_expanded, self.features)  # [batch, num_prototypes, num_features]

    #     # a · fk > 0
    #     x_present = torch.nn.functional.relu(x_features)
    #     p_present = torch.nn.functional.relu(p_features)

    #     # Set operations
    #     both_present = x_present * p_present  # A ∩ B
    #     x_only = x_present * (1 - p_present)  # A - B
    #     p_only = p_present * (1 - x_present)  # B - A

    #     # Feature measures
    #     common = (x_features * p_features * both_present).sum(dim=2)  # [batch, num_prototypes]
    #     x_distinctive = (x_features * x_only).sum(dim=2)  # [batch, num_prototypes]
    #     p_distinctive = (p_features * p_only).sum(dim=2)  # [batch, num_prototypes]

    #     # Tversky similarity for all pairs
    #     return self.theta * common - self.alpha * x_distinctive - self.beta * p_distinctive

    # Faster version, we need to then chunk this over the prototype dimension but this is much better.
    def forward(self, x):
        batch_size = x.shape[0]

        x_features = torch.matmul(x, self.features.T)  # [batch, features]
        p_features = torch.matmul(self.prototypes, self.features.T)  # [prototypes, features]
        x_present = F.relu(x_features)  # [batch, features]
        p_present = F.relu(p_features)  # [prototypes, features]

        # Reformulated to avoid materializing large tensors:
        # Original: (x_features * p_features * both_present).sum(dim=2)
        # where both_present = x_present * p_present (broadcasted to [batch, prototypes, features])
        # Equivalent: sum_f(x_features[f] * p_features[f] * x_present[f] * p_present[f])
        # = sum_f((x_features[f] * x_present[f]) * (p_features[f] * p_present[f]))
        # = (x_features * x_present) @ (p_features * p_present).T

        x_weighted = x_features * x_present  # [batch, features]
        p_weighted = p_features * p_present  # [prototypes, features]
        common = torch.matmul(x_weighted, p_weighted.T)  # [batch, prototypes]

        # Original: (x_features * x_only).sum(dim=2)
        # where x_only = x_present * (1 - p_present) (broadcasted)
        # = sum_f(x_features[f] * x_present[f] * (1 - p_present[f]))
        # = sum_f(x_features[f] * x_present[f]) - sum_f(x_features[f] * x_present[f] * p_present[f])
        # = x_weighted.sum(1) - (x_weighted @ p_present.T)

        x_weighted_sum = x_weighted.sum(dim=1, keepdim=True)  # [batch, 1]
        x_p_interaction = torch.matmul(x_weighted, p_present.T)  # [batch, prototypes]
        x_distinctive = x_weighted_sum - x_p_interaction  # [batch, prototypes]

        # Original: (p_features * p_only).sum(dim=2)
        # where p_only = p_present * (1 - x_present) (broadcasted)
        # = sum_f(p_features[f] * p_present[f] * (1 - x_present[f]))
        # = sum_f(p_features[f] * p_present[f]) - sum_f(p_features[f] * p_present[f] * x_present[f])
        # = p_weighted.sum(1) - (x_present @ p_weighted.T)

        p_weighted_sum = p_weighted.sum(dim=1).unsqueeze(0)  # [1, prototypes]
        x_p_weighted_interaction = torch.matmul(x_present, p_weighted.T)  # [batch, prototypes]
        p_distinctive = p_weighted_sum - x_p_weighted_interaction  # [batch, prototypes]

        return self.theta * common - self.alpha * x_distinctive - self.beta * p_distinctive
