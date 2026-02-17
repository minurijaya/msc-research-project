import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, reduction="mean")

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
        Returns:
            loss: scalar
        """
        # Create pairwise distance matrix
        # dist[i, j] = ||emb[i] - emb[j]||_2
        # (a-b)^2 = a^2 + b^2 - 2ab
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=1e-16).sqrt()

        # Mask for valid triplets
        # valid_positive: labels[i] == labels[j] and i != j
        # valid_negative: labels[i] != labels[j]
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_anchor_positive = labels_equal.float()
        mask_anchor_negative = 1 - labels_equal.float()
        
        # We don't want the diagonal (self-distance) to be the hardest positive (it's 0)
        # So we set diagonal to 0 in mask_anchor_positive? No, we want to maximize distance for hard positive
        # Hardest positive: Max distance for same label
        # Hardest negative: Min distance for diff label
        
        # Hardest Positive
        # We replace 0s (negatives) with -inf to ignore them in max
        anchor_positive_dist = distances * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # Hardest Negative
        # We replace 0s (positives + self) with +inf to ignore them in min
        max_dist = distances.max().item()
        anchor_negative_dist = distances + max_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        
        # Compute Loss
        # We can use functional triplet loss or manual
        # loss = max(0, hp - hn + margin)
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return loss.mean()
