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
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=1e-16).sqrt()

        # Mask for valid triplets
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_anchor_positive = labels_equal.float()
        mask_anchor_negative = 1 - labels_equal.float()
        
    
        # Hardest Positive per anchor
        anchor_positive_dist = distances * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # Hardest Negative per anchor
        max_dist = distances.max().item()
        anchor_negative_dist = distances + max_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        
        # Compute Loss
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return loss.mean()
