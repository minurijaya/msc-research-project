import torch
import numpy as np

def compute_recall_at_k(anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor, k_values=[1, 5, 10]):
    """
    Computes Recall@K for retrieval.
    
    Args:
        anchor_embeddings: (N, D) embeddings (e.g., text)
        target_embeddings: (M, D) embeddings (e.g., images)
        
    Assuming 1-to-1 correspondence where anchor[i] matches target[i].
    """
    # Normalize embeddings
    anchor_embeddings = anchor_embeddings / anchor_embeddings.norm(dim=-1, keepdim=True)
    target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute Similarity Matrix (N, M)
    logits = anchor_embeddings @ target_embeddings.t()
    
    n_samples = logits.shape[0]
    # Ground truth: diagonal elements are positives if N=M and aligned
    # For general case, we assume anchor[i] should retrieve target[i]
    targets = torch.arange(n_samples, device=logits.device)
    
    recalls = {}
    for k in k_values:
        _, indices = logits.topk(k, dim=1)
        correct = indices.eq(targets.view(-1, 1).expand_as(indices))
        recall = correct.sum().float() / n_samples
        recalls[f"R@{k}"] = recall.item()
        
    return recalls

def compute_mrr(anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor):
    """
    Computes Mean Reciprocal Rank.
    """
    anchor_embeddings = anchor_embeddings / anchor_embeddings.norm(dim=-1, keepdim=True)
    target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)
    
    logits = anchor_embeddings @ target_embeddings.t()
    n_samples = logits.shape[0]
    targets = torch.arange(n_samples, device=logits.device)
    
    # Sort all to find rank
    sorted_indices = logits.argsort(dim=1, descending=True)
    # Find where targets are
    ranks = (sorted_indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1]
    # Ranks are 0-indexed, so +1
    reciprocal_ranks = 1.0 / (ranks.float() + 1.0)
    mrr = reciprocal_ranks.mean().item()
    
    return mrr
