"""
Simplified InfoNCE loss for temporal emotion contrastive learning
FIXED: Reports RAW cosine similarities (not temperature-scaled)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class InfoNCELoss(nn.Module):
    """InfoNCE Loss with in-batch negatives"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute InfoNCE loss
        
        Returns RAW cosine similarities in metrics (range [-1, 1])
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device
        
        # Normalize
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        num_positives_total = positive_embeddings.shape[0]
        num_pos_per_anchor = num_positives_total // batch_size
        
        # Reshape positives
        positive_embeddings = positive_embeddings.view(batch_size, num_pos_per_anchor, -1)
        
        # Compute anchor-anchor similarities (RAW, for negatives)
        anchor_anchor_sim = torch.mm(anchor_embeddings, anchor_embeddings.T)
        
        # Create mask to exclude self-similarities
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Loss computation
        losses = []
        pos_sims_raw = []
        neg_sims_raw = []
        
        for i in range(batch_size):
            anchor = anchor_embeddings[i:i+1]  # (1, embed_dim)
            
            # Positive similarities: RAW cosine (before temperature)
            pos_sim_raw = torch.mm(anchor, positive_embeddings[i].T).squeeze(0)  # (num_pos,)
            
            # Negative similarities: RAW (from pre-computed matrix, exclude self)
            neg_sim_raw = anchor_anchor_sim[i][~mask[i]]  # (batch_size - 1,)
            
            # Store RAW similarities for metrics
            pos_sims_raw.append(pos_sim_raw.mean().item())
            neg_sims_raw.append(neg_sim_raw.mean().item())
            
            # Apply temperature for loss computation
            pos_sim = pos_sim_raw / self.temperature
            neg_sim = neg_sim_raw / self.temperature
            
            # InfoNCE loss
            for pos_score in pos_sim:
                numerator = torch.exp(pos_score)
                denominator = numerator + torch.sum(torch.exp(neg_sim))
                loss = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss)
        
        total_loss = torch.stack(losses).mean()
        
        # Metrics use RAW similarities
        mean_pos = sum(pos_sims_raw) / len(pos_sims_raw)
        mean_neg = sum(neg_sims_raw) / len(neg_sims_raw)
        
        metrics = {
            'loss': total_loss.item(),
            'mean_pos_sim': mean_pos,
            'mean_neg_sim': mean_neg,
            'pos_neg_gap': mean_pos - mean_neg,
            'temperature': self.temperature,
            'num_positives': num_positives_total,
            'num_negatives': batch_size - 1,
        }
        
        return total_loss, metrics


def compute_contrastive_accuracy(anchor_embeddings: torch.Tensor,
                                 positive_embeddings: torch.Tensor,
                                 top_k: int = 1) -> float:
    """Compute contrastive accuracy"""
    batch_size = anchor_embeddings.shape[0]
    num_pos_total = positive_embeddings.shape[0]
    num_pos_per_anchor = num_pos_total // batch_size
    
    anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
    positive_embeddings = positive_embeddings.view(batch_size, num_pos_per_anchor, -1)
    
    correct = 0
    for i in range(batch_size):
        anchor = anchor_embeddings[i:i+1]
        pos_sim = torch.mm(anchor, positive_embeddings[i].T).squeeze(0)
        min_pos_sim = pos_sim.min().item()
        
        neg_sim = torch.mm(anchor, anchor_embeddings.T).squeeze(0)
        neg_sim[i] = -float('inf')
        max_neg_sim = neg_sim.max().item()
        
        if min_pos_sim > max_neg_sim:
            correct += 1
    
    return correct / batch_size


if __name__ == "__main__":
    print("Testing loss...")
    loss_fn = InfoNCELoss(0.07)
    anchors = torch.randn(8, 128, requires_grad=True)
    positives = torch.randn(16, 128, requires_grad=True)
    
    loss, metrics = loss_fn(anchors, positives)
    print(f"Loss: {loss.item():.4f}")
    print(f"Pos sim: {metrics['mean_pos_sim']:.4f} (should be ~-0.5 to 0.5)")
    print(f"Neg sim: {metrics['mean_neg_sim']:.4f} (should be ~-0.3 to 0.3)")
    
    if abs(metrics['mean_pos_sim']) < 2.0:
        print(" CORRECT - similarities in valid range!")
    else:
        print(" WRONG - similarities too high!")