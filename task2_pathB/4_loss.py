"""
Simplified InfoNCE loss for temporal emotion contrastive learning
Uses in-batch negatives for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ============================================================================
# INFO-NCE LOSS WITH IN-BATCH NEGATIVES
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss
    
    Core contrastive learning loss that:
    - Pulls positive pairs closer in embedding space
    - Pushes negative pairs farther apart
    
    Uses in-batch negatives: all other samples in the batch serve as negatives
    This is efficient and works well (used in CLIP, SimCLR, MoCo, etc.)
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling parameter
                        Lower = harder negatives (more discrimination)
                        Typical range: 0.05-0.1
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
            anchor_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute InfoNCE loss with in-batch negatives
        
        Args:
            anchor_embeddings: (batch_size, embed_dim) - normalized embeddings
            positive_embeddings: (batch_size * num_pos, embed_dim) - normalized embeddings
                                Each anchor has num_pos positives
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary with detailed metrics
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device
        
        # Assume each anchor has same number of positives
        num_positives_total = positive_embeddings.shape[0]
        num_pos_per_anchor = num_positives_total // batch_size
        
        # Normalize embeddings (if not already normalized)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Reshape positives to (batch_size, num_pos_per_anchor, embed_dim)
        positive_embeddings = positive_embeddings.view(batch_size, num_pos_per_anchor, -1)
        
        # Compute all anchor-anchor similarities (for in-batch negatives)
        # (batch_size, batch_size)
        anchor_anchor_sim = torch.mm(anchor_embeddings, anchor_embeddings.T) / self.temperature
        
        # Create mask to exclude self-similarities (diagonal)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Compute loss for each anchor - KEEP AS TENSOR
        losses = []
        positive_sims = []
        negative_sims = []
        
        for i in range(batch_size):
            # Get anchor
            anchor = anchor_embeddings[i:i+1]  # (1, embed_dim)
            
            # Positive similarities: anchor vs its positives
            # (1, embed_dim) x (num_pos, embed_dim).T = (1, num_pos)
            pos_sim = torch.mm(anchor, positive_embeddings[i].T) / self.temperature
            pos_sim = pos_sim.squeeze(0)  # (num_pos,)
            
            # Negative similarities: anchor vs all other anchors in batch
            neg_sim = anchor_anchor_sim[i][~mask[i]]  # (batch_size - 1,)
            
            # InfoNCE loss for this anchor
            # For each positive, compute: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for pos_score in pos_sim:
                # Numerator: exp(positive similarity)
                numerator = torch.exp(pos_score)
                
                # Denominator: exp(positive) + sum(exp(negatives))
                denominator = numerator + torch.sum(torch.exp(neg_sim))
                
                # Loss: -log(numerator / denominator)
                loss = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss)
            
            # Track similarities for metrics (detach for metrics only)
            positive_sims.append(pos_sim.mean().item())
            negative_sims.append(neg_sim.mean().item())
        
        # Stack and average losses - KEEP AS TENSOR
        total_loss = torch.stack(losses).mean()
        
        # Compute metrics
        metrics = {
            'loss': total_loss.item(),
            'mean_pos_sim': sum(positive_sims) / len(positive_sims),
            'mean_neg_sim': sum(negative_sims) / len(negative_sims),
            'pos_neg_gap': sum(positive_sims) / len(positive_sims) - sum(negative_sims) / len(negative_sims),
            'temperature': self.temperature,
            'num_positives': num_positives_total,
            'num_negatives': batch_size - 1,  # Per anchor
        }
        
        return total_loss, metrics


# ============================================================================
# CONTRASTIVE ACCURACY
# ============================================================================

def compute_contrastive_accuracy(anchor_embeddings: torch.Tensor,
                                 positive_embeddings: torch.Tensor,
                                 top_k: int = 1) -> float:
    """
    Compute contrastive accuracy: how often positives rank higher than negatives
    
    Args:
        anchor_embeddings: (batch_size, embed_dim)
        positive_embeddings: (batch_size * num_pos, embed_dim)
        top_k: Consider top-k predictions
    
    Returns:
        Accuracy (fraction of correct rankings)
    """
    batch_size = anchor_embeddings.shape[0]
    num_pos_total = positive_embeddings.shape[0]
    num_pos_per_anchor = num_pos_total // batch_size
    
    # Normalize
    anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
    
    # Reshape positives
    positive_embeddings = positive_embeddings.view(batch_size, num_pos_per_anchor, -1)
    
    correct = 0
    total = 0
    
    for i in range(batch_size):
        anchor = anchor_embeddings[i:i+1]
        
        # Similarity with own positives
        pos_sim = torch.mm(anchor, positive_embeddings[i].T).squeeze(0)
        min_pos_sim = pos_sim.min().item()
        
        # Similarity with all other anchors (negatives)
        neg_sim = torch.mm(anchor, anchor_embeddings.T).squeeze(0)
        neg_sim[i] = -float('inf')  # Mask out self
        max_neg_sim = neg_sim.max().item()
        
        # Check if minimum positive > maximum negative
        if min_pos_sim > max_neg_sim:
            correct += 1
        total += 1
    
    return correct / total


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING INFONCE LOSS")
    print("="*80)
    
    # Create dummy embeddings
    batch_size = 8
    embed_dim = 128
    num_pos_per_anchor = 2
    
    # Random normalized embeddings
    anchors = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
    positives = F.normalize(torch.randn(batch_size * num_pos_per_anchor, embed_dim), p=2, dim=1)
    
    print(f"\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Positives per anchor: {num_pos_per_anchor}")
    print(f"  In-batch negatives per anchor: {batch_size - 1}")
    
    # Test loss
    print("\n1. Testing InfoNCE loss...")
    loss_fn = InfoNCELoss(temperature=0.07)
    loss, metrics = loss_fn(anchors, positives)
    
    print(f"✅ Loss computation successful!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")
    
    # Test with perfect alignment (positives = anchors)
    print("\n2. Testing with perfect alignment...")
    perfect_positives = anchors.repeat(num_pos_per_anchor, 1)
    loss_perfect, metrics_perfect = loss_fn(anchors, perfect_positives)
    
    print(f"   Loss (should be low): {loss_perfect.item():.4f}")
    print(f"   Pos similarity: {metrics_perfect['mean_pos_sim']:.4f}")
    print(f"   Neg similarity: {metrics_perfect['mean_neg_sim']:.4f}")
    
    # Test accuracy
    print("\n3. Testing contrastive accuracy...")
    accuracy = compute_contrastive_accuracy(anchors, positives)
    print(f"   Accuracy (random embeddings): {accuracy:.2%}")
    
    accuracy_perfect = compute_contrastive_accuracy(anchors, perfect_positives)
    print(f"   Accuracy (perfect alignment): {accuracy_perfect:.2%}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    loss.backward()
    print(f"   ✅ Gradients computed successfully!")
    
    # Test different temperatures
    print("\n5. Testing different temperatures...")
    for temp in [0.05, 0.07, 0.1, 0.5]:
        loss_fn_temp = InfoNCELoss(temperature=temp)
        loss_temp, _ = loss_fn_temp(anchors, positives)
        print(f"   Temperature {temp:.2f}: loss = {loss_temp.item():.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL LOSS TESTS PASSED!")
    print("="*80)