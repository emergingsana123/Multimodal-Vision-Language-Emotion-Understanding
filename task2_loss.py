"""
task2_phase2_losses.py
Phase 2B: Loss Functions for Temporal Emotion Contrastive Learning

Implements:
1. InfoNCE Contrastive Loss
2. Temporal Smoothness Loss
3. Transition Ranking Loss
4. Pseudo-Label Consistency Loss
5. Combined Weighted Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


# ============================================================================
# INFO-NCE CONTRASTIVE LOSS
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss
    
    The core contrastive learning loss that:
    - Pulls positive pairs closer in embedding space
    - Pushes negative pairs farther apart
    
    Also known as NT-Xent loss, used in SimCLR, MoCo, etc.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling parameter
                        Lower values = harder negatives
                        Typical range: 0.05-0.1
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute InfoNCE loss
        
        Args:
            anchor_embeddings: (batch_size, embed_dim)
            positive_embeddings: (num_positives, embed_dim)
            negative_embeddings: (num_negatives, embed_dim)
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary with detailed metrics
        """
        batch_size = anchor_embeddings.shape[0]
        num_positives = positive_embeddings.shape[0]
        num_negatives = negative_embeddings.shape[0]
        
        # Normalize embeddings (if not already normalized)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute similarity matrices
        # Positive similarities: (batch_size, num_positives)
        pos_sim = torch.mm(anchor_embeddings, positive_embeddings.T) / self.temperature
        
        # Negative similarities: (batch_size, num_negatives)
        neg_sim = torch.mm(anchor_embeddings, negative_embeddings.T) / self.temperature
        
        # For each anchor, compute loss
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        losses = []
        
        for i in range(batch_size):
            # Get positives for this anchor
            anchor_pos_sim = pos_sim[i]  # (num_positives,)
            anchor_neg_sim = neg_sim[i]  # (num_negatives,)
            
            # Compute loss for each positive
            for pos_score in anchor_pos_sim:
                # Numerator: exp(positive similarity)
                numerator = torch.exp(pos_score)
                
                # Denominator: exp(positive) + sum(exp(negatives))
                denominator = numerator + torch.sum(torch.exp(anchor_neg_sim))
                
                # Loss: -log(numerator / denominator)
                loss = -torch.log(numerator / denominator + 1e-8)
                losses.append(loss)
        
        # Average loss
        total_loss = torch.stack(losses).mean()
        
        # Compute metrics for monitoring
        metrics = {
            'contrastive_loss': total_loss.item(),
            'mean_pos_sim': pos_sim.mean().item(),
            'mean_neg_sim': neg_sim.mean().item(),
            'pos_neg_gap': (pos_sim.mean() - neg_sim.mean()).item(),
            'num_positives': num_positives,
            'num_negatives': num_negatives,
        }
        
        return total_loss, metrics


# ============================================================================
# TEMPORAL SMOOTHNESS LOSS
# ============================================================================

class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal Smoothness Loss
    
    Encourages embeddings of consecutive frames to be similar
    when emotions don't change drastically.
    
    Key idea: If emotion is stable, embeddings should be stable too.
    """
    
    def __init__(self, 
                 emotion_change_weight: bool = True,
                 smoothness_type: str = 'l2'):
        """
        Args:
            emotion_change_weight: If True, weight loss by emotion change
                                  (allow larger embedding changes when emotion changes)
            smoothness_type: 'l2' or 'cosine'
        """
        super().__init__()
        self.emotion_change_weight = emotion_change_weight
        self.smoothness_type = smoothness_type
    
    def forward(self,
                embeddings_t: torch.Tensor,
                embeddings_t1: torch.Tensor,
                valence_t: torch.Tensor,
                arousal_t: torch.Tensor,
                valence_t1: torch.Tensor,
                arousal_t1: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute temporal smoothness loss
        
        Args:
            embeddings_t: Embeddings at time t (batch_size, embed_dim)
            embeddings_t1: Embeddings at time t+1 (batch_size, embed_dim)
            valence_t: Valence at time t (batch_size,)
            arousal_t: Arousal at time t (batch_size,)
            valence_t1: Valence at time t+1 (batch_size,)
            arousal_t1: Arousal at time t+1 (batch_size,)
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary with detailed metrics
        """
        batch_size = embeddings_t.shape[0]
        
        # Normalize embeddings
        embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        embeddings_t1 = F.normalize(embeddings_t1, p=2, dim=1)
        
        # Compute embedding distance
        if self.smoothness_type == 'l2':
            # L2 distance
            embedding_distance = torch.norm(embeddings_t - embeddings_t1, p=2, dim=1)
        elif self.smoothness_type == 'cosine':
            # Cosine distance (1 - cosine similarity)
            cosine_sim = F.cosine_similarity(embeddings_t, embeddings_t1, dim=1)
            embedding_distance = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown smoothness_type: {self.smoothness_type}")
        
        if self.emotion_change_weight:
            # Compute emotion change magnitude
            valence_change = torch.abs(valence_t - valence_t1)
            arousal_change = torch.abs(arousal_t - arousal_t1)
            emotion_change = torch.sqrt(valence_change**2 + arousal_change**2)
            
            # Normalize emotion change to [0, 1] range
            # Assume max emotion change is ~10 (from -10 to +10)
            emotion_change_norm = emotion_change / 10.0
            emotion_change_norm = torch.clamp(emotion_change_norm, 0, 1)
            
            # Weight: inverse of emotion change
            # If emotion doesn't change (0), weight = 1 (high penalty for embedding change)
            # If emotion changes a lot (1), weight = 0.1 (low penalty for embedding change)
            weights = 1.0 - 0.9 * emotion_change_norm
            
            # Weighted loss
            weighted_distances = embedding_distance * weights
            loss = weighted_distances.mean()
            
            metrics = {
                'smoothness_loss': loss.item(),
                'mean_embedding_distance': embedding_distance.mean().item(),
                'mean_emotion_change': emotion_change.mean().item(),
                'mean_weight': weights.mean().item(),
            }
        else:
            # Unweighted loss - penalize all embedding changes equally
            loss = embedding_distance.mean()
            
            metrics = {
                'smoothness_loss': loss.item(),
                'mean_embedding_distance': embedding_distance.mean().item(),
            }
        
        return loss, metrics


# ============================================================================
# TRANSITION RANKING LOSS
# ============================================================================

class TransitionRankingLoss(nn.Module):
    """
    Transition Ranking Loss
    
    Enforces ordinal relationships in emotion transitions.
    
    Example: For neutral → happy → excited:
    - distance(neutral, happy) < distance(neutral, excited)
    - distance(happy, excited) < distance(neutral, excited)
    
    This helps the model learn smooth emotion trajectories.
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Margin for ranking loss (larger = stricter ordering)
        """
        super().__init__()
        self.margin = margin
    
    def forward(self,
                anchor_embeddings: torch.Tensor,
                intermediate_embeddings: torch.Tensor,
                far_embeddings: torch.Tensor,
                anchor_emotions: torch.Tensor,
                intermediate_emotions: torch.Tensor,
                far_emotions: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute transition ranking loss
        
        Args:
            anchor_embeddings: Starting emotion (batch_size, embed_dim)
            intermediate_embeddings: Middle emotion (batch_size, embed_dim)
            far_embeddings: End emotion (batch_size, embed_dim)
            anchor_emotions: (batch_size, 2) - [valence, arousal]
            intermediate_emotions: (batch_size, 2)
            far_emotions: (batch_size, 2)
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary with detailed metrics
        """
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        intermediate_embeddings = F.normalize(intermediate_embeddings, p=2, dim=1)
        far_embeddings = F.normalize(far_embeddings, p=2, dim=1)
        
        # Compute embedding distances
        dist_anchor_intermediate = torch.norm(
            anchor_embeddings - intermediate_embeddings, p=2, dim=1
        )
        dist_anchor_far = torch.norm(
            anchor_embeddings - far_embeddings, p=2, dim=1
        )
        
        # Compute emotion distances (ground truth)
        emotion_dist_anchor_intermediate = torch.norm(
            anchor_emotions - intermediate_emotions, p=2, dim=1
        )
        emotion_dist_anchor_far = torch.norm(
            anchor_emotions - far_emotions, p=2, dim=1
        )
        
        # Ranking constraint: dist(A, B) + margin < dist(A, C)
        # where emotion_dist(A, B) < emotion_dist(A, C)
        
        # Find valid triplets (where intermediate is actually closer in emotion space)
        valid_mask = emotion_dist_anchor_intermediate < emotion_dist_anchor_far
        
        if valid_mask.sum() == 0:
            # No valid triplets in this batch
            return torch.tensor(0.0, device=anchor_embeddings.device), {
                'ranking_loss': 0.0,
                'num_valid_triplets': 0,
            }
        
        # Compute ranking loss (hinge loss)
        # Loss = max(0, dist(A, intermediate) - dist(A, far) + margin)
        ranking_loss = torch.relu(
            dist_anchor_intermediate - dist_anchor_far + self.margin
        )
        
        # Only use valid triplets
        ranking_loss = ranking_loss[valid_mask].mean()
        
        # Compute violation rate (how many triplets violate the ordering)
        violations = (dist_anchor_intermediate >= dist_anchor_far)[valid_mask]
        violation_rate = violations.float().mean().item()
        
        metrics = {
            'ranking_loss': ranking_loss.item(),
            'num_valid_triplets': valid_mask.sum().item(),
            'violation_rate': violation_rate,
            'mean_dist_intermediate': dist_anchor_intermediate[valid_mask].mean().item(),
            'mean_dist_far': dist_anchor_far[valid_mask].mean().item(),
        }
        
        return ranking_loss, metrics


# ============================================================================
# PSEUDO-LABEL CONSISTENCY LOSS
# ============================================================================

class PseudoLabelConsistencyLoss(nn.Module):
    """
    Pseudo-Label Consistency Loss
    
    Uses pseudo-labels from pretrained emotion models as weak supervision.
    Encourages embeddings to be similar to pseudo-emotion predictions.
    """
    
    def __init__(self, 
                 consistency_type: str = 'kl',
                 temperature: float = 1.0):
        """
        Args:
            consistency_type: 'kl' (KL divergence) or 'mse' (mean squared error)
            temperature: Temperature for softening distributions
        """
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
    
    def forward(self,
                embeddings: torch.Tensor,
                pseudo_labels: torch.Tensor,
                pseudo_confidence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute pseudo-label consistency loss
        
        Args:
            embeddings: (batch_size, embed_dim)
            pseudo_labels: (batch_size, num_emotions) - probability distribution
            pseudo_confidence: (batch_size,) - confidence scores
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary with detailed metrics
        """
        # This is a placeholder implementation
        # In practice, you would:
        # 1. Project embeddings to emotion logits
        # 2. Compute consistency between projected logits and pseudo-labels
        # 3. Weight by pseudo-confidence
        
        # For now, return zero loss if not using pseudo-labels
        device = embeddings.device
        
        metrics = {
            'pseudo_consistency_loss': 0.0,
            'mean_pseudo_confidence': pseudo_confidence.mean().item() if pseudo_confidence is not None else 0.0,
        }
        
        return torch.tensor(0.0, device=device), metrics


# ============================================================================
# COMBINED TEMPORAL CONTRASTIVE LOSS
# ============================================================================

class TemporalContrastiveLoss(nn.Module):
    """
    Combined loss for temporal emotion contrastive learning
    
    Combines:
    1. InfoNCE contrastive loss (main objective)
    2. Temporal smoothness loss (regularization)
    3. Transition ranking loss (structure learning)
    4. Pseudo-label consistency (weak supervision)
    """
    
    def __init__(self, config):
        """
        Args:
            config: TemporalEmotionConfig with loss weights
        """
        super().__init__()
        
        self.config = config
        
        # Initialize loss components
        self.info_nce = InfoNCELoss(temperature=config.temperature)
        self.temporal_smoothness = TemporalSmoothnessLoss(
            emotion_change_weight=True,
            smoothness_type='l2'
        )
        self.transition_ranking = TransitionRankingLoss(margin=0.5)
        self.pseudo_consistency = PseudoLabelConsistencyLoss()
        
        # Loss weights
        self.lambda_contrastive = config.lambda_contrastive
        self.lambda_smooth = config.lambda_temporal_smooth
        self.lambda_transition = config.lambda_transition
        self.lambda_pseudo = config.lambda_pseudo
    
    def forward(self, batch_data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        Args:
            batch_data: Dictionary containing:
                - anchor_embeddings: (batch_size, embed_dim)
                - positive_embeddings: (num_positives, embed_dim)
                - negative_embeddings: (num_negatives, embed_dim)
                - anchor_valence: (batch_size,)
                - anchor_arousal: (batch_size,)
                - (optional) temporal_pairs for smoothness loss
                - (optional) transition_triplets for ranking loss
        
        Returns:
            total_loss: Scalar tensor
            metrics: Dictionary with all metrics
        """
        device = batch_data['anchor_embeddings'].device
        total_loss = torch.tensor(0.0, device=device)
        all_metrics = {}
        
        # 1. InfoNCE Contrastive Loss (always computed)
        contrastive_loss, contrastive_metrics = self.info_nce(
            anchor_embeddings=batch_data['anchor_embeddings'],
            positive_embeddings=batch_data['positive_embeddings'],
            negative_embeddings=batch_data['negative_embeddings'],
        )
        total_loss += self.lambda_contrastive * contrastive_loss
        all_metrics.update(contrastive_metrics)
        
        # 2. Temporal Smoothness Loss (if temporal pairs provided)
        if 'temporal_pairs' in batch_data and batch_data['temporal_pairs'] is not None:
            pairs = batch_data['temporal_pairs']
            smoothness_loss, smoothness_metrics = self.temporal_smoothness(
                embeddings_t=pairs['embeddings_t'],
                embeddings_t1=pairs['embeddings_t1'],
                valence_t=pairs['valence_t'],
                arousal_t=pairs['arousal_t'],
                valence_t1=pairs['valence_t1'],
                arousal_t1=pairs['arousal_t1'],
            )
            total_loss += self.lambda_smooth * smoothness_loss
            all_metrics.update(smoothness_metrics)
        
        # 3. Transition Ranking Loss (if triplets provided)
        if 'transition_triplets' in batch_data and batch_data['transition_triplets'] is not None:
            triplets = batch_data['transition_triplets']
            ranking_loss, ranking_metrics = self.transition_ranking(
                anchor_embeddings=triplets['anchor_embeddings'],
                intermediate_embeddings=triplets['intermediate_embeddings'],
                far_embeddings=triplets['far_embeddings'],
                anchor_emotions=triplets['anchor_emotions'],
                intermediate_emotions=triplets['intermediate_emotions'],
                far_emotions=triplets['far_emotions'],
            )
            total_loss += self.lambda_transition * ranking_loss
            all_metrics.update(ranking_metrics)
        
        # 4. Pseudo-Label Consistency (if pseudo-labels provided)
        if 'pseudo_labels' in batch_data and batch_data['pseudo_labels'] is not None:
            pseudo_loss, pseudo_metrics = self.pseudo_consistency(
                embeddings=batch_data['anchor_embeddings'],
                pseudo_labels=batch_data['pseudo_labels'],
                pseudo_confidence=batch_data['pseudo_confidence'],
            )
            total_loss += self.lambda_pseudo * pseudo_loss
            all_metrics.update(pseudo_metrics)
        
        # Add total loss to metrics
        all_metrics['total_loss'] = total_loss.item()
        all_metrics['weighted_contrastive'] = (self.lambda_contrastive * contrastive_loss).item()
        
        return total_loss, all_metrics


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_accuracy_from_similarity(pos_sim: torch.Tensor, 
                                     neg_sim: torch.Tensor) -> float:
    """
    Compute contrastive accuracy: how often positives are more similar than negatives
    
    Args:
        pos_sim: Positive similarities (batch_size, num_positives)
        neg_sim: Negative similarities (batch_size, num_negatives)
    
    Returns:
        accuracy: Fraction of correct rankings
    """
    # For each anchor, check if max positive > max negative
    max_pos = pos_sim.max(dim=1)[0]  # (batch_size,)
    max_neg = neg_sim.max(dim=1)[0]  # (batch_size,)
    
    correct = (max_pos > max_neg).float()
    accuracy = correct.mean().item()
    
    return accuracy


def test_losses():
    """Test all loss functions"""
    print("="*80)
    print("TESTING LOSS FUNCTIONS")
    print("="*80)
    
    # Create dummy data
    batch_size = 4
    embed_dim = 768
    num_positives = 8
    num_negatives = 32
    
    # Random embeddings
    anchor_emb = torch.randn(batch_size, embed_dim)
    pos_emb = torch.randn(num_positives, embed_dim)
    neg_emb = torch.randn(num_negatives, embed_dim)
    
    # Random emotions
    valence_t = torch.randn(batch_size) * 5
    arousal_t = torch.randn(batch_size) * 5
    valence_t1 = valence_t + torch.randn(batch_size) * 0.5  # Small change
    arousal_t1 = arousal_t + torch.randn(batch_size) * 0.5
    
    print("\n1. Testing InfoNCE Loss...")
    info_nce = InfoNCELoss(temperature=0.07)
    loss, metrics = info_nce(anchor_emb, pos_emb, neg_emb)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    print("\n2. Testing Temporal Smoothness Loss...")
    smoothness = TemporalSmoothnessLoss(emotion_change_weight=True)
    loss, metrics = smoothness(
        anchor_emb, pos_emb[:batch_size],
        valence_t, arousal_t, valence_t1, arousal_t1
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    print("\n3. Testing Transition Ranking Loss...")
    ranking = TransitionRankingLoss(margin=0.5)
    emotions_t = torch.stack([valence_t, arousal_t], dim=1)
    emotions_t1 = torch.stack([valence_t1, arousal_t1], dim=1)
    emotions_t2 = emotions_t1 + torch.randn(batch_size, 2) * 0.5
    loss, metrics = ranking(
        anchor_emb, pos_emb[:batch_size], neg_emb[:batch_size],
        emotions_t, emotions_t1, emotions_t2
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    print("\n" + "="*80)
    print("✅ ALL LOSS TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    test_losses()