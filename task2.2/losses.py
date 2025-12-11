"""
Loss functions for temporal emotion recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def ccc_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Concordance Correlation Coefficient (CCC) loss
    
    Args:
        y_true: [N] ground truth values
        y_pred: [N] predicted values
    Returns:
        loss: scalar, 1 - CCC (to minimize)
    """
    y_true_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    
    covariance = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean()
    var_true = ((y_true - y_true_mean) ** 2).mean()
    var_pred = ((y_pred - y_pred_mean) ** 2).mean()
    
    ccc = (2 * covariance) / (var_true + var_pred + (y_true_mean - y_pred_mean) ** 2 + 1e-8)
    
    return 1 - ccc


def compute_ccc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute CCC metric (for evaluation)
    
    Args:
        y_true: [N] ground truth values
        y_pred: [N] predicted values
    Returns:
        ccc: scalar CCC value
    """
    with torch.no_grad():
        return 1 - ccc_loss(y_true, y_pred).item()


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with temperature scaling
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
        positive_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            anchors: [N, D] anchor embeddings (L2 normalized)
            positives: [N, D] positive embeddings (L2 normalized)
            negatives: [M, D] negative embeddings (L2 normalized), optional
            positive_weights: [N] weights for positive pairs, optional
        Returns:
            loss: scalar
        """
        # Compute similarities
        pos_sim = (anchors * positives).sum(dim=-1) / self.temperature  # [N]
        
        # Get negatives from all anchors if not provided
        if negatives is None:
            negatives = anchors  # Use all embeddings as negatives
        
        # Compute negative similarities
        neg_sim = torch.matmul(anchors, negatives.T) / self.temperature  # [N, M]
        
        # Compute log-sum-exp of negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, 1+M]
        
        # Apply positive weights if provided
        if positive_weights is not None:
            pos_sim = pos_sim * positive_weights
        
        # InfoNCE loss
        labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class LocalLocalContrastiveLoss(nn.Module):
    """
    Local-Local contrastive loss for adjacent frame pairs
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.infonce = InfoNCELoss(temperature)
        
    def forward(
        self,
        z_t: torch.Tensor,
        va_values: torch.Tensor,
        memory_queue: Optional[torch.Tensor] = None,
        va_threshold: float = 0.1,
        va_weight_beta: float = 10.0
    ) -> torch.Tensor:
        """
        Args:
            z_t: [B, L, D] frame embeddings
            va_values: [B, L, 2] valence/arousal values
            memory_queue: [K, D] memory queue embeddings
            va_threshold: threshold for VA-guided positives
            va_weight_beta: beta for VA weighting
        Returns:
            loss: scalar
        """
        B, L, D = z_t.shape
        
        # Flatten embeddings and VA values
        z_flat = z_t.reshape(B * L, D)  # [B*L, D]
        va_flat = va_values.reshape(B * L, 2)  # [B*L, 2]
        
        losses = []
        
        # Build local positive pairs
        for b in range(B):
            for t in range(L - 1):
                anchor_idx = b * L + t
                pos_idx = b * L + t + 1
                
                anchor = z_flat[anchor_idx:anchor_idx+1]  # [1, D]
                positive = z_flat[pos_idx:pos_idx+1]  # [1, D]
                
                # Compute VA-based weight
                va_diff = torch.norm(va_flat[anchor_idx] - va_flat[pos_idx])
                weight = torch.exp(-va_weight_beta * va_diff)
                
                # Get negatives (all other frames + memory queue)
                neg_mask = torch.ones(B * L, dtype=torch.bool, device=z_t.device)
                neg_mask[anchor_idx] = False
                neg_mask[pos_idx] = False
                negatives = z_flat[neg_mask]
                
                if memory_queue is not None:
                    negatives = torch.cat([negatives, memory_queue], dim=0)
                
                # Compute loss
                loss = self.infonce(anchor, positive, negatives, weight.unsqueeze(0))
                losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z_t.device)


class GlobalLocalContrastiveLoss(nn.Module):
    """
    Global-Local contrastive loss between clip and frame embeddings
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.infonce = InfoNCELoss(temperature)
        
    def forward(
        self,
        g: torch.Tensor,
        z_t: torch.Tensor,
        memory_queue: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            g: [B, D] clip embeddings
            z_t: [B, L, D] frame embeddings
            memory_queue: [K, D] memory queue embeddings
        Returns:
            loss: scalar
        """
        B, L, D = z_t.shape
        
        losses = []
        
        # For each clip, match with all its frames
        for b in range(B):
            clip_emb = g[b:b+1]  # [1, D]
            
            for t in range(L):
                frame_emb = z_t[b, t:t+1]  # [1, D]
                
                # Negatives: all other frames from other clips
                neg_mask = torch.ones(B, L, dtype=torch.bool, device=z_t.device)
                neg_mask[b, :] = False
                negatives = z_t[neg_mask].reshape(-1, D)
                
                if memory_queue is not None:
                    negatives = torch.cat([negatives, memory_queue], dim=0)
                
                loss = self.infonce(clip_emb, frame_emb, negatives)
                losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z_t.device)


class SmoothnessLoss(nn.Module):
    """
    Temporal smoothness regularizer
    """
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [B, L, D] frame embeddings
        Returns:
            loss: scalar
        """
        B, L, D = z_t.shape
        
        if L < 2:
            return torch.tensor(0.0, device=z_t.device)
        
        # Compute differences between consecutive frames
        diff = z_t[:, 1:, :] - z_t[:, :-1, :]  # [B, L-1, D]
        
        # L2 norm
        loss = (diff ** 2).sum(dim=-1).mean()
        
        return loss


class CombinedPretrainLoss(nn.Module):
    """
    Combined loss for contrastive pretraining
    """
    def __init__(
        self,
        lambda_ll: float = 1.0,
        lambda_gl: float = 0.5,
        lambda_smooth: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.lambda_ll = lambda_ll
        self.lambda_gl = lambda_gl
        self.lambda_smooth = lambda_smooth
        
        self.local_local_loss = LocalLocalContrastiveLoss(temperature)
        self.global_local_loss = GlobalLocalContrastiveLoss(temperature)
        self.smoothness_loss = SmoothnessLoss()
        
    def forward(
        self,
        z_t: torch.Tensor,
        g: torch.Tensor,
        va_values: torch.Tensor,
        memory_queue: Optional[torch.Tensor] = None,
        config = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z_t: [B, L, D] frame embeddings
            g: [B, D] clip embeddings
            va_values: [B, L, 2] VA values
            memory_queue: [K, D] optional memory queue
            config: config object with thresholds
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual losses
        """
        # Compute individual losses
        loss_ll = self.local_local_loss(
            z_t, va_values, memory_queue,
            va_threshold=config.va_threshold if config else 0.1,
            va_weight_beta=config.va_weight_beta if config else 10.0
        )
        
        loss_gl = self.global_local_loss(g, z_t, memory_queue)
        
        loss_smooth = self.smoothness_loss(z_t)
        
        # Combine losses
        total_loss = (
            self.lambda_ll * loss_ll +
            self.lambda_gl * loss_gl +
            self.lambda_smooth * loss_smooth
        )
        
        loss_dict = {
            'loss_ll': loss_ll.item(),
            'loss_gl': loss_gl.item(),
            'loss_smooth': loss_smooth.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class RegressionLoss(nn.Module):
    """
    Combined MSE + CCC loss for regression
    """
    def __init__(self, mse_weight: float = 0.5, ccc_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.ccc_weight = ccc_weight
        
    def forward(
        self,
        pred_va: torch.Tensor,
        true_va: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_va: [B, L, 2] predicted VA
            true_va: [B, L, 2] ground truth VA
        Returns:
            total_loss: scalar
            metrics: dictionary with metrics
        """
        # Flatten for loss computation
        pred_va_flat = pred_va.reshape(-1, 2)
        true_va_flat = true_va.reshape(-1, 2)
        
        # MSE loss
        mse = F.mse_loss(pred_va_flat, true_va_flat)
        
        # CCC loss for valence and arousal separately
        ccc_v = ccc_loss(true_va_flat[:, 0], pred_va_flat[:, 0])
        ccc_a = ccc_loss(true_va_flat[:, 1], pred_va_flat[:, 1])
        ccc_combined = (ccc_v + ccc_a) / 2
        
        # Total loss
        total_loss = self.mse_weight * mse + self.ccc_weight * ccc_combined
        
        metrics = {
            'mse': mse.item(),
            'ccc_valence': 1 - ccc_v.item(),
            'ccc_arousal': 1 - ccc_a.item(),
            'ccc_mean': 1 - ccc_combined.item(),
            'total': total_loss.item()
        }
        
        return total_loss, metrics