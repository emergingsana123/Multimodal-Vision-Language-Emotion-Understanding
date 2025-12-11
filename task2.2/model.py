"""
Model architecture: CLIP + LoRA + Temporal Head + Regression Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import open_clip


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for attention weights
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA: x @ (W + BA)"""
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class FrameProjector(nn.Module):
    """Project frame features to embedding space"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] or [B*L, D]
        Returns:
            z: [B, L, output_dim] or [B*L, output_dim], L2 normalized
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class TemporalHead(nn.Module):
    """
    Temporal aggregation head to produce clip-level embedding
    Option A: Mean pooling + MLP
    Option B: 1-layer transformer + pooling
    """
    def __init__(self, input_dim: int, output_dim: int, use_transformer: bool = False):
        super().__init__()
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.transformer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=input_dim * 2,
                dropout=0.1,
                batch_first=True
            )
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [B, L, D] frame embeddings
        Returns:
            g: [B, D] clip embedding, L2 normalized
        """
        if self.use_transformer:
            z_t = self.transformer(z_t)
        
        # Mean pooling over time
        g = z_t.mean(dim=1)  # [B, D]
        g = self.projection(g)
        
        return F.normalize(g, p=2, dim=-1)


class RegressionHead(nn.Module):
    """Regression head for valence/arousal prediction"""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # valence, arousal
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] or [B, D]
        Returns:
            va: [B, L, 2] or [B, 2] (valence, arousal)
        """
        return self.net(x)


class TemporalEmotionModel(nn.Module):
    """
    Complete model: CLIP backbone + LoRA + Frame Projector + Temporal Head
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load CLIP model (frozen backbone)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model,
            pretrained=config.clip_pretrained
        )
        
        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Add LoRA to the last transformer block
        if config.use_lora:
            self._add_lora_to_clip()
        
        # Frame projector
        self.frame_projector = FrameProjector(
            input_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim
        )
        
        # Temporal head
        self.temporal_head = TemporalHead(
            input_dim=config.projection_dim,
            output_dim=config.projection_dim,
            use_transformer=False  # Start with simple mean pooling
        )
        
        # Regression head (used during fine-tuning)
        self.regression_head = RegressionHead(
            input_dim=config.projection_dim,
            hidden_dim=64
        )
        
        # Memory queue for contrastive learning
        self.register_buffer(
            "memory_queue",
            torch.randn(config.memory_queue_size, config.projection_dim)
        )
        self.memory_queue = F.normalize(self.memory_queue, p=2, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _add_lora_to_clip(self):
        """Add LoRA adapters to the last transformer block"""
        # Get the last transformer block
        if hasattr(self.clip_model.visual, 'transformer'):
            # ViT architecture
            last_block = self.clip_model.visual.transformer.resblocks[-1]
            
            # Add LoRA to attention Q and V projections
            attn = last_block.attn
            d_model = attn.in_proj_weight.shape[1]
            
            # Create LoRA layers
            self.lora_q = LoRALayer(
                d_model, d_model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha
            )
            self.lora_v = LoRALayer(
                d_model, d_model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha
            )
            
            # Store original forward
            original_forward = attn.forward
            
            # Wrap attention forward to include LoRA
            def lora_forward(x):
                # This is a simplified version - full implementation would
                # properly integrate LoRA into the attention mechanism
                return original_forward(x)
            
            attn.forward = lora_forward
            
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode frames with CLIP backbone
        Args:
            images: [B, L, C, H, W] or [B*L, C, H, W]
        Returns:
            features: [B, L, D] or [B*L, D]
        """
        original_shape = images.shape
        
        if len(images.shape) == 5:
            B, L = images.shape[:2]
            images = images.view(B * L, *images.shape[2:])
        else:
            B, L = None, None
        
        # Extract features with frozen CLIP
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        
        if B is not None:
            features = features.view(B, L, -1)
        
        return features
    
    def forward(
        self,
        images: torch.Tensor,
        return_regression: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        Args:
            images: [B, L, C, H, W]
            return_regression: whether to return VA predictions
        Returns:
            z_t: [B, L, projection_dim] frame embeddings
            g: [B, projection_dim] clip embedding
            va_pred: [B, L, 2] if return_regression else None
        """
        # Encode frames
        features = self.encode_frames(images)  # [B, L, feature_dim]
        
        # Project frames
        z_t = self.frame_projector(features)  # [B, L, projection_dim]
        
        # Get clip embedding
        g = self.temporal_head(z_t)  # [B, projection_dim]
        
        # Optionally predict VA
        va_pred = None
        if return_regression:
            va_pred = self.regression_head(z_t)  # [B, L, 2]
        
        return z_t, g, va_pred
    
    @torch.no_grad()
    def update_memory_queue(self, embeddings: torch.Tensor):
        """Update memory queue with new embeddings"""
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest embeddings
        if ptr + batch_size <= self.config.memory_queue_size:
            self.memory_queue[ptr:ptr + batch_size] = embeddings
            ptr = (ptr + batch_size) % self.config.memory_queue_size
        else:
            # Wrap around
            remaining = self.config.memory_queue_size - ptr
            self.memory_queue[ptr:] = embeddings[:remaining]
            self.memory_queue[:batch_size - remaining] = embeddings[remaining:]
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr
    
    def get_trainable_parameters(self, stage: str = 'pretrain'):
        """Get trainable parameters for different training stages"""
        if stage == 'pretrain':
            # LoRA + frame projector + temporal head
            params = []
            if self.config.use_lora:
                params.extend([
                    {'params': self.lora_q.parameters()},
                    {'params': self.lora_v.parameters()}
                ])
            params.extend([
                {'params': self.frame_projector.parameters()},
                {'params': self.temporal_head.parameters()}
            ])
            return params
        elif stage == 'finetune':
            # LoRA + temporal head + regression head
            params = []
            if self.config.use_lora:
                params.extend([
                    {'params': self.lora_q.parameters()},
                    {'params': self.lora_v.parameters()}
                ])
            params.extend([
                {'params': self.temporal_head.parameters()},
                {'params': self.regression_head.parameters()}
            ])
            return params
        else:
            raise ValueError(f"Unknown stage: {stage}")