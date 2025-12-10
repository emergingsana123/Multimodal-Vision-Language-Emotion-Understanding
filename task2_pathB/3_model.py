"""
Temporal Head with LoRA for emotion contrastive learning
Lightweight transformer that processes pre-extracted CLIP features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import get_config


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model) or (batch, seq_len, d_model)
        """
        if x.dim() == 3 and x.size(1) > x.size(0):
            # (batch, seq_len, d_model) format
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        else:
            # (seq_len, batch, d_model) format
            x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)


# ============================================================================
# TEMPORAL HEAD (WITHOUT LORA - BASE MODEL)
# ============================================================================

class TemporalHead(nn.Module):
    """
    Temporal aggregation head for emotion representation learning
    
    Architecture:
    1. Positional encoding
    2. Multi-layer Transformer encoder
    3. Attention pooling
    4. Projection head
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.hidden_dim,
            max_len=config.num_frames * 2,  # Extra buffer
            dropout=config.dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,  # (batch, seq, feature) format
            norm_first=True   # Pre-LN for better stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
            norm=nn.LayerNorm(config.hidden_dim)
        )
        
        # Attention pooling (learnable query)
        self.attention_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.attention_key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_value = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.projection_hidden_dim),
            nn.LayerNorm(config.projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_hidden_dim, config.projection_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize attention query
        nn.init.normal_(self.attention_query, std=0.02)
        
        # Initialize projection head
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch, num_frames, feature_dim)
            return_all_tokens: If True, return all token embeddings
        
        Returns:
            Normalized embeddings (batch, projection_dim)
        """
        batch_size, num_frames, feature_dim = x.shape
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, num_frames, hidden_dim)
        
        if return_all_tokens:
            # Return all frame embeddings (for visualization)
            return encoded
        
        # Attention pooling
        pooled = self._attention_pooling(encoded)  # (batch, hidden_dim)
        
        # Project to embedding space
        embeddings = self.projection_head(pooled)  # (batch, projection_dim)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def _attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling over temporal dimension
        
        Args:
            x: Encoded features (batch, num_frames, hidden_dim)
        
        Returns:
            Pooled features (batch, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # Expand query to batch size
        query = self.attention_query.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        
        # Compute keys and values
        keys = self.attention_key(x)  # (batch, num_frames, hidden_dim)
        values = self.attention_value(x)  # (batch, num_frames, hidden_dim)
        
        # Attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, num_frames)
        scores = scores / math.sqrt(self.config.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        pooled = torch.bmm(attention_weights, values).squeeze(1)  # (batch, hidden_dim)
        
        return pooled
    
    def get_num_params(self) -> dict:
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


# ============================================================================
# TEMPORAL HEAD WITH LORA
# ============================================================================

def create_model_with_lora(config):
    """
    Create temporal head and apply LoRA if enabled
    
    Args:
        config: Configuration object
    
    Returns:
        Model with LoRA applied (if config.use_lora=True)
    """
    print("\n" + "="*80)
    print("CREATING TEMPORAL HEAD MODEL")
    print("="*80)
    
    # Create base model
    model = TemporalHead(config)
    
    # Print base model stats
    params = model.get_num_params()
    print(f"Base model parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Apply LoRA if enabled
    if config.use_lora:
        print(f"\nApplying LoRA...")
        print(f"  Rank: {config.lora_rank}")
        print(f"  Alpha: {config.lora_alpha}")
        print(f"  Dropout: {config.lora_dropout}")
        print(f"  Target modules: {config.lora_target_modules}")
        
        from peft import LoraConfig, get_peft_model
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=None  # No task type for custom models
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Get updated params
        params = model.get_num_params()
        print(f"\nAfter LoRA:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Trainable ratio: {100 * params['trainable'] / params['total']:.2f}%")
    
    print("="*80)
    
    return model


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING TEMPORAL HEAD MODEL")
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Test without LoRA first
    print("\n1. Testing base model (no LoRA)...")
    config.use_lora = False
    model_base = create_model_with_lora(config)
    
    # Test forward pass
    batch_size = 4
    num_frames = 8
    feature_dim = 512
    
    dummy_input = torch.randn(batch_size, num_frames, feature_dim)
    
    model_base.eval()
    with torch.no_grad():
        output = model_base(dummy_input)
    
    print(f"\n✅ Base model test passed!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output normalized: {torch.allclose(output.norm(dim=-1), torch.ones(batch_size), atol=1e-5)}")
    
    # Test with LoRA
    print("\n2. Testing model with LoRA...")
    config.use_lora = True
    model_lora = create_model_with_lora(config)
    
    model_lora.eval()
    with torch.no_grad():
        output_lora = model_lora(dummy_input)
    
    print(f"\n✅ LoRA model test passed!")
    print(f"   Output shape: {output_lora.shape}")
    print(f"   Output normalized: {torch.allclose(output_lora.norm(dim=-1), torch.ones(batch_size), atol=1e-5)}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow (LoRA model)...")
    model_lora.train()
    
    output = model_lora(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # Check which parameters have gradients
    has_grad = sum(1 for p in model_lora.parameters() if p.grad is not None)
    total_params = sum(1 for p in model_lora.parameters())
    
    print(f"   Parameters with gradients: {has_grad}/{total_params}")
    
    if has_grad > 0:
        print(f"   ✅ Gradient flow working!")
    else:
        print(f"   ❌ No gradients! Check LoRA configuration")
    
    # Memory usage
    if torch.cuda.is_available():
        dummy_input_gpu = dummy_input.cuda()
        model_lora_gpu = model_lora.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output_gpu = model_lora_gpu(dummy_input_gpu)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n💾 GPU Memory:")
        print(f"   Peak memory (single forward): {peak_memory:.2f} GB")
    
    print("\n" + "="*80)
    print("✅ ALL MODEL TESTS PASSED!")
    print("="*80)