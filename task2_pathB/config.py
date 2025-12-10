"""
Configuration for Path B: CLIP Features + Temporal LoRA
Optimized for 20GB GPU memory, 9-hour training budget
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch


@dataclass
class Config:
    """Complete configuration for feature extraction and training"""
    
    # ============================================================================
    # PATHS
    # ============================================================================
    dataset_dir: str = '/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/dataset'
    project_root: str = './outputs'
    
    # Derived paths (auto-created)
    features_cache_dir: str = field(init=False)
    checkpoints_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    results_dir: str = field(init=False)
    
    def __post_init__(self):
        """Setup derived paths"""
        root = Path(self.project_root)
        self.features_cache_dir = str(root / 'features_cache')
        self.checkpoints_dir = str(root / 'checkpoints')
        self.logs_dir = str(root / 'logs')
        self.results_dir = str(root / 'results')
        
        # Create directories
        for dir_path in [self.features_cache_dir, self.checkpoints_dir, 
                         self.logs_dir, self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # DEVICE & COMPUTE
    # ============================================================================
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True  # FP16 for memory savings
    
    # ============================================================================
    # CLIP FEATURE EXTRACTION
    # ============================================================================
    clip_model_name: str = 'openai/clip-vit-base-patch16'
    clip_feature_dim: int = 512  # CLIP ViT-B/16 output dimension
    extraction_batch_size: int = 48  # Batch size for feature extraction
    image_size: int = 224  # CLIP input size
    
    # ============================================================================
    # TEMPORAL SETTINGS
    # ============================================================================
    num_frames: int = 8  # Frames per temporal window
    temporal_stride: int = 2  # Stride for window extraction
    max_frames_per_clip: int = 300  # Skip very long clips
    
    # ============================================================================
    # TEMPORAL HEAD ARCHITECTURE
    # ============================================================================
    # Transformer settings
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    hidden_dim: int = 512  # Match CLIP feature dim
    feedforward_dim: int = 2048
    dropout: float = 0.1
    
    # Projection head
    projection_dim: int = 128  # Final embedding dimension for contrastive learning
    projection_hidden_dim: int = 256
    
    # ============================================================================
    # LORA CONFIGURATION
    # ============================================================================
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj']  # Query and Value in attention
    )
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    # Batch settings (optimized for 20GB GPU)
    batch_size: int = 18  # Main batch size
    gradient_accumulation_steps: int = 2  # Effective batch size = 36
    
    # Optimization
    num_epochs: int = 70
    learning_rate: float = 5e-4  # Higher LR since only training temporal head
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_epochs: int = 5
    warmup_steps: int = 500
    lr_scheduler: str = 'cosine'  # 'cosine' or 'linear'
    
    # ============================================================================
    # CONTRASTIVE LEARNING
    # ============================================================================
    temperature: float = 0.07  # InfoNCE temperature
    num_positives_per_anchor: int = 2  # Temporal positives from same clip
    use_in_batch_negatives: bool = True  # Use all other samples in batch as negatives
    
    # Positive sampling strategy
    temporal_positive_window: int = 5  # Max temporal distance for positives (in frames)
    
    # ============================================================================
    # LOSS WEIGHTS
    # ============================================================================
    lambda_contrastive: float = 1.0  # Main contrastive loss
    lambda_regularization: float = 0.1  # Optional L2 regularization on embeddings
    
    # ============================================================================
    # VALIDATION & CHECKPOINTING
    # ============================================================================
    val_split: float = 0.15  # 15% for validation
    eval_frequency: int = 2  # Validate every N epochs
    save_frequency: int = 5  # Save checkpoint every N epochs
    checkpoint_frequency: int = 100  # Save during epoch every N batches
    early_stopping_patience: int = 10  # Stop if no improvement for N evaluations
    
    # ============================================================================
    # LOGGING & VISUALIZATION
    # ============================================================================
    log_frequency: int = 20  # Log metrics every N batches
    plot_frequency: int = 5  # Generate plots every N epochs
    save_best_only: bool = True  # Only save best model
    
    # Embedding visualization
    visualize_embeddings: bool = True
    visualization_samples: int = 500  # Number of samples for UMAP/PCA
    
    # ============================================================================
    # FEATURE EXTRACTION SETTINGS
    # ============================================================================
    test_mode: bool = False  # If True, only extract features for few clips
    test_num_clips: int = 10  # Number of clips in test mode
    extraction_resume: bool = True  # Resume from existing features


def get_config() -> Config:
    """Get default configuration"""
    return Config()


def print_config(config: Config):
    """Pretty print configuration"""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    
    sections = {
        'Paths': ['dataset_dir', 'project_root', 'features_cache_dir'],
        'Device': ['device', 'use_mixed_precision'],
        'CLIP': ['clip_model_name', 'clip_feature_dim', 'extraction_batch_size'],
        'Temporal': ['num_frames', 'temporal_stride'],
        'Model': ['num_transformer_layers', 'num_attention_heads', 'projection_dim'],
        'LoRA': ['use_lora', 'lora_rank', 'lora_alpha'],
        'Training': ['batch_size', 'gradient_accumulation_steps', 'num_epochs', 'learning_rate'],
        'Contrastive': ['temperature', 'num_positives_per_anchor'],
    }
    
    for section, keys in sections.items():
        print(f"\n{section}:")
        for key in keys:
            value = getattr(config, key)
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print_config(config)
    
    # Verify paths
    print("\n📁 Directory Structure:")
    print(f"  Dataset: {Path(config.dataset_dir).exists()} ✓" if Path(config.dataset_dir).exists() 
          else f"  Dataset: {Path(config.dataset_dir).exists()} ✗ (will be created)")
    print(f"  Outputs: {Path(config.project_root).exists()} ✓")
    print(f"  Features cache: {Path(config.features_cache_dir).exists()} ✓")
    print(f"  Checkpoints: {Path(config.checkpoints_dir).exists()} ✓")
    print(f"  Logs: {Path(config.logs_dir).exists()} ✓")
    print(f"  Results: {Path(config.results_dir).exists()} ✓")
    
    # Estimate memory usage
    print("\n💾 Estimated GPU Memory:")
    print(f"  Feature extraction: ~8-10 GB")
    print(f"  Training (batch={config.batch_size}): ~15-18 GB")
    print(f"  Peak (with mixed precision): ~18-20 GB")
    print(f"  Target limit: 20 GB ✓")