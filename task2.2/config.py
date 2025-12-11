"""
Configuration file for Temporal Emotion Recognition Project
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Dataset paths
    dataset_root: str = "/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/dataset"
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    
    # Model architecture
    clip_model: str = "ViT-B/32"
    clip_pretrained: str = "openai"
    feature_dim: int = 512  # CLIP output dimension
    projection_dim: int = 128
    hidden_dim: int = 512
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Training hyperparameters
    batch_size: int = 32
    num_frames: int = 16  # L
    num_epochs_pretrain: int = 30
    num_epochs_finetune: int = 15
    
    # Optimizer settings
    lr_pretrain: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    
    # Loss weights
    lambda_ll: float = 1.0  # local-local
    lambda_gl: float = 0.5  # global-local
    lambda_smooth: float = 0.1  # smoothness
    temperature: float = 0.07
    
    # Pair construction
    va_threshold: float = 0.1  # for VA-guided positives
    va_weight_beta: float = 10.0
    memory_queue_size: int = 4096
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.3
    color_jitter: bool = True
    temporal_jitter: int = 1
    
    # Training options
    use_amp: bool = True  # mixed precision
    grad_accumulation_steps: int = 1
    use_lora: bool = True
    cache_features: bool = False  # cache CLIP features for faster iteration
    
    # Evaluation
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    visualize_samples: int = 5
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)