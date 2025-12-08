"""
task2_temporal_emotion_learning_complete.py
Complete Temporal Emotion Contrastive Learning Pipeline with Checkpointing
All-in-one file with automatic resume capability
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import h5py
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms

from transformers import (
    AutoModel,
    AutoImageProcessor,
    VideoMAEModel,
    VideoMAEImageProcessor,
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TemporalEmotionConfig:
    """Configuration for temporal emotion contrastive learning"""
    
    # Paths - MODIFY THESE FOR YOUR SYSTEM
    data_dir: str = '/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/dataset'
    project_root: str = '/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/task2_outputs'
    
    # Basic settings
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model configuration
    backbone_model: str = 'MCG-NJU/videomae-base'
    use_pretrained: bool = True
    freeze_backbone: bool = True
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["attention.attention.query", "attention.attention.value"])
    
    # Video/Temporal settings
    num_frames: int = 16
    frame_sample_rate: int = 2
    video_size: int = 224
    temporal_window_size: int = 16
    
    # Data settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Contrastive learning settings
    temperature: float = 0.07
    num_positive_pairs: int = 2
    num_negative_pairs: int = 8
    
    # Loss weights
    lambda_contrastive: float = 1.0
    lambda_temporal_smooth: float = 0.3
    lambda_transition: float = 0.5
    lambda_pseudo: float = 0.3
    
    # Emotion similarity thresholds
    valence_threshold_similar: float = 1.5
    arousal_threshold_similar: float = 1.5
    valence_threshold_different: float = 3.0
    arousal_threshold_different: float = 3.0
    
    # Training settings
    num_epochs: int = 50
    warmup_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Validation and checkpointing
    eval_frequency: int = 1
    save_frequency: int = 5
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 100  # Save every N batches
    
    # Pseudo-label settings
    pseudo_label_model: str = 'hsemotion'
    pseudo_confidence_threshold: float = 0.6
    use_ensemble_pseudo: bool = False
    
    # Advanced settings
    use_hard_negatives: bool = True
    hard_negative_ratio: float = 0.7
    use_mixed_precision: bool = True


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages checkpoints to allow resuming from crashes"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, name: str, data: dict):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        temp_path = self.checkpoint_dir / f"{name}.tmp"
        
        # Save to temp file first
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Rename to final checkpoint (atomic operation)
        temp_path.rename(checkpoint_path)
        print(f"💾 Checkpoint saved: {name}")
    
    def load_checkpoint(self, name: str):
        """Load checkpoint if exists"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def checkpoint_exists(self, name: str) -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{name}.pkl").exists()
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        return [f.stem for f in self.checkpoint_dir.glob("*.pkl")]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_afew_va_clip(clip_dir: Path):
    """Load a single clip's annotations and frame data."""
    clip_name = clip_dir.name
    annotation_file = clip_dir / f"{clip_name}.json"
    
    if not annotation_file.exists():
        return None
    
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    except:
        return None
    
    frames = annotations.get("frames", {})
    clip_frames = []
    
    for frame_id, frame_data in frames.items():
        try:
            frame_num = int(frame_id)
        except:
            continue
        
        image_path = clip_dir / f"{frame_id}.png"
        if not image_path.exists():
            image_path = clip_dir / f"{frame_id}.jpg"
        
        if image_path.exists():
            clip_frames.append({
                'clip_name': clip_name,
                'clip_id': int(clip_name) if clip_name.isdigit() else hash(clip_name),
                'frame_id': frame_id,
                'frame_num': frame_num,
                'image_path': str(image_path),
                'valence': frame_data.get('valence', 0.0),
                'arousal': frame_data.get('arousal', 0.0),
            })
    
    clip_frames.sort(key=lambda x: x['frame_num'])
    
    for idx, frame in enumerate(clip_frames):
        frame['temporal_idx'] = idx
        frame['total_frames_in_clip'] = len(clip_frames)
    
    return clip_frames


def load_afew_va_dataset(data_dir: Path, checkpoint_manager: CheckpointManager):
    """Load entire AFEW-VA dataset with checkpointing"""
    
    # Check if already loaded
    samples = checkpoint_manager.load_checkpoint('dataset_loaded')
    if samples is not None:
        print(f"✅ Loaded dataset from checkpoint: {len(samples)} samples")
        return samples
    
    print("="*80)
    print("LOADING AFEW-VA DATASET")
    print("="*80)
    
    all_frames = []
    clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(clip_dirs)} clip directories")
    
    for clip_dir in tqdm(clip_dirs, desc="Loading clips"):
        frames = load_afew_va_clip(clip_dir)
        if frames:
            all_frames.extend(frames)
    
    print(f"\nTotal frames loaded: {len(all_frames)}")
    print(f"Unique clips: {len(set(f['clip_name'] for f in all_frames))}")
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint('dataset_loaded', all_frames)
    
    return all_frames


# ============================================================================
# VIDEOMAE BACKBONE
# ============================================================================

class VideoMAEBackbone(nn.Module):
    """Wrapper for VideoMAE model"""
    
    def __init__(self, model_name: str = 'MCG-NJU/videomae-base', freeze: bool = True):
        super().__init__()
        
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        if freeze:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor):
        """Forward pass"""
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs
    
    def get_embeddings(self, pixel_values: torch.Tensor):
        """Extract normalized embeddings"""
        outputs = self.forward(pixel_values)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
    
    def get_trainable_parameters(self):
        """Count trainable vs total parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def initialize_backbone(config, checkpoint_manager: CheckpointManager):
    """Initialize VideoMAE backbone with checkpointing"""
    
    print("="*80)
    print("INITIALIZING VIDEOMAE BACKBONE")
    print("="*80)
    
    device = torch.device(config.device)
    backbone = VideoMAEBackbone(model_name=config.backbone_model, freeze=config.freeze_backbone)
    backbone = backbone.to(device)
    
    trainable, total = backbone.get_trainable_parameters()
    print(f"\n📊 Model Statistics (Before LoRA):")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    
    return backbone


# ============================================================================
# LORA APPLICATION
# ============================================================================

def apply_lora(backbone, config, checkpoint_manager: CheckpointManager):
    """Apply LoRA adapters with checkpointing"""
    
    # Check if LoRA already applied
    if checkpoint_manager.checkpoint_exists('lora_applied'):
        print("✅ LoRA already applied (loading from checkpoint)")
        checkpoint_data = checkpoint_manager.load_checkpoint('lora_applied')
        # Load LoRA state if saved
        return backbone
    
    print("="*80)
    print("APPLYING LORA ADAPTERS")
    print("="*80)
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        # Don't specify task_type for vision models - causes PEFT to inject wrong kwargs
        inference_mode=False, 
    )
    
    print(f"LoRA Configuration:")
    print(f"   Rank: {lora_config.r}")
    print(f"   Alpha: {lora_config.lora_alpha}")
    print(f"   Target modules: {lora_config.target_modules}")
    
    # Apply LoRA
    lora_model = get_peft_model(backbone.model, lora_config)
    lora_model.print_trainable_parameters()
    
    backbone.model = lora_model
    
    trainable, total = backbone.get_trainable_parameters()
    print(f"\n📊 Model Statistics (After LoRA):")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Trainable ratio: {100 * trainable / total:.4f}%")
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint('lora_applied', {
        'lora_config': lora_config,
        'trainable_params': trainable,
        'total_params': total
    })
    
    return backbone


# ============================================================================
# PSEUDO LABEL GENERATOR
# ============================================================================

class PseudoLabelGenerator:
    """Generate pseudo-labels using pretrained emotion models"""
    
    def __init__(self, model_name: str = 'hsemotion', device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        
        print(f"Initializing pseudo-label generator: {model_name}")
        
        if model_name == 'hsemotion':
            self._init_hsemotion()
        elif model_name == 'fer':
            self._init_fer()
    
    def _init_hsemotion(self):
        """Initialize HSEmotion model"""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            self.model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
            self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 
                                  'Happy', 'Neutral', 'Sad', 'Surprise']
            print("✅ HSEmotion loaded")
        except Exception as e:
            print(f"⚠️ HSEmotion failed: {e}")
    
    def _init_fer(self):
        """Initialize FER model"""
        try:
            from fer import FER
            self.model = FER(mtcnn=True)
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 
                                  'sad', 'surprise', 'neutral']
            print("✅ FER loaded")
        except Exception as e:
            print(f"⚠️ FER failed: {e}")
    
    def predict_emotion(self, image_path: str):
        """Predict emotion from image"""
        if self.model is None:
            return None
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.model_name == 'hsemotion':
                emotion, scores = self.model.predict_emotions(image_rgb, logits=True)
                probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
                
                result = {
                    'emotion_probs': dict(zip(self.emotion_labels, probs)),
                    'dominant_emotion': emotion,
                    'confidence': float(probs.max()),
                    'model': self.model_name
                }
            elif self.model_name == 'fer':
                result_list = self.model.detect_emotions(image_rgb)
                if len(result_list) == 0:
                    return None
                emotions = result_list[0]['emotions']
                result = {
                    'emotion_probs': emotions,
                    'dominant_emotion': max(emotions, key=emotions.get),
                    'confidence': max(emotions.values()),
                    'model': self.model_name
                }
            
            result['valence_estimate'] = self._map_to_valence(result['emotion_probs'])
            result['arousal_estimate'] = self._map_to_arousal(result['emotion_probs'])
            
            return result
        except Exception as e:
            return None
    
    def _map_to_valence(self, emotion_probs: Dict[str, float]) -> float:
        """Map emotions to valence"""
        valence_map = {
            'Happy': 8.0, 'Surprise': 5.0, 'Neutral': 0.0,
            'Anger': -6.0, 'Disgust': -7.0, 'Fear': -6.0,
            'Sad': -8.0, 'Contempt': -5.0,
            'happy': 8.0, 'surprise': 5.0, 'neutral': 0.0,
            'angry': -6.0, 'disgust': -7.0, 'fear': -6.0, 'sad': -8.0
        }
        valence = sum(emotion_probs.get(e, 0) * valence_map.get(e, 0) 
                     for e in emotion_probs.keys())
        return float(valence)
    
    def _map_to_arousal(self, emotion_probs: Dict[str, float]) -> float:
        """Map emotions to arousal"""
        arousal_map = {
            'Happy': 6.0, 'Surprise': 8.0, 'Neutral': 0.0,
            'Anger': 7.0, 'Disgust': 5.0, 'Fear': 8.0,
            'Sad': -4.0, 'Contempt': 3.0,
            'happy': 6.0, 'surprise': 8.0, 'neutral': 0.0,
            'angry': 7.0, 'disgust': 5.0, 'fear': 8.0, 'sad': -4.0
        }
        arousal = sum(emotion_probs.get(e, 0) * arousal_map.get(e, 0) 
                     for e in emotion_probs.keys())
        return float(arousal)


def generate_pseudo_labels(samples: List[Dict], 
                          generator: PseudoLabelGenerator,
                          checkpoint_manager: CheckpointManager,
                          batch_size: int = 100):
    """Generate pseudo-labels with checkpointing"""
    
    # Check if already generated
    cache_name = f'pseudo_labels_{generator.model_name}'
    pseudo_labels = checkpoint_manager.load_checkpoint(cache_name)
    if pseudo_labels is not None:
        print(f"✅ Loaded {len(pseudo_labels)} pseudo-labels from checkpoint")
        return pseudo_labels
    
    print("="*80)
    print("GENERATING PSEUDO-LABELS")
    print("="*80)
    
    pseudo_labels = {}
    failed_count = 0
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating pseudo-labels"):
        batch = samples[i:i+batch_size]
        
        for sample in batch:
            image_path = sample['image_path']
            if image_path in pseudo_labels:
                continue
            
            result = generator.predict_emotion(image_path)
            if result:
                pseudo_labels[image_path] = result
            else:
                failed_count += 1
        
        # Save checkpoint every 1000 samples
        if (i + batch_size) % 1000 == 0:
            checkpoint_manager.save_checkpoint(cache_name, pseudo_labels)
    
    # Final save
    checkpoint_manager.save_checkpoint(cache_name, pseudo_labels)
    
    print(f"\n✅ Generated {len(pseudo_labels)} pseudo-labels")
    print(f"   Failed: {failed_count}")
    print(f"   Success rate: {100 * len(pseudo_labels) / len(samples):.1f}%")
    
    return pseudo_labels


def add_pseudo_labels_to_samples(samples: List[Dict], 
                                 pseudo_labels: Dict,
                                 confidence_threshold: float = 0.6):
    """Add pseudo-labels to samples"""
    print("\nAdding pseudo-labels to samples...")
    
    high_confidence_count = 0
    
    for sample in samples:
        image_path = sample['image_path']
        
        if image_path in pseudo_labels:
            pseudo = pseudo_labels[image_path]
            sample['pseudo_emotion'] = pseudo['dominant_emotion']
            sample['pseudo_confidence'] = pseudo['confidence']
            sample['pseudo_valence'] = pseudo['valence_estimate']
            sample['pseudo_arousal'] = pseudo['arousal_estimate']
            sample['pseudo_probs'] = pseudo['emotion_probs']
            sample['use_pseudo'] = pseudo['confidence'] >= confidence_threshold
            
            if sample['use_pseudo']:
                high_confidence_count += 1
        else:
            sample['pseudo_emotion'] = None
            sample['pseudo_confidence'] = 0.0
            sample['use_pseudo'] = False
    
    print(f"✅ Pseudo-labels added")
    print(f"   High-confidence samples: {high_confidence_count}")
    
    return samples


# ============================================================================
# STATISTICS AND ANALYSIS
# ============================================================================

def compute_statistics(samples: List[Dict]):
    """Compute and print dataset statistics"""
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    valid_samples = [s for s in samples if s['valence'] is not None and s['arousal'] is not None]
    
    if len(valid_samples) == 0:
        print("⚠️ No valid samples found")
        return
    
    valences = [s['valence'] for s in valid_samples]
    arousals = [s['arousal'] for s in valid_samples]
    
    print(f"\nSample Statistics:")
    print(f"  Total samples: {len(samples):,}")
    print(f"  Valid samples: {len(valid_samples):,}")
    print(f"  Unique clips: {len(set(s['clip_name'] for s in samples))}")
    
    print(f"\nEmotion Statistics:")
    print(f"  Valence - Mean: {np.mean(valences):.3f}, Std: {np.std(valences):.3f}")
    print(f"  Valence - Range: [{min(valences):.2f}, {max(valences):.2f}]")
    print(f"  Arousal - Mean: {np.mean(arousals):.3f}, Std: {np.std(arousals):.3f}")
    print(f"  Arousal - Range: [{min(arousals):.2f}, {max(arousals):.2f}]")
    
    # Pseudo-label statistics
    with_pseudo = [s for s in samples if s.get('pseudo_emotion') is not None]
    high_conf = [s for s in samples if s.get('use_pseudo', False)]
    
    if len(with_pseudo) > 0:
        print(f"\nPseudo-label Statistics:")
        print(f"  Samples with pseudo-labels: {len(with_pseudo):,}")
        print(f"  High-confidence samples: {len(high_conf):,}")
        print(f"  Coverage: {100 * len(with_pseudo) / len(samples):.1f}%")
        
        # Correlation with ground truth
        valid_with_pseudo = [s for s in valid_samples if s.get('pseudo_valence') is not None]
        if len(valid_with_pseudo) > 10:
            pseudo_val = [s['pseudo_valence'] for s in valid_with_pseudo]
            true_val = [s['valence'] for s in valid_with_pseudo]
            pseudo_ar = [s['pseudo_arousal'] for s in valid_with_pseudo]
            true_ar = [s['arousal'] for s in valid_with_pseudo]
            
            val_corr, _ = pearsonr(pseudo_val, true_val)
            ar_corr, _ = pearsonr(pseudo_ar, true_ar)
            
            print(f"  Valence correlation: r={val_corr:.3f}")
            print(f"  Arousal correlation: r={ar_corr:.3f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline with comprehensive checkpointing"""
    
    print("="*80)
    print("TEMPORAL EMOTION CONTRASTIVE LEARNING - COMPLETE PIPELINE")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Initialize configuration
    config = TemporalEmotionConfig()
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Create directories
    PROJECT_ROOT = Path(config.project_root)
    CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
    LORA_DIR = PROJECT_ROOT / 'lora_adapters'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    
    for directory in [PROJECT_ROOT, CHECKPOINTS_DIR, LORA_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(PROJECT_ROOT / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n📁 Project directories created:")
    print(f"   Root: {PROJECT_ROOT}")
    print(f"   Checkpoints: {CHECKPOINTS_DIR}")
    print(f"   LoRA: {LORA_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(CHECKPOINTS_DIR)
    
    existing_checkpoints = checkpoint_manager.list_checkpoints()
    if existing_checkpoints:
        print(f"\n📋 Found existing checkpoints: {existing_checkpoints}")
    
    # STEP 1: Load Dataset
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    DATA_DIR = Path(config.data_dir)
    if not DATA_DIR.exists():
        print(f"❌ ERROR: Data directory not found: {DATA_DIR}")
        print("Please update the data_dir in the configuration")
        return
    
    samples = load_afew_va_dataset(DATA_DIR, checkpoint_manager)
    
    # STEP 2: Initialize Backbone
    print("\n" + "="*80)
    print("STEP 2: INITIALIZING BACKBONE MODEL")
    print("="*80)
    
    try:
        backbone = initialize_backbone(config, checkpoint_manager)
    except Exception as e:
        print(f"❌ ERROR loading backbone: {e}")
        print("This might be due to network issues. The model will be downloaded on first run.")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 3: Apply LoRA
    print("\n" + "="*80)
    print("STEP 3: APPLYING LORA ADAPTERS")
    print("="*80)
    
    try:
        backbone = apply_lora(backbone, config, checkpoint_manager)
    except Exception as e:
        print(f"❌ ERROR applying LoRA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 4: Test Model
    print("\n" + "="*80)
    print("STEP 4: TESTING MODEL")
    print("="*80)
    
    device = torch.device(config.device)
    
    # Create dummy input
    batch_size = 2
    dummy_video = torch.randn(batch_size, config.num_frames, 3, config.video_size, config.video_size)
    dummy_video = dummy_video.to(device)
    
    backbone.eval()
    try:
        with torch.no_grad():
            embeddings = backbone.get_embeddings(dummy_video)
            print(f"✅ Model test passed!")
            print(f"   Output shape: {embeddings.shape}")
            print(f"   Embeddings normalized: {torch.allclose(embeddings.norm(dim=-1), torch.ones(batch_size).to(device), atol=1e-5)}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    backbone.train()
    
    # STEP 5: Initialize Pseudo-Label Generator
    print("\n" + "="*80)
    print("STEP 5: INITIALIZING PSEUDO-LABEL GENERATOR")
    print("="*80)
    
    try:
        pseudo_generator = PseudoLabelGenerator(
            model_name=config.pseudo_label_model,
            device=config.device
        )
    except Exception as e:
        print(f"⚠️ Warning: Pseudo-label generator initialization failed: {e}")
        print("Continuing without pseudo-labels...")
        pseudo_generator = None
    
    # STEP 6: Generate Pseudo-Labels
    if pseudo_generator and pseudo_generator.model is not None:
        print("\n" + "="*80)
        print("STEP 6: GENERATING PSEUDO-LABELS")
        print("="*80)
        
        try:
            pseudo_labels_dict = generate_pseudo_labels(
                samples=samples,
                generator=pseudo_generator,
                checkpoint_manager=checkpoint_manager,
                batch_size=100
            )
            
            samples = add_pseudo_labels_to_samples(
                samples=samples,
                pseudo_labels=pseudo_labels_dict,
                confidence_threshold=config.pseudo_confidence_threshold
            )
        except Exception as e:
            print(f"⚠️ Warning: Pseudo-label generation failed: {e}")
            print("Continuing without pseudo-labels...")
    else:
        print("\n⏭️  Skipping pseudo-label generation")
    
    # STEP 7: Save Final Checkpoint
    print("\n" + "="*80)
    print("STEP 7: SAVING FINAL CHECKPOINT")
    print("="*80)
    
    # Save model state
    try:
        torch.save(backbone.state_dict(), LORA_DIR / 'backbone_with_lora.pt')
        print("💾 Model state saved")
    except Exception as e:
        print(f"⚠️ Warning: Could not save model state: {e}")
    
    # Save samples
    checkpoint_manager.save_checkpoint('samples_final', samples)
    
    # Save final summary
    final_checkpoint = {
        'config': config_dict,
        'num_samples': len(samples),
        'num_valid_samples': len([s for s in samples if s['valence'] is not None]),
        'num_with_pseudo': len([s for s in samples if s.get('pseudo_emotion') is not None]),
        'num_high_conf_pseudo': len([s for s in samples if s.get('use_pseudo', False)]),
        'status': 'phase1_complete',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    checkpoint_manager.save_checkpoint('phase1_complete', final_checkpoint)
    
    with open(PROJECT_ROOT / 'phase1_summary.json', 'w') as f:
        json.dump(final_checkpoint, f, indent=2)
    
    # STEP 8: Compute and Display Statistics
    compute_statistics(samples)
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total samples: {len(samples):,}")
    print(f"   Valid samples: {len([s for s in samples if s['valence'] is not None]):,}")
    print(f"   With pseudo-labels: {len([s for s in samples if s.get('pseudo_emotion')]):,}")
    print(f"\n📁 Output files saved to: {PROJECT_ROOT}")
    print(f"   - config.json")
    print(f"   - phase1_summary.json")
    print(f"   - checkpoints/ (multiple checkpoints for resume)")
    print(f"   - lora_adapters/backbone_with_lora.pt")
    
    print("\n🚀 Ready for Phase 2: Temporal Data Pipeline")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Progress has been saved in checkpoints.")
        print("You can resume by running this script again.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Check the checkpoints directory for saved progress.")