"""
task2_phase2_train.py
Phase 2C: Training Loop for Temporal Emotion Contrastive Learning

Complete training pipeline with:
- Multi-phase training (warmup → full → fine-tuning)
- Checkpoint management and resume capability
- Comprehensive metrics tracking
- Validation and early stopping
- UMAP visualization of learned embeddings
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import from previous phases
from task2_temporal_analysis_backbone import (
    TemporalEmotionConfig, 
    CheckpointManager,
    VideoMAEBackbone
)
from task2_dataset import build_temporal_dataloaders
from task2_loss import TemporalContrastiveLoss

# For visualization
try:
    import umap
    from sklearn.decomposition import PCA
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not available. Install with: pip install umap-learn")


# ============================================================================
# TRAINING STATE MANAGER
# ============================================================================

class TrainingState:
    """Manages training state for checkpointing and resume"""
    
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_phase = 'warmup'  # 'warmup', 'full', 'finetune'
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        
        # Learning rate history
        self.lr_history = []
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'training_phase': self.training_phase,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': dict(self.train_metrics_history),
            'val_metrics_history': dict(self.val_metrics_history),
            'lr_history': self.lr_history,
        }
    
    @classmethod
    def from_dict(cls, config, state_dict):
        """Load from dictionary"""
        state = cls(config)
        state.epoch = state_dict['epoch']
        state.global_step = state_dict['global_step']
        state.best_val_loss = state_dict['best_val_loss']
        state.epochs_without_improvement = state_dict['epochs_without_improvement']
        state.training_phase = state_dict['training_phase']
        state.train_losses = state_dict['train_losses']
        state.val_losses = state_dict['val_losses']
        state.train_metrics_history = defaultdict(list, state_dict['train_metrics_history'])
        state.val_metrics_history = defaultdict(list, state_dict['val_metrics_history'])
        state.lr_history = state_dict['lr_history']
        return state


# ============================================================================
# TRAINER CLASS
# ============================================================================

class TemporalEmotionTrainer:
    """Main trainer for temporal emotion contrastive learning"""
    
    def __init__(self, 
                 model: VideoMAEBackbone,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TemporalEmotionConfig,
                 checkpoint_manager: CheckpointManager,
                 device: torch.device):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        
        # Initialize loss
        self.criterion = TemporalContrastiveLoss(config).to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.state = TrainingState(config)
        
        # Try to load existing checkpoint
        self._load_checkpoint_if_exists()
    
    def _create_optimizer(self):
        """Create optimizer for LoRA parameters"""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        print(f"✅ Optimizer created:")
        print(f"   Trainable parameters: {len(trainable_params)}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Weight decay: {self.config.weight_decay}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = self.config.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"✅ Scheduler created:")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
        
        return scheduler
    
    def _load_checkpoint_if_exists(self):
        """Try to load existing training checkpoint"""
        checkpoint_data = self.checkpoint_manager.load_checkpoint('training_state')
        
        if checkpoint_data is not None:
            print(f"\n{'='*80}")
            print("RESUMING FROM CHECKPOINT")
            print(f"{'='*80}")
            
            # Load model state
            self.model.load_state_dict(checkpoint_data['model_state'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            
            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state'])
            
            # Load training state
            self.state = TrainingState.from_dict(self.config, checkpoint_data['training_state'])
            
            print(f"Resumed from epoch {self.state.epoch}")
            print(f"Best val loss: {self.state.best_val_loss:.4f}")
            print(f"Training phase: {self.state.training_phase}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_data = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_state': self.state.to_dict(),
            'config': self.config.__dict__,
        }
        
        # Save regular checkpoint
        self.checkpoint_manager.save_checkpoint('training_state', checkpoint_data)
        
        # Save best model
        if is_best:
            self.checkpoint_manager.save_checkpoint('best_model', checkpoint_data)
            print(f"💾 Saved best model (val_loss: {self.state.best_val_loss:.4f})")
        
        # Save epoch checkpoint
        if self.state.epoch % self.config.save_frequency == 0:
            epoch_name = f'epoch_{self.state.epoch:03d}'
            self.checkpoint_manager.save_checkpoint(epoch_name, checkpoint_data)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device) if batch['positive'] is not None else None
            negative = batch['negative'].to(self.device) if batch['negative'] is not None else None
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_mixed_precision):
                # Get embeddings
                anchor_embeddings = self.model.get_embeddings(anchor)
                
                if positive is not None:
                    positive_embeddings = self.model.get_embeddings(positive)
                else:
                    positive_embeddings = torch.zeros_like(anchor_embeddings[:0])
                
                if negative is not None:
                    negative_embeddings = self.model.get_embeddings(negative)
                else:
                    negative_embeddings = torch.zeros_like(anchor_embeddings[:0])
                
                # Prepare batch data for loss
                loss_input = {
                    'anchor_embeddings': anchor_embeddings,
                    'positive_embeddings': positive_embeddings,
                    'negative_embeddings': negative_embeddings,
                }
                
                # Compute loss
                loss, metrics = self.criterion(loss_input)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Record metrics
            epoch_losses.append(loss.item())
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'pos_sim': f"{metrics.get('mean_pos_sim', 0):.3f}",
                'neg_sim': f"{metrics.get('mean_neg_sim', 0):.3f}",
            })
            
            self.state.global_step += 1
            
            # Save checkpoint every N batches
            if batch_idx % self.config.checkpoint_frequency == 0 and batch_idx > 0:
                self.save_checkpoint()
        
        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Record in state
        self.state.train_losses.append(avg_loss)
        for key, value in avg_metrics.items():
            self.state.train_metrics_history[key].append(value)
        
        current_lr = self.scheduler.get_last_lr()[0]
        self.state.lr_history.append(current_lr)
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        val_losses = []
        val_metrics = defaultdict(list)
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch in progress_bar:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device) if batch['positive'] is not None else None
            negative = batch['negative'].to(self.device) if batch['negative'] is not None else None
            
            # Get embeddings
            anchor_embeddings = self.model.get_embeddings(anchor)
            
            if positive is not None:
                positive_embeddings = self.model.get_embeddings(positive)
            else:
                positive_embeddings = torch.zeros_like(anchor_embeddings[:0])
            
            if negative is not None:
                negative_embeddings = self.model.get_embeddings(negative)
            else:
                negative_embeddings = torch.zeros_like(anchor_embeddings[:0])
            
            # Prepare batch data for loss
            loss_input = {
                'anchor_embeddings': anchor_embeddings,
                'positive_embeddings': positive_embeddings,
                'negative_embeddings': negative_embeddings,
            }
            
            # Compute loss
            loss, metrics = self.criterion(loss_input)
            
            # Record metrics
            val_losses.append(loss.item())
            for key, value in metrics.items():
                val_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
            })
        
        # Compute validation statistics
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        # Record in state
        self.state.val_losses.append(avg_loss)
        for key, value in avg_metrics.items():
            self.state.val_metrics_history[key].append(value)
        
        return avg_loss, avg_metrics
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}")
        print(f"Training phase: {self.state.training_phase}")
        print(f"Starting epoch: {self.state.epoch}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        start_epoch = self.state.epoch
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*80}")
            
            # Update training phase
            if epoch < self.config.warmup_epochs:
                self.state.training_phase = 'warmup'
            elif epoch < self.config.num_epochs - 5:
                self.state.training_phase = 'full'
            else:
                self.state.training_phase = 'finetune'
            
            print(f"Training phase: {self.state.training_phase}")
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            print(f"\n📊 Training Results:")
            print(f"   Loss: {train_loss:.4f}")
            print(f"   Contrastive loss: {train_metrics.get('contrastive_loss', 0):.4f}")
            print(f"   Pos similarity: {train_metrics.get('mean_pos_sim', 0):.4f}")
            print(f"   Neg similarity: {train_metrics.get('mean_neg_sim', 0):.4f}")
            print(f"   Gap: {train_metrics.get('pos_neg_gap', 0):.4f}")
            
            # Validate
            if (epoch + 1) % self.config.eval_frequency == 0:
                val_loss, val_metrics = self.validate()
                
                print(f"\n📊 Validation Results:")
                print(f"   Loss: {val_loss:.4f}")
                print(f"   Contrastive loss: {val_metrics.get('contrastive_loss', 0):.4f}")
                
                # Check if best model
                is_best = val_loss < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_loss
                    self.state.epochs_without_improvement = 0
                    print(f"   🎉 New best validation loss!")
                else:
                    self.state.epochs_without_improvement += 1
                
                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
                
                # Early stopping
                if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"\n⚠️ Early stopping triggered after {self.state.epochs_without_improvement} epochs without improvement")
                    break
            else:
                # Save checkpoint even without validation
                self.save_checkpoint()
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()
        
        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.state.best_val_loss:.4f}")
        print(f"Total epochs: {self.state.epoch + 1}")
        print(f"{'='*80}\n")
    
    def plot_training_curves(self):
        """Plot training curves"""
        results_dir = Path(self.config.project_root) / 'results'
        results_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax = axes[0, 0]
        epochs = range(1, len(self.state.train_losses) + 1)
        ax.plot(epochs, self.state.train_losses, label='Train', marker='o')
        if self.state.val_losses:
            val_epochs = range(self.config.eval_frequency, 
                             len(self.state.val_losses) * self.config.eval_frequency + 1,
                             self.config.eval_frequency)
            ax.plot(val_epochs, self.state.val_losses, label='Val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[0, 1]
        ax.plot(self.state.lr_history)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Similarity gap
        ax = axes[1, 0]
        if 'pos_neg_gap' in self.state.train_metrics_history:
            gaps = self.state.train_metrics_history['pos_neg_gap']
            ax.plot(range(1, len(gaps) + 1), gaps, marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Pos-Neg Similarity Gap')
            ax.set_title('Contrastive Learning Progress')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
        
        # Positive vs Negative similarities
        ax = axes[1, 1]
        if 'mean_pos_sim' in self.state.train_metrics_history:
            pos_sims = self.state.train_metrics_history['mean_pos_sim']
            neg_sims = self.state.train_metrics_history['mean_neg_sim']
            epochs = range(1, len(pos_sims) + 1)
            ax.plot(epochs, pos_sims, label='Positive', marker='o')
            ax.plot(epochs, neg_sims, label='Negative', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title('Positive vs Negative Similarities')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f'training_curves_epoch_{self.state.epoch:03d}.png', dpi=150)
        plt.close()
        
        print(f"📈 Training curves saved to {results_dir}")


# ============================================================================
# EMBEDDING EXTRACTOR
# ============================================================================

@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=1000):
    """Extract embeddings from dataset for visualization"""
    model.eval()
    
    all_embeddings = []
    all_valences = []
    all_arousals = []
    all_clip_ids = []
    
    samples_collected = 0
    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if samples_collected >= max_samples:
            break
        
        anchor = batch['anchor'].to(device)
        metadata = batch['metadata']
        
        # Get embeddings
        embeddings = model.get_embeddings(anchor)
        
        # Collect
        all_embeddings.append(embeddings.cpu())
        all_valences.append(metadata['valences'].cpu())
        all_arousals.append(metadata['arousals'].cpu())
        all_clip_ids.extend(metadata['clip_ids'])
        
        samples_collected += anchor.shape[0]
    
    # Concatenate
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    valences = torch.cat(all_valences, dim=0).numpy()
    arousals = torch.cat(all_arousals, dim=0).numpy()
    
    return embeddings, valences, arousals, all_clip_ids


def visualize_embeddings(embeddings, valences, arousals, save_path, method='umap'):
    """Visualize embeddings with UMAP or PCA"""
    
    if method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Valence
    ax = axes[0]
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=valences, cmap='RdYlGn', s=20, alpha=0.6)
    ax.set_title(f'Embeddings colored by Valence ({method.upper()})')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Valence')
    
    # Arousal
    ax = axes[1]
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                        c=arousals, cmap='coolwarm', s=20, alpha=0.6)
    ax.set_title(f'Embeddings colored by Arousal ({method.upper()})')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Arousal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"📊 Embeddings visualization saved to {save_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("PHASE 2C: TEMPORAL EMOTION CONTRASTIVE TRAINING")
    print("="*80)
    
    # Load configuration
    config = TemporalEmotionConfig()
    
    # Set device
    device = torch.device(config.device)
    print(f"\n💻 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize checkpoint manager
    checkpoint_dir = Path(config.project_root) / 'checkpoints'
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Load samples
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    samples = checkpoint_manager.load_checkpoint('samples_final')
    if samples is None:
        print("❌ No samples found. Run Phase 1 first.")
        return
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Load model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    
    from transformers import VideoMAEImageProcessor
    
    # Initialize backbone
    backbone = VideoMAEBackbone(
        model_name=config.backbone_model,
        freeze=config.freeze_backbone
    )
    
    # Load LoRA weights
    lora_path = Path(config.project_root) / 'lora_adapters' / 'backbone_with_lora.pt'
    if lora_path.exists():
        print(f"Loading LoRA weights from {lora_path}")
        backbone.load_state_dict(torch.load(lora_path, map_location='cpu'))
        print("✅ LoRA weights loaded")
    else:
        print("⚠️ No LoRA weights found, starting from scratch")
    
    backbone = backbone.to(device)
    
    # Initialize processor
    processor = VideoMAEImageProcessor.from_pretrained(config.backbone_model)
    
    # Build dataloaders
    print(f"\n{'='*80}")
    print("BUILDING DATALOADERS")
    print(f"{'='*80}")
    
    train_loader, val_loader = build_temporal_dataloaders(
        samples=samples,
        config=config,
        image_processor=processor,
        train_ratio=0.8
    )
    
    # Initialize trainer
    print(f"\n{'='*80}")
    print("INITIALIZING TRAINER")
    print(f"{'='*80}")
    
    trainer = TemporalEmotionTrainer(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device=device
    )
    
    # Start training
    trainer.train()
    
    # Extract and visualize final embeddings
    print(f"\n{'='*80}")
    print("EXTRACTING FINAL EMBEDDINGS")
    print(f"{'='*80}")
    
    embeddings, valences, arousals, clip_ids = extract_embeddings(
        model=backbone,
        dataloader=val_loader,
        device=device,
        max_samples=1000
    )
    
    results_dir = Path(config.project_root) / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Visualize with UMAP
    if UMAP_AVAILABLE:
        visualize_embeddings(
            embeddings, valences, arousals,
            save_path=results_dir / 'final_embeddings_umap.png',
            method='umap'
        )
    
    # Visualize with PCA (always available)
    visualize_embeddings(
        embeddings, valences, arousals,
        save_path=results_dir / 'final_embeddings_pca.png',
        method='pca'
    )
    
    print(f"\n{'='*80}")
    print("✅ PHASE 2C COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {config.project_root}")
    print(f"   - checkpoints/training_state.pkl")
    print(f"   - checkpoints/best_model.pkl")
    print(f"   - results/training_curves_*.png")
    print(f"   - results/final_embeddings_*.png")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted. Progress saved in checkpoints.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()