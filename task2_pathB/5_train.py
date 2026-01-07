"""
Complete Training Loop for Temporal Emotion Contrastive Learning
Optimized for 20GB GPU, 7-10 hour training budget
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from config import get_config, print_config
from utils import (
    setup_logging,
    CheckpointManager,
    ProgressTracker,
    print_gpu_memory,
    clear_gpu_cache,
    set_seed
)
from dataset import build_dataloaders
from model import create_model
from loss import InfoNCELoss, compute_contrastive_accuracy


# ============================================================================
# TRAINING STATE
# ============================================================================

class TrainingState:
    """Track training progress and metrics"""
    
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
    
    def to_dict(self):
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
        }
    
    @classmethod
    def from_dict(cls, data):
        state = cls()
        for key, value in data.items():
            setattr(state, key, value)
        return state


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Main trainer class"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        
        # Initialize components
        self._init_data()
        self._init_model()
        self._init_loss()
        self._init_optimizer()
        self._init_scheduler()
        self._init_training_state()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.checkpoints_dir)
        
        # Progress tracker
        self.progress_tracker = ProgressTracker(
            total_epochs=config.num_epochs,
            steps_per_epoch=len(self.train_loader)
        )
        
        # Try to resume from checkpoint
        self._resume_from_checkpoint()
    
    def _init_data(self):
        """Initialize dataloaders"""
        self.logger.info("Initializing dataloaders...")
        self.train_loader, self.val_loader = build_dataloaders(self.config)
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _init_model(self):
        """Initialize model"""
        self.logger.info("Initializing model...")
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'transformer_encoder'):
            if hasattr(self.model.transformer_encoder, 'layers'):
                for layer in self.model.transformer_encoder.layers:
                    if hasattr(layer, 'checkpoint'):
                        layer.checkpoint = True
        
        params = self.model.get_num_params()
        self.logger.info(f"Model parameters: {params['total']:,} ({params['total']*4/1e6:.1f}MB)")
    
    def _init_loss(self):
        """Initialize loss function"""
        self.logger.info(f"Initializing loss (temperature={self.config.temperature})...")
        self.criterion = InfoNCELoss(temperature=self.config.temperature)
    
    def _init_optimizer(self):
        """Initialize optimizer"""
        self.logger.info("Initializing optimizer...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Weight decay: {self.config.weight_decay}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler with warmup"""
        self.logger.info("Initializing scheduler...")
        
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
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.logger.info(f"Warmup steps: {warmup_steps}")
        self.logger.info(f"Total steps: {total_steps}")
    
    def _init_training_state(self):
        """Initialize training state"""
        self.state = TrainingState()
    
    def _resume_from_checkpoint(self):
        """Try to resume from latest checkpoint"""
        checkpoint = self.checkpoint_manager.load('latest')
        
        if checkpoint is not None:
            self.logger.info("="*80)
            self.logger.info("RESUMING FROM CHECKPOINT")
            self.logger.info("="*80)
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.state = TrainingState.from_dict(checkpoint['training_state'])
            
            if self.scaler and 'scaler_state' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state'])
            
            self.logger.info(f"Resumed from epoch {self.state.epoch}")
            self.logger.info(f"Best val loss: {self.state.best_val_loss:.4f}")
        else:
            self.logger.info("Starting training from scratch")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_state': self.state.to_dict(),
            'config': self.config.__dict__,
        }
        
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        # Save latest
        self.checkpoint_manager.save('latest', checkpoint)
        
        # Save best
        if is_best:
            self.checkpoint_manager.save('best_model', checkpoint, is_best=True)
        
        # Save periodic epoch checkpoint
        if (self.state.epoch + 1) % self.config.save_frequency == 0:
            self.checkpoint_manager.save(f'epoch_{self.state.epoch:03d}', checkpoint)
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch+1}/{self.config.num_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            anchor = batch['anchor'].to(self.device)  # (batch, frames, feat_dim)
            positives = batch['positives'].to(self.device)  # (batch*num_pos, frames, feat_dim)
            
            # Forward pass through model
            with autocast(enabled=self.config.use_mixed_precision):
                # Get embeddings
                anchor_emb = self.model(anchor)  # (batch, embed_dim)
                positive_emb = self.model(positives)  # (batch*num_pos, embed_dim)
                
                # Compute loss
                loss, metrics = self.criterion(anchor_emb, positive_emb)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1
            
            # Record metrics (unscale loss for logging)
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            epoch_metrics.append(metrics)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.3f}",
                'lr': f"{current_lr:.2e}",
                'gap': f"{metrics['pos_neg_gap']:.3f}"
            })
            
            # Log periodically
            if (batch_idx + 1) % self.config.log_frequency == 0:
                self.logger.info(
                    f"Epoch {self.state.epoch+1} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {np.mean(epoch_losses[-10:]):.4f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        val_losses = []
        val_metrics = []
        all_anchors = []
        all_positives = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            anchor = batch['anchor'].to(self.device)
            positives = batch['positives'].to(self.device)
            
            # Forward pass
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positives)
            
            # Compute loss
            loss, metrics = self.criterion(anchor_emb, positive_emb)
            
            val_losses.append(loss.item())
            val_metrics.append(metrics)
            
            # Collect for accuracy
            all_anchors.append(anchor_emb.cpu())
            all_positives.append(positive_emb.cpu())
        
        # Average metrics
        avg_loss = np.mean(val_losses)
        avg_metrics = {
            key: np.mean([m[key] for m in val_metrics])
            for key in val_metrics[0].keys()
        }
        
        # Compute accuracy
        all_anchors = torch.cat(all_anchors)
        all_positives = torch.cat(all_positives)
        accuracy = compute_contrastive_accuracy(all_anchors, all_positives)
        avg_metrics['accuracy'] = accuracy
        
        return avg_loss, avg_metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("="*80)
        self.logger.info("STARTING TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print_gpu_memory("Initial ")
        self.logger.info("="*80)
        
        self.progress_tracker.start()
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            self.progress_tracker.start_epoch(epoch)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Record
            self.state.train_losses.append(train_loss)
            self.state.train_metrics.append(train_metrics)
            self.state.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            # Log training results
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch+1}/{self.config.num_epochs} - TRAINING")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Loss: {train_loss:.4f}")
            self.logger.info(f"Pos similarity: {train_metrics['mean_pos_sim']:.4f}")
            self.logger.info(f"Neg similarity: {train_metrics['mean_neg_sim']:.4f}")
            self.logger.info(f"Gap: {train_metrics['pos_neg_gap']:.4f}")
            self.logger.info(f"Learning rate: {self.state.learning_rates[-1]:.2e}")
            self.logger.info(f"Epoch time: {self.progress_tracker.format_time(self.progress_tracker.get_epoch_time())}")
            
            # Validate
            if (epoch + 1) % self.config.eval_frequency == 0:
                val_loss, val_metrics = self.validate()
                
                self.state.val_losses.append(val_loss)
                self.state.val_metrics.append(val_metrics)
                
                # Log validation results
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"EPOCH {epoch+1}/{self.config.num_epochs} - VALIDATION")
                self.logger.info(f"{'='*80}")
                self.logger.info(f"Loss: {val_loss:.4f}")
                self.logger.info(f"Accuracy: {val_metrics['accuracy']:.2%}")
                self.logger.info(f"Pos similarity: {val_metrics['mean_pos_sim']:.4f}")
                self.logger.info(f"Neg similarity: {val_metrics['mean_neg_sim']:.4f}")
                self.logger.info(f"Gap: {val_metrics['pos_neg_gap']:.4f}")
                
                # Check for improvement
                is_best = val_loss < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_loss
                    self.state.patience_counter = 0
                    self.logger.info(f" New best validation loss: {val_loss:.4f}")
                else:
                    self.state.patience_counter += 1
                    self.logger.info(f"No improvement ({self.state.patience_counter}/{self.config.early_stopping_patience})")
                
                # Save checkpoint
                self._save_checkpoint(is_best=is_best)
                
                # Early stopping
                if self.state.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"\n Early stopping triggered!")
                    break
            else:
                # Save checkpoint even without validation
                self._save_checkpoint()
            
            # ETA
            eta = self.progress_tracker.get_eta(epoch)
            self.logger.info(f"ETA: {eta}")
            
            # Clear cache
            clear_gpu_cache()
        
        self.logger.info("\n" + "="*80)
        self.logger.info(" TRAINING COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"Best validation loss: {self.state.best_val_loss:.4f}")
        self.logger.info(f"Total time: {self.progress_tracker.format_time(self.progress_tracker.get_total_time())}")
        self.logger.info(f"Final epoch: {self.state.epoch + 1}")
        self.logger.info("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train temporal emotion contrastive model")
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Setup logging
    logger = setup_logging(config.logs_dir, 'training')
    
    # Print config
    logger.info("\n" + "="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    print_config(config)
    
    # Set seed
    set_seed(config.random_seed)
    logger.info(f"Random seed: {config.random_seed}")
    
    # Check feature extraction
    features_index = Path(config.features_cache_dir) / 'index.json'
    if not features_index.exists():
        logger.error(f" Features not extracted!")
        logger.error(f"Run: python 1_extract_features.py")
        sys.exit(1)
    
    try:
        # Initialize trainer
        trainer = Trainer(config, logger)
        
        # Train
        trainer.train()
        
        logger.info("\n Training completed successfully!")
        logger.info(f"Checkpoints saved to: {config.checkpoints_dir}")
        logger.info(f"Logs saved to: {config.logs_dir}")
        
    except KeyboardInterrupt:
        logger.warning("\n Training interrupted by user")
        logger.info("Progress saved in checkpoints. Run with --resume to continue.")
    
    except Exception as e:
        logger.error(f"\n Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()