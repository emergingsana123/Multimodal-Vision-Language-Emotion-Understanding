"""
Training script for contrastive pretraining
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from config import Config
from dataset import create_dataloaders, split_dataset
from model import TemporalEmotionModel
from losses import CombinedPretrainLoss
from utils import AverageMeter, save_checkpoint, load_checkpoint, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: CombinedPretrainLoss,
    optimizer,
    scaler,
    config: Config,
    epoch: int,
    device: torch.device
) -> dict:
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    ll_losses = AverageMeter()
    gl_losses = AverageMeter()
    smooth_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)  # [B, L, C, H, W]
        valences = batch['valences'].to(device)  # [B, L]
        arousals = batch['arousals'].to(device)  # [B, L]
        
        # Combine VA values
        va_values = torch.stack([valences, arousals], dim=-1)  # [B, L, 2]
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_amp):
            z_t, g, _ = model(images, return_regression=False)
            
            # Compute loss
            loss, loss_dict = criterion(
                z_t, g, va_values,
                memory_queue=model.memory_queue if config.memory_queue_size > 0 else None,
                config=config
            )
        
        # Backward pass
        if config.use_amp:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Update memory queue
        if config.memory_queue_size > 0:
            with torch.no_grad():
                # Flatten frame embeddings
                z_flat = z_t.reshape(-1, z_t.shape[-1])  # [B*L, D]
                model.update_memory_queue(z_flat)
        
        # Update metrics
        losses.update(loss_dict['total'], images.size(0))
        ll_losses.update(loss_dict['loss_ll'], images.size(0))
        gl_losses.update(loss_dict['loss_gl'], images.size(0))
        smooth_losses.update(loss_dict['loss_smooth'], images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'll': f"{ll_losses.avg:.4f}",
            'gl': f"{gl_losses.avg:.4f}",
            'sm': f"{smooth_losses.avg:.4f}"
        })
    
    return {
        'loss': losses.avg,
        'loss_ll': ll_losses.avg,
        'loss_gl': gl_losses.avg,
        'loss_smooth': smooth_losses.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: CombinedPretrainLoss,
    config: Config,
    device: torch.device
) -> dict:
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    ll_losses = AverageMeter()
    gl_losses = AverageMeter()
    smooth_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        images = batch['images'].to(device)
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        
        va_values = torch.stack([valences, arousals], dim=-1)
        
        # Forward pass
        z_t, g, _ = model(images, return_regression=False)
        
        # Compute loss
        loss, loss_dict = criterion(
            z_t, g, va_values,
            memory_queue=None,  # Don't use queue during validation
            config=config
        )
        
        losses.update(loss_dict['total'], images.size(0))
        ll_losses.update(loss_dict['loss_ll'], images.size(0))
        gl_losses.update(loss_dict['loss_gl'], images.size(0))
        smooth_losses.update(loss_dict['loss_smooth'], images.size(0))
        
        pbar.set_postfix({'val_loss': f"{losses.avg:.4f}"})
    
    return {
        'loss': losses.avg,
        'loss_ll': ll_losses.avg,
        'loss_gl': gl_losses.avg,
        'loss_smooth': smooth_losses.avg
    }


def train_pretrain(config: Config):
    """Main training loop for contrastive pretraining"""
    
    # Set seed
    set_seed(config.random_seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split dataset
    train_clips, val_clips, test_clips = split_dataset(
        config.dataset_root,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.random_seed
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_clips, val_clips, test_clips
    )
    
    # Create model
    model = TemporalEmotionModel(config).to(device)
    
    # Get trainable parameters
    params = model.get_trainable_parameters(stage='pretrain')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        params,
        lr=config.lr_pretrain,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs_pretrain,
        eta_min=config.lr_pretrain * 0.01
    )
    
    # Loss function
    criterion = CombinedPretrainLoss(
        lambda_ll=config.lambda_ll,
        lambda_gl=config.lambda_gl,
        lambda_smooth=config.lambda_smooth,
        temperature=config.temperature
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting Contrastive Pretraining")
    print("="*60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training clips: {len(train_clips)}")
    print(f"Validation clips: {len(val_clips)}")
    print(f"Epochs: {config.num_epochs_pretrain}")
    print("="*60 + "\n")
    
    for epoch in range(1, config.num_epochs_pretrain + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config, device)
        
        # Step scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.num_epochs_pretrain}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(LL: {train_metrics['loss_ll']:.4f}, "
              f"GL: {train_metrics['loss_gl']:.4f}, "
              f"Smooth: {train_metrics['loss_smooth']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % config.save_interval == 0 or val_metrics['loss'] < best_val_loss:
            is_best = val_metrics['loss'] < best_val_loss
            best_val_loss = min(val_metrics['loss'], best_val_loss)
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config,
                    'history': history
                },
                is_best,
                config.checkpoint_dir,
                filename=f'pretrain_epoch_{epoch}.pth'
            )
    
    print("\n" + "="*60)
    print("Pretraining Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60 + "\n")
    
    # Save final history
    with open(os.path.join(config.output_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == '__main__':
    config = Config()
    train_pretrain(config)