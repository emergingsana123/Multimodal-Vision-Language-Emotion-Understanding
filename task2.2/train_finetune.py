"""
Training script for regression fine-tuning
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import json

from config import Config
from dataset import create_dataloaders, split_dataset
from model import TemporalEmotionModel
from losses import RegressionLoss, compute_ccc
from utils import AverageMeter, save_checkpoint, load_checkpoint, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: RegressionLoss,
    optimizer,
    scaler,
    config: Config,
    epoch: int,
    device: torch.device
) -> dict:
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    mse_meter = AverageMeter()
    ccc_v_meter = AverageMeter()
    ccc_a_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)  # [B, L, C, H, W]
        valences = batch['valences'].to(device)  # [B, L]
        arousals = batch['arousals'].to(device)  # [B, L]
        
        # Stack ground truth
        true_va = torch.stack([valences, arousals], dim=-1)  # [B, L, 2]
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_amp):
            _, _, pred_va = model(images, return_regression=True)
            
            # Compute loss
            loss, metrics = criterion(pred_va, true_va)
        
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
        
        # Update metrics
        losses.update(metrics['total'], images.size(0))
        mse_meter.update(metrics['mse'], images.size(0))
        ccc_v_meter.update(metrics['ccc_valence'], images.size(0))
        ccc_a_meter.update(metrics['ccc_arousal'], images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'mse': f"{mse_meter.avg:.4f}",
            'ccc_v': f"{ccc_v_meter.avg:.4f}",
            'ccc_a': f"{ccc_a_meter.avg:.4f}"
        })
    
    return {
        'loss': losses.avg,
        'mse': mse_meter.avg,
        'ccc_valence': ccc_v_meter.avg,
        'ccc_arousal': ccc_a_meter.avg,
        'ccc_mean': (ccc_v_meter.avg + ccc_a_meter.avg) / 2
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: RegressionLoss,
    config: Config,
    device: torch.device
) -> dict:
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    mse_meter = AverageMeter()
    
    all_pred_v = []
    all_true_v = []
    all_pred_a = []
    all_true_a = []
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        images = batch['images'].to(device)
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        
        true_va = torch.stack([valences, arousals], dim=-1)
        
        # Forward pass
        _, _, pred_va = model(images, return_regression=True)
        
        # Compute loss
        loss, metrics = criterion(pred_va, true_va)
        
        losses.update(metrics['total'], images.size(0))
        mse_meter.update(metrics['mse'], images.size(0))
        
        # Collect predictions for CCC computation
        all_pred_v.append(pred_va[:, :, 0].cpu())
        all_true_v.append(true_va[:, :, 0].cpu())
        all_pred_a.append(pred_va[:, :, 1].cpu())
        all_true_a.append(true_va[:, :, 1].cpu())
        
        pbar.set_postfix({'val_loss': f"{losses.avg:.4f}"})
    
    # Concatenate all predictions
    all_pred_v = torch.cat(all_pred_v, dim=0).flatten()
    all_true_v = torch.cat(all_true_v, dim=0).flatten()
    all_pred_a = torch.cat(all_pred_a, dim=0).flatten()
    all_true_a = torch.cat(all_true_a, dim=0).flatten()
    
    # Compute CCC
    ccc_v = compute_ccc(all_true_v, all_pred_v)
    ccc_a = compute_ccc(all_true_a, all_pred_a)
    ccc_mean = (ccc_v + ccc_a) / 2
    
    return {
        'loss': losses.avg,
        'mse': mse_meter.avg,
        'ccc_valence': ccc_v,
        'ccc_arousal': ccc_a,
        'ccc_mean': ccc_mean
    }


def train_finetune(config: Config, pretrain_checkpoint: str = None):
    """Main training loop for regression fine-tuning"""
    
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
    
    # Load pretrained weights
    if pretrain_checkpoint is not None:
        print(f"Loading pretrained weights from {pretrain_checkpoint}")
        checkpoint = load_checkpoint(pretrain_checkpoint, device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("Warning: Starting fine-tuning without pretrained weights!")
    
    # Get trainable parameters (LoRA + temporal + regression heads)
    params = model.get_trainable_parameters(stage='finetune')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        params,
        lr=config.lr_finetune,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs_finetune,
        eta_min=config.lr_finetune * 0.01
    )
    
    # Loss function
    criterion = RegressionLoss(mse_weight=0.5, ccc_weight=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training history
    history = {
        'train_loss': [],
        'train_ccc': [],
        'val_loss': [],
        'val_ccc': [],
        'lr': []
    }
    
    best_val_ccc = -float('inf')
    
    print("\n" + "="*60)
    print("Starting Regression Fine-tuning")
    print("="*60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training clips: {len(train_clips)}")
    print(f"Validation clips: {len(val_clips)}")
    print(f"Epochs: {config.num_epochs_finetune}")
    print("="*60 + "\n")
    
    for epoch in range(1, config.num_epochs_finetune + 1):
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
        history['train_ccc'].append(train_metrics['ccc_mean'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ccc'].append(val_metrics['ccc_mean'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.num_epochs_finetune}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"CCC: {train_metrics['ccc_mean']:.4f} "
              f"(V: {train_metrics['ccc_valence']:.4f}, A: {train_metrics['ccc_arousal']:.4f})")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"CCC: {val_metrics['ccc_mean']:.4f} "
              f"(V: {val_metrics['ccc_valence']:.4f}, A: {val_metrics['ccc_arousal']:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % config.save_interval == 0 or val_metrics['ccc_mean'] > best_val_ccc:
            is_best = val_metrics['ccc_mean'] > best_val_ccc
            best_val_ccc = max(val_metrics['ccc_mean'], best_val_ccc)
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_ccc': val_metrics['ccc_mean'],
                    'config': config,
                    'history': history
                },
                is_best,
                config.checkpoint_dir,
                filename=f'finetune_epoch_{epoch}.pth'
            )
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print(f"Best validation CCC: {best_val_ccc:.4f}")
    print("="*60 + "\n")
    
    # Test on test set
    print("Evaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, config, device)
    print(f"Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  CCC: {test_metrics['ccc_mean']:.4f} "
          f"(V: {test_metrics['ccc_valence']:.4f}, A: {test_metrics['ccc_arousal']:.4f})")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    
    # Save final history
    history['test_metrics'] = test_metrics
    with open(os.path.join(config.output_dir, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == '__main__':
    config = Config()
    
    # Path to best pretrained checkpoint
    pretrain_checkpoint = os.path.join(config.checkpoint_dir, 'pretrain_best.pth')
    
    train_finetune(config, pretrain_checkpoint)