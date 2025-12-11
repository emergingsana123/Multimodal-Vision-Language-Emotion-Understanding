"""
Utility functions for training and evaluation
"""
import os
import torch
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth'
):
    """Save checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, filename.replace('epoch', 'best').replace('.pth', '_best.pth'))
        torch.save(state, best_path)
        print(f"  ✓ Saved best checkpoint: {best_path}")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def visualize_embeddings(
    embeddings: np.ndarray,
    va_values: np.ndarray,
    save_path: str,
    title: str = "Embedding Trajectory"
):
    """
    Visualize embedding trajectories with VA coloring
    
    Args:
        embeddings: [L, D] frame embeddings
        va_values: [L, 2] valence/arousal values
        save_path: path to save figure
        title: plot title
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Trajectory colored by time
    ax = axes[0]
    colors = np.arange(len(embeddings))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=colors, cmap='viridis', s=50, alpha=0.6)
    ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.set_title('Trajectory (colored by time)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Frame')
    
    # Plot 2: Colored by valence
    ax = axes[1]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=va_values[:, 0], cmap='RdYlGn', s=50, alpha=0.6, vmin=-1, vmax=1)
    ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.set_title('Colored by Valence')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Valence')
    
    # Plot 3: Colored by arousal
    ax = axes[2]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=va_values[:, 1], cmap='YlOrRd', s=50, alpha=0.6, vmin=-1, vmax=1)
    ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.set_title('Colored by Arousal')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Arousal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(
    true_va: np.ndarray,
    pred_va: np.ndarray,
    save_path: str,
    clip_id: str = ""
):
    """
    Plot ground truth vs predicted VA values over time
    
    Args:
        true_va: [L, 2] ground truth VA
        pred_va: [L, 2] predicted VA
        save_path: path to save figure
        clip_id: clip identifier for title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    frames = np.arange(len(true_va))
    
    # Valence
    ax = axes[0]
    ax.plot(frames, true_va[:, 0], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(frames, pred_va[:, 0], 'r--', label='Predicted', linewidth=2)
    ax.fill_between(frames, true_va[:, 0], pred_va[:, 0], alpha=0.2)
    ax.set_ylabel('Valence', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Valence Prediction - {clip_id}')
    
    # Arousal
    ax = axes[1]
    ax.plot(frames, true_va[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(frames, pred_va[:, 1], 'r--', label='Predicted', linewidth=2)
    ax.fill_between(frames, true_va[:, 1], pred_va[:, 1], alpha=0.2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Arousal Prediction - {clip_id}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(
    history: Dict[str, List],
    save_path: str,
    metric_name: str = "loss"
):
    """
    Plot training history
    
    Args:
        history: dictionary with 'train_{metric}' and 'val_{metric}' keys
        save_path: path to save figure
        metric_name: name of metric to plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history[f'train_{metric_name}']) + 1)
    
    ax.plot(epochs, history[f'train_{metric_name}'], 'b-o', label=f'Train {metric_name}')
    ax.plot(epochs, history[f'val_{metric_name}'], 'r-s', label=f'Val {metric_name}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f'Training History - {metric_name.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_temporal_consistency(embeddings: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between consecutive frames
    
    Args:
        embeddings: [B, L, D] or [L, D]
    Returns:
        consistency: mean cosine similarity
    """
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    if len(embeddings) < 2:
        return 0.0
    
    # Compute cosine similarity between consecutive frames
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = torch.nn.functional.cosine_similarity(
            embeddings[i:i+1], embeddings[i+1:i+2], dim=1
        )
        similarities.append(sim.item())
    
    return np.mean(similarities)


def analyze_embeddings(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 5
):
    """
    Analyze and visualize embeddings from validation set
    
    Args:
        model: trained model
        dataloader: validation dataloader
        device: torch device
        output_dir: directory to save visualizations
        num_samples: number of samples to visualize
    """
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    consistency_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            images = batch['images'].to(device)
            valences = batch['valences'].to(device)
            arousals = batch['arousals'].to(device)
            clip_ids = batch['clip_id']
            
            # Get embeddings
            z_t, g, _ = model(images, return_regression=False)
            
            # Process each clip in batch
            for i in range(len(clip_ids)):
                clip_id = clip_ids[i]
                z = z_t[i].cpu().numpy()  # [L, D]
                va = torch.stack([valences[i], arousals[i]], dim=-1).cpu().numpy()  # [L, 2]
                
                # Compute temporal consistency
                consistency = compute_temporal_consistency(z_t[i:i+1])
                consistency_scores.append(consistency)
                
                # Visualize trajectory
                save_path = os.path.join(output_dir, f'{clip_id}_trajectory.png')
                visualize_embeddings(z, va, save_path, title=f'Clip {clip_id} - Consistency: {consistency:.3f}')
    
    print(f"\nEmbedding Analysis:")
    print(f"  Mean temporal consistency: {np.mean(consistency_scores):.4f}")
    print(f"  Std temporal consistency: {np.std(consistency_scores):.4f}")
    print(f"  Saved {len(consistency_scores)} trajectory visualizations to {output_dir}")


def create_submission_format(
    predictions: Dict[str, np.ndarray],
    output_path: str
):
    """
    Create submission file in required format
    
    Args:
        predictions: dict mapping clip_id to [L, 2] predictions
        output_path: path to save submission
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'frame', 'valence', 'arousal'])
        
        for clip_id, preds in predictions.items():
            for frame_idx, (v, a) in enumerate(preds):
                writer.writerow([clip_id, frame_idx, f'{v:.6f}', f'{a:.6f}'])
    
    print(f"Saved submission to {output_path}")