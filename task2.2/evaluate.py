"""
Evaluation and inference script
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

from config import Config
from dataset import create_dataloaders, split_dataset, AFEWVADataset, get_clip_transform
from model import TemporalEmotionModel
from losses import compute_ccc
from utils import (
    load_checkpoint, plot_predictions, analyze_embeddings,
    compute_temporal_consistency, create_submission_format
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_visualizations: bool = True,
    output_dir: str = 'outputs/evaluation'
) -> Dict:
    """
    Comprehensive evaluation of the model
    
    Args:
        model: trained model
        dataloader: test dataloader
        device: torch device
        save_visualizations: whether to save prediction plots
        output_dir: directory for outputs
    
    Returns:
        results: dictionary with all metrics
    """
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_pred_v = []
    all_true_v = []
    all_pred_a = []
    all_true_a = []
    
    all_predictions = {}
    consistency_scores = []
    
    mse_list = []
    
    print("Evaluating model...")
    pbar = tqdm(dataloader, desc="Evaluation")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        valences = batch['valences'].to(device)
        arousals = batch['arousals'].to(device)
        clip_ids = batch['clip_id']
        
        # Forward pass
        z_t, g, pred_va = model(images, return_regression=True)
        
        # Collect predictions
        for i in range(len(clip_ids)):
            clip_id = clip_ids[i]
            
            # Get predictions and ground truth for this clip
            pred = pred_va[i].cpu().numpy()  # [L, 2]
            true = torch.stack([valences[i], arousals[i]], dim=-1).cpu().numpy()  # [L, 2]
            
            # Store predictions
            all_predictions[clip_id] = pred
            
            # Collect for overall metrics
            all_pred_v.append(pred[:, 0])
            all_true_v.append(true[:, 0])
            all_pred_a.append(pred[:, 1])
            all_true_a.append(true[:, 1])
            
            # Compute MSE for this clip
            mse = np.mean((pred - true) ** 2)
            mse_list.append(mse)
            
            # Compute temporal consistency
            consistency = compute_temporal_consistency(z_t[i:i+1])
            consistency_scores.append(consistency)
            
            # Save visualization for first few clips
            if save_visualizations and batch_idx < 5:
                save_path = os.path.join(output_dir, f'{clip_id}_predictions.png')
                plot_predictions(true, pred, save_path, clip_id)
    
    # Concatenate all predictions
    all_pred_v = np.concatenate(all_pred_v)
    all_true_v = np.concatenate(all_true_v)
    all_pred_a = np.concatenate(all_pred_a)
    all_true_a = np.concatenate(all_true_a)
    
    # Compute overall metrics
    ccc_v = compute_ccc(
        torch.tensor(all_true_v),
        torch.tensor(all_pred_v)
    )
    ccc_a = compute_ccc(
        torch.tensor(all_true_a),
        torch.tensor(all_pred_a)
    )
    ccc_mean = (ccc_v + ccc_a) / 2
    
    mse_overall = np.mean(mse_list)
    rmse_overall = np.sqrt(mse_overall)
    
    # Pearson correlation
    corr_v = np.corrcoef(all_true_v, all_pred_v)[0, 1]
    corr_a = np.corrcoef(all_true_a, all_pred_a)[0, 1]
    
    results = {
        'ccc_valence': ccc_v,
        'ccc_arousal': ccc_a,
        'ccc_mean': ccc_mean,
        'mse': mse_overall,
        'rmse': rmse_overall,
        'pearson_valence': corr_v,
        'pearson_arousal': corr_a,
        'temporal_consistency_mean': np.mean(consistency_scores),
        'temporal_consistency_std': np.std(consistency_scores),
        'num_clips': len(all_predictions)
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of clips evaluated: {results['num_clips']}")
    print(f"\nCCC Scores:")
    print(f"  Valence: {results['ccc_valence']:.4f}")
    print(f"  Arousal: {results['ccc_arousal']:.4f}")
    print(f"  Mean:    {results['ccc_mean']:.4f}")
    print(f"\nError Metrics:")
    print(f"  MSE:  {results['mse']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"\nPearson Correlation:")
    print(f"  Valence: {results['pearson_valence']:.4f}")
    print(f"  Arousal: {results['pearson_arousal']:.4f}")
    print(f"\nTemporal Consistency:")
    print(f"  Mean: {results['temporal_consistency_mean']:.4f}")
    print(f"  Std:  {results['temporal_consistency_std']:.4f}")
    print("="*60 + "\n")
    
    # Save results
    import json
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Create submission file
    submission_path = os.path.join(output_dir, 'submission.csv')
    create_submission_format(all_predictions, submission_path)
    
    return results


@torch.no_grad()
def inference_on_video(
    model: torch.nn.Module,
    video_frames: torch.Tensor,
    device: torch.device,
    window_size: int = 16,
    overlap: int = 8
) -> np.ndarray:
    """
    Run inference on a full video with sliding window
    
    Args:
        model: trained model
        video_frames: [T, C, H, W] video frames
        device: torch device
        window_size: number of frames per window
        overlap: overlap between windows
    
    Returns:
        predictions: [T, 2] VA predictions
    """
    model.eval()
    
    T = video_frames.shape[0]
    stride = window_size - overlap
    
    # Collect predictions for each frame
    frame_predictions = [[] for _ in range(T)]
    
    # Sliding window
    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Get window
        window = video_frames[start_idx:end_idx].unsqueeze(0).to(device)  # [1, L, C, H, W]
        
        # Predict
        _, _, pred_va = model(window, return_regression=True)
        pred_va = pred_va[0].cpu().numpy()  # [L, 2]
        
        # Store predictions
        for i, frame_idx in enumerate(range(start_idx, end_idx)):
            frame_predictions[frame_idx].append(pred_va[i])
    
    # Handle remaining frames
    if T % stride != 0:
        # Pad the last window
        remaining = T - (T // stride) * stride
        if remaining > 0:
            last_window_start = T - window_size
            if last_window_start >= 0:
                window = video_frames[last_window_start:].unsqueeze(0).to(device)
                _, _, pred_va = model(window, return_regression=True)
                pred_va = pred_va[0].cpu().numpy()
                
                for i, frame_idx in enumerate(range(last_window_start, T)):
                    if len(frame_predictions[frame_idx]) == 0:
                        frame_predictions[frame_idx].append(pred_va[i])
    
    # Average predictions for each frame
    predictions = np.zeros((T, 2))
    for i in range(T):
        if len(frame_predictions[i]) > 0:
            predictions[i] = np.mean(frame_predictions[i], axis=0)
        else:
            # If no prediction, use nearest neighbor
            if i > 0:
                predictions[i] = predictions[i-1]
    
    return predictions


def main():
    """Main evaluation function"""
    config = Config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    checkpoint_path = os.path.join(config.checkpoint_dir, 'finetune_best_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_pretrain.py and train_finetune.py")
        return
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Create model
    model = TemporalEmotionModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Split dataset
    train_clips, val_clips, test_clips = split_dataset(
        config.dataset_root,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.random_seed
    )
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        config, train_clips, val_clips, test_clips
    )
    
    # Evaluate
    output_dir = os.path.join(config.output_dir, 'evaluation')
    results = evaluate_model(
        model, test_loader, device,
        save_visualizations=True,
        output_dir=output_dir
    )
    
    # Analyze embeddings
    print("\nAnalyzing embeddings...")
    embed_dir = os.path.join(output_dir, 'embeddings')
    analyze_embeddings(model, test_loader, device, embed_dir, num_samples=5)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()