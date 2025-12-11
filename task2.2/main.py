#!/usr/bin/env python3
"""
Main training pipeline for Temporal Emotion Recognition
Orchestrates the complete training workflow
"""
import os
import argparse
from pathlib import Path

from config import Config
from train_pretrain import train_pretrain
from train_finetune import train_finetune
from evaluate import evaluate_model
from dataset import create_dataloaders, split_dataset
from model import TemporalEmotionModel
from utils import load_checkpoint, set_seed

import torch


def check_dataset(dataset_root: str) -> bool:
    """Check if dataset exists and has correct structure"""
    root = Path(dataset_root)
    
    if not root.exists():
        print(f"Error: Dataset directory not found: {dataset_root}")
        return False
    
    # Check for clip directories
    clip_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    if len(clip_dirs) == 0:
        print(f"Error: No clip directories found in {dataset_root}")
        print("Expected structure:")
        print("  AFEW-VA/")
        print("    001/")
        print("      00000.png")
        print("      00001.png")
        print("      ...")
        print("      001.json")
        return False
    
    print(f"✓ Found {len(clip_dirs)} clip directories")
    
    # Check a sample directory
    sample_dir = clip_dirs[0]
    json_files = list(sample_dir.glob("*.json"))
    png_files = list(sample_dir.glob("*.png"))
    
    if len(json_files) == 0:
        print(f"Warning: No JSON annotation files found in {sample_dir}")
    if len(png_files) == 0:
        print(f"Warning: No PNG frame files found in {sample_dir}")
    
    print(f"✓ Sample clip {sample_dir.name}: {len(png_files)} frames, {len(json_files)} annotations")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Emotion Recognition Model')
    parser.add_argument('--dataset', type=str, default='AFEW-VA',
                       help='Path to AFEW-VA dataset')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['pretrain', 'finetune', 'evaluate', 'all'],
                       help='Training stage to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming or evaluation')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Use LoRA for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Override number of frames per clip')
    parser.add_argument('--epochs_pretrain', type=int, default=None,
                       help='Override pretrain epochs')
    parser.add_argument('--epochs_finetune', type=int, default=None,
                       help='Override finetune epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.dataset_root = args.dataset
    config.random_seed = args.seed
    config.use_lora = args.use_lora
    
    # Override config if specified
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_frames is not None:
        config.num_frames = args.num_frames
    if args.epochs_pretrain is not None:
        config.num_epochs_pretrain = args.epochs_pretrain
    if args.epochs_finetune is not None:
        config.num_epochs_finetune = args.epochs_finetune
    
    # Set seed
    set_seed(config.random_seed)
    
    print("\n" + "="*80)
    print("TEMPORAL EMOTION RECOGNITION - TRAINING PIPELINE")
    print("="*80)
    print(f"Dataset: {config.dataset_root}")
    print(f"Stage: {args.stage}")
    print(f"Batch size: {config.batch_size}")
    print(f"Frames per clip: {config.num_frames}")
    print(f"Use LoRA: {config.use_lora}")
    print(f"Random seed: {config.random_seed}")
    print("="*80 + "\n")
    
    # Check dataset
    if not check_dataset(config.dataset_root):
        print("\nPlease ensure your dataset follows the expected structure:")
        print("AFEW-VA/")
        print("  001/")
        print("    00000.png")
        print("    00001.png")
        print("    ...")
        print("    001.json")
        print("  002/")
        print("    ...")
        return
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, using CPU (training will be slow)")
    
    print()
    
    # Run appropriate stage
    if args.stage in ['pretrain', 'all']:
        print("\n" + "="*80)
        print("STAGE 1: CONTRASTIVE PRETRAINING")
        print("="*80 + "\n")
        
        model, history = train_pretrain(config)
        
        print(f"\n✓ Pretraining complete!")
        print(f"  Best validation loss: {min(history['val_loss']):.4f}")
        print(f"  Checkpoint saved to: {config.checkpoint_dir}/pretrain_best.pth")
    
    if args.stage in ['finetune', 'all']:
        print("\n" + "="*80)
        print("STAGE 2: REGRESSION FINE-TUNING")
        print("="*80 + "\n")
        
        # Determine pretrain checkpoint
        if args.checkpoint:
            pretrain_checkpoint = args.checkpoint
        else:
            pretrain_checkpoint = os.path.join(config.checkpoint_dir, 'pretrain_best.pth')
        
        if not os.path.exists(pretrain_checkpoint):
            print(f"Error: Pretrain checkpoint not found: {pretrain_checkpoint}")
            if args.stage == 'finetune':
                print("Please run pretraining first or specify --checkpoint")
                return
        
        model, history = train_finetune(config, pretrain_checkpoint)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"  Best validation CCC: {max(history['val_ccc']):.4f}")
        print(f"  Test CCC: {history['test_metrics']['ccc_mean']:.4f}")
        print(f"  Checkpoint saved to: {config.checkpoint_dir}/finetune_best_best.pth")
    
    if args.stage == 'evaluate':
        print("\n" + "="*80)
        print("STAGE 3: EVALUATION")
        print("="*80 + "\n")
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint_path = args.checkpoint if args.checkpoint else \
                         os.path.join(config.checkpoint_dir, 'finetune_best_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        model = TemporalEmotionModel(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test dataloader
        train_clips, val_clips, test_clips = split_dataset(
            config.dataset_root,
            val_split=config.val_split,
            test_split=config.test_split,
            seed=config.random_seed
        )
        
        _, _, test_loader = create_dataloaders(
            config, train_clips, val_clips, test_clips
        )
        
        # Evaluate
        from evaluate import evaluate_model, analyze_embeddings
        
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
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Results: {config.output_dir}")
    print(f"  Visualizations: {config.output_dir}/evaluation")
    print("\nTo evaluate a trained model:")
    print(f"  python main.py --stage evaluate --checkpoint <path_to_checkpoint>")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()