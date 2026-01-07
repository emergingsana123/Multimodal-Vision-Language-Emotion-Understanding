"""
task2_phase2_train_test.py
Quick test script to verify training pipeline works before full training
Uses minimal data and small batches to test quickly and efficiently
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# Import from previous phases
from task2_temporal_analysis_backbone import (
    TemporalEmotionConfig, 
    CheckpointManager,
    VideoMAEBackbone
)
from task2_dataset import build_temporal_dataloaders
from task2_loss import TemporalContrastiveLoss
from transformers import VideoMAEImageProcessor

def test_training_pipeline():
    """Test the training pipeline with minimal resources"""
    
    print("="*80)
    print("TESTING TRAINING PIPELINE (LIGHTWEIGHT)")
    print("="*80)
    
    # Load configuration with memory-efficient settings
    config = TemporalEmotionConfig()
    
    # Override settings for testing
    config.batch_size = 2  # Very small batch
    config.num_workers = 0  # No multiprocessing for testing
    config.num_positive_pairs = 1  # Fewer pairs
    config.num_negative_pairs = 4  # Fewer negatives
    config.use_mixed_precision = True  # Save memory
    
    print(f"\n🔧 Test Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Num workers: {config.num_workers}")
    print(f"   Positive pairs: {config.num_positive_pairs}")
    print(f"   Negative pairs: {config.num_negative_pairs}")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    
    # Set device
    device = torch.device(config.device)
    print(f"\n💻 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        # Clear cache
        torch.cuda.empty_cache()
        print("   GPU cache cleared")
    
    # Initialize checkpoint manager
    checkpoint_dir = Path(config.project_root) / 'checkpoints'
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Load samples (subset for testing)
    print(f"\n{'='*80}")
    print("LOADING DATA (SUBSET)")
    print(f"{'='*80}")
    
    samples = checkpoint_manager.load_checkpoint('samples_final')
    if samples is None:
        print(" No samples found. Run Phase 1 first.")
        return False
    
    # Use only first 1000 samples for quick test
    samples = samples[:1000]
    print(f" Using {len(samples)} samples for testing")
    
    # Load model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    
    # Initialize backbone WITHOUT LoRA (LoRA has gradient flow issues with VideoMAE)
    print(" Note: LoRA has gradient flow issues with VideoMAE")
    print("   Training full model instead (86M params)")
    
    backbone = VideoMAEBackbone(
        model_name=config.backbone_model,
        freeze=False  # Don't freeze - train full model
    )
    
    print(" Backbone initialized (full model trainable)")
    print(f"   This works fine with batch_size=2-4 + gradient accumulation")
    
    # Skip LoRA application (doesn't work with VideoMAE)
    print("\n Using full model (no LoRA)")
    print("   All 86,817,024 parameters trainable")
    print("   Memory efficient with small batch + gradient accumulation")
    
    backbone = backbone.to(device)
    
    # Check trainable parameters
    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in backbone.model.parameters())
    
    print(f"\n Final Model Statistics:")
    print(f"   Trainable parameters: {trainable_count:,}")
    print(f"   Total parameters: {total_count:,}")
    print(f"   Trainable ratio: {100*trainable_count/total_count:.4f}%")
    
    if trainable_count == 0:
        print(f"\n ERROR: No trainable parameters!")
        return False
    
    # Initialize processor
    processor = VideoMAEImageProcessor.from_pretrained(config.backbone_model)
    
    # Build dataloaders
    print(f"\n{'='*80}")
    print("BUILDING DATALOADERS")
    print(f"{'='*80}")
    
    try:
        train_loader, val_loader = build_temporal_dataloaders(
            samples=samples,
            config=config,
            image_processor=processor,
            train_ratio=0.8
        )
    except Exception as e:
        print(f" Failed to build dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data loading
    print(f"\n{'='*80}")
    print("TESTING DATA LOADING")
    print(f"{'='*80}")
    
    try:
        batch = next(iter(train_loader))
        print(f" Successfully loaded one batch")
        print(f"   Anchor shape: {batch['anchor'].shape}")
        print(f"   Positive shape: {batch['positive'].shape if batch['positive'] is not None else None}")
        print(f"   Negative shape: {batch['negative'].shape if batch['negative'] is not None else None}")
    except Exception as e:
        print(f" Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    print(f"\n{'='*80}")
    print("TESTING FORWARD PASS")
    print(f"{'='*80}")
    
    backbone.eval()
    
    try:
        with torch.no_grad():
            anchor = batch['anchor'].to(device)
            print(f"   Processing {anchor.shape[0]} samples...")
            
            # Process one at a time to save memory
            embeddings_list = []
            for i in range(anchor.shape[0]):
                single_video = anchor[i:i+1]
                emb = backbone.get_embeddings(single_video)
                embeddings_list.append(emb)
            
            embeddings = torch.cat(embeddings_list, dim=0)
            
            print(f" Forward pass successful")
            print(f"   Output shape: {embeddings.shape}")
            print(f"   Output dtype: {embeddings.dtype}")
            print(f"   Output device: {embeddings.device}")
            
            # Check if normalized
            norms = torch.norm(embeddings, p=2, dim=1)
            print(f"   Embedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OUT OF MEMORY ERROR")
            print(f"   Try reducing:")
            print(f"   - batch_size (currently {config.batch_size})")
            print(f"   - num_frames (currently {config.num_frames})")
            print(f"   - Or use gradient_accumulation_steps")
            return False
        else:
            print(f" Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f" Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print(f"\n{'='*80}")
    print("TESTING LOSS COMPUTATION")
    print(f"{'='*80}")
    
    try:
        criterion = TemporalContrastiveLoss(config).to(device)
        
        with torch.no_grad():
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device) if batch['positive'] is not None else None
            negative = batch['negative'].to(device) if batch['negative'] is not None else None
            
            # Get embeddings one at a time
            anchor_emb_list = []
            for i in range(anchor.shape[0]):
                emb = backbone.get_embeddings(anchor[i:i+1])
                anchor_emb_list.append(emb)
            anchor_embeddings = torch.cat(anchor_emb_list, dim=0)
            
            if positive is not None:
                pos_emb_list = []
                for i in range(min(4, positive.shape[0])):  # Only process first 4
                    emb = backbone.get_embeddings(positive[i:i+1])
                    pos_emb_list.append(emb)
                positive_embeddings = torch.cat(pos_emb_list, dim=0)
            else:
                positive_embeddings = torch.zeros_like(anchor_embeddings[:0])
            
            if negative is not None:
                neg_emb_list = []
                for i in range(min(8, negative.shape[0])):  # Only process first 8
                    emb = backbone.get_embeddings(negative[i:i+1])
                    neg_emb_list.append(emb)
                negative_embeddings = torch.cat(neg_emb_list, dim=0)
            else:
                negative_embeddings = torch.zeros_like(anchor_embeddings[:0])
            
            # Compute loss
            loss_input = {
                'anchor_embeddings': anchor_embeddings,
                'positive_embeddings': positive_embeddings,
                'negative_embeddings': negative_embeddings,
            }
            
            loss, metrics = criterion(loss_input)
            
            print(f" Loss computation successful")
            print(f"   Total loss: {loss.item():.4f}")
            print(f"   Contrastive loss: {metrics.get('contrastive_loss', 0):.4f}")
            print(f"   Pos similarity: {metrics.get('mean_pos_sim', 0):.4f}")
            print(f"   Neg similarity: {metrics.get('mean_neg_sim', 0):.4f}")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OUT OF MEMORY during loss computation")
            print(f"   Recommendations:")
            print(f"   1. Reduce batch_size to 1")
            print(f"   2. Process embeddings one at a time (already doing this)")
            print(f"   3. Reduce num_positive_pairs and num_negative_pairs")
            return False
        else:
            print(f" Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f" Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print(f"\n{'='*80}")
    print("TESTING BACKWARD PASS")
    print(f"{'='*80}")
    
    # CRITICAL: Set model to training mode BEFORE moving to device
    backbone.train()
    backbone.model.train()
    
    # Verify LoRA is in training mode
    print(f"   Backbone training mode: {backbone.training}")
    print(f"   Backbone.model training mode: {backbone.model.training}")
    
    # Move to device
    backbone = backbone.to(device)
    
    try:
        # Get one sample
        anchor = batch['anchor'][:1].to(device)  # Shape: (1, 16, 3, 224, 224)
        
        print(f"   Input shape: {anchor.shape}")
        print(f"   Input requires_grad: {anchor.requires_grad}")
        print(f"   Input device: {anchor.device}")
        
        # Create optimizer - all trainable parameters
        trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
        print(f"   Found {len(trainable_params)} trainable parameter tensors")
        
        if len(trainable_params) == 0:
            print(f" No trainable parameters found!")
            return False
        
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        
        # Forward pass - call model directly to ensure gradients
        print(f"\n   Computing forward pass...")
        outputs = backbone.model(pixel_values=anchor, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
        
        print(f"   Raw embeddings shape: {embeddings.shape}")
        print(f"   Raw embeddings require_grad: {embeddings.requires_grad}")
        
        if not embeddings.requires_grad:
            print(f"\n    ERROR: Embeddings don't have gradients after forward pass!")
            print(f"   Checking LoRA parameters...")
            
            # Debug: Check if any LoRA params actually require grad
            lora_params_with_grad = 0
            for name, param in backbone.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_params_with_grad += 1
                    print(f"      {name}: requires_grad={param.requires_grad}")
                    if lora_params_with_grad >= 3:  # Just show first 3
                        break
            
            if lora_params_with_grad == 0:
                print(f"    No LoRA parameters have requires_grad=True!")
                return False
            
            print(f"\n   This means the forward pass is not using LoRA parameters.")
            print(f"   Possible issue: Model needs to be unwrapped or in correct mode.")
            return False
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        print(f"   Normalized embeddings require_grad: {embeddings.requires_grad}")
        print(f"   Embeddings grad_fn: {embeddings.grad_fn}")
        
        # Simple loss
        loss = embeddings.mean()
        
        print(f"\n   Loss value: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        print(f"   Loss is_leaf: {loss.is_leaf}")
        
        # Backward
        optimizer.zero_grad()
        
        print(f"\n   Calling backward...")
        loss.backward()
        
        print(f"   Backward completed, checking gradients...")
        
        # Check if gradients were computed
        grad_count = sum(1 for p in trainable_params if p.grad is not None)
        print(f"   Parameters with gradients: {grad_count}/{len(trainable_params)}")
        
        if grad_count == 0:
            print(f"    No gradients computed!")
            return False
        
        # Check gradient magnitudes
        grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
        print(f"   Gradient norms - mean: {np.mean(grad_norms):.6f}, max: {max(grad_norms):.6f}")
        
        optimizer.step()
        
        print(f"\n Backward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Gradients computed and applied successfully")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OUT OF MEMORY during backward pass")
            print(f"   This is the most memory-intensive operation")
            print(f"   Try:")
            print(f"   1. batch_size = 1")
            print(f"   2. gradient_accumulation_steps = 4 or 8")
            print(f"   3. Clear CUDA cache between steps")
            return False
        else:
            print(f" Backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f" Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Memory report
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print("MEMORY REPORT")
        print(f"{'='*80}")
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Reserved: {memory_reserved:.2f} GB")
        print(f"   Total: {memory_total:.2f} GB")
        print(f"   Usage: {100 * memory_allocated / memory_total:.1f}%")
        
        if memory_allocated / memory_total > 0.9:
            print(f"\n WARNING: High memory usage!")
            print(f"   Recommended settings for full training:")
            print(f"   - batch_size = 1 or 2")
            print(f"   - gradient_accumulation_steps = 8")
            print(f"   - num_workers = 0")
        elif memory_allocated / memory_total > 0.7:
            print(f"\n Moderate memory usage")
            print(f"   Recommended settings:")
            print(f"   - batch_size = 2 or 4")
            print(f"   - gradient_accumulation_steps = 4")
        else:
            print(f"\n Good memory usage")
            print(f"   You can use:")
            print(f"   - batch_size = 4 or 8")
            print(f"   - gradient_accumulation_steps = 2")
    
    print(f"\n{'='*80}")
    print(" ALL TESTS PASSED!")
    print(f"{'='*80}")
    print("\nYou can now run the full training with:")
    print("python task2_phase2_train.py")
    print("\nRecommended config changes in TemporalEmotionConfig:")
    print("   batch_size: 2-4 (instead of 32)")
    print("   gradient_accumulation_steps: 4-8")
    print("   num_workers: 0-2 (instead of 4)")
    
    return True


if __name__ == "__main__":
    try:
        success = test_training_pipeline()
        if not success:
            print("\n Some tests failed. Fix the issues before full training.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n Test interrupted by user.")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)