#!/usr/bin/env python3
"""
Quick test script to verify installation and setup
"""
import sys
import torch

def check_installation():
    """Check if all required packages are installed"""
    print("="*60)
    print("CHECKING INSTALLATION")
    print("="*60 + "\n")
    
    errors = []
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        errors.append("Python 3.9+ required")
    else:
        print("  ✓ Python version OK")
    
    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA not available (CPU mode)")
    
    # Check other packages
    packages_to_check = [
        'open_clip',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'tqdm',
        'PIL',
        'einops',
        'timm'
    ]
    
    print("\nChecking required packages:")
    for package in packages_to_check:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'open_clip':
                __import__('open_clip')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            errors.append(f"{package} not installed")
    
    # Test model creation
    print("\nTesting model creation:")
    try:
        from config import Config
        from model import TemporalEmotionModel
        
        config = Config()
        config.batch_size = 2
        config.num_frames = 4
        
        model = TemporalEmotionModel(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Model created successfully")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Trainable ratio: {100*trainable_params/total_params:.2f}%")
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        errors.append(f"Model creation failed: {e}")
    
    # Test forward pass
    print("\nTesting forward pass:")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy input
        batch_size = 2
        num_frames = 4
        dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
        
        with torch.no_grad():
            z_t, g, va_pred = model(dummy_input, return_regression=True)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Frame embeddings: {z_t.shape}")
        print(f"    Clip embedding: {g.shape}")
        print(f"    VA predictions: {va_pred.shape}")
        
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        errors.append(f"Forward pass failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("SETUP INCOMPLETE - ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    else:
        print("✓ ALL CHECKS PASSED - READY TO TRAIN!")
    print("="*60 + "\n")
    
    return len(errors) == 0


def test_dataset_loading():
    """Test dataset loading with dummy data"""
    print("="*60)
    print("TESTING DATASET LOADING")
    print("="*60 + "\n")
    
    try:
        from dataset import AFEWVADataset, get_clip_transform
        import tempfile
        import os
        import json
        from PIL import Image
        import numpy as np
        
        # Create temporary dataset structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy clip
            clip_dir = os.path.join(tmpdir, '001')
            os.makedirs(clip_dir, exist_ok=True)
            
            # Create dummy frames
            num_frames = 20
            annotations = []
            for i in range(num_frames):
                # Create dummy image
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(clip_dir, f'{i:05d}.png'))
                
                # Create annotation
                annotations.append({
                    'frame': i,
                    'valence': np.random.uniform(-1, 1),
                    'arousal': np.random.uniform(-1, 1)
                })
            
            # Save annotations
            with open(os.path.join(clip_dir, '001.json'), 'w') as f:
                json.dump(annotations, f)
            
            # Create dataset
            transform = get_clip_transform(is_train=False, augment=False)
            dataset = AFEWVADataset(
                root_dir=tmpdir,
                num_frames=16,
                split='test',
                transform=transform
            )
            
            print(f"  ✓ Dataset created with {len(dataset)} clips")
            
            # Test loading
            sample = dataset[0]
            print(f"  ✓ Sample loaded successfully")
            print(f"    Images shape: {sample['images'].shape}")
            print(f"    Valences shape: {sample['valences'].shape}")
            print(f"    Arousals shape: {sample['arousals'].shape}")
            print(f"    Clip ID: {sample['clip_id']}")
            
        print("\n✓ Dataset loading test passed!")
        
    except Exception as e:
        print(f"\n✗ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    success = check_installation()
    
    if success:
        print("\nRunning dataset loading test...")
        test_dataset_loading()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Prepare your AFEW-VA dataset in the correct format")
        print("2. Run the training pipeline:")
        print("     python main.py --dataset AFEW-VA --stage all")
        print("\n3. Or run individual stages:")
        print("     python main.py --stage pretrain")
        print("     python main.py --stage finetune")
        print("     python main.py --stage evaluate")
        print("\n4. Monitor training:")
        print("     - Checkpoints: ./checkpoints/")
        print("     - Outputs: ./outputs/")
        print("     - Visualizations: ./outputs/evaluation/")
        print("="*60 + "\n")
    else:
        print("\nPlease fix the errors above before proceeding.")
        sys.exit(1)