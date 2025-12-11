#!/usr/bin/env python3
"""
Comprehensive integration test for the entire pipeline
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image

def test_model_architecture():
    """Test model creation and forward pass"""
    print("Testing model architecture...")
    
    from config import Config
    from model import TemporalEmotionModel
    
    config = Config()
    config.batch_size = 2
    config.num_frames = 4
    config.use_lora = True
    
    model = TemporalEmotionModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Model created")
    print(f"    Total params: {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    print(f"    Ratio: {100*trainable_params/total_params:.2f}%")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    batch_size = 2
    num_frames = 4
    dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    
    with torch.no_grad():
        z_t, g, va_pred = model(dummy_input, return_regression=True)
    
    assert z_t.shape == (batch_size, num_frames, config.projection_dim)
    assert g.shape == (batch_size, config.projection_dim)
    assert va_pred.shape == (batch_size, num_frames, 2)
    
    # Check normalization
    z_t_norm = torch.norm(z_t, p=2, dim=-1)
    g_norm = torch.norm(g, p=2, dim=-1)
    assert torch.allclose(z_t_norm, torch.ones_like(z_t_norm), atol=1e-5), "Frame embeddings not normalized"
    assert torch.allclose(g_norm, torch.ones_like(g_norm), atol=1e-5), "Clip embeddings not normalized"
    
    print(f"  ✓ Forward pass successful")
    print(f"    Frame embeddings: {z_t.shape}")
    print(f"    Clip embeddings: {g.shape}")
    print(f"    VA predictions: {va_pred.shape}")
    print(f"    Embeddings properly normalized")
    
    return True


def test_losses():
    """Test all loss functions"""
    print("\nTesting loss functions...")
    
    from losses import (
        ccc_loss, compute_ccc, InfoNCELoss,
        LocalLocalContrastiveLoss, GlobalLocalContrastiveLoss,
        SmoothnessLoss, CombinedPretrainLoss, RegressionLoss
    )
    from config import Config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CCC
    y_true = torch.randn(100).to(device)
    y_pred = y_true + torch.randn(100).to(device) * 0.1
    ccc_val = compute_ccc(y_true, y_pred)
    assert 0.5 < ccc_val < 1.0, f"CCC should be high for correlated data, got {ccc_val}"
    print(f"  ✓ CCC loss: {ccc_val:.4f}")
    
    # Test InfoNCE
    B, D = 4, 128
    anchors = F.normalize(torch.randn(B, D).to(device), p=2, dim=1)
    positives = F.normalize(torch.randn(B, D).to(device), p=2, dim=1)
    negatives = F.normalize(torch.randn(16, D).to(device), p=2, dim=1)
    
    infonce = InfoNCELoss(temperature=0.07)
    loss = infonce(anchors, positives, negatives)
    assert loss.item() > 0, "InfoNCE loss should be positive"
    print(f"  ✓ InfoNCE loss: {loss.item():.4f}")
    
    # Test LocalLocal (vectorized)
    B, L, D = 2, 8, 128
    z_t = F.normalize(torch.randn(B, L, D).to(device), p=2, dim=2)
    va_values = torch.randn(B, L, 2).to(device)
    
    ll_loss = LocalLocalContrastiveLoss(temperature=0.07)
    loss = ll_loss(z_t, va_values)
    assert loss.item() > 0, "LocalLocal loss should be positive"
    print(f"  ✓ LocalLocal loss (vectorized): {loss.item():.4f}")
    
    # Test GlobalLocal (vectorized)
    g = F.normalize(torch.randn(B, D).to(device), p=2, dim=1)
    
    gl_loss = GlobalLocalContrastiveLoss(temperature=0.07)
    loss = gl_loss(g, z_t)
    assert loss.item() > 0, "GlobalLocal loss should be positive"
    print(f"  ✓ GlobalLocal loss (vectorized): {loss.item():.4f}")
    
    # Test Smoothness
    smooth_loss = SmoothnessLoss()
    loss = smooth_loss(z_t)
    assert loss.item() >= 0, "Smoothness loss should be non-negative"
    print(f"  ✓ Smoothness loss: {loss.item():.4f}")
    
    # Test Combined Pretrain Loss
    config = Config()
    combined_loss = CombinedPretrainLoss(
        lambda_ll=config.lambda_ll,
        lambda_gl=config.lambda_gl,
        lambda_smooth=config.lambda_smooth,
        temperature=config.temperature
    )
    
    loss, loss_dict = combined_loss(z_t, g, va_values, config=config)
    assert loss.item() > 0, "Combined loss should be positive"
    assert 'loss_ll' in loss_dict
    assert 'loss_gl' in loss_dict
    assert 'loss_smooth' in loss_dict
    print(f"  ✓ Combined pretrain loss: {loss.item():.4f}")
    print(f"    Components: LL={loss_dict['loss_ll']:.4f}, GL={loss_dict['loss_gl']:.4f}, Smooth={loss_dict['loss_smooth']:.4f}")
    
    # Test Regression Loss
    pred_va = torch.randn(B, L, 2).to(device)
    true_va = torch.randn(B, L, 2).to(device)
    
    reg_loss = RegressionLoss()
    loss, metrics = reg_loss(pred_va, true_va)
    assert loss.item() > 0, "Regression loss should be positive"
    assert 'mse' in metrics
    assert 'ccc_valence' in metrics
    assert 'ccc_arousal' in metrics
    print(f"  ✓ Regression loss: {loss.item():.4f}")
    print(f"    MSE={metrics['mse']:.4f}, CCC_v={metrics['ccc_valence']:.4f}, CCC_a={metrics['ccc_arousal']:.4f}")
    
    return True


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    from dataset import AFEWVADataset, get_clip_transform
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy clip
        clip_dir = Path(tmpdir) / '001'
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        # Create frames and annotations
        num_frames = 20
        annotations = []
        for i in range(num_frames):
            # Create dummy image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(clip_dir / f'{i:05d}.png')
            
            # Create annotation
            annotations.append({
                'frame': i,
                'valence': float(np.random.uniform(-1, 1)),
                'arousal': float(np.random.uniform(-1, 1))
            })
        
        # Save annotations
        with open(clip_dir / '001.json', 'w') as f:
            json.dump(annotations, f)
        
        # Create dataset
        transform = get_clip_transform(is_train=False, augment=False)
        dataset = AFEWVADataset(
            root_dir=tmpdir,
            num_frames=16,
            split='test',
            transform=transform
        )
        
        assert len(dataset) == 1, f"Expected 1 clip, got {len(dataset)}"
        print(f"  ✓ Dataset created with {len(dataset)} clip")
        
        # Test loading
        sample = dataset[0]
        assert 'images' in sample
        assert 'valences' in sample
        assert 'arousals' in sample
        assert 'clip_id' in sample
        
        assert sample['images'].shape == (16, 3, 224, 224)
        assert sample['valences'].shape == (16,)
        assert sample['arousals'].shape == (16,)
        
        print(f"  ✓ Sample loaded successfully")
        print(f"    Images: {sample['images'].shape}")
        print(f"    Valences: {sample['valences'].shape}")
        print(f"    Arousals: {sample['arousals'].shape}")
    
    return True


def test_memory_queue():
    """Test memory queue updates"""
    print("\nTesting memory queue...")
    
    from config import Config
    from model import TemporalEmotionModel
    
    config = Config()
    config.memory_queue_size = 100
    config.batch_size = 2
    config.num_frames = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalEmotionModel(config).to(device)
    
    # Initial queue
    initial_queue = model.memory_queue.clone()
    
    # Create embeddings and update queue
    embeddings = F.normalize(torch.randn(8, config.projection_dim).to(device), p=2, dim=1)
    model.update_memory_queue(embeddings)
    
    # Check queue was updated
    assert not torch.allclose(model.memory_queue, initial_queue), "Memory queue should be updated"
    assert int(model.queue_ptr) == 8, f"Queue pointer should be 8, got {int(model.queue_ptr)}"
    
    print(f"  ✓ Memory queue updated")
    print(f"    Queue size: {config.memory_queue_size}")
    print(f"    Current pointer: {int(model.queue_ptr)}")
    
    return True


def test_trainable_parameters():
    """Test trainable parameters selection"""
    print("\nTesting trainable parameters...")
    
    from config import Config
    from model import TemporalEmotionModel
    
    config = Config()
    config.use_lora = True
    
    model = TemporalEmotionModel(config)
    
    # Test pretrain stage
    pretrain_params = model.get_trainable_parameters(stage='pretrain')
    pretrain_count = sum(p.numel() for group in pretrain_params for p in group['params'])
    
    # Test finetune stage
    finetune_params = model.get_trainable_parameters(stage='finetune')
    finetune_count = sum(p.numel() for group in finetune_params for p in group['params'])
    
    print(f"  ✓ Pretrain parameters: {pretrain_count:,}")
    print(f"  ✓ Finetune parameters: {finetune_count:,}")
    
    # Test without LoRA
    config.use_lora = False
    model_no_lora = TemporalEmotionModel(config)
    
    no_lora_params = model_no_lora.get_trainable_parameters(stage='pretrain')
    no_lora_count = sum(p.numel() for group in no_lora_params for p in group['params'])
    
    print(f"  ✓ Without LoRA: {no_lora_count:,}")
    
    assert pretrain_count > no_lora_count, "LoRA should add trainable parameters"
    
    return True


def test_gradient_flow():
    """Test that gradients flow correctly"""
    print("\nTesting gradient flow...")
    
    from config import Config
    from model import TemporalEmotionModel
    from losses import CombinedPretrainLoss
    
    config = Config()
    config.batch_size = 2
    config.num_frames = 4
    config.use_lora = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalEmotionModel(config).to(device)
    
    # Get trainable parameters
    params = model.get_trainable_parameters(stage='pretrain')
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    # Create dummy batch
    images = torch.randn(2, 4, 3, 224, 224).to(device)
    va_values = torch.randn(2, 4, 2).to(device)
    
    # Forward pass
    z_t, g, _ = model(images, return_regression=False)
    
    # Compute loss
    criterion = CombinedPretrainLoss(temperature=config.temperature)
    loss, _ = criterion(z_t, g, va_values, config=config)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for param_group in params:
        for p in param_group['params']:
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
    
    assert has_grad, "No gradients found in trainable parameters"
    print(f"  ✓ Gradients computed successfully")
    print(f"    Loss: {loss.item():.4f}")
    
    # Test optimization step
    optimizer.step()
    print(f"  ✓ Optimization step successful")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("="*80)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Loss Functions", test_losses),
        ("Dataset Loading", test_dataset),
        ("Memory Queue", test_memory_queue),
        ("Trainable Parameters", test_trainable_parameters),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed}/{len(tests)} passed")
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {failed} tests failed")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)