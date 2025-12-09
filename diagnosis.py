"""
diagnose_lora.py
Quick diagnostic to check if LoRA was applied correctly in Phase 1
"""

import torch
from pathlib import Path
from task2_temporal_analysis_backbone import TemporalEmotionConfig, VideoMAEBackbone
from peft import LoraConfig, get_peft_model

print("="*80)
print("DIAGNOSING LORA APPLICATION")
print("="*80)

config = TemporalEmotionConfig()

# Check saved model
lora_path = Path(config.project_root) / 'lora_adapters' / 'backbone_with_lora.pt'

if not lora_path.exists():
    print(f"❌ No saved model found at {lora_path}")
    print("You need to run Phase 1 first!")
    exit(1)

print(f"✅ Found saved model at {lora_path}")

# Load and inspect
state_dict = torch.load(lora_path, map_location='cpu')

print(f"\n📊 Saved State Dict Analysis:")
print(f"   Total keys: {len(state_dict.keys())}")

# Count LoRA-specific keys
lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
print(f"   LoRA keys: {len(lora_keys)}")

if len(lora_keys) == 0:
    print("\n❌ WARNING: No LoRA keys found in saved model!")
    print("This means LoRA was not applied during Phase 1")
    print("\nSolution: Re-run Phase 1 with correct LoRA application")
else:
    print(f"\n✅ LoRA keys found! Sample keys:")
    for key in list(lora_keys)[:5]:
        print(f"      {key}")

# Check what target modules are in config
print(f"\n⚙️ Config Settings:")
print(f"   Target modules: {config.lora_target_modules}")
print(f"   LoRA rank: {config.lora_rank}")
print(f"   LoRA alpha: {config.lora_alpha}")

# Try to load model and see what happens
print(f"\n{'='*80}")
print("TESTING MODEL LOADING")
print(f"{'='*80}")

# Method 1: Load without reapplying LoRA (wrong way)
print("\n1. Loading WITHOUT reapplying LoRA (wrong way):")
backbone1 = VideoMAEBackbone(config.backbone_model, freeze=True)
try:
    backbone1.load_state_dict(state_dict, strict=False)
    trainable1 = sum(p.numel() for p in backbone1.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable1:,}")
    if trainable1 < 100000:
        print(f"   ❌ Too few trainable parameters!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Method 2: Reapply LoRA then load (right way)
print("\n2. REAPPLYING LoRA then loading (right way):")
backbone2 = VideoMAEBackbone(config.backbone_model, freeze=True)

# Try with config's target modules
print(f"   Trying target modules: {config.lora_target_modules}")
try:
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        inference_mode=False,
    )
    backbone2.model = get_peft_model(backbone2.model, lora_config)
    backbone2.model.print_trainable_parameters()
    
    # Now load weights
    missing, unexpected = backbone2.load_state_dict(state_dict, strict=False)
    trainable2 = sum(p.numel() for p in backbone2.parameters() if p.requires_grad)
    print(f"\n   ✅ Trainable parameters: {trainable2:,}")
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    
    if trainable2 > 500000:
        print(f"\n   ✅ SUCCESS! LoRA properly applied with {trainable2:,} parameters")
    else:
        print(f"\n   ⚠️ Only {trainable2:,} trainable parameters (expected ~590,000)")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")
    print("\n   Trying alternative target modules...")
    
    # Try simpler target modules
    for target_attempt in [
        ["query", "value"],
        ["attention.query", "attention.value"],
        ["encoder.layer.*.attention.attention.query", "encoder.layer.*.attention.attention.value"],
    ]:
        print(f"\n   Trying: {target_attempt}")
        try:
            backbone3 = VideoMAEBackbone(config.backbone_model, freeze=True)
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=target_attempt,
                lora_dropout=config.lora_dropout,
                bias="none",
                inference_mode=False,
            )
            backbone3.model = get_peft_model(backbone3.model, lora_config)
            backbone3.model.print_trainable_parameters()
            
            trainable3 = sum(p.numel() for p in backbone3.parameters() if p.requires_grad)
            print(f"      Trainable: {trainable3:,}")
            
            if trainable3 > 500000:
                print(f"      ✅ This works! Use these target modules: {target_attempt}")
                break
        except Exception as e2:
            print(f"      ❌ Failed: {e2}")

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*80}")

print("\n💡 RECOMMENDATIONS:")
print("1. If LoRA keys found in saved model BUT trainable params < 500k:")
print("   → Target modules mismatch. Update config and re-run Phase 1")
print("\n2. If NO LoRA keys in saved model:")
print("   → LoRA was never applied. Re-run Phase 1")
print("\n3. If trainable params > 500k:")
print("   → Everything is OK! Proceed with training")