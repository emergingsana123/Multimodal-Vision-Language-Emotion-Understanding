"""
Phase 1: Extract CLIP features from AFEW-VA dataset
Runs once, saves features to disk for fast training
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import get_config
from utils import (
    setup_logging, 
    save_clip_features, 
    create_feature_index,
    print_gpu_memory,
    clear_gpu_cache,
    set_seed
)


# ============================================================================
# AFEW-VA DATA LOADER
# ============================================================================

def load_afew_clip_metadata(clip_dir: Path) -> Optional[Dict]:
    """
    Load metadata for a single AFEW-VA clip
    
    Args:
        clip_dir: Path to clip directory
    
    Returns:
        Dictionary with clip metadata or None if invalid
    """
    clip_name = clip_dir.name
    annotation_file = clip_dir / f"{clip_name}.json"
    
    if not annotation_file.exists():
        return None
    
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    except:
        return None
    
    frames_data = annotations.get("frames", {})
    if not frames_data:
        return None
    
    # Collect frame information
    frames = []
    for frame_id, frame_data in frames_data.items():
        # Find image file
        image_path = clip_dir / f"{frame_id}.png"
        if not image_path.exists():
            image_path = clip_dir / f"{frame_id}.jpg"
        
        if not image_path.exists():
            continue
        
        try:
            frame_num = int(frame_id)
        except:
            continue
        
        frames.append({
            'frame_id': frame_id,
            'frame_num': frame_num,
            'image_path': str(image_path),
            'valence': frame_data.get('valence', 0.0),
            'arousal': frame_data.get('arousal', 0.0),
        })
    
    if not frames:
        return None
    
    # Sort by frame number
    frames.sort(key=lambda x: x['frame_num'])
    
    # Add temporal index
    for idx, frame in enumerate(frames):
        frame['temporal_idx'] = idx
    
    return {
        'clip_name': clip_name,
        'clip_dir': str(clip_dir),
        'num_frames': len(frames),
        'frames': frames
    }


def load_afew_dataset(dataset_dir: Path, config) -> List[Dict]:
    """
    Load all AFEW-VA clips
    
    Args:
        dataset_dir: Root directory of AFEW-VA dataset
        config: Configuration object
    
    Returns:
        List of clip metadata dictionaries
    """
    print("=" * 80)
    print("LOADING AFEW-VA DATASET")
    print("=" * 80)
    
    clip_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(clip_dirs)} clip directories")
    
    clips = []
    for clip_dir in tqdm(clip_dirs, desc="Loading clips"):
        clip_metadata = load_afew_clip_metadata(clip_dir)
        if clip_metadata:
            # Skip very long clips
            if clip_metadata['num_frames'] <= config.max_frames_per_clip:
                clips.append(clip_metadata)
    
    print(f"\n✅ Loaded {len(clips)} valid clips")
    print(f"   Total frames: {sum(c['num_frames'] for c in clips):,}")
    print(f"   Avg frames per clip: {np.mean([c['num_frames'] for c in clips]):.1f}")
    
    return clips


# ============================================================================
# CLIP FEATURE EXTRACTOR
# ============================================================================

class CLIPFeatureExtractor:
    """Extract features using frozen CLIP model"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        print("\n" + "=" * 80)
        print("INITIALIZING CLIP MODEL")
        print("=" * 80)
        
        # Load CLIP model and processor
        print(f"Loading {config.clip_model_name}...")
        self.model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        
        # Move to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✅ CLIP model loaded on {device}")
        print(f"   Feature dimension: {config.clip_feature_dim}")
        print_gpu_memory("   ")
    
    @torch.no_grad()
    def extract_frame_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features for a batch of frames
        
        Args:
            image_paths: List of paths to images
        
        Returns:
            Numpy array of shape (batch_size, feature_dim)
        """
        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                # Fallback: create black image
                print(f"Warning: Could not load {img_path}: {e}")
                images.append(Image.new('RGB', (224, 224), color='black'))
        
        # Process with CLIP processor
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features using vision encoder
        outputs = self.model.get_image_features(**inputs)
        
        # Normalize features
        features = F.normalize(outputs, p=2, dim=-1)
        
        return features.cpu().numpy()
    
    def extract_clip_features(self, clip_metadata: Dict) -> tuple:
        """
        Extract features for all frames in a clip
        
        Args:
            clip_metadata: Clip metadata dictionary
        
        Returns:
            (features, metadata) tuple
        """
        frames = clip_metadata['frames']
        num_frames = len(frames)
        
        # Extract features in batches
        all_features = []
        
        for i in range(0, num_frames, self.config.extraction_batch_size):
            batch_frames = frames[i:i + self.config.extraction_batch_size]
            batch_paths = [f['image_path'] for f in batch_frames]
            
            batch_features = self.extract_frame_features(batch_paths)
            all_features.append(batch_features)
        
        # Concatenate all batches
        features = np.concatenate(all_features, axis=0)
        
        # Prepare metadata
        metadata = {
            'clip_name': clip_metadata['clip_name'],
            'num_frames': num_frames,
            'feature_dim': features.shape[1],
            'frames': [
                {
                    'frame_id': f['frame_id'],
                    'frame_num': f['frame_num'],
                    'temporal_idx': f['temporal_idx'],
                    'valence': f['valence'],
                    'arousal': f['arousal'],
                }
                for f in frames
            ]
        }
        
        return features, metadata


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_all_features(config, logger):
    """
    Main feature extraction pipeline
    
    Args:
        config: Configuration object
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("STARTING FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config.random_seed)
    
    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    dataset_dir = Path(config.dataset_dir)
    clips = load_afew_dataset(dataset_dir, config)
    
    # Test mode: only process subset
    if config.test_mode:
        clips = clips[:config.test_num_clips]
        logger.info(f"\n⚠️  TEST MODE: Processing only {len(clips)} clips")
    
    # Initialize CLIP extractor
    extractor = CLIPFeatureExtractor(config, device)
    
    # Feature cache directory
    cache_dir = Path(config.features_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features for each clip
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTING FEATURES")
    logger.info("=" * 80)
    
    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    
    for clip_metadata in tqdm(clips, desc="Processing clips"):
        clip_name = clip_metadata['clip_name']
        clip_cache_dir = cache_dir / clip_name
        
        # Skip if already extracted (unless not resuming)
        if config.extraction_resum