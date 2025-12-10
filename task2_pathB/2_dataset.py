"""
Feature-based dataset for temporal emotion contrastive learning
Loads pre-extracted CLIP features (fast!) instead of raw images
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import get_config


# ============================================================================
# TEMPORAL WINDOW CREATION
# ============================================================================

def create_temporal_windows(clip_metadata: Dict, 
                           num_frames: int = 8, 
                           stride: int = 2) -> List[Dict]:
    """
    Create sliding temporal windows from a clip
    
    Args:
        clip_metadata: Clip metadata from feature index
        num_frames: Number of frames per window
        stride: Stride for sliding window
    
    Returns:
        List of window dictionaries
    """
    frames = clip_metadata['frames']
    clip_name = clip_metadata['clip_name']
    
    if len(frames) < num_frames:
        # Clip too short - return single window with padding info
        return [{
            'clip_name': clip_name,
            'frame_indices': list(range(len(frames))),
            'num_frames': len(frames),
            'needs_padding': True,
            'valences': [f['valence'] for f in frames],
            'arousals': [f['arousal'] for f in frames],
            'temporal_indices': [f['temporal_idx'] for f in frames],
        }]
    
    windows = []
    for start_idx in range(0, len(frames) - num_frames + 1, stride):
        end_idx = start_idx + num_frames
        window_frames = frames[start_idx:end_idx]
        
        window = {
            'clip_name': clip_name,
            'frame_indices': list(range(start_idx, end_idx)),
            'num_frames': num_frames,
            'needs_padding': False,
            'valences': [f['valence'] for f in window_frames],
            'arousals': [f['arousal'] for f in window_frames],
            'temporal_indices': [f['temporal_idx'] for f in window_frames],
        }
        windows.append(window)
    
    return windows


# ============================================================================
# FEATURE-BASED TEMPORAL DATASET
# ============================================================================

class TemporalFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted CLIP features for temporal learning
    
    Much faster than loading images!
    """
    
    def __init__(self, 
                 features_cache_dir: str,
                 config,
                 split: str = 'train',
                 train_ratio: float = 0.85):
        """
        Args:
            features_cache_dir: Directory with extracted features
            config: Configuration object
            split: 'train' or 'val'
            train_ratio: Ratio of clips for training
        """
        self.features_cache_dir = Path(features_cache_dir)
        self.config = config
        self.split = split
        
        print(f"\n{'='*80}")
        print(f"BUILDING {split.upper()} DATASET")
        print(f"{'='*80}")
        
        # Load feature index
        self._load_feature_index()
        
        # Split into train/val
        self._create_split(train_ratio)
        
        # Create temporal windows
        self._create_temporal_windows()
        
        # Group windows by clip for positive sampling
        self._group_windows_by_clip()
        
        # Statistics
        self._print_statistics()
    
    def _load_feature_index(self):
        """Load feature index file"""
        index_path = self.features_cache_dir / 'index.json'
        
        if not index_path.exists():
            raise FileNotFoundError(
                f"Feature index not found: {index_path}\n"
                f"Run feature extraction first: python 1_extract_features.py"
            )
        
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        self.all_clips = self.index['clips']
        print(f"Loaded feature index: {len(self.all_clips)} clips, "
              f"{self.index['total_frames']} frames")
    
    def _create_split(self, train_ratio: float):
        """Split clips into train/val"""
        # Shuffle clips deterministically
        random.seed(self.config.random_seed)
        clips = self.all_clips.copy()
        random.shuffle(clips)
        
        # Split
        split_idx = int(len(clips) * train_ratio)
        train_clips = clips[:split_idx]
        val_clips = clips[split_idx:]
        
        self.clips = train_clips if self.split == 'train' else val_clips
        
        print(f"Split: {len(train_clips)} train, {len(val_clips)} val")
        print(f"Using {len(self.clips)} clips for {self.split}")
    
    def _create_temporal_windows(self):
        """Create temporal windows from clips"""
        print(f"Creating temporal windows (frames={self.config.num_frames}, "
              f"stride={self.config.temporal_stride})...")
        
        self.windows = []
        
        for clip in self.clips:
            windows = create_temporal_windows(
                clip, 
                num_frames=self.config.num_frames,
                stride=self.config.temporal_stride
            )
            self.windows.extend(windows)
        
        print(f"Created {len(self.windows)} temporal windows")
    
    def _group_windows_by_clip(self):
        """Group windows by clip for positive sampling"""
        self.clip_to_windows = defaultdict(list)
        
        for idx, window in enumerate(self.windows):
            self.clip_to_windows[window['clip_name']].append(idx)
    
    def _print_statistics(self):
        """Print dataset statistics"""
        all_valences = []
        all_arousals = []
        
        for window in self.windows:
            all_valences.extend(window['valences'])
            all_arousals.extend(window['arousals'])
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total windows: {len(self.windows)}")
        print(f"   Clips: {len(self.clips)}")
        print(f"   Valence: mean={np.mean(all_valences):.2f}, std={np.std(all_valences):.2f}")
        print(f"   Arousal: mean={np.mean(all_arousals):.2f}, std={np.std(all_arousals):.2f}")
    
    def __len__(self):
        return len(self.windows)
    
    def _load_window_features(self, window: Dict) -> torch.Tensor:
        """
        Load features for a temporal window
        
        Args:
            window: Window metadata
        
        Returns:
            Tensor of shape (num_frames, feature_dim)
        """
        clip_name = window['clip_name']
        clip_dir = self.features_cache_dir / clip_name
        
        # Load full clip features
        features = np.load(clip_dir / 'features.npy')
        
        # Extract window frames
        frame_indices = window['frame_indices']
        window_features = features[frame_indices]
        
        # Handle padding if needed
        if window['needs_padding']:
            # Pad by repeating last frame
            num_pad = self.config.num_frames - window['num_frames']
            if num_pad > 0:
                last_frame = window_features[-1:].repeat(num_pad, axis=0)
                window_features = np.concatenate([window_features, last_frame], axis=0)
        
        return torch.from_numpy(window_features).float()
    
    def _sample_positive(self, anchor_idx: int) -> int:
        """
        Sample a positive window (same clip, temporally close)
        
        Args:
            anchor_idx: Index of anchor window
        
        Returns:
            Index of positive window
        """
        anchor_window = self.windows[anchor_idx]
        clip_name = anchor_window['clip_name']
        
        # Get all windows from same clip
        clip_windows = self.clip_to_windows[clip_name]
        
        if len(clip_windows) == 1:
            # Only one window in clip, return same
            return anchor_idx
        
        # Find temporally close windows
        anchor_temporal_idx = anchor_window['temporal_indices'][0]
        candidates = []
        
        for idx in clip_windows:
            if idx == anchor_idx:
                continue
            
            window = self.windows[idx]
            window_temporal_idx = window['temporal_indices'][0]
            
            # Check temporal distance
            temporal_distance = abs(window_temporal_idx - anchor_temporal_idx)
            
            if temporal_distance <= self.config.temporal_positive_window:
                candidates.append(idx)
        
        if not candidates:
            # No close windows, sample any from same clip
            candidates = [idx for idx in clip_windows if idx != anchor_idx]
        
        return random.choice(candidates) if candidates else anchor_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            Dictionary with:
            - anchor: (num_frames, feature_dim)
            - positives: list of (num_frames, feature_dim)
            - metadata: valence, arousal, etc.
        """
        # Load anchor
        anchor_window = self.windows[idx]
        anchor_features = self._load_window_features(anchor_window)
        
        # Sample positives
        positives = []
        for _ in range(self.config.num_positives_per_anchor):
            pos_idx = self._sample_positive(idx)
            pos_window = self.windows[pos_idx]
            pos_features = self._load_window_features(pos_window)
            positives.append(pos_features)
        
        # Metadata
        metadata = {
            'clip_name': anchor_window['clip_name'],
            'valence': np.mean(anchor_window['valences']),
            'arousal': np.mean(anchor_window['arousals']),
            'num_frames': anchor_window['num_frames'],
        }
        
        return {
            'anchor': anchor_features,
            'positives': positives,
            'metadata': metadata
        }


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Handles:
    - In-batch negatives (all other anchors are negatives)
    - Variable number of positives
    """
    # Stack anchors
    anchors = torch.stack([item['anchor'] for item in batch])  # (batch, frames, feat_dim)
    
    # Collect all positives
    all_positives = []
    for item in batch:
        all_positives.extend(item['positives'])
    positives = torch.stack(all_positives) if all_positives else None  # (num_pos_total, frames, feat_dim)
    
    # Metadata
    metadata = {
        'clip_names': [item['metadata']['clip_name'] for item in batch],
        'valences': torch.tensor([item['metadata']['valence'] for item in batch]),
        'arousals': torch.tensor([item['metadata']['arousal'] for item in batch]),
    }
    
    return {
        'anchor': anchors,
        'positives': positives,
        'metadata': metadata
    }


# ============================================================================
# DATALOADER BUILDERS
# ============================================================================

def build_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders
    
    Args:
        config: Configuration object
    
    Returns:
        (train_loader, val_loader)
    """
    print("\n" + "="*80)
    print("BUILDING DATALOADERS")
    print("="*80)
    
    # Create datasets
    train_dataset = TemporalFeatureDataset(
        features_cache_dir=config.features_cache_dir,
        config=config,
        split='train',
        train_ratio=1 - config.val_split
    )
    
    val_dataset = TemporalFeatureDataset(
        features_cache_dir=config.features_cache_dir,
        config=config,
        split='val',
        train_ratio=1 - config.val_split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True  # For stable batch size in contrastive learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Effective batch size (with grad accum): {config.batch_size * config.gradient_accumulation_steps}")
    
    return train_loader, val_loader


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING DATASET")
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Build dataloaders
    train_loader, val_loader = build_dataloaders(config)
    
    # Test one batch
    print("\n" + "="*80)
    print("TESTING BATCH LOADING")
    print("="*80)
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Anchor shape: {batch['anchor'].shape}")
        print(f"  Positives shape: {batch['positives'].shape if batch['positives'] is not None else None}")
        print(f"  Valence range: [{batch['metadata']['valences'].min():.2f}, {batch['metadata']['valences'].max():.2f}]")
        print(f"  Arousal range: [{batch['metadata']['arousals'].min():.2f}, {batch['metadata']['arousals'].max():.2f}]")
        
        # In-batch negatives info
        batch_size = batch['anchor'].shape[0]
        num_positives = batch['positives'].shape[0] if batch['positives'] is not None else 0
        num_negatives = batch_size - 1  # All other samples in batch
        
        print(f"  In-batch negatives: {num_negatives} per anchor")
        print(f"  Total positives: {num_positives}")
        
        if batch_idx >= 2:
            break
    
    print("\n" + "="*80)
    print("✅ DATASET TEST COMPLETE!")
    print("="*80)