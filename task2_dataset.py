"""
task2_phase2_dataset.py
Phase 2: Temporal Emotion Contrastive Dataset
Creates temporal windows and samples positive/negative pairs for contrastive learning
"""

import os
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import cv2
from tqdm.auto import tqdm


# ============================================================================
# TEMPORAL WINDOW EXTRACTOR
# ============================================================================

class TemporalWindowExtractor:
    """Extract temporal windows from video clips"""
    
    def __init__(self, num_frames: int = 16, stride: int = 1):
        self.num_frames = num_frames
        self.stride = stride
    
    def extract_windows(self, clip_samples: List[Dict]) -> List[Dict]:
        """
        Extract all possible temporal windows from a clip
        
        Args:
            clip_samples: List of frames from same clip, sorted by temporal_idx
        
        Returns:
            List of windows, each containing metadata
        """
        if len(clip_samples) < self.num_frames:
            # Clip too short, return single window with padding
            return [self._create_padded_window(clip_samples)]
        
        windows = []
        for start_idx in range(0, len(clip_samples) - self.num_frames + 1, self.stride):
            end_idx = start_idx + self.num_frames
            window_samples = clip_samples[start_idx:end_idx]
            
            window = {
                'clip_id': clip_samples[0]['clip_id'],
                'clip_name': clip_samples[0]['clip_name'],
                'start_frame': window_samples[0]['frame_num'],
                'end_frame': window_samples[-1]['frame_num'],
                'start_temporal_idx': window_samples[0]['temporal_idx'],
                'end_temporal_idx': window_samples[-1]['temporal_idx'],
                'samples': window_samples,
                'num_frames': len(window_samples),
                'mean_valence': np.mean([s['valence'] for s in window_samples]),
                'mean_arousal': np.mean([s['arousal'] for s in window_samples]),
                'std_valence': np.std([s['valence'] for s in window_samples]),
                'std_arousal': np.std([s['arousal'] for s in window_samples]),
            }
            windows.append(window)
        
        return windows
    
    def _create_padded_window(self, clip_samples: List[Dict]) -> Dict:
        """Create a window from short clip with padding info"""
        return {
            'clip_id': clip_samples[0]['clip_id'],
            'clip_name': clip_samples[0]['clip_name'],
            'start_frame': clip_samples[0]['frame_num'],
            'end_frame': clip_samples[-1]['frame_num'],
            'start_temporal_idx': clip_samples[0]['temporal_idx'],
            'end_temporal_idx': clip_samples[-1]['temporal_idx'],
            'samples': clip_samples,
            'num_frames': len(clip_samples),
            'mean_valence': np.mean([s['valence'] for s in clip_samples]),
            'mean_arousal': np.mean([s['arousal'] for s in clip_samples]),
            'std_valence': np.std([s['valence'] for s in clip_samples]),
            'std_arousal': np.std([s['arousal'] for s in clip_samples]),
            'is_padded': True,
        }


# ============================================================================
# EMOTION SIMILARITY CALCULATOR
# ============================================================================

class EmotionSimilarityCalculator:
    """Calculate emotion similarity between frames/windows"""
    
    def __init__(self, 
                 valence_threshold_similar: float = 1.5,
                 arousal_threshold_similar: float = 1.5,
                 valence_threshold_different: float = 3.0,
                 arousal_threshold_different: float = 3.0):
        self.val_sim = valence_threshold_similar
        self.ar_sim = arousal_threshold_similar
        self.val_diff = valence_threshold_different
        self.ar_diff = arousal_threshold_different
    
    def are_similar(self, sample1: Dict, sample2: Dict) -> bool:
        """Check if two samples have similar emotions"""
        val_dist = abs(sample1['mean_valence'] - sample2['mean_valence'])
        ar_dist = abs(sample1['mean_arousal'] - sample2['mean_arousal'])
        return val_dist < self.val_sim and ar_dist < self.ar_sim
    
    def are_different(self, sample1: Dict, sample2: Dict) -> bool:
        """Check if two samples have different emotions"""
        val_dist = abs(sample1['mean_valence'] - sample2['mean_valence'])
        ar_dist = abs(sample1['mean_arousal'] - sample2['mean_arousal'])
        return val_dist > self.val_diff or ar_dist > self.ar_diff
    
    def emotion_distance(self, sample1: Dict, sample2: Dict) -> float:
        """Calculate Euclidean distance in emotion space"""
        val_dist = sample1['mean_valence'] - sample2['mean_valence']
        ar_dist = sample1['mean_arousal'] - sample2['mean_arousal']
        return np.sqrt(val_dist**2 + ar_dist**2)
    
    def is_transition(self, sample1: Dict, sample2: Dict, 
                     max_frame_gap: int = 3) -> bool:
        """Check if two samples are part of an emotion transition"""
        # Must be from same clip
        if sample1['clip_id'] != sample2['clip_id']:
            return False
        
        # Must be temporally close
        frame_gap = abs(sample1['start_temporal_idx'] - sample2['start_temporal_idx'])
        if frame_gap > max_frame_gap:
            return False
        
        # Must have some emotion change (not too similar, not too different)
        distance = self.emotion_distance(sample1, sample2)
        return 0.5 < distance < 2.5  # Moderate change


# ============================================================================
# CONTRASTIVE PAIR SAMPLER
# ============================================================================

class ContrastivePairSampler:
    """Sample positive and negative pairs for contrastive learning"""
    
    def __init__(self, 
                 emotion_calculator: EmotionSimilarityCalculator,
                 num_positive_pairs: int = 2,
                 num_negative_pairs: int = 8,
                 hard_negative_ratio: float = 0.7):
        self.emotion_calc = emotion_calculator
        self.num_pos = num_positive_pairs
        self.num_neg = num_negative_pairs
        self.hard_neg_ratio = hard_negative_ratio
    
    def sample_positive_pairs(self, 
                             anchor: Dict, 
                             all_windows: List[Dict],
                             clip_to_windows: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Sample positive pairs for an anchor
        
        Strategy:
        - 50% temporal positives (same clip, nearby)
        - 30% emotion positives (different clip, similar emotion)
        - 20% transition positives (consecutive frames with emotion change)
        """
        positives = []
        
        # Type 1: Temporal Positives (same clip, nearby)
        num_temporal = int(self.num_pos * 0.5)
        temporal_positives = self._sample_temporal_positives(
            anchor, clip_to_windows, num_temporal
        )
        positives.extend(temporal_positives)
        
        # Type 2: Emotion Positives (different clip, similar emotion)
        num_emotion = int(self.num_pos * 0.3)
        emotion_positives = self._sample_emotion_positives(
            anchor, all_windows, num_emotion
        )
        positives.extend(emotion_positives)
        
        # Type 3: Transition Positives (emotion transitions)
        num_transition = self.num_pos - len(positives)
        transition_positives = self._sample_transition_positives(
            anchor, clip_to_windows, num_transition
        )
        positives.extend(transition_positives)
        
        # Ensure we have exactly num_pos positives
        while len(positives) < self.num_pos:
            # Fallback: sample any similar window
            candidates = [w for w in all_windows 
                         if self.emotion_calc.are_similar(anchor, w)]
            if candidates:
                positives.append(random.choice(candidates))
            else:
                break
        
        return positives[:self.num_pos]
    
    def sample_negative_pairs(self, 
                             anchor: Dict, 
                             all_windows: List[Dict],
                             clip_to_windows: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Sample negative pairs for an anchor
        
        Strategy:
        - 70% hard negatives (same clip or person, different emotion)
        - 30% random negatives (random samples)
        """
        negatives = []
        
        # Hard negatives
        num_hard = int(self.num_neg * self.hard_neg_ratio)
        hard_negatives = self._sample_hard_negatives(
            anchor, all_windows, clip_to_windows, num_hard
        )
        negatives.extend(hard_negatives)
        
        # Random negatives
        num_random = self.num_neg - len(negatives)
        random_negatives = self._sample_random_negatives(
            anchor, all_windows, num_random
        )
        negatives.extend(random_negatives)
        
        return negatives[:self.num_neg]
    
    def _sample_temporal_positives(self, 
                                   anchor: Dict, 
                                   clip_to_windows: Dict,
                                   num_samples: int) -> List[Dict]:
        """Sample from same clip, temporally close"""
        clip_id = anchor['clip_id']
        clip_windows = clip_to_windows.get(clip_id, [])
        
        # Find windows close in time with similar emotion
        candidates = []
        for window in clip_windows:
            if window == anchor:
                continue
            
            # Temporal distance
            temporal_dist = abs(window['start_temporal_idx'] - anchor['start_temporal_idx'])
            
            # Must be nearby (within 5 frames) and similar emotion
            if temporal_dist <= 5 and self.emotion_calc.are_similar(anchor, window):
                candidates.append((window, temporal_dist))
        
        # Sort by temporal distance (prefer closer)
        candidates.sort(key=lambda x: x[1])
        
        return [c[0] for c in candidates[:num_samples]]
    
    def _sample_emotion_positives(self, 
                                  anchor: Dict, 
                                  all_windows: List[Dict],
                                  num_samples: int) -> List[Dict]:
        """Sample from different clips with similar emotion"""
        candidates = []
        
        for window in all_windows:
            # Must be different clip
            if window['clip_id'] == anchor['clip_id']:
                continue
            
            # Must have similar emotion
            if self.emotion_calc.are_similar(anchor, window):
                distance = self.emotion_calc.emotion_distance(anchor, window)
                candidates.append((window, distance))
        
        # Sort by emotion distance (prefer most similar)
        candidates.sort(key=lambda x: x[1])
        
        return [c[0] for c in candidates[:num_samples]]
    
    def _sample_transition_positives(self, 
                                    anchor: Dict, 
                                    clip_to_windows: Dict,
                                    num_samples: int) -> List[Dict]:
        """Sample consecutive frames with emotion transitions"""
        clip_id = anchor['clip_id']
        clip_windows = clip_to_windows.get(clip_id, [])
        
        candidates = []
        for window in clip_windows:
            if window == anchor:
                continue
            
            if self.emotion_calc.is_transition(anchor, window):
                distance = self.emotion_calc.emotion_distance(anchor, window)
                candidates.append((window, distance))
        
        # Prefer moderate distances (transitions, not jumps)
        candidates.sort(key=lambda x: abs(x[1] - 1.5))
        
        return [c[0] for c in candidates[:num_samples]]
    
    def _sample_hard_negatives(self, 
                              anchor: Dict, 
                              all_windows: List[Dict],
                              clip_to_windows: Dict,
                              num_samples: int) -> List[Dict]:
        """Sample hard negatives: same context, different emotion"""
        candidates = []
        
        # Strategy 1: Same clip, different emotion
        clip_id = anchor['clip_id']
        clip_windows = clip_to_windows.get(clip_id, [])
        
        for window in clip_windows:
            if window == anchor:
                continue
            
            if self.emotion_calc.are_different(anchor, window):
                distance = self.emotion_calc.emotion_distance(anchor, window)
                candidates.append((window, distance))
        
        # Strategy 2: Different clip, very different emotion
        for window in all_windows:
            if window['clip_id'] == anchor['clip_id']:
                continue
            
            if self.emotion_calc.are_different(anchor, window):
                distance = self.emotion_calc.emotion_distance(anchor, window)
                candidates.append((window, distance))
        
        # Sort by distance (prefer moderately different, not opposite extremes)
        candidates.sort(key=lambda x: abs(x[1] - 4.0))
        
        return [c[0] for c in candidates[:num_samples]]
    
    def _sample_random_negatives(self, 
                                anchor: Dict, 
                                all_windows: List[Dict],
                                num_samples: int) -> List[Dict]:
        """Sample random negatives"""
        candidates = [w for w in all_windows if w != anchor]
        return random.sample(candidates, min(num_samples, len(candidates)))


# ============================================================================
# TEMPORAL EMOTION DATASET
# ============================================================================

class TemporalEmotionDataset(Dataset):
    """
    Dataset for temporal emotion contrastive learning
    
    Returns batches with:
    - Anchor windows
    - Positive pairs
    - Negative pairs
    """
    
    def __init__(self,
                 samples: List[Dict],
                 config,
                 image_processor,
                 split: str = 'train'):
        """
        Args:
            samples: List of frame samples from Phase 1
            config: TemporalEmotionConfig
            image_processor: VideoMAE image processor
            split: 'train' or 'val'
        """
        self.config = config
        self.processor = image_processor
        self.split = split
        
        print(f"\n{'='*80}")
        print(f"BUILDING TEMPORAL EMOTION DATASET ({split})")
        print(f"{'='*80}")
        
        # Group samples by clip
        self.clip_to_samples = defaultdict(list)
        for sample in samples:
            self.clip_to_samples[sample['clip_id']].append(sample)
        
        # Sort each clip by temporal index
        for clip_id in self.clip_to_samples:
            self.clip_to_samples[clip_id].sort(key=lambda x: x['temporal_idx'])
        
        print(f"Loaded {len(samples)} samples from {len(self.clip_to_samples)} clips")
        
        # Extract temporal windows
        self.window_extractor = TemporalWindowExtractor(
            num_frames=config.num_frames,
            stride=4 if split == 'train' else 8  # More overlap for training
        )
        
        self.windows = []
        self.clip_to_windows = defaultdict(list)
        
        for clip_id, clip_samples in tqdm(self.clip_to_samples.items(), 
                                         desc="Extracting temporal windows"):
            windows = self.window_extractor.extract_windows(clip_samples)
            for window in windows:
                self.windows.append(window)
                self.clip_to_windows[clip_id].append(window)
        
        print(f"Created {len(self.windows)} temporal windows")
        
        # Initialize emotion similarity calculator
        self.emotion_calc = EmotionSimilarityCalculator(
            valence_threshold_similar=config.valence_threshold_similar,
            arousal_threshold_similar=config.arousal_threshold_similar,
            valence_threshold_different=config.valence_threshold_different,
            arousal_threshold_different=config.arousal_threshold_different,
        )
        
        # Initialize pair sampler
        self.pair_sampler = ContrastivePairSampler(
            emotion_calculator=self.emotion_calc,
            num_positive_pairs=config.num_positive_pairs,
            num_negative_pairs=config.num_negative_pairs,
            hard_negative_ratio=config.hard_negative_ratio,
        )
        
        # Analyze dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Print dataset statistics"""
        print(f"\n Dataset Analysis:")
        print(f"   Total windows: {len(self.windows)}")
        print(f"   Clips: {len(self.clip_to_windows)}")
        
        # Window size distribution
        window_sizes = [w['num_frames'] for w in self.windows]
        print(f"   Window sizes - Mean: {np.mean(window_sizes):.1f}, "
              f"Min: {min(window_sizes)}, Max: {max(window_sizes)}")
        
        # Emotion distribution
        valences = [w['mean_valence'] for w in self.windows]
        arousals = [w['mean_arousal'] for w in self.windows]
        print(f"   Valence - Mean: {np.mean(valences):.2f}, Std: {np.std(valences):.2f}")
        print(f"   Arousal - Mean: {np.mean(arousals):.2f}, Std: {np.std(arousals):.2f}")
        
        # Emotion variability
        val_stds = [w['std_valence'] for w in self.windows]
        ar_stds = [w['std_arousal'] for w in self.windows]
        print(f"   Within-window variability:")
        print(f"      Valence std: {np.mean(val_stds):.2f}")
        print(f"      Arousal std: {np.mean(ar_stds):.2f}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a training sample with anchor, positives, and negatives
        
        Returns:
            Dictionary with:
            - anchor_frames: Tensor (num_frames, 3, H, W)
            - positive_frames: List of tensors
            - negative_frames: List of tensors
            - anchor_metadata: Dict
        """
        anchor = self.windows[idx]
        
        # Sample positive and negative pairs
        positives = self.pair_sampler.sample_positive_pairs(
            anchor, self.windows, self.clip_to_windows
        )
        negatives = self.pair_sampler.sample_negative_pairs(
            anchor, self.windows, self.clip_to_windows
        )
        
        # Load frames
        anchor_frames = self._load_window_frames(anchor)
        positive_frames = [self._load_window_frames(pos) for pos in positives]
        negative_frames = [self._load_window_frames(neg) for neg in negatives]
        
        return {
            'anchor_frames': anchor_frames,
            'positive_frames': positive_frames,
            'negative_frames': negative_frames,
            'anchor_metadata': {
                'clip_id': anchor['clip_id'],
                'mean_valence': anchor['mean_valence'],
                'mean_arousal': anchor['mean_arousal'],
                'num_positives': len(positives),
                'num_negatives': len(negatives),
            }
        }
    
    def _load_window_frames(self, window: Dict) -> torch.Tensor:
        """
        Load frames for a window and process them
        
        Returns:
            Tensor of shape (num_frames, 3, H, W)
        """
        frames = []
        
        for sample in window['samples']:
            try:
                # Load image
                img = Image.open(sample['image_path']).convert('RGB')
                frames.append(img)
            except Exception as e:
                print(f"Warning: Could not load {sample['image_path']}: {e}")
                # Create blank frame as fallback
                frames.append(Image.new('RGB', (224, 224), color='black'))
        
        # Handle padding for short clips
        while len(frames) < self.config.num_frames:
            # Repeat last frame
            frames.append(frames[-1])
        
        # Process frames using VideoMAE processor
        # Processor expects list of PIL images
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim
        
        return pixel_values


# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================

def temporal_emotion_collate_fn(batch):
    """
    Custom collate function for batching
    
    Handles variable number of positive/negative pairs
    """
    anchor_frames = torch.stack([item['anchor_frames'] for item in batch])
    
    # Collect all positive frames
    positive_frames = []
    for item in batch:
        positive_frames.extend(item['positive_frames'])
    positive_frames = torch.stack(positive_frames) if positive_frames else None
    
    # Collect all negative frames
    negative_frames = []
    for item in batch:
        negative_frames.extend(item['negative_frames'])
    negative_frames = torch.stack(negative_frames) if negative_frames else None
    
    # Metadata
    metadata = {
        'clip_ids': [item['anchor_metadata']['clip_id'] for item in batch],
        'valences': torch.tensor([item['anchor_metadata']['mean_valence'] for item in batch]),
        'arousals': torch.tensor([item['anchor_metadata']['mean_arousal'] for item in batch]),
        'num_positives': sum(item['anchor_metadata']['num_positives'] for item in batch),
        'num_negatives': sum(item['anchor_metadata']['num_negatives'] for item in batch),
    }
    
    return {
        'anchor': anchor_frames,
        'positive': positive_frames,
        'negative': negative_frames,
        'metadata': metadata,
    }


# ============================================================================
# DATASET BUILDER
# ============================================================================

def build_temporal_dataloaders(samples: List[Dict],
                               config,
                               image_processor,
                               train_ratio: float = 0.8):
    """
    Build train and validation dataloaders
    
    Args:
        samples: List of frame samples from Phase 1
        config: TemporalEmotionConfig
        image_processor: VideoMAE image processor
        train_ratio: Ratio of data for training
    
    Returns:
        train_loader, val_loader
    """
    # Split samples by clip to avoid leakage
    clip_ids = list(set(s['clip_id'] for s in samples))
    random.shuffle(clip_ids)
    
    split_idx = int(len(clip_ids) * train_ratio)
    train_clip_ids = set(clip_ids[:split_idx])
    val_clip_ids = set(clip_ids[split_idx:])
    
    train_samples = [s for s in samples if s['clip_id'] in train_clip_ids]
    val_samples = [s for s in samples if s['clip_id'] in val_clip_ids]
    
    print(f"\n{'='*80}")
    print(f"DATASET SPLIT")
    print(f"{'='*80}")
    print(f"Train clips: {len(train_clip_ids)} ({len(train_samples)} frames)")
    print(f"Val clips: {len(val_clip_ids)} ({len(val_samples)} frames)")
    
    # Create datasets
    train_dataset = TemporalEmotionDataset(
        samples=train_samples,
        config=config,
        image_processor=image_processor,
        split='train'
    )
    
    val_dataset = TemporalEmotionDataset(
        samples=val_samples,
        config=config,
        image_processor=image_processor,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=temporal_emotion_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=temporal_emotion_collate_fn,
    )
    
    print(f"\n Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_dataset():
    """Test the dataset implementation"""
    print("="*80)
    print("TESTING TEMPORAL EMOTION DATASET")
    print("="*80)
    
    # Load checkpoint
    from pathlib import Path
    import sys
    
    # Get the root directory (where this script is located)
    root_dir = Path(__file__).parent
    
    # Import config and checkpoint manager from phase 1
    # The phase 1 script should be in the same root directory
    sys.path.insert(0, str(root_dir))
    from task2_temporal_analysis_backbone import TemporalEmotionConfig, CheckpointManager
    
    # Initialize config (it already points to task2_outputs)
    config = TemporalEmotionConfig()
    
    # Checkpoints are in: task2_outputs/checkpoints
    checkpoint_dir = Path(config.project_root) / 'checkpoints'
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    print(f"Loading checkpoints from: {checkpoint_dir}")
    
    # Load samples
    samples = checkpoint_manager.load_checkpoint('samples_final')
    if samples is None:
        print(" No samples found. Run Phase 1 first.")
        return
    
    print(f" Loaded {len(samples)} samples")
    
    # Initialize processor
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(config.backbone_model)
    
    # Build dataloaders
    train_loader, val_loader = build_temporal_dataloaders(
        samples=samples,
        config=config,
        image_processor=processor,
        train_ratio=0.8
    )
    
    # Test one batch
    print(f"\n{'='*80}")
    print("TESTING BATCH LOADING")
    print(f"{'='*80}")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"   Anchor shape: {batch['anchor'].shape}")
        print(f"   Positive shape: {batch['positive'].shape if batch['positive'] is not None else None}")
        print(f"   Negative shape: {batch['negative'].shape if batch['negative'] is not None else None}")
        print(f"   Valence range: [{batch['metadata']['valences'].min():.2f}, {batch['metadata']['valences'].max():.2f}]")
        print(f"   Arousal range: [{batch['metadata']['arousals'].min():.2f}, {batch['metadata']['arousals'].max():.2f}]")
        print(f"   Num positives: {batch['metadata']['num_positives']}")
        print(f"   Num negatives: {batch['metadata']['num_negatives']}")
        
        if batch_idx >= 2:  # Test 3 batches
            break
    
    print(f"\n{'='*80}")
    print(" DATASET TEST COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_dataset()