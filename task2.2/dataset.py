"""
Dataset loader for AFEW-VA with temporal frame sampling
FIXED: Handles multiple JSON formats
"""
import os
import json
import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class AFEWVADataset(Dataset):
    """
    AFEW-VA dataset with temporal sampling support
    
    Expected structure:
    AFEW-VA/
      001/
        00000.png
        00001.png
        ...
        001.json
    """
    
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 16,
        split: str = 'train',
        transform=None,
        cache_features: bool = False,
        clip_ids: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self.cache_features = cache_features
        
        # Load all clips
        self.clips = self._load_clips(clip_ids)
        
        print(f"Loaded {len(self.clips)} clips for {split} split")
        
    def _load_clips(self, clip_ids: Optional[List[str]] = None) -> List[Dict]:
        """Load all clips with frame paths and VA annotations"""
        clips = []
        
        # Get all clip directories
        if clip_ids is not None:
            clip_dirs = [self.root_dir / cid for cid in clip_ids]
        else:
            clip_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            json_path = clip_dir / f"{clip_id}.json"
            
            if not json_path.exists():
                continue
            
            # Load annotations
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            # Get frame files
            frame_files = sorted(glob.glob(str(clip_dir / "*.png")))
            
            if len(frame_files) == 0:
                continue
            
            # Parse annotations based on format
            anno_dict = self._parse_annotations(annotations)
            
            # Build frame list with VA values
            frames = []
            for frame_path in frame_files:
                frame_name = Path(frame_path).stem
                
                # Get VA values for this frame
                if frame_name in anno_dict:
                    valence, arousal = anno_dict[frame_name]
                    frames.append({
                        'path': frame_path,
                        'valence': valence,
                        'arousal': arousal
                    })
            
            if len(frames) >= self.num_frames:
                clips.append({
                    'clip_id': clip_id,
                    'frames': frames
                })
        
        return clips
    
    def _parse_annotations(self, annotations) -> Dict[str, Tuple[float, float]]:
        """
        Parse annotations in various formats
        Returns: dict mapping frame_name -> (valence, arousal)
        """
        anno_dict = {}
        
        # Format 1: List of dicts [{"frame": 0, "valence": 0.5, "arousal": 0.2}, ...]
        if isinstance(annotations, list):
            for anno in annotations:
                if isinstance(anno, dict):
                    frame_num = anno.get('frame', None)
                    if frame_num is not None:
                        frame_name = str(frame_num).zfill(5)
                        valence = float(anno.get('valence', 0.0))
                        arousal = float(anno.get('arousal', 0.0))
                        anno_dict[frame_name] = (valence, arousal)
        
        # Format 2: Dict with frame keys {"00000": {"valence": 0.5, "arousal": 0.2}, ...}
        elif isinstance(annotations, dict):
            for frame_key, values in annotations.items():
                if isinstance(values, dict):
                    frame_name = str(frame_key).zfill(5)
                    valence = float(values.get('valence', 0.0))
                    arousal = float(values.get('arousal', 0.0))
                    anno_dict[frame_name] = (valence, arousal)
                elif isinstance(values, (list, tuple)) and len(values) >= 2:
                    # Format: {"00000": [valence, arousal], ...}
                    frame_name = str(frame_key).zfill(5)
                    valence = float(values[0])
                    arousal = float(values[1])
                    anno_dict[frame_name] = (valence, arousal)
        
        return anno_dict
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a window of frames from a clip
        """
        clip = self.clips[idx]
        frames = clip['frames']
        
        # Sample a contiguous window
        if len(frames) > self.num_frames:
            start_idx = np.random.randint(0, len(frames) - self.num_frames + 1)
            sampled_frames = frames[start_idx:start_idx + self.num_frames]
        else:
            sampled_frames = frames[:self.num_frames]
            # Pad if needed
            while len(sampled_frames) < self.num_frames:
                sampled_frames.append(sampled_frames[-1])
        
        # Load images with error handling
        images = []
        valences = []
        arousals = []
        
        for frame in sampled_frames:
            try:
                img = Image.open(frame['path']).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                images.append(img)
                valences.append(frame['valence'])
                arousals.append(frame['arousal'])
            except Exception as e:
                print(f"Warning: Failed to load {frame['path']}: {e}")
                # Use previous frame if available
                if len(images) > 0:
                    images.append(images[-1])
                    valences.append(valences[-1])
                    arousals.append(arousals[-1])
        
        # Ensure we have enough frames
        if len(images) == 0:
            raise RuntimeError(f"Failed to load any frames from clip {clip['clip_id']}")
        
        images = torch.stack(images)  # [L, C, H, W]
        valences = torch.tensor(valences, dtype=torch.float32)
        arousals = torch.tensor(arousals, dtype=torch.float32)
        
        return {
            'images': images,
            'valences': valences,
            'arousals': arousals,
            'clip_id': clip['clip_id']
        }


def get_clip_transform(is_train: bool = True, augment: bool = True) -> transforms.Compose:
    """Get CLIP-compatible image transforms"""
    
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    if is_train and augment:
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(
    config,
    train_clips: List[str],
    val_clips: List[str],
    test_clips: List[str]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    train_dataset = AFEWVADataset(
        root_dir=config.dataset_root,
        num_frames=config.num_frames,
        split='train',
        transform=get_clip_transform(is_train=True, augment=config.use_augmentation),
        clip_ids=train_clips
    )
    
    val_dataset = AFEWVADataset(
        root_dir=config.dataset_root,
        num_frames=config.num_frames,
        split='val',
        transform=get_clip_transform(is_train=False, augment=False),
        clip_ids=val_clips
    )
    
    test_dataset = AFEWVADataset(
        root_dir=config.dataset_root,
        num_frames=config.num_frames,
        split='test',
        transform=get_clip_transform(is_train=False, augment=False),
        clip_ids=test_clips
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def split_dataset(root_dir: str, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
    """Split clips into train/val/test sets"""
    np.random.seed(seed)
    
    # Get all clip IDs
    root = Path(root_dir)
    clip_ids = sorted([d.name for d in root.iterdir() if d.is_dir()])
    
    # Shuffle
    np.random.shuffle(clip_ids)
    
    n_clips = len(clip_ids)
    n_test = int(n_clips * test_split)
    n_val = int(n_clips * val_split)
    n_train = n_clips - n_test - n_val
    
    train_clips = clip_ids[:n_train]
    val_clips = clip_ids[n_train:n_train + n_val]
    test_clips = clip_ids[n_train + n_val:]
    
    print(f"Split: {n_train} train, {n_val} val, {n_test} test clips")
    
    return train_clips, val_clips, test_clips