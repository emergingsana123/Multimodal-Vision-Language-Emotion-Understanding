"""
Dataset loader for AFEW-VA with temporal frame sampling
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
            
            # Build frame list with VA values
            frames = []
            for frame_path in frame_files:
                frame_name = Path(frame_path).stem
                
                # Find matching annotation
                frame_anno = None
                for anno in annotations:
                    if str(anno.get('frame', '')).zfill(5) == frame_name:
                        frame_anno = anno
                        break
                
                if frame_anno is not None:
                    frames.append({
                        'path': frame_path,
                        'valence': frame_anno.get('valence', 0.0),
                        'arousal': frame_anno.get('arousal', 0.0)
                    })
            
            if len(frames) >= self.num_frames:
                clips.append({
                    'clip_id': clip_id,
                    'frames': frames
                })
        
        return clips
    
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
        
        # Load images
        images = []
        valences = []
        arousals = []
        
        for frame in sampled_frames:
            img = Image.open(frame['path']).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
            valences.append(frame['valence'])
            arousals.append(frame['arousal'])
        
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