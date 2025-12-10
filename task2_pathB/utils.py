"""
Utility functions for Path B pipeline
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str, name: str = 'training') -> logging.Logger:
    """
    Setup logging to both file and console
    
    Args:
        log_dir: Directory to save log files
        name: Logger name
    
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove existing handlers
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manage training checkpoints with save/load/resume functionality"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, 
             name: str, 
             data: Dict[str, Any],
             is_best: bool = False):
        """
        Save checkpoint
        
        Args:
            name: Checkpoint name (e.g., 'epoch_010', 'latest')
            data: Dictionary containing state to save
            is_best: If True, also save as 'best_model'
        """
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        
        # Save with atomic write (write to temp, then rename)
        temp_path = self.checkpoint_dir / f'{name}.tmp'
        torch.save(data, temp_path)
        temp_path.rename(checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(data, best_path)
            print(f"🏆 Best model saved: {best_path.name}")
    
    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint
        
        Args:
            name: Checkpoint name
        
        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        if not checkpoint_path.exists():
            return None
        
        try:
            data = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ Checkpoint loaded: {checkpoint_path.name}")
            return data
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path.name}: {e}")
            return None
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint"""
        return self.load('latest')
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best model checkpoint"""
        return self.load('best_model')
    
    def exists(self, name: str) -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f'{name}.pt').exists()
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        return sorted([f.stem for f in self.checkpoint_dir.glob('*.pt')])


# ============================================================================
# FEATURE CACHE MANAGEMENT
# ============================================================================

def save_clip_features(clip_dir: Path, 
                       features: np.ndarray, 
                       metadata: Dict[str, Any]):
    """
    Save features and metadata for a single clip
    
    Args:
        clip_dir: Directory to save clip data
        features: Numpy array of shape (num_frames, feature_dim)
        metadata: Dictionary with clip information
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    np.save(clip_dir / 'features.npy', features)
    
    # Save metadata
    with open(clip_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_clip_features(clip_dir: Path) -> tuple:
    """
    Load features and metadata for a single clip
    
    Args:
        clip_dir: Directory containing clip data
    
    Returns:
        (features, metadata) tuple
    """
    features = np.load(clip_dir / 'features.npy')
    
    with open(clip_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return features, metadata


def create_feature_index(features_cache_dir: Path) -> Dict[str, Any]:
    """
    Create index of all extracted features
    
    Args:
        features_cache_dir: Root directory with all clip features
    
    Returns:
        Index dictionary with clip information
    """
    index = {
        'clips': [],
        'total_frames': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    clip_dirs = sorted([d for d in features_cache_dir.iterdir() if d.is_dir()])
    
    for clip_dir in clip_dirs:
        if (clip_dir / 'features.npy').exists():
            features, metadata = load_clip_features(clip_dir)
            
            clip_info = {
                'clip_name': clip_dir.name,
                'clip_dir': str(clip_dir),
                'num_frames': features.shape[0],
                'feature_dim': features.shape[1],
                **metadata
            }
            
            index['clips'].append(clip_info)
            index['total_frames'] += features.shape[0]
    
    # Save index
    index_path = features_cache_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\n✅ Feature index created:")
    print(f"   Clips: {len(index['clips'])}")
    print(f"   Total frames: {index['total_frames']}")
    print(f"   Saved to: {index_path}")
    
    return index


def load_feature_index(features_cache_dir: Path) -> Optional[Dict[str, Any]]:
    """Load feature index"""
    index_path = Path(features_cache_dir) / 'index.json'
    
    if not index_path.exists():
        return None
    
    with open(index_path, 'r') as f:
        return json.load(f)


# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


def print_gpu_memory(prefix: str = ""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory()
        print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def clear_gpu_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================