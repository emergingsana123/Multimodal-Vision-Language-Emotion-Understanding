#!/usr/bin/env python3
"""
AFEW-VA Dataset Complete Analysis Pipeline - Enhanced Version
Includes ALL analyses from the original notebook:
- BLIP-2 embedding extraction
- UMAP and t-SNE dimensionality reduction
- PCA validation analysis
- Comprehensive correlation analysis (Pearson & Spearman)
- Temporal dynamics analysis
- Multiple visualization outputs
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import h5py
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from transformers import Blip2Processor, Blip2Model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR VM
# ============================================================================

# Update this to your actual dataset path
DATA_DIR = Path('/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/dataset')

# Project output directories
PROJECT_ROOT = Path('/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/outputs')
EMBEDDINGS_DIR = PROJECT_ROOT / 'embeddings'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
VISUALIZATIONS_DIR = PROJECT_ROOT / 'visualizations'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create all output directories
#for directory in [PROJECT_ROOT, EMBEDDINGS_DIR, CHECKPOINTS_DIR, VISUALIZATIONS_DIR, RESULTS_DIR]:
<<<<<<< HEAD
#    directory.mkdir(parents=True, exist_ok=True)
=======
#   directory.mkdir(parents=True, exist_ok=True)
# for directory in [PROJECT_ROOT, EMBEDDINGS_DIR, CHECKPOINTS_DIR, VISUALIZATIONS_DIR, RESULTS_DIR]:
#    directory.mkdir(parents=True, exist_ok=True)

>>>>>>> d80814a (change)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

CONFIG = {
    'random_seed': RANDOM_SEED,
    'model_name': 'Salesforce/blip2-opt-2.7b',
    'batch_size': 16,
    'image_size': 224,
    'use_fp16': True,
    'embedding_dim': 768,
    'umap_n_neighbors': 30,
    'umap_min_dist': 0.1,
    'umap_n_components': 2,
    'umap_metric': 'cosine',
    'pca_n_components': 50,
    'checkpoint_frequency': 100,
}

# Save configuration
with open(PROJECT_ROOT / 'config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("="*80)
print("AFEW-VA COMPLETE ANALYSIS PIPELINE - Enhanced Version")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {PROJECT_ROOT}")
print("="*80)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_afew_va_clip(clip_dir: Path):
    """Load a single clip's annotations and frame data."""
    clip_name = clip_dir.name
    annotation_file = clip_dir / f"{clip_name}.json"
    
    if not annotation_file.exists():
        return None
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    frames = annotations.get("frames", {})
    
    frames_data = []
    for frame_id, frame_data in frames.items():
        frame_num = int(frame_id)
        
        # Support png or jpg
        image_path = clip_dir / f"{frame_id}.png"
        if not image_path.exists():
            image_path = clip_dir / f"{frame_id}.jpg"
        
        if image_path.exists():
            frames_data.append({
                'clip_name': clip_name,
                'frame_id': frame_id,
                'frame_num': frame_num,
                'image_path': str(image_path),
                'valence': frame_data.get('valence', 0.0),
                'arousal': frame_data.get('arousal', 0.0),
            })
    
    return frames_data


def load_afew_va_dataset(data_dir: Path):
    """Load entire AFEW-VA dataset."""
    print("\n" + "="*80)
    print("LOADING AFEW-VA DATASET")
    print("="*80)
    
    all_frames = []
    clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(clip_dirs)} clip directories")
    
    for clip_dir in tqdm(clip_dirs, desc="Loading clips"):
        frames = load_afew_va_clip(clip_dir)
        if frames:
            all_frames.extend(frames)
    
    df = pd.DataFrame(all_frames)
    
    print(f"\nDataset Statistics:")
    print(f"  Total frames: {len(df)}")
    print(f"  Total clips: {df['clip_name'].nunique()}")
    print(f"  Valence range: [{df['valence'].min():.3f}, {df['valence'].max():.3f}]")
    print(f"  Arousal range: [{df['arousal'].min():.3f}, {df['arousal'].max():.3f}]")
    print(f"  Valence distribution: μ={df['valence'].mean():.2f}, σ={df['valence'].std():.2f}")
    print(f"  Arousal distribution: μ={df['arousal'].mean():.2f}, σ={df['arousal'].std():.2f}")
    
    return df


# ============================================================================
# DATASET CLASS
# ============================================================================

class AFEWVADataset(Dataset):
    """PyTorch Dataset for AFEW-VA."""
    
    def __init__(self, dataframe: pd.DataFrame, processor, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image = Image.open(row['image_path']).convert('RGB')
        
        # Process with BLIP-2 processor
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'clip_name': row['clip_name'],
            'frame_id': row['frame_id'],
            'valence': torch.tensor(row['valence'], dtype=torch.float32),
            'arousal': torch.tensor(row['arousal'], dtype=torch.float32),
            'idx': idx
        }


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(model, processor, dataframe, batch_size=16, device='cuda'):
    """Extract BLIP-2 embeddings for all frames."""
    print("\n" + "="*80)
    print("EXTRACTING BLIP-2 EMBEDDINGS")
    print("="*80)
    
    dataset = AFEWVADataset(dataframe, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            pixel_values = batch['pixel_values'].to(device)
            
            # Extract vision embeddings
            vision_outputs = model.vision_model(pixel_values)
            embeddings = vision_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Store embeddings and metadata
            all_embeddings.append(embeddings.cpu().numpy())
            
            for i in range(len(batch['clip_name'])):
                all_metadata.append({
                    'clip_name': batch['clip_name'][i],
                    'frame_id': batch['frame_id'][i],
                    'valence': batch['valence'][i].item(),
                    'arousal': batch['arousal'][i].item(),
                    'idx': batch['idx'][i].item()
                })
            
            # Save checkpoint
            if (batch_idx + 1) % CONFIG['checkpoint_frequency'] == 0:
                checkpoint_path = CHECKPOINTS_DIR / f'checkpoint_batch_{batch_idx+1}.pkl'
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'embeddings': np.vstack(all_embeddings),
                        'metadata': all_metadata
                    }, f)
                print(f"  Saved checkpoint at batch {batch_idx+1}")
    
    embeddings_array = np.vstack(all_embeddings)
    metadata_df = pd.DataFrame(all_metadata)
    
    print(f"\nExtracted embeddings shape: {embeddings_array.shape}")
    
    # Save final embeddings
    embeddings_path = EMBEDDINGS_DIR / 'blip2_embeddings.h5'
    with h5py.File(embeddings_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings_array)
        f.create_dataset('valence', data=metadata_df['valence'].values)
        f.create_dataset('arousal', data=metadata_df['arousal'].values)
    
    # Also save as pickle for compatibility
    with open(EMBEDDINGS_DIR / 'embeddings.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_array,
            'valences': metadata_df['valence'].values,
            'arousals': metadata_df['arousal'].values,
            'metadata': metadata_df
        }, f)
    
    metadata_df.to_csv(EMBEDDINGS_DIR / 'metadata.csv', index=False)
    print(f"Saved embeddings to: {embeddings_path}")
    
    return embeddings_array, metadata_df


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def apply_umap(embeddings, config):
    """Apply UMAP dimensionality reduction."""
    print("\n" + "="*80)
    print("APPLYING UMAP DIMENSIONALITY REDUCTION")
    print("="*80)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    reducer = umap.UMAP(
        n_neighbors=config['umap_n_neighbors'],
        min_dist=config['umap_min_dist'],
        n_components=config['umap_n_components'],
        metric=config['umap_metric'],
        random_state=config['random_seed'],
        verbose=True
    )
    
    embeddings_2d = reducer.fit_transform(embeddings_scaled)
    
    print(f"Reduced embeddings shape: {embeddings_2d.shape}")
    
    # Save UMAP results
    np.save(RESULTS_DIR / 'umap_embeddings_2d.npy', embeddings_2d)
    
    return embeddings_2d


def apply_tsne(embeddings, n_components=2):
    """Apply t-SNE dimensionality reduction."""
    print("\n" + "="*80)
    print("APPLYING t-SNE DIMENSIONALITY REDUCTION")
    print("="*80)
    
    tsne = TSNE(n_components=n_components, random_state=RANDOM_SEED, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print(f"Reduced embeddings shape: {embeddings_2d.shape}")
    
    np.save(RESULTS_DIR / 'tsne_embeddings_2d.npy', embeddings_2d)
    
    return embeddings_2d


def apply_pca(embeddings, n_components=50):
    """Apply PCA for validation and analysis."""
    print("\n" + "="*80)
    print("APPLYING PCA (VALIDATION & ANALYSIS)")
    print("="*80)
    
    # Standardize first
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_embeddings = pca.fit_transform(embeddings_scaled)
    
    print(f"PCA embeddings shape: {pca_embeddings.shape}")
    print(f"Explained variance by first 10 components: {pca.explained_variance_ratio_[:10].sum():.2%}")
    print(f"Components needed for 80% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1}")
    print(f"Components needed for 90% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1}")
    
    # Save PCA results
    np.save(RESULTS_DIR / 'pca_embeddings.npy', pca_embeddings)
    np.save(RESULTS_DIR / 'pca_explained_variance.npy', pca.explained_variance_ratio_)
    
    return pca_embeddings, pca


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_comprehensive_correlations(embeddings, pca_embeddings, umap_embeddings, metadata_df):
    """Compute comprehensive correlation analysis (Pearson & Spearman)."""
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*80)
    
    valences = metadata_df['valence'].values
    arousals = metadata_df['arousal'].values
    
    results = {
        'embedding_dim_correlations': {
            'valence_pearson': [],
            'arousal_pearson': [],
            'valence_spearman': [],
            'arousal_spearman': []
        },
        'pca_correlations': {
            'valence_pearson': [],
            'arousal_pearson': [],
            'valence_spearman': [],
            'arousal_spearman': [],
            'valence_pvalues': [],
            'arousal_pvalues': []
        },
        'umap_correlations': {
            'valence_pearson': [],
            'arousal_pearson': [],
            'valence_spearman': [],
            'arousal_spearman': []
        }
    }
    
    # 1. Raw embedding dimensions (first 50)
    print("\nAnalyzing raw embedding dimensions...")
    for dim in range(min(50, embeddings.shape[1])):
        val_corr_p, _ = pearsonr(embeddings[:, dim], valences)
        aro_corr_p, _ = pearsonr(embeddings[:, dim], arousals)
        val_corr_s, _ = spearmanr(embeddings[:, dim], valences)
        aro_corr_s, _ = spearmanr(embeddings[:, dim], arousals)
        
        results['embedding_dim_correlations']['valence_pearson'].append(val_corr_p)
        results['embedding_dim_correlations']['arousal_pearson'].append(aro_corr_p)
        results['embedding_dim_correlations']['valence_spearman'].append(val_corr_s)
        results['embedding_dim_correlations']['arousal_spearman'].append(aro_corr_s)
    
    # 2. PCA components
    print("Analyzing PCA components...")
    print("-" * 50)
    for i in range(min(5, pca_embeddings.shape[1])):
        val_corr_p, val_p = pearsonr(pca_embeddings[:, i], valences)
        aro_corr_p, aro_p = pearsonr(pca_embeddings[:, i], arousals)
        val_corr_s, _ = spearmanr(pca_embeddings[:, i], valences)
        aro_corr_s, _ = spearmanr(pca_embeddings[:, i], arousals)
        
        results['pca_correlations']['valence_pearson'].append(val_corr_p)
        results['pca_correlations']['arousal_pearson'].append(aro_corr_p)
        results['pca_correlations']['valence_spearman'].append(val_corr_s)
        results['pca_correlations']['arousal_spearman'].append(aro_corr_s)
        results['pca_correlations']['valence_pvalues'].append(val_p)
        results['pca_correlations']['arousal_pvalues'].append(aro_p)
        
        print(f"PC{i+1}:")
        print(f"  Valence: Pearson={val_corr_p:+.3f} (p={val_p:.2e}), Spearman={val_corr_s:+.3f}")
        print(f"  Arousal: Pearson={aro_corr_p:+.3f} (p={aro_p:.2e}), Spearman={aro_corr_s:+.3f}")
    
    # 3. UMAP dimensions
    print("\n" + "-" * 50)
    print("Analyzing UMAP dimensions...")
    print("-" * 50)
    for i in range(2):
        val_corr_p, val_p = pearsonr(umap_embeddings[:, i], valences)
        aro_corr_p, aro_p = pearsonr(umap_embeddings[:, i], arousals)
        val_corr_s, _ = spearmanr(umap_embeddings[:, i], valences)
        aro_corr_s, _ = spearmanr(umap_embeddings[:, i], arousals)
        
        results['umap_correlations']['valence_pearson'].append(val_corr_p)
        results['umap_correlations']['arousal_pearson'].append(aro_corr_p)
        results['umap_correlations']['valence_spearman'].append(val_corr_s)
        results['umap_correlations']['arousal_spearman'].append(aro_corr_s)
        
        print(f"UMAP Dim {i+1}:")
        print(f"  Valence: Pearson={val_corr_p:+.3f} (p={val_p:.2e}), Spearman={val_corr_s:+.3f}")
        print(f"  Arousal: Pearson={aro_corr_p:+.3f} (p={aro_p:.2e}), Spearman={aro_corr_s:+.3f}")
    
    # Save results
    with open(RESULTS_DIR / 'comprehensive_correlations.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            json_results[key] = {k: [float(x) for x in v] for k, v in value.items()}
        json.dump(json_results, f, indent=2)
    
    # Create correlation summary
    summary = {
        'max_valence_pearson_raw': float(max([abs(x) for x in results['embedding_dim_correlations']['valence_pearson']])),
        'max_arousal_pearson_raw': float(max([abs(x) for x in results['embedding_dim_correlations']['arousal_pearson']])),
        'max_valence_pearson_pca': float(max([abs(x) for x in results['pca_correlations']['valence_pearson']])),
        'max_arousal_pearson_pca': float(max([abs(x) for x in results['pca_correlations']['arousal_pearson']])),
    }
    
    print("\n" + "="*80)
    print("CORRELATION SUMMARY:")
    print(f"  Max valence correlation (raw): {summary['max_valence_pearson_raw']:.4f}")
    print(f"  Max arousal correlation (raw): {summary['max_arousal_pearson_raw']:.4f}")
    print(f"  Max valence correlation (PCA): {summary['max_valence_pearson_pca']:.4f}")
    print(f"  Max arousal correlation (PCA): {summary['max_arousal_pearson_pca']:.4f}")
    print("="*80)
    
    return results


def compute_temporal_analysis(embeddings, metadata_df):
    """Analyze temporal dynamics within clips."""
    print("\n" + "="*80)
    print("COMPUTING TEMPORAL ANALYSIS")
    print("="*80)
    
    temporal_stats = []
    
    for clip_name in metadata_df['clip_name'].unique():
        clip_mask = metadata_df['clip_name'] == clip_name
        clip_embeddings = embeddings[clip_mask]
        clip_meta = metadata_df[clip_mask].sort_values('frame_id')
        
        if len(clip_embeddings) > 1:
            # Compute temporal smoothness
            differences = np.diff(clip_embeddings, axis=0)
            smoothness = np.mean(np.linalg.norm(differences, axis=1))
            
            # Valence/arousal dynamics
            valence_change = clip_meta['valence'].max() - clip_meta['valence'].min()
            arousal_change = clip_meta['arousal'].max() - clip_meta['arousal'].min()
            
            temporal_stats.append({
                'clip_name': clip_name,
                'num_frames': len(clip_embeddings),
                'embedding_smoothness': smoothness,
                'valence_change': valence_change,
                'arousal_change': arousal_change,
                'mean_valence': clip_meta['valence'].mean(),
                'mean_arousal': clip_meta['arousal'].mean(),
                'std_valence': clip_meta['valence'].std(),
                'std_arousal': clip_meta['arousal'].std()
            })
    
    temporal_df = pd.DataFrame(temporal_stats)
    temporal_df.to_csv(RESULTS_DIR / 'temporal_analysis.csv', index=False)
    
    print(f"Analyzed {len(temporal_df)} clips")
    print(f"Mean embedding smoothness: {temporal_df['embedding_smoothness'].mean():.4f}")
    print(f"Mean valence change: {temporal_df['valence_change'].mean():.4f}")
    print(f"Mean arousal change: {temporal_df['arousal_change'].mean():.4f}")
    
    return temporal_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_umap_primary_visualization(umap_embeddings, metadata_df):
    """Create primary UMAP visualization (matches notebook output)."""
    print("\n" + "="*80)
    print("CREATING PRIMARY UMAP VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # UMAP colored by Valence
    scatter1 = axes[0].scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=metadata_df['valence'],
        cmap='RdYlBu_r',  # Red (negative) to Blue (positive)
        s=8,
        alpha=0.6,
        edgecolors='none'
    )
    axes[0].set_title('UMAP Projection - Colored by Valence', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Valence (Negative ← → Positive)', fontsize=10)
    
    # UMAP colored by Arousal
    scatter2 = axes[1].scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=metadata_df['arousal'],
        cmap='YlOrRd',  # Light (low arousal) to Dark (high arousal)
        s=8,
        alpha=0.6,
        edgecolors='none'
    )
    axes[1].set_title('UMAP Projection - Colored by Arousal', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Arousal (Low ← → High)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'umap_primary_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: umap_primary_visualization.png")


def create_pca_validation_plots(pca_embeddings, pca, metadata_df):
    """Create comprehensive PCA validation plots (matches notebook)."""
    print("\n" + "="*80)
    print("CREATING PCA VALIDATION VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # PCA colored by Valence
    scatter1 = axes[0, 0].scatter(
        pca_embeddings[:, 0],
        pca_embeddings[:, 1],
        c=metadata_df['valence'],
        cmap='RdYlBu_r',
        s=8,
        alpha=0.6,
        edgecolors='none'
    )
    axes[0, 0].set_title('PCA Projection - Colored by Valence', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Valence', fontsize=10)
    
    # PCA colored by Arousal
    scatter2 = axes[0, 1].scatter(
        pca_embeddings[:, 0],
        pca_embeddings[:, 1],
        c=metadata_df['arousal'],
        cmap='YlOrRd',
        s=8,
        alpha=0.6,
        edgecolors='none'
    )
    axes[0, 1].set_title('PCA Projection - Colored by Arousal', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Arousal', fontsize=10)
    
    # Scree plot - Explained variance
    axes[1, 0].plot(range(1, 21), pca.explained_variance_ratio_[:20], 'bo-', linewidth=2, markersize=6)
    axes[1, 0].set_title('PCA Scree Plot - Explained Variance', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Principal Component', fontsize=11)
    axes[1, 0].set_ylabel('Explained Variance Ratio', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_[:20])
    axes[1, 1].plot(range(1, 21), cumsum, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=0.8, color='g', linestyle='--', label='80% variance', linewidth=2)
    axes[1, 1].axhline(y=0.9, color='orange', linestyle='--', label='90% variance', linewidth=2)
    axes[1, 1].set_title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Components', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Explained Variance', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'pca_validation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: pca_validation_analysis.png")


def create_2d_va_space_plot(metadata_df):
    """Create 2D valence-arousal space plot."""
    print("\nCreating 2D Valence-Arousal space plot...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(metadata_df['valence'], metadata_df['arousal'], 
                        alpha=0.5, s=30, c='blue', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Valence', fontsize=14)
    ax.set_ylabel('Arousal', fontsize=14)
    ax.set_title('2D Valence-Arousal Space', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / '2d_valence_arousal_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 2d_valence_arousal_space.png")


def create_temporal_plots(temporal_df):
    """Create temporal analysis visualizations."""
    print("\nCreating temporal analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Smoothness distribution
    axes[0, 0].hist(temporal_df['embedding_smoothness'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Embedding Smoothness')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Temporal Smoothness')
    
    # Valence change distribution
    axes[0, 1].hist(temporal_df['valence_change'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Valence Change')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Valence Change per Clip')
    
    # Arousal change distribution
    axes[1, 0].hist(temporal_df['arousal_change'], bins=30, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Arousal Change')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Arousal Change per Clip')
    
    # Smoothness vs valence change
    axes[1, 1].scatter(temporal_df['embedding_smoothness'], temporal_df['valence_change'], 
                       alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Embedding Smoothness')
    axes[1, 1].set_ylabel('Valence Change')
    axes[1, 1].set_title('Smoothness vs Valence Change')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: temporal_analysis.png")


def create_summary_report(metadata_df, embeddings, pca, correlations, temporal_df):
    """Create a comprehensive summary report."""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    report = {
        'dataset_statistics': {
            'total_frames': len(metadata_df),
            'total_clips': metadata_df['clip_name'].nunique(),
            'valence_mean': float(metadata_df['valence'].mean()),
            'valence_std': float(metadata_df['valence'].std()),
            'valence_min': float(metadata_df['valence'].min()),
            'valence_max': float(metadata_df['valence'].max()),
            'arousal_mean': float(metadata_df['arousal'].mean()),
            'arousal_std': float(metadata_df['arousal'].std()),
            'arousal_min': float(metadata_df['arousal'].min()),
            'arousal_max': float(metadata_df['arousal'].max()),
        },
        'embedding_statistics': {
            'embedding_dim': embeddings.shape[1],
            'embedding_mean': float(embeddings.mean()),
            'embedding_std': float(embeddings.std()),
        },
        'pca_statistics': {
            'n_components': pca.n_components_,
            'variance_explained_by_first_2': float(pca.explained_variance_ratio_[:2].sum()),
            'variance_explained_by_first_10': float(pca.explained_variance_ratio_[:10].sum()),
            'components_for_80_percent': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1),
            'components_for_90_percent': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1),
        },
        'correlation_analysis': {
            'max_valence_pearson_raw': float(max([abs(x) for x in correlations['embedding_dim_correlations']['valence_pearson']])),
            'max_arousal_pearson_raw': float(max([abs(x) for x in correlations['embedding_dim_correlations']['arousal_pearson']])),
            'max_valence_pearson_pca': float(max([abs(x) for x in correlations['pca_correlations']['valence_pearson']])),
            'max_arousal_pearson_pca': float(max([abs(x) for x in correlations['pca_correlations']['arousal_pearson']])),
        },
        'temporal_analysis': {
            'mean_embedding_smoothness': float(temporal_df['embedding_smoothness'].mean()),
            'mean_valence_change': float(temporal_df['valence_change'].mean()),
            'mean_arousal_change': float(temporal_df['arousal_change'].mean()),
            'std_embedding_smoothness': float(temporal_df['embedding_smoothness'].std()),
        }
    }
    
    with open(RESULTS_DIR / 'comprehensive_summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("DATASET STATISTICS:")
    print(f"  Total frames: {report['dataset_statistics']['total_frames']:,}")
    print(f"  Total clips: {report['dataset_statistics']['total_clips']}")
    print(f"  Valence: μ={report['dataset_statistics']['valence_mean']:.2f}, σ={report['dataset_statistics']['valence_std']:.2f}")
    print(f"  Valence range: [{report['dataset_statistics']['valence_min']:.2f}, {report['dataset_statistics']['valence_max']:.2f}]")
    print(f"  Arousal: μ={report['dataset_statistics']['arousal_mean']:.2f}, σ={report['dataset_statistics']['arousal_std']:.2f}")
    print(f"  Arousal range: [{report['dataset_statistics']['arousal_min']:.2f}, {report['dataset_statistics']['arousal_max']:.2f}]")
    
    print("\nPCA STATISTICS:")
    print(f"  First 2 components explain: {report['pca_statistics']['variance_explained_by_first_2']:.2%}")
    print(f"  First 10 components explain: {report['pca_statistics']['variance_explained_by_first_10']:.2%}")
    print(f"  Components for 80% variance: {report['pca_statistics']['components_for_80_percent']}")
    print(f"  Components for 90% variance: {report['pca_statistics']['components_for_90_percent']}")
    
    print("\nCORRELATION ANALYSIS:")
    print(f"  Max valence correlation (raw): {report['correlation_analysis']['max_valence_pearson_raw']:.4f}")
    print(f"  Max arousal correlation (raw): {report['correlation_analysis']['max_arousal_pearson_raw']:.4f}")
    print(f"  Max valence correlation (PCA): {report['correlation_analysis']['max_valence_pearson_pca']:.4f}")
    print(f"  Max arousal correlation (PCA): {report['correlation_analysis']['max_arousal_pearson_pca']:.4f}")
    
    print("\nTEMPORAL ANALYSIS:")
    print(f"  Mean smoothness: {report['temporal_analysis']['mean_embedding_smoothness']:.4f} ± {report['temporal_analysis']['std_embedding_smoothness']:.4f}")
    print(f"  Mean valence change: {report['temporal_analysis']['mean_valence_change']:.4f}")
    print(f"  Mean arousal change: {report['temporal_analysis']['mean_arousal_change']:.4f}")
    
    print("\n" + "="*80)
    print(f"Saved comprehensive summary to: {RESULTS_DIR / 'comprehensive_summary_report.json'}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline - Complete analysis."""
    
    # Step 1: Load dataset
    df = load_afew_va_dataset(DATA_DIR)
    
    # Step 2: Check if embeddings already exist
    embeddings_pkl_path = EMBEDDINGS_DIR / 'embeddings.pkl'
    
    if embeddings_pkl_path.exists():
        print("\n" + "="*80)
        print("⚡ LOADING EXISTING EMBEDDINGS (skipping extraction)")
        print("="*80)
        print(f"Found existing embeddings at: {embeddings_pkl_path}")
        
        # Load from pickle file
        with open(embeddings_pkl_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        embeddings = saved_data['embeddings']
        
        # Load metadata
        if 'metadata' in saved_data:
            if isinstance(saved_data['metadata'], pd.DataFrame):
                metadata_df = saved_data['metadata']
            else:
                metadata_df = pd.DataFrame(saved_data['metadata'])
        else:
            metadata_df = pd.read_csv(EMBEDDINGS_DIR / 'metadata.csv')
        
        print(f"✅ Loaded embeddings shape: {embeddings.shape}")
        print(f"✅ Loaded metadata: {len(metadata_df)} frames")
        print("⏭️  Skipping BLIP-2 model loading and extraction (saving time!)...")
        
    else:
        print("\n" + "="*80)
        print("No existing embeddings found - will extract from scratch")
        print("="*80)
        
        # Step 2: Initialize BLIP-2 model (only if needed)
        print("\n" + "="*80)
        print("LOADING BLIP-2 MODEL")
        print("="*80)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = Blip2Processor.from_pretrained(CONFIG['model_name'])
        model = Blip2Model.from_pretrained(CONFIG['model_name'])
        
        if CONFIG['use_fp16'] and device == 'cuda':
            model = model.half()
        
        print(f"Model loaded on: {device}")
        
        # Step 3: Extract embeddings
        embeddings, metadata_df = extract_embeddings(model, processor, df, 
                                                     CONFIG['batch_size'], device)
        
        # Free up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Step 4: Dimensionality reduction
    print("\n" + "="*80)
    print("PERFORMING DIMENSIONALITY REDUCTION")
    print("="*80)
    
    embeddings_2d_umap = apply_umap(embeddings, CONFIG)
    embeddings_2d_tsne = apply_tsne(embeddings)
    pca_embeddings, pca = apply_pca(embeddings, CONFIG['pca_n_components'])
    
    # Step 5: Comprehensive correlation analysis
    correlations = compute_comprehensive_correlations(
        embeddings, pca_embeddings, embeddings_2d_umap, metadata_df
    )
    
    # Step 6: Temporal analysis
    temporal_df = compute_temporal_analysis(embeddings, metadata_df)
    
    # Step 7: Create ALL visualizations
    print("\n" + "="*80)
    print("CREATING ALL VISUALIZATIONS")
    print("="*80)
    
    create_umap_primary_visualization(embeddings_2d_umap, metadata_df)
    create_pca_validation_plots(pca_embeddings, pca, metadata_df)
    create_2d_va_space_plot(metadata_df)
    create_temporal_plots(temporal_df)
    
    # Step 8: Generate comprehensive summary
    create_summary_report(metadata_df, embeddings, pca, correlations, temporal_df)
    
    print("\n" + "="*80)
    print("✅ COMPLETE ANALYSIS PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll outputs saved to: {PROJECT_ROOT}")
    print(f"\n📁 Output Structure:")
    print(f"  📊 Embeddings: {EMBEDDINGS_DIR}")
    print(f"     - blip2_embeddings.h5 (HDF5 format)")
    print(f"     - embeddings.pkl (Pickle format)")
    print(f"     - metadata.csv")
    print(f"\n  🎨 Visualizations: {VISUALIZATIONS_DIR}")
    print(f"     - umap_primary_visualization.png")
    print(f"     - pca_validation_analysis.png")
    print(f"     - 2d_valence_arousal_space.png")
    print(f"     - temporal_analysis.png")
    print(f"\n  📈 Results: {RESULTS_DIR}")
    print(f"     - umap_embeddings_2d.npy")
    print(f"     - tsne_embeddings_2d.npy")
    print(f"     - pca_embeddings.npy")
    print(f"     - pca_explained_variance.npy")
    print(f"     - comprehensive_correlations.json")
    print(f"     - temporal_analysis.csv")
    print(f"     - comprehensive_summary_report.json")
    print(f"\n  💾 Checkpoints: {CHECKPOINTS_DIR}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
