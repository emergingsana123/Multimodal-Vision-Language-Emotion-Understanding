#!/usr/bin/env python3
"""
AFEW-VA Dataset Analysis Script - Modified for University VM
Extracts embeddings, performs analysis, and generates visualizations
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

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
#   directory.mkdir(parents=True, exist_ok=True)

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
    'checkpoint_frequency': 100,
}

# Save configuration
with open(PROJECT_ROOT / 'config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("="*80)
print("AFEW-VA Analysis Pipeline - University VM Version")
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
    
    # Standardize embeddings for better visualization
    print("Standardizing embeddings...")
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
    
    return embeddings_2d, embeddings_scaled


def apply_pca(embeddings_scaled, n_components=50):
    """Apply PCA dimensionality reduction with comprehensive analysis."""
    print("\n" + "="*80)
    print("APPLYING PCA ANALYSIS")
    print("="*80)
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_embeddings = pca.fit_transform(embeddings_scaled)
    
    print(f"PCA complete! Shape: {pca_embeddings.shape}")
    print(f"Explained variance by first 2 components: {pca.explained_variance_ratio_[:2].sum():.2%}")
    print(f"Explained variance by first 10 components: {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    # Calculate components needed for variance thresholds
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_80 = np.argmax(cumsum >= 0.8) + 1
    n_90 = np.argmax(cumsum >= 0.9) + 1
    
    print(f"Components for 80% variance: {n_80}")
    print(f"Components for 90% variance: {n_90}")
    
    np.save(RESULTS_DIR / 'pca_embeddings_2d.npy', pca_embeddings[:, :2])
    
    return pca_embeddings, pca


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_correlations(embeddings_2d_umap, pca_embeddings, metadata_df):
    """Compute comprehensive correlation analysis between embeddings and valence/arousal."""
    print("\n" + "="*80)
    print("COMPUTING CORRELATION ANALYSIS")
    print("="*80)
    
    results = {
        'pca_valence_pearson': [],
        'pca_arousal_pearson': [],
        'pca_valence_spearman': [],
        'pca_arousal_spearman': [],
        'pca_valence_pvalue': [],
        'pca_arousal_pvalue': []
    }
    
    # PCA correlations (first 10 components)
    print("\nPCA Component Correlations:")
    print("-" * 50)
    for i in range(min(10, pca_embeddings.shape[1])):
        val_corr_p, val_p = pearsonr(pca_embeddings[:, i], metadata_df['valence'])
        aro_corr_p, aro_p = pearsonr(pca_embeddings[:, i], metadata_df['arousal'])
        val_corr_s, _ = spearmanr(pca_embeddings[:, i], metadata_df['valence'])
        aro_corr_s, _ = spearmanr(pca_embeddings[:, i], metadata_df['arousal'])
        
        results['pca_valence_pearson'].append(val_corr_p)
        results['pca_arousal_pearson'].append(aro_corr_p)
        results['pca_valence_spearman'].append(val_corr_s)
        results['pca_arousal_spearman'].append(aro_corr_s)
        results['pca_valence_pvalue'].append(val_p)
        results['pca_arousal_pvalue'].append(aro_p)
        
        if i < 5:  # Print first 5
            print(f"PC{i+1}:")
            print(f"  Valence: Pearson={val_corr_p:+.3f} (p={val_p:.2e}), Spearman={val_corr_s:+.3f}")
            print(f"  Arousal: Pearson={aro_corr_p:+.3f} (p={aro_p:.2e}), Spearman={aro_corr_s:+.3f}")
    
    # UMAP correlations
    print("\n" + "-" * 50)
    print("UMAP Dimension Correlations:")
    print("-" * 50)
    results['umap_correlations'] = []
    for i in range(2):
        val_corr_p, val_p = pearsonr(embeddings_2d_umap[:, i], metadata_df['valence'])
        aro_corr_p, aro_p = pearsonr(embeddings_2d_umap[:, i], metadata_df['arousal'])
        val_corr_s, _ = spearmanr(embeddings_2d_umap[:, i], metadata_df['valence'])
        aro_corr_s, _ = spearmanr(embeddings_2d_umap[:, i], metadata_df['arousal'])
        
        results['umap_correlations'].append({
            'dimension': i+1,
            'valence_pearson': val_corr_p,
            'arousal_pearson': aro_corr_p,
            'valence_spearman': val_corr_s,
            'arousal_spearman': aro_corr_s,
            'valence_pvalue': val_p,
            'arousal_pvalue': aro_p
        })
        
        print(f"UMAP Dim {i+1}:")
        print(f"  Valence: Pearson={val_corr_p:+.3f} (p={val_p:.2e}), Spearman={val_corr_s:+.3f}")
        print(f"  Arousal: Pearson={aro_corr_p:+.3f} (p={aro_p:.2e}), Spearman={aro_corr_s:+.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'correlation_analysis.csv', index=False)
    
    print(f"\nMax PCA valence correlation: {max(np.abs(results['pca_valence_pearson'])):.4f}")
    print(f"Max PCA arousal correlation: {max(np.abs(results['pca_arousal_pearson'])):.4f}")
    
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
                'mean_arousal': clip_meta['arousal'].mean()
            })
    
    temporal_df = pd.DataFrame(temporal_stats)
    temporal_df.to_csv(RESULTS_DIR / 'temporal_analysis.csv', index=False)
    
    print(f"Analyzed {len(temporal_df)} clips")
    print(f"Mean embedding smoothness: {temporal_df['embedding_smoothness'].mean():.4f}")
    
    return temporal_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_valence_arousal_scatter(embeddings_2d, metadata_df):
    """Create scatter plot colored by valence and arousal."""
    print("\n" + "="*80)
    print("CREATING VALENCE-AROUSAL VISUALIZATIONS")
    print("="*80)
    
    # Valence plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=metadata_df['valence'], cmap='RdYlGn', 
                               alpha=0.6, s=20)
    axes[0].set_title('UMAP Projection colored by Valence', fontsize=14)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=axes[0], label='Valence')
    
    scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=metadata_df['arousal'], cmap='coolwarm', 
                               alpha=0.6, s=20)
    axes[1].set_title('UMAP Projection colored by Arousal', fontsize=14)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[1], label='Arousal')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'valence_arousal_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: valence_arousal_scatter.png")


def create_2d_va_space_plot(metadata_df):
    """Create 2D valence-arousal space plot."""
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
    
    print(f"Saved: 2d_valence_arousal_space.png")


def create_correlation_heatmap(correlations):
    """Create heatmap of dimension correlations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    corr_matrix = np.array([correlations['valence_correlations'], 
                           correlations['arousal_correlations']])
    
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                yticklabels=['Valence', 'Arousal'],
                xticklabels=[f'Dim{i}' for i in range(corr_matrix.shape[1])],
                ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Embedding Dimension Correlations with V/A', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'dimension_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: dimension_correlations.png")


def create_temporal_plots(temporal_df):
    """Create temporal analysis visualizations."""
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
    
    print(f"Saved: temporal_analysis.png")


def create_summary_report(metadata_df, embeddings, correlations, temporal_df):
    """Create a summary report of all analyses."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report = {
        'dataset_statistics': {
            'total_frames': len(metadata_df),
            'total_clips': metadata_df['clip_name'].nunique(),
            'valence_mean': float(metadata_df['valence'].mean()),
            'valence_std': float(metadata_df['valence'].std()),
            'arousal_mean': float(metadata_df['arousal'].mean()),
            'arousal_std': float(metadata_df['arousal'].std()),
        },
        'embedding_statistics': {
            'embedding_dim': embeddings.shape[1],
            'embedding_mean': float(embeddings.mean()),
            'embedding_std': float(embeddings.std()),
        },
        'correlation_analysis': {
            'max_valence_correlation': float(max(correlations['valence_correlations'])),
            'max_arousal_correlation': float(max(correlations['arousal_correlations'])),
            'mean_abs_valence_correlation': float(np.mean(np.abs(correlations['valence_correlations']))),
            'mean_abs_arousal_correlation': float(np.mean(np.abs(correlations['arousal_correlations']))),
        },
        'temporal_analysis': {
            'mean_embedding_smoothness': float(temporal_df['embedding_smoothness'].mean()),
            'mean_valence_change': float(temporal_df['valence_change'].mean()),
            'mean_arousal_change': float(temporal_df['arousal_change'].mean()),
        }
    }
    
    with open(RESULTS_DIR / 'summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nDATASET STATISTICS:")
    print(f"  Total frames: {report['dataset_statistics']['total_frames']}")
    print(f"  Total clips: {report['dataset_statistics']['total_clips']}")
    print(f"  Valence: {report['dataset_statistics']['valence_mean']:.3f} ± {report['dataset_statistics']['valence_std']:.3f}")
    print(f"  Arousal: {report['dataset_statistics']['arousal_mean']:.3f} ± {report['dataset_statistics']['arousal_std']:.3f}")
    
    print("\nCORRELATION ANALYSIS:")
    print(f"  Max valence correlation: {report['correlation_analysis']['max_valence_correlation']:.4f}")
    print(f"  Max arousal correlation: {report['correlation_analysis']['max_arousal_correlation']:.4f}")
    
    print("\nTEMPORAL ANALYSIS:")
    print(f"  Mean smoothness: {report['temporal_analysis']['mean_embedding_smoothness']:.4f}")
    print(f"  Mean valence change: {report['temporal_analysis']['mean_valence_change']:.4f}")
    print(f"  Mean arousal change: {report['temporal_analysis']['mean_arousal_change']:.4f}")
    
    print(f"\nSaved summary report to: {RESULTS_DIR / 'summary_report.json'}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Step 1: Load dataset
    df = load_afew_va_dataset(DATA_DIR)
    
    # Step 2: Initialize BLIP-2 model
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
    embeddings_2d_umap = apply_umap(embeddings, CONFIG)
    embeddings_2d_tsne = apply_tsne(embeddings)
    
    # Step 5: Compute analyses
    correlations = compute_valence_arousal_correlation(embeddings, metadata_df)
    temporal_df = compute_temporal_analysis(embeddings, metadata_df)
    
    # Step 6: Create visualizations
    create_valence_arousal_scatter(embeddings_2d_umap, metadata_df)
    create_2d_va_space_plot(metadata_df)
    create_correlation_heatmap(correlations)
    create_temporal_plots(temporal_df)
    
    # Step 7: Generate summary report
    create_summary_report(metadata_df, embeddings, correlations, temporal_df)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll outputs saved to: {PROJECT_ROOT}")
    print(f"  - Embeddings: {EMBEDDINGS_DIR}")
    print(f"  - Visualizations: {VISUALIZATIONS_DIR}")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()