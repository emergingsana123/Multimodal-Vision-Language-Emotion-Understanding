#!/usr/bin/env python3
"""
SUPPLEMENTARY: Enhanced PCA and Correlation Analysis
This adds the detailed PCA scree plots and correlation analysis from the notebook
Run this AFTER the main afew_va_analysis.py completes
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

#============================================================================
# CONFIGURATION - Match your main script paths
# ===========================================================================

PROJECT_ROOT = Path('/home_local/sdeshmukh/Multimodal-Vision-Language-Emotion-Understanding/outputs')
EMBEDDINGS_DIR = PROJECT_ROOT / 'embeddings'
VISUALIZATIONS_DIR = PROJECT_ROOT / 'visualizations'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("="*80)
print("SUPPLEMENTARY PCA & CORRELATION ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading embeddings and metadata...")
with h5py.File(EMBEDDINGS_DIR / 'blip2_embeddings.h5', 'r') as f:
    embeddings = f['embeddings'][:]
    valences = f['valence'][:]
    arousals = f['arousal'][:]

metadata_df = pd.read_csv(EMBEDDINGS_DIR / 'metadata.csv')

print(f"Loaded {len(embeddings)} samples")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Valence range: [{valences.min():.2f}, {valences.max():.2f}]")
print(f"Arousal range: [{arousals.min():.2f}, {arousals.max():.2f}]")

# ============================================================================
# STANDARDIZE AND RUN PCA
# ============================================================================

print("\nStandardizing embeddings...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

print("\nRunning PCA with 50 components...")
pca = PCA(n_components=50, random_state=42)
pca_embeddings = pca.fit_transform(embeddings_scaled)

print(f"PCA complete! Shape: {pca_embeddings.shape}")
print(f"Explained variance by first 10 components: {pca.explained_variance_ratio_[:10].sum():.2%}")

# ============================================================================
# COMPREHENSIVE PCA VISUALIZATION (4-panel plot from notebook)
# ============================================================================

print("\n" + "="*80)
print("CREATING COMPREHENSIVE PCA VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: PCA colored by Valence
scatter1 = axes[0, 0].scatter(
    pca_embeddings[:, 0],
    pca_embeddings[:, 1],
    c=valences,
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

# Panel 2: PCA colored by Arousal
scatter2 = axes[0, 1].scatter(
    pca_embeddings[:, 0],
    pca_embeddings[:, 1],
    c=arousals,
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

# Panel 3: Scree plot - Explained variance
axes[1, 0].plot(range(1, 21), pca.explained_variance_ratio_[:20], 'bo-', linewidth=2, markersize=6)
axes[1, 0].set_title('PCA Scree Plot - Explained Variance', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Principal Component', fontsize=11)
axes[1, 0].set_ylabel('Explained Variance Ratio', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Cumulative explained variance
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
plt.savefig(VISUALIZATIONS_DIR / 'pca_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: pca_comprehensive_analysis.png")

# ============================================================================
# DETAILED CORRELATION ANALYSIS (Pearson + Spearman)
# ============================================================================

print("\n" + "="*80)
print("DETAILED CORRELATION ANALYSIS")
print("="*80)

# Load UMAP embeddings if they exist
umap_path = RESULTS_DIR / 'umap_embeddings_2d.npy'
if umap_path.exists():
    umap_embeddings = np.load(umap_path)
    has_umap = True
else:
    print("Warning: UMAP embeddings not found, skipping UMAP correlations")
    has_umap = False

# PCA correlations
print("\nPCA Component Correlations:")
print("-" * 60)

correlation_results = []

for i in range(min(10, pca_embeddings.shape[1])):
    valence_corr_p, valence_p = pearsonr(pca_embeddings[:, i], valences)
    arousal_corr_p, arousal_p = pearsonr(pca_embeddings[:, i], arousals)
    valence_corr_s, _ = spearmanr(pca_embeddings[:, i], valences)
    arousal_corr_s, _ = spearmanr(pca_embeddings[:, i], arousals)
    
    correlation_results.append({
        'component': f'PC{i+1}',
        'valence_pearson': valence_corr_p,
        'valence_pvalue': valence_p,
        'valence_spearman': valence_corr_s,
        'arousal_pearson': arousal_corr_p,
        'arousal_pvalue': arousal_p,
        'arousal_spearman': arousal_corr_s,
    })
    
    if i < 5:  # Print details for first 5
        print(f"PC{i+1}:")
        print(f"  Valence: Pearson={valence_corr_p:+.3f} (p={valence_p:.2e}), Spearman={valence_corr_s:+.3f}")
        print(f"  Arousal: Pearson={arousal_corr_p:+.3f} (p={arousal_p:.2e}), Spearman={arousal_corr_s:+.3f}")

# UMAP correlations
if has_umap:
    print("\n" + "-" * 60)
    print("UMAP Dimension Correlations:")
    print("-" * 60)
    
    for i in range(2):
        valence_corr_p, valence_p = pearsonr(umap_embeddings[:, i], valences)
        arousal_corr_p, arousal_p = pearsonr(umap_embeddings[:, i], arousals)
        valence_corr_s, _ = spearmanr(umap_embeddings[:, i], valences)
        arousal_corr_s, _ = spearmanr(umap_embeddings[:, i], arousals)
        
        correlation_results.append({
            'component': f'UMAP{i+1}',
            'valence_pearson': valence_corr_p,
            'valence_pvalue': valence_p,
            'valence_spearman': valence_corr_s,
            'arousal_pearson': arousal_corr_p,
            'arousal_pvalue': arousal_p,
            'arousal_spearman': arousal_corr_s,
        })
        
        print(f"UMAP Dim {i+1}:")
        print(f"  Valence: Pearson={valence_corr_p:+.3f} (p={valence_p:.2e}), Spearman={valence_corr_s:+.3f}")
        print(f"  Arousal: Pearson={arousal_corr_p:+.3f} (p={arousal_p:.2e}), Spearman={arousal_corr_s:+.3f}")

# Save correlation results
corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv(RESULTS_DIR / 'detailed_correlation_analysis.csv', index=False)

print(f"\n✅ Saved: detailed_correlation_analysis.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total samples: {len(embeddings):,}")
print(f"  Original embedding dimension: {embeddings.shape[1]}")
print(f"  Valence distribution: μ={valences.mean():.2f}, σ={valences.std():.2f}")
print(f"  Arousal distribution: μ={arousals.mean():.2f}, σ={arousals.std():.2f}")

print(f"\nPCA Results:")
print(f"  First 2 components explain: {pca.explained_variance_ratio_[:2].sum():.2%} of variance")
print(f"  First 10 components explain: {pca.explained_variance_ratio_[:10].sum():.2%} of variance")
print(f"  Components needed for 80% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1}")
print(f"  Components needed for 90% variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1}")

print(f"\nCorrelation Insights:")
pca_corrs = corr_df[corr_df['component'].str.startswith('PC')]
print(f"  Strongest PCA-Valence correlation: {pca_corrs['valence_pearson'].abs().max():.4f} ({pca_corrs.loc[pca_corrs['valence_pearson'].abs().idxmax(), 'component']})")
print(f"  Strongest PCA-Arousal correlation: {pca_corrs['arousal_pearson'].abs().max():.4f} ({pca_corrs.loc[pca_corrs['arousal_pearson'].abs().idxmax(), 'component']})")

print("\n" + "="*80)
print("✅ SUPPLEMENTARY ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. pca_comprehensive_analysis.png - 4-panel PCA visualization with scree plots")
print("  2. detailed_correlation_analysis.csv - Complete correlation table")
print(f"\nCheck {VISUALIZATIONS_DIR} for results!")