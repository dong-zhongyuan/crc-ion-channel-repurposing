#!/usr/bin/env python3
"""Strategy 6: Perturbation Effect Score (Mixscape-like)"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

PAIRS = [
    ('GALK1', 'KCNA5'), ('LSM7', 'CLIC1'), ('RIPK2', 'CFTR'),
    ('TRMT112', 'CFTR'), ('RPS21', 'KCNQ2'), ('EXOSC5', 'AQP9'),
]

_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(_STEP_DIR, 'result', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)
OUT_TBL = os.path.join(_STEP_DIR, 'result', 'tables')
os.makedirs(OUT_TBL, exist_ok=True)

print("Loading data...")
adata = ad.read_h5ad(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'subset_6pairs_allgenes.h5ad'))
X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
gene2idx = {g: i for i, g in enumerate(adata.var_names)}

# Normalize for PCA
print("Normalizing for PCA...")
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)
sc.pp.highly_variable_genes(adata_norm, n_top_genes=2000, flavor='seurat_v3')
sc.pp.pca(adata_norm, n_comps=50, use_highly_variable=True)

# ============================================================
# Perturbation scoring
# ============================================================
print("\n=== Strategy 6: Perturbation Effect Score ===")
perturb_results = []

ctrl_mask = (adata.obs['gene_target'] == 'Non-Targeting').values
ctrl_pca = adata_norm.obsm['X_pca'][ctrl_mask]
ctrl_centroid = ctrl_pca.mean(axis=0)

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---")
    ko_mask = (adata.obs['gene_target'] == ko).values
    ko_pca = adata_norm.obsm['X_pca'][ko_mask]
    
    # Compute perturbation score: distance from control centroid
    ko_distances = np.sqrt(((ko_pca - ctrl_centroid) ** 2).sum(axis=1))
    ctrl_distances = np.sqrt(((ctrl_pca - ctrl_centroid) ** 2).sum(axis=1))
    
    # Z-score relative to control distribution
    ctrl_mean_dist = ctrl_distances.mean()
    ctrl_std_dist = ctrl_distances.std()
    ko_zscores = (ko_distances - ctrl_mean_dist) / ctrl_std_dist
    
    print(f"  KO cells: {ko_mask.sum()}, mean z-score: {ko_zscores.mean():.3f}")
    
    # Classify responders vs non-responders using GMM
    if len(ko_zscores) >= 10:
        gmm = GaussianMixture(n_components=2, random_state=42)
        labels = gmm.fit_predict(ko_zscores.reshape(-1, 1))
        
        # The cluster with higher mean z-score = responders
        mean0 = ko_zscores[labels == 0].mean()
        mean1 = ko_zscores[labels == 1].mean()
        responder_label = 0 if mean0 > mean1 else 1
        
        responder_mask = labels == responder_label
        n_resp = responder_mask.sum()
        n_nonresp = (~responder_mask).sum()
        print(f"  Responders: {n_resp}, Non-responders: {n_nonresp}")
    else:
        # Too few cells for GMM, use median split
        median_z = np.median(ko_zscores)
        responder_mask = ko_zscores > median_z
        n_resp = responder_mask.sum()
        n_nonresp = (~responder_mask).sum()
        print(f"  Median split - Responders: {n_resp}, Non-responders: {n_nonresp}")
    
    # Get target gene expression in responders vs non-responders vs control
    if target in gene2idx:
        target_idx = gene2idx[target]
        ko_target = np.asarray(X[ko_mask, target_idx].todense()).flatten()
        ctrl_target = np.asarray(X[ctrl_mask, target_idx].todense()).flatten()
        
        resp_target = ko_target[responder_mask]
        nonresp_target = ko_target[~responder_mask]
        
        # Stats
        for group_name, group_vals in [('responder', resp_target), ('non-responder', nonresp_target), ('all_KO', ko_target)]:
            if len(group_vals) > 0 and len(ctrl_target) > 0:
                try:
                    stat, pval = stats.mannwhitneyu(group_vals, ctrl_target, alternative='two-sided')
                except:
                    pval = np.nan
                lfc = np.log2((np.mean(group_vals) + 0.01) / (np.mean(ctrl_target) + 0.01))
            else:
                pval, lfc = np.nan, np.nan
            
            perturb_results.append({
                'ko': ko, 'target': target, 'group': group_name,
                'n_cells': len(group_vals),
                'mean_target': np.mean(group_vals) if len(group_vals) > 0 else 0,
                'detect_rate': (group_vals > 0).mean() if len(group_vals) > 0 else 0,
                'log2FC_vs_ctrl': lfc, 'pvalue': pval,
                'mean_perturb_zscore': ko_zscores[responder_mask].mean() if group_name == 'responder' else (ko_zscores[~responder_mask].mean() if group_name == 'non-responder' else ko_zscores.mean()),
            })
        
        # Control group
        perturb_results.append({
            'ko': ko, 'target': target, 'group': 'control',
            'n_cells': len(ctrl_target),
            'mean_target': np.mean(ctrl_target),
            'detect_rate': (ctrl_target > 0).mean(),
            'log2FC_vs_ctrl': 0, 'pvalue': 1.0,
            'mean_perturb_zscore': 0,
        })
        
        print(f"  Target {target} in responders: mean={np.mean(resp_target):.4f}, detect={((resp_target>0).mean()):.3f}")
        print(f"  Target {target} in non-resp:   mean={np.mean(nonresp_target):.4f}, detect={((nonresp_target>0).mean()):.3f}")
        print(f"  Target {target} in control:    mean={np.mean(ctrl_target):.4f}, detect={((ctrl_target>0).mean()):.3f}")

perturb_df = pd.DataFrame(perturb_results)
perturb_df.to_csv(f'{OUT_TBL}/strategy6_perturbation.csv', index=False)

# ============================================================
# FIGURE 8: Perturbation score
# ============================================================
print("\nGenerating Figure 8: Perturbation effect score...")

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    pair_data = perturb_df[(perturb_df['ko'] == ko) & (perturb_df['target'] == target)]
    
    if len(pair_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue
    
    groups = ['control', 'non-responder', 'responder']
    group_labels = ['Control', 'Non-resp', 'Responder']
    colors_g = ['#4DBBD5', '#999999', '#E64B35']
    
    positions = []
    data_vals = []
    
    for j, grp in enumerate(groups):
        grp_data = pair_data[pair_data['group'] == grp]
        if len(grp_data) > 0:
            positions.append(j)
            data_vals.append(grp_data.iloc[0])
    
    # Bar plot of detection rate and mean expression
    x = np.arange(len(positions))
    detect_rates = [d['detect_rate'] for d in data_vals]
    mean_exprs = [d['mean_target'] for d in data_vals]
    
    width = 0.35
    bars1 = ax.bar(x - width/2, detect_rates, width, label='Detection rate', color='#4DBBD5', alpha=0.7)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, mean_exprs, width, label='Mean expression', color='#E64B35', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([group_labels[p] for p in positions], fontsize=6, rotation=15)
    ax.set_ylabel('Detection rate', fontsize=7, color='#4DBBD5')
    ax2.set_ylabel('Mean expression', fontsize=7, color='#E64B35')
    
    # Add p-values
    for j, d in enumerate(data_vals):
        if d['group'] != 'control' and pd.notna(d['pvalue']):
            pv = d['pvalue']
            if pv < 0.05:
                ax.text(j, max(detect_rates) * 1.05, f'p={pv:.2e}', ha='center', fontsize=5, color='#E64B35')
    
    ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    
    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc='upper right')

plt.suptitle('Strategy 6: Perturbation Effect Score (Responder vs Non-responder)', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig8_perturbation.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig8_perturbation.pdf', bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

print("\n✓ Strategy 6 complete")
