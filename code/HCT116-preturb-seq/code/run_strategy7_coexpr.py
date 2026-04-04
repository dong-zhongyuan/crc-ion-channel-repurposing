#!/usr/bin/env python3
"""Strategy 7: Co-expression Network Analysis (optimized)"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, stats
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

print("Loading data...", flush=True)
adata = ad.read_h5ad(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'subset_6pairs_allgenes.h5ad'))
X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
gene2idx = {g: i for i, g in enumerate(adata.var_names)}
print(f"Loaded: {adata.shape}", flush=True)

ctrl_mask = (adata.obs['gene_target'] == 'Non-Targeting').values
ctrl_X = X[ctrl_mask]

# ============================================================
# Co-expression analysis (simplified, efficient)
# ============================================================
print("\n=== Strategy 7: Co-expression Network ===", flush=True)
coexpr_results = []
module_results = []

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---", flush=True)
    
    if ko not in gene2idx or target not in gene2idx:
        print(f"  Gene not found", flush=True)
        continue
    
    ko_mask = (adata.obs['gene_target'] == ko).values
    ko_X = X[ko_mask]
    
    # Get expression vectors
    ko_expr_ctrl = np.asarray(ctrl_X[:, gene2idx[ko]].todense()).flatten().astype(float)
    target_expr_ctrl = np.asarray(ctrl_X[:, gene2idx[target]].todense()).flatten().astype(float)
    
    ko_expr_ko = np.asarray(ko_X[:, gene2idx[ko]].todense()).flatten().astype(float)
    target_expr_ko = np.asarray(ko_X[:, gene2idx[target]].todense()).flatten().astype(float)
    
    # Spearman correlation
    if np.std(ko_expr_ctrl) > 0 and np.std(target_expr_ctrl) > 0:
        rho_ctrl, pval_ctrl = stats.spearmanr(ko_expr_ctrl, target_expr_ctrl)
    else:
        rho_ctrl, pval_ctrl = 0, 1.0
    
    if np.std(ko_expr_ko) > 0 and np.std(target_expr_ko) > 0:
        rho_ko, pval_ko = stats.spearmanr(ko_expr_ko, target_expr_ko)
    else:
        rho_ko, pval_ko = 0, 1.0
    
    print(f"  Control: rho={rho_ctrl:.4f}, p={pval_ctrl:.4e}", flush=True)
    print(f"  KO:      rho={rho_ko:.4f}, p={pval_ko:.4e}", flush=True)
    
    coexpr_results.append({
        'ko': ko, 'target': target,
        'ctrl_rho': rho_ctrl, 'ctrl_pval': pval_ctrl,
        'ko_rho': rho_ko, 'ko_pval': pval_ko,
        'delta_rho': rho_ko - rho_ctrl,
    })
    
    # Module analysis: find top 20 genes correlated with KO gene in controls
    # Use a subset of genes for speed
    print(f"  Computing module...", flush=True)
    
    # Sample 500 genes with highest variance
    gene_vars = np.asarray(ctrl_X.power(2).mean(axis=0) - np.power(ctrl_X.mean(axis=0), 2)).flatten()
    top_var_idx = np.argsort(gene_vars)[-500:]
    
    # Make sure KO and target genes are included
    for g in [ko, target]:
        gidx = gene2idx[g]
        if gidx not in top_var_idx:
            top_var_idx = np.append(top_var_idx, gidx)
    
    # Compute correlations with KO gene
    ko_corrs = {}
    for gidx in top_var_idx:
        gname = adata.var_names[gidx]
        expr = np.asarray(ctrl_X[:, gidx].todense()).flatten().astype(float)
        if np.std(expr) > 0 and np.std(ko_expr_ctrl) > 0:
            r, p = stats.spearmanr(ko_expr_ctrl, expr)
            ko_corrs[gname] = r
    
    # Sort and get top module
    sorted_corrs = sorted(ko_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_module_genes = [g for g, r in sorted_corrs[:20] if g != ko][:20]
    
    # Check target position
    target_corr = ko_corrs.get(target, 0)
    target_rank = None
    for rank, (g, r) in enumerate(sorted_corrs):
        if g == target:
            target_rank = rank
            break
    
    print(f"  Target {target} correlation with {ko}: {target_corr:.4f}, rank: {target_rank}/{len(sorted_corrs)}", flush=True)
    
    # Module disruption: compare mean correlation in control vs KO
    module_genes_idx = [gene2idx[g] for g in top_module_genes if g in gene2idx]
    
    if len(module_genes_idx) >= 5:
        # Control correlation matrix
        ctrl_module = np.asarray(ctrl_X[:, module_genes_idx].todense())
        ctrl_corr = np.corrcoef(ctrl_module.T)
        ctrl_mean = np.mean(np.abs(ctrl_corr[np.triu_indices(len(module_genes_idx), k=1)]))
        
        # KO correlation matrix
        ko_module = np.asarray(ko_X[:, module_genes_idx].todense())
        ko_corr = np.corrcoef(ko_module.T)
        ko_mean = np.mean(np.abs(ko_corr[np.triu_indices(len(module_genes_idx), k=1)]))
        
        disruption = ctrl_mean - ko_mean
        print(f"  Module |corr|: ctrl={ctrl_mean:.4f}, KO={ko_mean:.4f}, disruption={disruption:.4f}", flush=True)
    else:
        ctrl_mean, ko_mean, disruption = 0, 0, 0
    
    module_results.append({
        'ko': ko, 'target': target,
        'module_genes': ','.join(top_module_genes[:10]),
        'ctrl_mean_corr': ctrl_mean, 'ko_mean_corr': ko_mean,
        'disruption_score': disruption,
        'target_in_module': target in top_module_genes,
        'target_corr_with_ko': target_corr,
        'target_rank': target_rank,
    })

coexpr_df = pd.DataFrame(coexpr_results)
coexpr_df.to_csv(f'{OUT_TBL}/strategy7_coexpression.csv', index=False)
module_df = pd.DataFrame(module_results)
module_df.to_csv(f'{OUT_TBL}/strategy7_modules.csv', index=False)

print("\nCo-expression results:", flush=True)
print(coexpr_df.to_string(), flush=True)
print("\nModule results:", flush=True)
print(module_df[['ko','target','disruption_score','target_in_module','target_corr_with_ko']].to_string(), flush=True)

# ============================================================
# FIGURE 9: Co-expression module changes
# ============================================================
print("\nGenerating Figure 9...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    r = coexpr_df[(coexpr_df['ko'] == ko) & (coexpr_df['target'] == target)]
    m = module_df[(module_df['ko'] == ko) & (module_df['target'] == target)]
    
    if len(r) == 0:
        continue
    
    r = r.iloc[0]
    
    # Two-panel: correlation + module disruption
    x = [0, 1]
    heights = [r['ctrl_rho'], r['ko_rho']]
    colors = ['#4DBBD5', '#E64B35']
    bars = ax.bar(x, heights, color=colors, width=0.6, edgecolor='none', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Control', f'{ko} KO'], fontsize=7)
    ax.set_ylabel(f'Spearman ρ ({ko} vs {target})', fontsize=6)
    
    if len(m) > 0:
        mr = m.iloc[0]
        info = f"Module disruption: {mr['disruption_score']:.3f}\nTarget in module: {mr['target_in_module']}"
        ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', alpha=0.8))
    
    ax.axhline(0, color='black', linewidth=0.3)
    ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Strategy 7: Co-expression Network Changes After KO', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig9_coexpression.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig9_coexpression.pdf', bbox_inches='tight')
plt.close()
print("Figure 9 saved.", flush=True)
print("\n✓ Strategy 7 complete", flush=True)
