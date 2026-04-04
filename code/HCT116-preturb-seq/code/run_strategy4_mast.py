#!/usr/bin/env python3
"""Strategy 4: Zero-inflated model (MAST-like) - Detection rate vs Expression"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.api import add_constant
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

GENE_FAMILIES = {
    'KCNA5': ['KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6','KCNA7','KCNA10',
              'KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4','KCND1','KCND2','KCND3'],
    'CLIC1': ['CLIC1','CLIC2','CLIC3','CLIC4','CLIC5','CLIC6',
              'CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7'],
    'CFTR': ['CFTR','ABCC1','ABCC2','ABCC3','ABCC4','ABCC5','ABCC6',
             'ABCB1','ABCB4','ABCG2','CLCN2','CLCN4','ANO1'],
    'KCNQ2': ['KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5',
              'KCNA1','KCNA2','KCNA5','KCNB1','KCNC1','KCND1','KCND2'],
    'AQP9': ['AQP1','AQP2','AQP3','AQP4','AQP5','AQP6','AQP7','AQP8','AQP9','AQP10','AQP11'],
}

_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(_STEP_DIR, 'result', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)
OUT_TBL = os.path.join(_STEP_DIR, 'result', 'tables')
os.makedirs(OUT_TBL, exist_ok=True)

print("Loading data...")
adata = ad.read_h5ad(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'subset_6pairs_allgenes.h5ad'))
X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
gene2idx = {g: i for i, g in enumerate(adata.var_names)}

def get_expr(gene, mask):
    if gene not in gene2idx:
        return np.array([])
    return np.asarray(X[mask, gene2idx[gene]].todense()).flatten()

# ============================================================
# MAST-like analysis: separate detection rate and expression level
# ============================================================
print("\n=== MAST-like Zero-Inflated Analysis ===")
mast_results = []

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---")
    ko_mask = (adata.obs['gene_target'] == ko).values
    ctrl_mask = (adata.obs['gene_target'] == 'Non-Targeting').values
    
    # Analyze target gene and family members
    genes_to_test = [target] + [g for g in GENE_FAMILIES.get(target, []) if g != target and g in gene2idx]
    
    for gene in genes_to_test:
        ko_expr = get_expr(gene, ko_mask)
        ctrl_expr = get_expr(gene, ctrl_mask)
        
        if len(ko_expr) == 0 or len(ctrl_expr) == 0:
            continue
        
        # Component 1: Detection rate (proportion of non-zero cells)
        ko_detect = (ko_expr > 0).mean()
        ctrl_detect = (ctrl_expr > 0).mean()
        
        # Fisher's exact test for detection rate
        ko_nz = (ko_expr > 0).sum()
        ko_z = (ko_expr == 0).sum()
        ctrl_nz = (ctrl_expr > 0).sum()
        ctrl_z = (ctrl_expr == 0).sum()
        
        _, detect_pval = stats.fisher_exact([[ko_nz, ko_z], [ctrl_nz, ctrl_z]])
        detect_or = (ko_nz / max(ko_z, 1)) / (ctrl_nz / max(ctrl_z, 1)) if ctrl_nz > 0 else np.nan
        
        # Component 2: Expression level (among non-zero cells only)
        ko_nonzero = ko_expr[ko_expr > 0]
        ctrl_nonzero = ctrl_expr[ctrl_expr > 0]
        
        if len(ko_nonzero) >= 3 and len(ctrl_nonzero) >= 3:
            expr_stat, expr_pval = stats.mannwhitneyu(ko_nonzero, ctrl_nonzero, alternative='two-sided')
            expr_lfc = np.log2((np.mean(ko_nonzero) + 0.01) / (np.mean(ctrl_nonzero) + 0.01))
        elif len(ko_nonzero) > 0 and len(ctrl_nonzero) > 0:
            expr_pval = np.nan
            expr_lfc = np.log2((np.mean(ko_nonzero) + 0.01) / (np.mean(ctrl_nonzero) + 0.01))
        else:
            expr_pval = np.nan
            expr_lfc = np.nan
        
        # Combined test (Fisher's method)
        pvals = [p for p in [detect_pval, expr_pval] if pd.notna(p) and p > 0]
        if len(pvals) == 2:
            combined_stat = -2 * sum(np.log(p) for p in pvals)
            combined_pval = 1 - stats.chi2.cdf(combined_stat, 2 * len(pvals))
        elif len(pvals) == 1:
            combined_pval = pvals[0]
        else:
            combined_pval = np.nan
        
        is_target = gene == target
        result = {
            'ko': ko, 'target_gene': target, 'gene': gene, 'is_target': is_target,
            'ko_detect_rate': ko_detect, 'ctrl_detect_rate': ctrl_detect,
            'detect_pval': detect_pval, 'detect_OR': detect_or,
            'ko_mean_nz': np.mean(ko_nonzero) if len(ko_nonzero) > 0 else 0,
            'ctrl_mean_nz': np.mean(ctrl_nonzero) if len(ctrl_nonzero) > 0 else 0,
            'expr_log2FC': expr_lfc, 'expr_pval': expr_pval,
            'combined_pval': combined_pval,
            'n_ko': len(ko_expr), 'n_ctrl': len(ctrl_expr),
        }
        mast_results.append(result)
        
        if is_target or (pd.notna(combined_pval) and combined_pval < 0.05):
            sig = '***' if pd.notna(combined_pval) and combined_pval < 0.001 else ('**' if pd.notna(combined_pval) and combined_pval < 0.01 else ('*' if pd.notna(combined_pval) and combined_pval < 0.05 else 'ns'))
            print(f"  {gene}: detect={ko_detect:.3f} vs {ctrl_detect:.3f} (p={detect_pval:.4e}), "
                  f"expr_lfc={expr_lfc:.3f} (p={expr_pval if pd.notna(expr_pval) else 'NA'}), "
                  f"combined={combined_pval if pd.notna(combined_pval) else 'NA'} {sig}")

mast_df = pd.DataFrame(mast_results)
mast_df.to_csv(f'{OUT_TBL}/strategy4_mast.csv', index=False)

# ============================================================
# FIGURE 6: MAST Detection Rate vs Expression
# ============================================================
print("\nGenerating Figure 6: MAST zero-inflated analysis...")

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    pair_data = mast_df[(mast_df['ko'] == ko) & (mast_df['target_gene'] == target)]
    
    if len(pair_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
        continue
    
    # Plot: x = detection rate change, y = expression change
    detect_change = pair_data['ko_detect_rate'] - pair_data['ctrl_detect_rate']
    expr_change = pair_data['expr_log2FC'].fillna(0)
    
    # Color by significance
    colors = []
    for _, row in pair_data.iterrows():
        if pd.notna(row['combined_pval']) and row['combined_pval'] < 0.05:
            colors.append('#E64B35')
        else:
            colors.append('#CCCCCC')
    
    sizes = [40 if row['is_target'] else 15 for _, row in pair_data.iterrows()]
    
    ax.scatter(detect_change, expr_change, c=colors, s=sizes, alpha=0.7, edgecolors='none', zorder=3)
    
    # Highlight target
    target_row = pair_data[pair_data['is_target']]
    if len(target_row) > 0:
        tr = target_row.iloc[0]
        dc = tr['ko_detect_rate'] - tr['ctrl_detect_rate']
        ec = tr['expr_log2FC'] if pd.notna(tr['expr_log2FC']) else 0
        ax.scatter([dc], [ec], c='#E64B35', s=60, zorder=5, edgecolors='black', linewidths=0.8, marker='*')
        ax.annotate(target, (dc, ec), fontsize=6, color='#E64B35', fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')
    
    # Label significant non-target genes
    sig_genes = pair_data[(pair_data['combined_pval'] < 0.05) & (~pair_data['is_target'])]
    for _, row in sig_genes.head(5).iterrows():
        dc = row['ko_detect_rate'] - row['ctrl_detect_rate']
        ec = row['expr_log2FC'] if pd.notna(row['expr_log2FC']) else 0
        ax.annotate(row['gene'], (dc, ec), fontsize=5, color='#3C5488',
                    xytext=(3, 3), textcoords='offset points')
    
    ax.axhline(0, color='grey', linewidth=0.3, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.3, linestyle='--')
    ax.set_xlabel('Δ Detection rate', fontsize=7)
    ax.set_ylabel('Expression log2FC\n(non-zero cells)', fontsize=7)
    ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Strategy 4: Zero-Inflated Analysis (Detection Rate vs Expression)', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig6_mast.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig6_mast.pdf', bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

print("\n✓ Strategy 4 complete")
