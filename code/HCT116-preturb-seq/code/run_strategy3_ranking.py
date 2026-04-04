#!/usr/bin/env python3
"""Strategy 3: Transcriptome-wide Perturbation Signature + Gene Family Analysis"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    'KCNA5': {
        'KCNA_family': ['KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6','KCNA7','KCNA10'],
        'Other_K_channels': ['KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4','KCND1','KCND2','KCND3',
                             'KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5'],
    },
    'CLIC1': {
        'CLIC_family': ['CLIC1','CLIC2','CLIC3','CLIC4','CLIC5','CLIC6'],
        'CLCN_family': ['CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7'],
        'Anoctamin': ['ANO1','ANO2','ANO3','ANO4','ANO5','ANO6','ANO7','ANO8','ANO9','ANO10'],
    },
    'CFTR': {
        'ABCC_family': ['CFTR','ABCC1','ABCC2','ABCC3','ABCC4','ABCC5','ABCC6'],
        'Other_ABC': ['ABCA1','ABCB1','ABCB4','ABCG1','ABCG2'],
        'Cl_channels': ['CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7','ANO1','ANO6'],
    },
    'KCNQ2': {
        'KCNQ_family': ['KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5'],
        'KCNA_family': ['KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6'],
        'Other_K_channels': ['KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4','KCND1','KCND2','KCND3'],
    },
    'AQP9': {
        'Aquaporin_family': ['AQP1','AQP2','AQP3','AQP4','AQP5','AQP6','AQP7','AQP8','AQP9','AQP10','AQP11'],
        'SLC2_glucose': ['SLC2A1','SLC2A2','SLC2A3','SLC2A4'],
        'Glycerol_metabolism': ['GK','GPD1','GPD2','LIPE','PNPLA2'],
    },
}

_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(_STEP_DIR, 'result', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)
OUT_TBL = os.path.join(_STEP_DIR, 'result', 'tables')
os.makedirs(OUT_TBL, exist_ok=True)

# Load DEG tables
print("Loading DEG tables...")
deg_tables = {}
for ko in set(p[0] for p in PAIRS):
    fpath = os.path.join(OUT_TBL, f'deg_{ko}.csv')  # pre-computed DEG table
    if os.path.exists(fpath):
        deg_tables[ko] = pd.read_csv(fpath, index_col=0)

# ============================================================
# Strategy 3a: Target gene ranking in transcriptome
# ============================================================
print("\n=== Strategy 3a: Target gene ranking ===")
ranking_results = []

for ko, target in PAIRS:
    if ko not in deg_tables:
        continue
    deg = deg_tables[ko].copy()
    
    # Rank by absolute log2FC
    deg['abs_log2FC'] = deg['log2FC'].abs()
    deg_sorted = deg.sort_values('abs_log2FC', ascending=False)
    deg_sorted['rank'] = range(1, len(deg_sorted) + 1)
    deg_sorted['percentile'] = deg_sorted['rank'] / len(deg_sorted) * 100
    
    if target in deg_sorted.index:
        row = deg_sorted.loc[target]
        print(f"{ko} -> {target}: rank={int(row['rank'])}/{len(deg_sorted)}, "
              f"percentile={row['percentile']:.1f}%, log2FC={row['log2FC']:.4f}, p={row['pvalue']:.4e}")
        ranking_results.append({
            'ko': ko, 'target': target,
            'rank': int(row['rank']), 'total_genes': len(deg_sorted),
            'percentile': row['percentile'],
            'log2FC': row['log2FC'], 'pvalue': row['pvalue'],
        })
    else:
        print(f"{ko} -> {target}: NOT FOUND in DEG table")

ranking_df = pd.DataFrame(ranking_results)
ranking_df.to_csv(f'{OUT_TBL}/strategy3_ranking.csv', index=False)

# ============================================================
# Strategy 3b: Gene family analysis
# ============================================================
print("\n=== Strategy 3b: Gene family analysis ===")
family_results = []

for ko, target in PAIRS:
    if ko not in deg_tables:
        continue
    deg = deg_tables[ko]
    families = GENE_FAMILIES.get(target, {})
    
    print(f"\n--- {ko} -> {target} ---")
    for fam_name, genes in families.items():
        for gene in genes:
            if gene in deg.index:
                row = deg.loc[gene]
                family_results.append({
                    'ko': ko, 'target': target, 'family': fam_name,
                    'gene': gene, 'log2FC': row['log2FC'],
                    'pvalue': row['pvalue'], 'padj': row['padj'],
                    'is_target': gene == target,
                })
                if row['padj'] < 0.05:
                    print(f"  {fam_name}/{gene}: log2FC={row['log2FC']:.4f}, padj={row['padj']:.4e} ***")

family_df = pd.DataFrame(family_results)
family_df.to_csv(f'{OUT_TBL}/strategy3_family.csv', index=False)

# ============================================================
# FIGURE 4: Waterfall / Rank plot
# ============================================================
print("\nGenerating Figure 4: Transcriptome-wide ranking...")
fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    if ko not in deg_tables:
        continue
    
    deg = deg_tables[ko].copy()
    deg['abs_log2FC'] = deg['log2FC'].abs()
    deg_sorted = deg.sort_values('abs_log2FC', ascending=False).reset_index()
    
    # Plot waterfall
    n = len(deg_sorted)
    x = np.arange(n)
    colors = np.where(deg_sorted['padj'] < 0.05, '#3C5488', '#CCCCCC')
    
    ax.scatter(x, deg_sorted['log2FC'], c=colors, s=0.3, alpha=0.3, rasterized=True)
    
    # Highlight target gene
    if target in deg_sorted['gene'].values:
        target_idx = deg_sorted[deg_sorted['gene'] == target].index[0]
        target_lfc = deg_sorted.loc[target_idx, 'log2FC']
        ax.scatter([target_idx], [target_lfc], c='#E64B35', s=30, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(target, (target_idx, target_lfc), fontsize=6, color='#E64B35',
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Highlight KO gene
    if ko in deg_sorted['gene'].values:
        ko_idx = deg_sorted[deg_sorted['gene'] == ko].index[0]
        ko_lfc = deg_sorted.loc[ko_idx, 'log2FC']
        ax.scatter([ko_idx], [ko_lfc], c='#00A087', s=30, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(ko, (ko_idx, ko_lfc), fontsize=6, color='#00A087',
                    xytext=(5, -10), textcoords='offset points', fontweight='bold')
    
    # Add percentile info
    if target in deg_sorted['gene'].values:
        pct = (target_idx + 1) / n * 100
        ax.text(0.95, 0.95, f'{target}: top {pct:.1f}%', transform=ax.transAxes,
                fontsize=6, ha='right', va='top', color='#E64B35',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E64B35', alpha=0.8))
    
    ax.set_xlabel('Gene rank', fontsize=7)
    ax.set_ylabel('log2FC', fontsize=7)
    ax.set_title(f'{ko} KO', fontsize=7, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Strategy 3: Target Gene Ranking in Transcriptome-wide DEG', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig4_ranking.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig4_ranking.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# FIGURE 5: Gene family heatmap
# ============================================================
print("\nGenerating Figure 5: Gene family heatmap...")

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

for i, (ko, target) in enumerate(PAIRS):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    
    fam_data = family_df[(family_df['ko'] == ko) & (family_df['target'] == target)]
    if len(fam_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
        continue
    
    # Pivot for heatmap
    families = fam_data['family'].unique()
    all_genes = []
    family_labels = []
    for fam in families:
        genes = fam_data[fam_data['family'] == fam]['gene'].tolist()
        all_genes.extend(genes)
        family_labels.extend([fam] * len(genes))
    
    lfc_vals = []
    sig_vals = []
    for gene in all_genes:
        row = fam_data[fam_data['gene'] == gene].iloc[0]
        lfc_vals.append(row['log2FC'])
        sig_vals.append(row['padj'] < 0.05 if pd.notna(row['padj']) else False)
    
    # Bar plot instead of heatmap for clarity
    colors = ['#E64B35' if s else '#CCCCCC' for s in sig_vals]
    bars = ax.barh(range(len(all_genes)), lfc_vals, color=colors, height=0.7, edgecolor='none')
    
    # Highlight target gene
    for j, gene in enumerate(all_genes):
        if gene == target:
            ax.barh(j, lfc_vals[j], color='#F39B7F', height=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(all_genes)))
    ax.set_yticklabels(all_genes, fontsize=5)
    ax.set_xlabel('log2FC', fontsize=7)
    ax.axvline(0, color='black', linewidth=0.3)
    ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance markers
    for j, (lfc, sig) in enumerate(zip(lfc_vals, sig_vals)):
        if sig:
            ax.text(lfc + 0.01 * np.sign(lfc), j, '*', fontsize=7, va='center', color='#E64B35')

plt.suptitle('Strategy 3: Gene Family Expression Changes After KO', fontsize=10, fontweight='bold', y=1.02)
fig.savefig(f'{OUT_FIG}/fig5_family.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig5_family.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print("\n✓ Strategy 3 complete")
