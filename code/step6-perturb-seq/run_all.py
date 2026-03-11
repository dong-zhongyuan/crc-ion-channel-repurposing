#!/usr/bin/env python3
"""
HCT116 Perturb-seq v2 — Complete Analysis Pipeline
7 strategies, 10+ figures, Nature quality
"""
import warnings
warnings.filterwarnings('ignore')
import os, sys, pickle, json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, stats
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Nature style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# Color palette (Nature-style)
COLORS = {
    'KO': '#E64B35',
    'Control': '#4DBBD5',
    'sig': '#E64B35',
    'ns': '#999999',
    'up': '#E64B35',
    'down': '#3C5488',
}
PAIR_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

PAIRS = [
    ('GALK1', 'KCNA5'),
    ('LSM7', 'CLIC1'),
    ('RIPK2', 'CFTR'),
    ('TRMT112', 'CFTR'),
    ('RPS21', 'KCNQ2'),
    ('EXOSC5', 'AQP9'),
]

# Gene family members for each target
GENE_FAMILIES = {
    'KCNA5': ['KCNA1', 'KCNA2', 'KCNA3', 'KCNA4', 'KCNA5', 'KCNA6', 'KCNA7', 'KCNA10',
              'KCNB1', 'KCNB2', 'KCNC1', 'KCNC2', 'KCNC3', 'KCNC4', 'KCND1', 'KCND2', 'KCND3'],
    'CLIC1': ['CLIC1', 'CLIC2', 'CLIC3', 'CLIC4', 'CLIC5', 'CLIC6',
              'CLCN1', 'CLCN2', 'CLCN3', 'CLCN4', 'CLCN5', 'CLCN6', 'CLCN7',
              'ANO1', 'ANO2', 'ANO6', 'BEST1'],
    'CFTR': ['CFTR', 'ABCC1', 'ABCC2', 'ABCC3', 'ABCC4', 'ABCC5', 'ABCC6',
             'ABCB1', 'ABCB4', 'ABCG2', 'SLC26A3', 'SLC26A6', 'SLC26A9',
             'CLCN2', 'CLCN4', 'ANO1'],
    'KCNQ2': ['KCNQ1', 'KCNQ2', 'KCNQ3', 'KCNQ4', 'KCNQ5',
              'KCNA1', 'KCNA2', 'KCNA5', 'KCNB1', 'KCNC1', 'KCND1', 'KCND2',
              'KCNJ1', 'KCNJ2', 'KCNJ11', 'KCNK1', 'KCNK2'],
    'AQP9': ['AQP1', 'AQP2', 'AQP3', 'AQP4', 'AQP5', 'AQP6', 'AQP7', 'AQP8', 'AQP9', 'AQP10', 'AQP11',
             'SLC2A1', 'SLC2A2', 'SLC2A3', 'SLC2A4'],
}

# Pathway gene sets for GSEA
PATHWAY_GENES = {
    'KCNA5': {
        'Potassium_channel': ['KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6','KCNA7','KCNA10',
                              'KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4','KCND1','KCND2','KCND3',
                              'KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5','KCNJ1','KCNJ2','KCNJ11',
                              'KCNK1','KCNK2','KCNK3','KCNK5','KCNK6'],
        'Ion_transport': ['SLC12A1','SLC12A2','SLC12A3','SLC12A4','SLC12A5','SLC12A6','SLC12A7',
                          'ATP1A1','ATP1A2','ATP1A3','ATP1B1','ATP1B2','ATP1B3'],
        'Membrane_potential': ['SCN1A','SCN2A','SCN3A','SCN5A','SCN8A','SCN9A','SCN10A',
                               'CACNA1A','CACNA1B','CACNA1C','CACNA1D','CACNA1E'],
    },
    'CLIC1': {
        'Chloride_channel': ['CLIC1','CLIC2','CLIC3','CLIC4','CLIC5','CLIC6',
                             'CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7',
                             'ANO1','ANO2','ANO6','BEST1','BEST2','BEST3','BEST4'],
        'Chloride_transport': ['SLC4A1','SLC4A2','SLC4A3','SLC26A3','SLC26A4','SLC26A6','SLC26A9',
                               'CFTR','GABRA1','GABRA2','GABRB1','GABRB2'],
        'Redox_signaling': ['GSTO1','GSTO2','GSR','GPX1','GPX2','GPX3','GPX4',
                            'SOD1','SOD2','CAT','PRDX1','PRDX2','PRDX3','PRDX4','PRDX5','PRDX6'],
    },
    'CFTR': {
        'ABC_transporter': ['CFTR','ABCA1','ABCA2','ABCA3','ABCB1','ABCB4','ABCB6','ABCB7',
                            'ABCC1','ABCC2','ABCC3','ABCC4','ABCC5','ABCC6','ABCG1','ABCG2'],
        'Chloride_transport': ['CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7',
                               'ANO1','ANO6','SLC26A3','SLC26A6','SLC26A9','CLIC1','CLIC4'],
        'Epithelial_ion_transport': ['ENaC','SCNN1A','SCNN1B','SCNN1G','ATP12A',
                                     'SLC9A1','SLC9A2','SLC9A3','NHE1','AQP1','AQP3','AQP5'],
    },
    'KCNQ2': {
        'Potassium_channel': ['KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5',
                              'KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6',
                              'KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4',
                              'KCND1','KCND2','KCND3','KCNJ1','KCNJ2','KCNJ11'],
        'Neuronal_excitability': ['SCN1A','SCN2A','SCN3A','SCN5A','SCN8A',
                                  'CACNA1A','CACNA1B','CACNA1C','CACNA1D',
                                  'GRIA1','GRIA2','GRIN1','GRIN2A','GRIN2B'],
        'M_current': ['KCNQ2','KCNQ3','KCNQ5','CALM1','CALM2','CALM3',
                      'PIP5K1A','PIP5K1B','PIP5K1C','PLCB1','PLCB2','PLCB3','PLCB4'],
    },
    'AQP9': {
        'Aquaporin': ['AQP1','AQP2','AQP3','AQP4','AQP5','AQP6','AQP7','AQP8','AQP9','AQP10','AQP11','AQP12A','AQP12B'],
        'Glycerol_transport': ['AQP3','AQP7','AQP9','AQP10','SLC2A1','SLC2A2','SLC2A3','SLC2A4',
                               'GK','GK2','GPD1','GPD2','GPAT1'],
        'Water_homeostasis': ['AQP1','AQP2','AQP3','AQP4','AQP5','SLC12A1','SLC12A2',
                              'AVP','AVPR1A','AVPR2','WNK1','WNK4','SGK1'],
    },
}

OUT_FIG = '/dawn/.openclaw/workspace/swap/downstream/hct116-v2/figures'
OUT_TBL = '/dawn/.openclaw/workspace/swap/downstream/hct116-v2/tables'

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading data...")
adata = ad.read_h5ad('/dawn/.openclaw/workspace/swap/downstream/hct116-perturbseq/subset_6pairs_allgenes.h5ad')
print(f"Loaded: {adata.shape}")

if sparse.issparse(adata.X):
    X = adata.X.tocsr()
else:
    X = sparse.csr_matrix(adata.X)

# Load DEG tables from v1
deg_tables = {}
for ko, _ in PAIRS:
    if ko not in deg_tables:
        fpath = f'/dawn/.openclaw/workspace/swap/downstream/hct116-perturbseq/tables/deg_{ko}.csv'
        if os.path.exists(fpath):
            deg_tables[ko] = pd.read_csv(fpath, index_col=0)
            print(f"  Loaded DEG table for {ko}: {deg_tables[ko].shape}")

# Gene name to index mapping
gene2idx = {g: i for i, g in enumerate(adata.var_names)}

def get_gene_expr(gene, mask):
    """Get expression values for a gene in masked cells"""
    if gene not in gene2idx:
        return np.array([])
    idx = gene2idx[gene]
    if hasattr(mask, "values"): mask = mask.values
    return np.asarray(X[mask, idx].todense()).flatten()

# ============================================================
# STRATEGY 1: PSEUDOBULK + NEGATIVE BINOMIAL
# ============================================================
print("\n" + "=" * 60)
print("STRATEGY 1: Pseudobulk + Negative Binomial")
print("=" * 60)

pb_results = []
deseq_results = []

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---")
    ko_mask = (adata.obs["gene_target"] == ko).values
    ctrl_mask = (adata.obs["gene_target"] == "Non-Targeting").values
    
    batches = adata.obs['sample'].unique()
    pb_data = []
    
    for batch in batches:
        batch_mask = (adata.obs["sample"] == batch).values
        for cond, cond_mask in [('KO', ko_mask), ('Control', ctrl_mask)]:
            cells = cond_mask & batch_mask
            n = cells.sum()
            if n == 0:
                continue
            
            target_expr = get_gene_expr(target, cells)
            total_counts = np.asarray(X[cells, :].sum(axis=1)).flatten().sum()
            
            pb_data.append({
                'batch': batch, 'condition': cond,
                'target_sum': target_expr.sum(),
                'target_mean': target_expr.mean(),
                'n_cells': int(n),
                'total_counts': total_counts,
                'nonzero_frac': (target_expr > 0).mean(),
            })
    
    pb_df = pd.DataFrame(pb_data)
    pb_df['ko'] = ko
    pb_df['target'] = target
    pb_df['target_cpm'] = pb_df['target_sum'] / pb_df['total_counts'] * 1e6
    pb_results.append(pb_df)
    
    # Wilcoxon test on pseudobulk CPM
    ko_cpm = pb_df[pb_df['condition'] == 'KO']['target_cpm'].dropna().values
    ctrl_cpm = pb_df[pb_df['condition'] == 'Control']['target_cpm'].dropna().values
    
    if len(ko_cpm) > 2 and len(ctrl_cpm) > 2:
        try:
            stat, pval = stats.mannwhitneyu(ko_cpm, ctrl_cpm, alternative='two-sided')
            l2fc = np.log2((np.mean(ko_cpm) + 0.01) / (np.mean(ctrl_cpm) + 0.01))
        except:
            pval, l2fc = np.nan, np.nan
    else:
        pval, l2fc = np.nan, np.nan
    
    print(f"  Pseudobulk MWU: p={pval:.4e}, log2FC={l2fc:.4f}" if pd.notna(pval) else "  Pseudobulk: insufficient data")
    
    deseq_results.append({
        'ko': ko, 'target': target,
        'log2FC_pb': l2fc, 'pvalue_pb': pval,
        'mean_ko_cpm': np.mean(ko_cpm) if len(ko_cpm) > 0 else 0,
        'mean_ctrl_cpm': np.mean(ctrl_cpm) if len(ctrl_cpm) > 0 else 0,
        'n_ko_batches': len(ko_cpm),
        'n_ctrl_batches': len(ctrl_cpm),
    })

all_pb = pd.concat(pb_results, ignore_index=True)
all_pb.to_csv(f'{OUT_TBL}/strategy1_pseudobulk.csv', index=False)
deseq_df = pd.DataFrame(deseq_results)
deseq_df.to_csv(f'{OUT_TBL}/strategy1_deseq_summary.csv', index=False)
print("\nPseudobulk summary:")
print(deseq_df[['ko','target','log2FC_pb','pvalue_pb','mean_ko_cpm','mean_ctrl_cpm']].to_string())

# Also try pyDESeq2 properly
print("\nRunning pyDESeq2...")
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

pydeseq_results = []
for ko, target in PAIRS:
    print(f"  pyDESeq2: {ko} -> {target}")
    ko_mask = (adata.obs["gene_target"] == ko).values
    ctrl_mask = (adata.obs["gene_target"] == "Non-Targeting").values
    batches = adata.obs['sample'].unique()
    
    # Build pseudobulk count matrix for top variable genes + target
    # First get HVGs
    genes_to_use = list(set(
        [target, ko] + 
        [g for g in adata.var_names[:500] if g in gene2idx]  # first 500 genes as background
    ))
    genes_to_use = [g for g in genes_to_use if g in gene2idx]
    
    count_rows = []
    meta_rows = []
    
    for batch in batches:
        batch_mask = (adata.obs["sample"] == batch).values
        for cond, cond_mask, label in [('KO', ko_mask, 'KO'), ('Control', ctrl_mask, 'Control')]:
            cells = cond_mask & batch_mask
            n = cells.sum()
            if n < 1:
                continue
            row = {}
            for g in genes_to_use:
                row[g] = int(np.asarray(X[cells, gene2idx[g]].todense()).sum())
            count_rows.append(row)
            meta_rows.append({'sample_id': f'{batch}_{label}', 'condition': label, 'n_cells': int(n)})
    
    counts_df = pd.DataFrame(count_rows)
    meta_df = pd.DataFrame(meta_rows)
    counts_df.index = meta_df['sample_id']
    meta_df.index = meta_df['sample_id']
    
    # Filter zero-sum genes
    counts_df = counts_df.loc[:, counts_df.sum() > 10]
    
    if target not in counts_df.columns:
        print(f"    {target} filtered out (zero counts)")
        pydeseq_results.append({'ko': ko, 'target': target, 'log2FC': np.nan, 'pvalue': np.nan, 'padj': np.nan, 'baseMean': 0})
        continue
    
    try:
        dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors='condition', refit_cooks=True, n_cpus=8)
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=['condition', 'KO', 'Control'], n_cpus=8)
        stat_res.summary()
        res = stat_res.results_df
        if target in res.index:
            r = res.loc[target]
            print(f"    log2FC={r['log2FoldChange']:.4f}, p={r['pvalue']:.4e}, padj={r['padj']:.4e}")
            pydeseq_results.append({'ko': ko, 'target': target, 'log2FC': r['log2FoldChange'], 
                                    'pvalue': r['pvalue'], 'padj': r['padj'], 'baseMean': r['baseMean']})
        else:
            pydeseq_results.append({'ko': ko, 'target': target, 'log2FC': np.nan, 'pvalue': np.nan, 'padj': np.nan, 'baseMean': 0})
    except Exception as e:
        print(f"    Error: {e}")
        pydeseq_results.append({'ko': ko, 'target': target, 'log2FC': np.nan, 'pvalue': np.nan, 'padj': np.nan, 'baseMean': 0, 'error': str(e)[:80]})

pydeseq_df = pd.DataFrame(pydeseq_results)
pydeseq_df.to_csv(f'{OUT_TBL}/strategy1_pydeseq2.csv', index=False)

# ============================================================
# FIGURE 2: Pseudobulk boxplots
# ============================================================
print("\nGenerating Figure 2: Pseudobulk boxplots...")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    pb = all_pb[(all_pb['ko'] == ko) & (all_pb['target'] == target)]
    
    ko_cpm = pb[pb['condition'] == 'KO']['target_cpm'].values
    ctrl_cpm = pb[pb['condition'] == 'Control']['target_cpm'].values
    
    bp = ax.boxplot([ctrl_cpm, ko_cpm], positions=[1, 2], widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=1))
    bp['boxes'][0].set_facecolor(COLORS['Control'])
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(COLORS['KO'])
    bp['boxes'][1].set_alpha(0.5)
    
    for j, (vals, pos) in enumerate([(ctrl_cpm, 1), (ko_cpm, 2)]):
        jitter = np.random.normal(0, 0.08, len(vals))
        c = COLORS['Control'] if j == 0 else COLORS['KO']
        ax.scatter(np.full_like(vals, pos) + jitter, vals, c=c, s=6, alpha=0.4, edgecolors='none', zorder=3)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control', f'{ko} KO'], fontsize=7)
    ax.set_ylabel(f'{target} (CPM)', fontsize=7)
    
    # Get p-value
    row = deseq_df[(deseq_df['ko'] == ko) & (deseq_df['target'] == target)]
    if len(row) > 0:
        pv = row.iloc[0]['pvalue_pb']
        lfc = row.iloc[0]['log2FC_pb']
        if pd.notna(pv):
            pstr = f'p={pv:.1e}' if pv < 0.01 else f'p={pv:.3f}'
            ax.set_title(f'{ko} → {target}\nlog2FC={lfc:.2f}, {pstr}', fontsize=7, fontweight='bold')
        else:
            ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Strategy 1: Pseudobulk Expression (CPM per batch)', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig2_pseudobulk.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig2_pseudobulk.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

print("\n✓ Strategy 1 complete")
