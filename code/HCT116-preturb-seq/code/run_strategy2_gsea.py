#!/usr/bin/env python3
"""Strategy 2: GSEA Pathway Analysis (custom gene sets only, no online queries)"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import gseapy as gp
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

PATHWAY_SETS = {
    'KCNA5': {
        'Potassium_channel': ['KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6','KCNA7','KCNA10',
            'KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4','KCND1','KCND2','KCND3',
            'KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5','KCNJ1','KCNJ2','KCNJ11','KCNK1','KCNK2','KCNK3'],
        'Voltage_gated_channel': ['SCN1A','SCN2A','SCN3A','SCN5A','SCN8A','SCN9A',
            'CACNA1A','CACNA1B','CACNA1C','CACNA1D','CACNA1E','CACNA1S',
            'KCNA1','KCNA2','KCNA5','KCNB1','KCNQ2','KCNQ3'],
        'Ion_homeostasis': ['ATP1A1','ATP1A2','ATP1A3','ATP1B1','ATP2A1','ATP2A2','ATP2B1',
            'SLC12A1','SLC12A2','SLC12A3','SLC12A4','SLC12A5','SLC12A6','SLC12A7'],
    },
    'CLIC1': {
        'Chloride_channel': ['CLIC1','CLIC2','CLIC3','CLIC4','CLIC5','CLIC6',
            'CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7',
            'ANO1','ANO2','ANO6','BEST1','BEST2','BEST3','BEST4'],
        'Glutathione_metabolism': ['GSTO1','GSTO2','GSTP1','GSTM1','GSTM2','GSTM3','GSTM4',
            'GSR','GPX1','GPX2','GPX3','GPX4','GSS','GCLC','GCLM'],
        'Redox_homeostasis': ['SOD1','SOD2','SOD3','CAT','PRDX1','PRDX2','PRDX3','PRDX4','PRDX5','PRDX6',
            'TXN','TXN2','TXNRD1','TXNRD2','NQO1','HMOX1'],
    },
    'CFTR': {
        'ABC_transporter': ['CFTR','ABCA1','ABCA2','ABCA3','ABCB1','ABCB4','ABCB6','ABCB7',
            'ABCC1','ABCC2','ABCC3','ABCC4','ABCC5','ABCC6','ABCG1','ABCG2'],
        'Chloride_transport': ['CLCN1','CLCN2','CLCN3','CLCN4','CLCN5','CLCN6','CLCN7',
            'ANO1','ANO6','SLC26A3','SLC26A6','SLC26A9','CLIC1','CLIC4','CFTR'],
        'Epithelial_transport': ['SCNN1A','SCNN1B','SCNN1G','ATP12A','SLC9A1','SLC9A2','SLC9A3',
            'AQP1','AQP3','AQP5','SLC4A1','SLC4A2','SLC4A4'],
    },
    'KCNQ2': {
        'Potassium_channel': ['KCNQ1','KCNQ2','KCNQ3','KCNQ4','KCNQ5',
            'KCNA1','KCNA2','KCNA3','KCNA4','KCNA5','KCNA6',
            'KCNB1','KCNB2','KCNC1','KCNC2','KCNC3','KCNC4',
            'KCND1','KCND2','KCND3','KCNJ1','KCNJ2','KCNJ11'],
        'Neuronal_signaling': ['GRIA1','GRIA2','GRIN1','GRIN2A','GRIN2B',
            'GABRA1','GABRA2','GABRB1','GABRB2','GABRG2',
            'SYN1','SYN2','SYP','SNAP25','VAMP2'],
        'Calmodulin_signaling': ['CALM1','CALM2','CALM3','CAMK2A','CAMK2B','CAMK2D','CAMK2G',
            'PIP5K1A','PIP5K1B','PIP5K1C','PLCB1','PLCB2','PLCB3','PLCB4'],
    },
    'AQP9': {
        'Aquaporin': ['AQP1','AQP2','AQP3','AQP4','AQP5','AQP6','AQP7','AQP8','AQP9','AQP10','AQP11','AQP12A','AQP12B'],
        'Glycerol_metabolism': ['AQP3','AQP7','AQP9','AQP10','GK','GK2','GPD1','GPD2',
            'SLC2A1','SLC2A2','SLC2A3','SLC2A4','LIPE','PNPLA2','DGAT1','DGAT2'],
        'Solute_transport': ['SLC2A1','SLC2A2','SLC2A3','SLC2A4','SLC5A1','SLC5A2',
            'SLC16A1','SLC16A3','SLC16A7','SLC22A1','SLC22A2','SLC22A3'],
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

# Run GSEA for each pair
print("\nRunning GSEA with custom pathway sets...")
gsea_results = []

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---")
    if ko not in deg_tables:
        continue
    
    deg = deg_tables[ko].copy()
    deg['rank_metric'] = -np.log10(deg['pvalue'].clip(1e-300)) * np.sign(deg['log2FC'])
    deg = deg.sort_values('rank_metric', ascending=False)
    ranked = deg['rank_metric'].dropna()
    ranked = ranked[~ranked.index.duplicated()]
    ranked = ranked.replace([np.inf, -np.inf], np.nan).dropna()
    
    pathways = PATHWAY_SETS.get(target, {})
    pathways_filtered = {}
    for pname, genes in pathways.items():
        present = [g for g in genes if g in ranked.index]
        if len(present) >= 3:
            pathways_filtered[pname] = present
            print(f"  {pname}: {len(present)}/{len(genes)} genes present")
    
    if not pathways_filtered:
        continue
    
    try:
        res = gp.prerank(
            rnk=ranked, gene_sets=pathways_filtered,
            min_size=3, max_size=500, permutation_num=1000,
            seed=42, no_plot=True, threads=8,
        )
        for idx, row in res.res2d.iterrows():
            gsea_results.append({
                'ko': ko, 'target': target, 'pathway': row['Term'],
                'es': row['ES'], 'nes': row['NES'],
                'pvalue': row['NOM p-val'], 'fdr': row['FDR q-val'],
                'lead_genes': row.get('Lead_genes', ''),
            })
            print(f"    {row['Term']}: NES={row['NES']:.3f}, p={row['NOM p-val']:.4f}")
    except Exception as e:
        print(f"  GSEA error: {e}")

gsea_df = pd.DataFrame(gsea_results)
gsea_df.to_csv(f'{OUT_TBL}/strategy2_gsea.csv', index=False)

# Also create empty MSigDB results file for compatibility
pd.DataFrame(columns=['ko','target','pathway','es','nes','pvalue','fdr','relevant']).to_csv(
    f'{OUT_TBL}/strategy2_gsea_msigdb.csv', index=False)

# ============================================================
# FIGURE 3: GSEA Results
# ============================================================
print("\nGenerating Figure 3...")
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    pair_gsea = gsea_df[(gsea_df['ko'] == ko) & (gsea_df['target'] == target)]
    
    if len(pair_gsea) == 0:
        ax.text(0.5, 0.5, 'No pathways\nwith enough genes', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='grey')
        ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        continue
    
    pair_gsea = pair_gsea.sort_values('nes')
    colors = ['#E64B35' if p < 0.05 else ('#F39B7F' if p < 0.1 else '#CCCCCC') for p in pair_gsea['pvalue']]
    
    bars = ax.barh(range(len(pair_gsea)), pair_gsea['nes'], color=colors, height=0.6, edgecolor='none')
    ax.set_yticks(range(len(pair_gsea)))
    ax.set_yticklabels(pair_gsea['pathway'], fontsize=6)
    ax.set_xlabel('NES', fontsize=7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f'{ko} → {target}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for j, (_, row) in enumerate(pair_gsea.iterrows()):
        pstr = f'p={row["pvalue"]:.3f}'
        color = '#E64B35' if row['pvalue'] < 0.05 else 'grey'
        ax.text(row['nes'] + 0.05 * np.sign(row['nes']), j, pstr, fontsize=5, va='center', color=color)

plt.suptitle('Strategy 2: GSEA Pathway Enrichment Analysis', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig3_gsea.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig3_gsea.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")
print("\n✓ Strategy 2 complete")
