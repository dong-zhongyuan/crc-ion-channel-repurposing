#!/usr/bin/env python3
"""Strategy 5: Indirect Regulatory Network (expanded search)"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
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

# Expanded known functional associations
KNOWN_ASSOCIATIONS = {
    'KCNA5': {
        'ion_transport': ['ATP1A1','ATP1B1','SLC12A2','SLC12A7','FXYD1','FXYD5','FXYD3','FXYD6',
                          'ATP2B1','ATP2B4','ATP2A2','SLC9A1','SLC9A3R1','SLC9A3R2'],
        'K_channel_regulation': ['KCNAB1','KCNAB2','DLG1','DLG4','KCNIP1','KCNIP2','KCNIP3','KCNIP4',
                                 'KCNMB1','KCNMB2','KCNMB4','KCNMA1'],
        'membrane_potential': ['SCN1A','SCN5A','CACNA1C','HCN1','HCN2','HCN4','KCNJ11','KCNJ2',
                               'KCNK1','KCNK2','KCNK3','KCNK5','KCNK6'],
        'galactose_metabolism': ['GALK2','GALE','GALT','UGP2','PGM1','PGM2','HK1','HK2','GPI',
                                 'GALM','B4GALT1','B4GALT2','B4GALT3','B4GALT4','B4GALT5','B4GALT6','B4GALT7',
                                 'ST3GAL1','ST3GAL2','ST3GAL3','ST3GAL4','ST3GAL5','ST3GAL6',
                                 'MGAT1','MGAT2','MGAT3','MGAT4A','MGAT4B','MGAT5'],
    },
    'CLIC1': {
        'redox': ['GSTO1','GSTO2','GSR','GPX1','GPX4','PRDX1','PRDX2','SOD1','SOD2','NQO1',
                  'TXN','TXNRD1','HMOX1','HMOX2','FTH1','FTL','NFE2L2','KEAP1'],
        'chloride_channel': ['CLCN3','CLCN5','CLCN7','ANO1','ANO6','CLIC4','CLIC2','CLIC3',
                             'CLCN2','CLCN4','CLCN6','BEST1','BEST3'],
        'cell_cycle': ['CDK1','CDK2','CDK4','CDK6','CCNB1','CCNB2','CCND1','CCNE1','CCNA2',
                       'TP53','RB1','E2F1','MYC','CDKN1A','CDKN1B','CDKN2A'],
        'RNA_processing': ['LSM1','LSM2','LSM3','LSM4','LSM5','LSM6','LSM8','LSM10','LSM11','LSM14A','LSM14B',
                           'SNRPD1','SNRPD2','SNRPD3','SNRPE','SNRPF','SNRPG',
                           'SF3B1','SF3B2','SF3B3','SF3A1','SF3A2','SF3A3',
                           'PRPF8','PRPF31','PRPF19','PRPF3','PRPF4','PRPF6',
                           'DDX5','DDX17','DDX39B','DDX46','DHX15','DHX38'],
    },
    'CFTR': {
        'ABC_transport': ['ABCC1','ABCC2','ABCC3','ABCC4','ABCB1','ABCG2','ABCA1','ABCA3',
                          'ABCB6','ABCB7','ABCB10','ABCD1','ABCD3'],
        'epithelial': ['SCNN1A','SCNN1B','SCNN1G','SLC9A3','SLC26A3','SLC26A6',
                       'CDH1','EPCAM','KRT8','KRT18','KRT19','MUC1','MUC2','MUC5AC'],
        'inflammation': ['IL6','IL8','TNF','NFKB1','NFKB2','RELA','IKBKB','IKBKG',
                         'IL1B','IL1A','CXCL1','CXCL2','CXCL8','CCL2','CCL5'],
        'NF_kB_signaling': ['RIPK1','RIPK2','RIPK3','TRAF2','TRAF6','BIRC2','BIRC3',
                            'NFKBIA','NFKBIB','NFKBIE','CHUK','IKBKB','IKBKG',
                            'MAP3K7','TAB1','TAB2','TAB3','TNFAIP3'],
        'translation': ['TRMT1','TRMT2A','TRMT2B','TRMT5','TRMT6','TRMT10A','TRMT10B','TRMT10C',
                        'TRMT11','TRMT12','TRMT13','TRMT44','TRMT61A','TRMT61B',
                        'METTL1','METTL3','METTL14','METTL16','NSUN2','NSUN3','NSUN4','NSUN5',
                        'EIF4A1','EIF4A2','EIF4E','EIF4G1','EIF3A','EIF3B','EIF3C','EIF3D'],
    },
    'KCNQ2': {
        'K_channel': ['KCNQ3','KCNQ5','KCNE1','KCNE2','KCNE3','KCNQ1','KCNQ4',
                      'KCNA1','KCNA2','KCNA5','KCNB1','KCNC1','KCND1','KCND2'],
        'calmodulin': ['CALM1','CALM2','CALM3','CAMK2A','CAMK2D','CAMK2G',
                       'CAMKK1','CAMKK2','CAMK1','CAMK4'],
        'ribosome': ['RPS3','RPS3A','RPS4X','RPS5','RPS6','RPS7','RPS8','RPS9','RPS10','RPS11',
                     'RPS12','RPS13','RPS14','RPS15','RPS15A','RPS16','RPS17','RPS18','RPS19','RPS20',
                     'RPS21','RPS23','RPS24','RPS25','RPS26','RPS27','RPS27A','RPS28','RPS29',
                     'RPL3','RPL4','RPL5','RPL6','RPL7','RPL7A','RPL8','RPL9','RPL10','RPL10A',
                     'RPL11','RPL12','RPL13','RPL13A','RPL14','RPL15','RPL17','RPL18','RPL18A',
                     'RPL19','RPL21','RPL22','RPL23','RPL23A','RPL24','RPL26','RPL27','RPL27A',
                     'RPL28','RPL29','RPL30','RPL31','RPL32','RPL34','RPL35','RPL35A','RPL36',
                     'RPL37','RPL37A','RPL38','RPL39','RPL40','RPL41','RPLP0','RPLP1','RPLP2'],
    },
    'AQP9': {
        'aquaporin': ['AQP1','AQP3','AQP4','AQP5','AQP7','AQP11'],
        'glycerol_metabolism': ['GK','GPD1','GPD2','LIPE','PNPLA2','DGAT1','DGAT2'],
        'solute_transport': ['SLC2A1','SLC2A3','SLC16A1','SLC16A3','SLC22A1'],
        'RNA_exosome': ['EXOSC1','EXOSC2','EXOSC3','EXOSC4','EXOSC6','EXOSC7','EXOSC8','EXOSC9','EXOSC10',
                        'DIS3','DIS3L','DIS3L2','MTREX','SKIV2L','SKIV2L2',
                        'RRP6','RRP41','RRP42','RRP43','RRP44','RRP45','RRP46',
                        'MPP6','C1D','ZFC3H1','ZCCHC8','RBM7','PABPN1'],
    },
}

_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIG = os.path.join(_STEP_DIR, 'result', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)
OUT_TBL = os.path.join(_STEP_DIR, 'result', 'tables')
os.makedirs(OUT_TBL, exist_ok=True)

# Load DEG tables
deg_tables = {}
for ko in set(p[0] for p in PAIRS):
    fpath = os.path.join(OUT_TBL, f'deg_{ko}.csv')  # pre-computed DEG table
    if os.path.exists(fpath):
        deg_tables[ko] = pd.read_csv(fpath, index_col=0)

print("=== Strategy 5: Indirect Regulatory Network ===", flush=True)
network_results = []

for ko, target in PAIRS:
    print(f"\n--- {ko} -> {target} ---", flush=True)
    if ko not in deg_tables:
        continue
    
    deg = deg_tables[ko]
    sig_deg = deg[deg['padj'] < 0.05].sort_values('padj')
    all_sig = set(sig_deg.index)
    top50 = set(sig_deg.head(50).index)
    
    print(f"  Total sig DEGs: {len(all_sig)}", flush=True)
    
    associations = KNOWN_ASSOCIATIONS.get(target, {})
    
    for category, assoc_genes in associations.items():
        # Check overlap with ALL significant DEGs (not just top 50)
        overlap_all = [g for g in assoc_genes if g in all_sig]
        overlap_top50 = [g for g in assoc_genes if g in top50]
        
        if overlap_all:
            print(f"  {category}: {len(overlap_all)} sig DEGs ({len(overlap_top50)} in top50)", flush=True)
            for g in overlap_all:
                row = deg.loc[g]
                in_top = g in top50
                network_results.append({
                    'ko': ko, 'target': target, 'category': category,
                    'mediator': g, 'log2FC': row['log2FC'], 'padj': row['padj'],
                    'in_top50': in_top,
                })
                if in_top or row['padj'] < 0.001:
                    print(f"    {g}: log2FC={row['log2FC']:.4f}, padj={row['padj']:.4e} {'[TOP50]' if in_top else ''}", flush=True)

network_df = pd.DataFrame(network_results)
network_df.to_csv(f'{OUT_TBL}/strategy5_network.csv', index=False)

# Summary
print("\nNetwork summary:", flush=True)
for ko, target in PAIRS:
    pair_net = network_df[(network_df['ko'] == ko) & (network_df['target'] == target)]
    n_med = len(pair_net)
    n_cat = pair_net['category'].nunique() if n_med > 0 else 0
    print(f"  {ko} -> {target}: {n_med} mediators in {n_cat} categories", flush=True)

# ============================================================
# FIGURE 7: Network diagram
# ============================================================
print("\nGenerating Figure 7...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

for i, (ko, target) in enumerate(PAIRS):
    ax = axes[i]
    pair_net = network_df[(network_df['ko'] == ko) & (network_df['target'] == target)]
    
    ax.set_xlim(-0.5, 2.5)
    ax.axis('off')
    
    # KO node
    n_med = len(pair_net)
    center_y = max(n_med, 3) / 2
    ax.set_ylim(-1, max(n_med + 1, 4))
    
    ax.add_patch(plt.Circle((0, center_y), 0.18, color='#00A087', zorder=5))
    ax.text(0, center_y, ko, ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=6)
    
    # Target node
    ax.add_patch(plt.Circle((2, center_y), 0.18, color='#E64B35', zorder=5))
    ax.text(2, center_y, target, ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=6)
    
    if n_med == 0:
        ax.annotate('', xy=(1.82, center_y), xytext=(0.18, center_y),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1, linestyle='--'))
        ax.text(1, center_y + 0.3, 'No shared\nmediators', ha='center', va='bottom', fontsize=6, color='grey')
    else:
        # Group by category, show top mediators
        categories = pair_net.groupby('category').agg(
            n=('mediator', 'count'),
            best_padj=('padj', 'min'),
            top_gene=('padj', lambda x: pair_net.loc[x.idxmin(), 'mediator']),
            top_lfc=('padj', lambda x: pair_net.loc[x.idxmin(), 'log2FC']),
        ).sort_values('best_padj')
        
        n_show = min(len(categories), 6)
        for j, (cat, row) in enumerate(categories.head(n_show).iterrows()):
            y = j * (max(n_med, 3) / max(n_show, 1))
            
            color = '#3C5488' if row['top_lfc'] < 0 else '#E64B35'
            size = 0.14
            
            ax.add_patch(plt.Circle((1, y), size, color=color, alpha=0.6, zorder=5))
            label = f"{cat[:8]}\n({int(row['n'])})"
            ax.text(1, y, label, ha='center', va='center', fontsize=3.5, color='white', zorder=6)
            
            # Arrows
            ax.annotate('', xy=(1 - size - 0.02, y), xytext=(0.18, center_y),
                        arrowprops=dict(arrowstyle='->', color='#3C5488', lw=0.4, alpha=0.5))
            ax.annotate('', xy=(1.82, center_y), xytext=(1 + size + 0.02, y),
                        arrowprops=dict(arrowstyle='->', color='#E64B35', lw=0.4, alpha=0.5, linestyle='--'))
    
    ax.set_title(f'{ko} → {target}\n({n_med} mediators)', fontsize=7, fontweight='bold')

plt.suptitle('Strategy 5: Indirect Regulatory Network (KO → DEG mediators → Target pathway)', 
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig7_network.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig7_network.pdf', bbox_inches='tight')
plt.close()
print("Figure 7 saved.", flush=True)
print("\n✓ Strategy 5 complete", flush=True)
