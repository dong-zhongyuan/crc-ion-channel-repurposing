#!/usr/bin/env python3
"""Generate Figure 1 (Overview) and Figure 10 (Comprehensive Summary)"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

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

# ============================================================
# FIGURE 1: Analysis Framework Overview
# ============================================================
print("Generating Figure 1: Analysis framework overview...")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(5, 7.5, 'HCT116 Perturb-seq v2: Multi-Strategy Analysis Framework',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Data box
data_box = FancyBboxPatch((0.5, 6.2), 3, 0.8, boxstyle="round,pad=0.1",
                           facecolor='#E8F4FD', edgecolor='#3C5488', linewidth=1.5)
ax.add_patch(data_box)
ax.text(2, 6.6, 'Perturb-seq Data\n8,445 cells × 38,606 genes\n6 KO × 109 batches',
        ha='center', va='center', fontsize=7, fontweight='bold')

# 6 pairs box
pairs_box = FancyBboxPatch((5.5, 6.2), 4, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#FFF3E0', edgecolor='#E64B35', linewidth=1.5)
ax.add_patch(pairs_box)
pair_text = 'GALK1→KCNA5  LSM7→CLIC1  RIPK2→CFTR\nTRMT112→CFTR  RPS21→KCNQ2  EXOSC5→AQP9'
ax.text(7.5, 6.6, f'6 KO-Target Pairs\n{pair_text}',
        ha='center', va='center', fontsize=6)

# Arrow from data to strategies
ax.annotate('', xy=(5, 5.8), xytext=(5, 6.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 7 Strategy boxes
strategies = [
    ('S1: Pseudobulk\n+ NB Model', '#E64B35'),
    ('S2: GSEA\nPathway', '#4DBBD5'),
    ('S3: Transcriptome\nRanking', '#00A087'),
    ('S4: Zero-Inflated\n(MAST)', '#3C5488'),
    ('S5: Indirect\nNetwork', '#F39B7F'),
    ('S6: Perturbation\nScore', '#8491B4'),
    ('S7: Co-expression\nModule', '#91D1C2'),
]

for j, (name, color) in enumerate(strategies):
    x = 0.5 + j * 1.3
    box = FancyBboxPatch((x, 4.2), 1.1, 1.2, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.3)
    ax.add_patch(box)
    ax.text(x + 0.55, 4.8, name, ha='center', va='center', fontsize=5.5, fontweight='bold')

# Arrow to results
ax.annotate('', xy=(5, 3.8), xytext=(5, 4.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Results box
results_box = FancyBboxPatch((1, 2.5), 8, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9', edgecolor='#00A087', linewidth=1.5)
ax.add_patch(results_box)
ax.text(5, 3.3, 'Comprehensive Evidence Matrix', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5, 2.9, 'Direct effects • Pathway-level changes • Gene family compensation\n'
        'Detection rate shifts • Indirect regulation • Perturbation response • Module disruption',
        ha='center', va='center', fontsize=6.5)

# Challenge annotation
challenge_box = FancyBboxPatch((1, 0.5), 8, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=1)
ax.add_patch(challenge_box)
ax.text(5, 1.6, 'Key Challenge: 4/6 target genes have near-zero baseline expression in HCT116',
        ha='center', va='center', fontsize=8, fontweight='bold', color='#E65100')
ax.text(5, 1.1, 'KCNA5, CFTR, KCNQ2, AQP9 — ion channel genes not expressed in colon cancer cells\n'
        'Solution: Multi-strategy approach captures indirect, pathway-level, and network-level evidence',
        ha='center', va='center', fontsize=6.5, color='#424242')

fig.savefig(f'{OUT_FIG}/fig1_overview.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig1_overview.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# FIGURE 10: Comprehensive Summary Heatmap
# ============================================================
print("\nGenerating Figure 10: Comprehensive summary heatmap...")

# Load all results
def safe_load(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

s1 = safe_load(f'{OUT_TBL}/strategy1_deseq_summary.csv')
s2 = safe_load(f'{OUT_TBL}/strategy2_gsea.csv')
s2m = safe_load(f'{OUT_TBL}/strategy2_gsea_msigdb.csv')
s3r = safe_load(f'{OUT_TBL}/strategy3_ranking.csv')
s3f = safe_load(f'{OUT_TBL}/strategy3_family.csv')
s4 = safe_load(f'{OUT_TBL}/strategy4_mast.csv')
s5 = safe_load(f'{OUT_TBL}/strategy5_network.csv')
s6 = safe_load(f'{OUT_TBL}/strategy6_perturbation.csv')
s7 = safe_load(f'{OUT_TBL}/strategy7_coexpression.csv')
s7m = safe_load(f'{OUT_TBL}/strategy7_modules.csv')

# Build evidence matrix
strategies = [
    'S1: Pseudobulk',
    'S2: GSEA Pathway',
    'S3: Transcriptome Rank',
    'S4: Zero-Inflated',
    'S5: Indirect Network',
    'S6: Perturbation Score',
    'S7: Co-expression',
]

pair_labels = [f'{ko}→{t}' for ko, t in PAIRS]

# Evidence strength: 0=no evidence, 1=weak, 2=moderate, 3=strong
evidence = np.zeros((len(PAIRS), len(strategies)))
pvalue_matrix = np.full((len(PAIRS), len(strategies)), np.nan)
detail_text = [['' for _ in strategies] for _ in PAIRS]

for i, (ko, target) in enumerate(PAIRS):
    # S1: Pseudobulk
    if len(s1) > 0:
        row = s1[(s1['ko'] == ko) & (s1['target'] == target)]
        if len(row) > 0:
            pv = row.iloc[0]['pvalue_pb']
            pvalue_matrix[i, 0] = pv
            if pd.notna(pv):
                if pv < 0.001: evidence[i, 0] = 3
                elif pv < 0.01: evidence[i, 0] = 2
                elif pv < 0.05: evidence[i, 0] = 1.5
                elif pv < 0.1: evidence[i, 0] = 1
                else: evidence[i, 0] = 0.5
                detail_text[i][0] = f'p={pv:.2e}'
            else:
                evidence[i, 0] = 0.3  # At least attempted
                detail_text[i][0] = 'low expr'
    
    # S2: GSEA
    if len(s2) > 0:
        pair_gsea = s2[(s2['ko'] == ko) & (s2['target'] == target)]
        if len(pair_gsea) > 0:
            best_p = pair_gsea['pvalue'].min()
            n_sig = (pair_gsea['pvalue'] < 0.05).sum()
            pvalue_matrix[i, 1] = best_p
            if n_sig >= 2: evidence[i, 1] = 3
            elif n_sig >= 1: evidence[i, 1] = 2
            elif best_p < 0.1: evidence[i, 1] = 1
            else: evidence[i, 1] = 0.5
            detail_text[i][1] = f'{n_sig} sig paths'
        else:
            evidence[i, 1] = 0.3
    # Also check MSigDB
    if len(s2m) > 0:
        pair_msig = s2m[(s2m['ko'] == ko) & (s2m['target'] == target) & (s2m['relevant'] == True)]
        if len(pair_msig) > 0:
            n_sig_m = (pair_msig['pvalue'] < 0.05).sum()
            if n_sig_m > 0:
                evidence[i, 1] = max(evidence[i, 1], 2)
                detail_text[i][1] += f'+{n_sig_m}GO'
    
    # S3: Ranking
    if len(s3r) > 0:
        row = s3r[(s3r['ko'] == ko) & (s3r['target'] == target)]
        if len(row) > 0:
            pct = row.iloc[0]['percentile']
            if pct < 5: evidence[i, 2] = 3
            elif pct < 10: evidence[i, 2] = 2
            elif pct < 25: evidence[i, 2] = 1.5
            elif pct < 50: evidence[i, 2] = 1
            else: evidence[i, 2] = 0.5
            detail_text[i][2] = f'top {pct:.0f}%'
    # Family compensation
    if len(s3f) > 0:
        pair_fam = s3f[(s3f['ko'] == ko) & (s3f['target'] == target)]
        n_sig_fam = (pair_fam['padj'] < 0.05).sum() if len(pair_fam) > 0 else 0
        if n_sig_fam > 0:
            evidence[i, 2] = max(evidence[i, 2], min(n_sig_fam, 3))
            detail_text[i][2] += f'\n{n_sig_fam} fam'
    
    # S4: MAST
    if len(s4) > 0:
        pair_mast = s4[(s4['ko'] == ko) & (s4['target_gene'] == target)]
        target_mast = pair_mast[pair_mast['is_target']]
        if len(target_mast) > 0:
            cp = target_mast.iloc[0]['combined_pval']
            dp = target_mast.iloc[0]['detect_pval']
            pvalue_matrix[i, 3] = cp
            if pd.notna(cp) and cp < 0.05: evidence[i, 3] = 2.5
            elif pd.notna(dp) and dp < 0.05: evidence[i, 3] = 2
            elif pd.notna(cp) and cp < 0.1: evidence[i, 3] = 1
            else: evidence[i, 3] = 0.5
            detail_text[i][3] = f'p={cp:.2e}' if pd.notna(cp) else 'NA'
        # Check family members
        sig_family = pair_mast[(pair_mast['combined_pval'] < 0.05) & (~pair_mast['is_target'])]
        if len(sig_family) > 0:
            evidence[i, 3] = max(evidence[i, 3], min(len(sig_family) * 0.5 + 1, 3))
            detail_text[i][3] += f'\n{len(sig_family)} fam'
    
    # S5: Network
    if len(s5) > 0:
        pair_net = s5[(s5['ko'] == ko) & (s5['target'] == target)]
        pair_net_genes = pair_net[pair_net['category'] != 'GO_enrichment']
        n_mediators = len(pair_net_genes)
        n_go = len(pair_net[pair_net['category'] == 'GO_enrichment'])
        if n_mediators >= 5: evidence[i, 4] = 3
        elif n_mediators >= 3: evidence[i, 4] = 2
        elif n_mediators >= 1: evidence[i, 4] = 1.5
        elif n_go > 0: evidence[i, 4] = 1
        else: evidence[i, 4] = 0.3
        detail_text[i][4] = f'{n_mediators} med'
        if n_go > 0:
            detail_text[i][4] += f'\n{n_go} GO'
    
    # S6: Perturbation
    if len(s6) > 0:
        resp = s6[(s6['ko'] == ko) & (s6['target'] == target) & (s6['group'] == 'responder')]
        if len(resp) > 0:
            pv = resp.iloc[0]['pvalue']
            pvalue_matrix[i, 5] = pv
            if pd.notna(pv) and pv < 0.01: evidence[i, 5] = 3
            elif pd.notna(pv) and pv < 0.05: evidence[i, 5] = 2
            elif pd.notna(pv) and pv < 0.1: evidence[i, 5] = 1.5
            else: evidence[i, 5] = 0.5
            detail_text[i][5] = f'p={pv:.2e}' if pd.notna(pv) else 'NA'
    
    # S7: Co-expression
    if len(s7) > 0:
        row = s7[(s7['ko'] == ko) & (s7['target'] == target)]
        if len(row) > 0:
            ctrl_rho = abs(row.iloc[0]['ctrl_rho'])
            delta = abs(row.iloc[0]['delta_rho'])
            if ctrl_rho > 0.1 and delta > 0.05: evidence[i, 6] = 2
            elif ctrl_rho > 0.05: evidence[i, 6] = 1
            else: evidence[i, 6] = 0.5
            detail_text[i][6] = f'ρ={row.iloc[0]["ctrl_rho"]:.2f}'
    if len(s7m) > 0:
        mod = s7m[(s7m['ko'] == ko) & (s7m['target'] == target)]
        if len(mod) > 0:
            disrupt = mod.iloc[0]['disruption_score']
            if disrupt > 0.05: evidence[i, 6] = max(evidence[i, 6], 2)
            elif disrupt > 0.02: evidence[i, 6] = max(evidence[i, 6], 1.5)
            detail_text[i][6] += f'\nΔ={disrupt:.3f}'

# Create heatmap
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

# Custom colormap: white -> light blue -> blue -> dark blue
cmap = LinearSegmentedColormap.from_list('evidence', 
    ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'], N=256)

im = ax.imshow(evidence, cmap=cmap, aspect='auto', vmin=0, vmax=3)

ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strategies, fontsize=7, rotation=30, ha='right')
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=8, fontweight='bold')

# Add text annotations
for i in range(len(PAIRS)):
    for j in range(len(strategies)):
        text = detail_text[i][j]
        if text:
            color = 'white' if evidence[i, j] > 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=4.5, color=color)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Evidence Strength', fontsize=8)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(['None', 'Weak', 'Moderate', 'Strong'], fontsize=7)

# Row summary: total evidence score
row_sums = evidence.sum(axis=1)
for i, s in enumerate(row_sums):
    ax.text(len(strategies) + 0.3, i, f'Σ={s:.1f}', fontsize=7, va='center', fontweight='bold',
            color='#1565C0' if s > 5 else '#424242')

ax.set_title('Comprehensive Evidence Matrix: 7 Strategies × 6 Gene Pairs', 
             fontsize=10, fontweight='bold', pad=15)

plt.tight_layout()
fig.savefig(f'{OUT_FIG}/fig10_summary.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_FIG}/fig10_summary.pdf', bbox_inches='tight')
plt.close()
print("Figure 10 saved.")

# ============================================================
# Save evidence matrix as table
# ============================================================
evidence_df = pd.DataFrame(evidence, index=pair_labels, columns=strategies)
evidence_df['Total'] = evidence_df.sum(axis=1)
evidence_df.to_csv(f'{OUT_TBL}/evidence_matrix.csv')
print("\nEvidence matrix:")
print(evidence_df.to_string())

print("\n✓ Summary figures complete")
