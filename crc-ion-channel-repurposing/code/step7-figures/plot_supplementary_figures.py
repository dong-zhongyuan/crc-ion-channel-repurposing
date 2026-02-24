#!/usr/bin/env python3
"""Supplementary Figures S1-S9 for IJMS manuscript.
Generates all supplementary figures referenced in the manuscript text.
"""

import pathlib, warnings, re, textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from scipy import stats
import networkx as nx

try:
    from adjustText import adjust_text

    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================
# PATHS
# ======================================================================
ROOT = pathlib.Path(r"D:\manuscript_2")
S1D = ROOT / "step1_deg_analysis" / "result" / "sourcedata16"
S2D = ROOT / "step2_wgcna" / "result" / "sourcedata16"
S3V = ROOT / "step3_validation" / "result" / "sourcedata16"
S3T = ROOT / "step3_tcga" / "result"
S4D = ROOT / "step4_network_pharmacology" / "result" / "sourcedata"
S5D = ROOT / "step5_vgae_ko" / "result" / "VGAE_KO_Unified"
HCT = ROOT / "hct116-v2" / "tables"
OUT = ROOT / "figures_IJMS" / "output" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================================
# STYLE
# ======================================================================
CASE = "#E64B35"
CTRL = "#4DBBD5"
BLUE = "#3C5488"
GREEN = "#00A087"
ORANGE = "#F39B7F"
PURPLE = "#8491B4"
DGRAY = "#333333"
LGRAY = "#E8E8E8"
MGRAY = "#999999"
FONT = "Arial"
DPI = 300

sns.set_style("ticks")
plt.rcParams.update(
    {
        "font.family": FONT,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

FUNC_CAT = {
    "RPS19": "Ribosomal",
    "RPS21": "Ribosomal",
    "RPS2": "Ribosomal",
    "RPL12": "Ribosomal",
    "RPL39": "Ribosomal",
    "ITGAL": "Immune",
    "CD27": "Immune",
    "LAG3": "Immune",
    "CD6": "Immune",
    "RIPK2": "Immune",
    "EXOSC5": "RNA Processing",
    "SNRPD2": "RNA Processing",
    "LSM7": "RNA Processing",
    "TRMT112": "RNA Processing",
    "NAA10": "Metabolism",
    "PDCD5": "Metabolism",
    "GALK1": "Metabolism",
    "PFDN4": "Metabolism",
    "LAGE3": "Metabolism",
    "CCDC167": "Metabolism",
    "NT5C3B": "Metabolism",
    "PSMG4": "Metabolism",
    "IGHV3-21": "Immune",
    "IGHV4-59": "Immune",
    "IGHV3-15": "Immune",
    "IGHV3-74": "Immune",
    "SNHG6": "RNA Processing",
    "ZFAS1": "RNA Processing",
}
CAT_COLOR = {
    "Ribosomal": CASE,
    "Immune": BLUE,
    "RNA Processing": GREEN,
    "Metabolism": ORANGE,
}


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(
        str(path).replace(".pdf", ".png"),
        bbox_inches="tight",
        dpi=DPI,
        facecolor="white",
    )
    plt.close(fig)
    print(f"  -> {pathlib.Path(path).name}")


def _despine(ax):
    sns.despine(ax=ax, top=True, right=True)


def _label(ax, txt, x=-0.10, y=1.08):
    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        fontfamily=FONT,
    )


def _wrap(s, maxlen=32):
    if len(s) <= maxlen:
        return s
    mid = len(s) // 2
    left = s.rfind(" ", 0, mid + 5)
    if left == -1:
        left = mid
    return s[:left] + "\n" + s[left:].lstrip()


# ======================================================================
# S1: Multi-Evidence Validation Heatmap
# ======================================================================
def make_supp_figure_s1(save_path=None):
    df = pd.read_csv(S3V / 'SourceData_Fig3_ABCD_Evidence.csv')
    df = df.sort_values('FinalScore', ascending=False).head(20).reset_index(drop=True)
    ev_cols = ['HubEvidence', 'DEGEvidence', 'PPIEvidence', 'RegPathEvidence']
    mat = df[ev_cols].values
    fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                              gridspec_kw={'width_ratios': [4, 0.6, 1.5], 'wspace': 0.08})
    ax_heat, ax_dir, ax_bar = axes
    im = ax_heat.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            tc = 'white' if mat[i, j] > 0.6 else 'black'
            ax_heat.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                         fontsize=7, color=tc, fontweight='bold')
    ax_heat.set_xticks(range(len(ev_cols)))
    ax_heat.set_xticklabels(['Hub', 'DEG', 'PPI', 'RegPath'], fontsize=9, rotation=30, ha='right')
    ax_heat.set_yticks(range(len(df)))
    ax_heat.set_yticklabels(df['gene'], fontsize=8, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02)
    cb.set_label('Evidence Score', fontsize=8)
    _label(ax_heat, 'A')
    # Direction arrows
    ax_dir.set_xlim(0, 1); ax_dir.set_ylim(-0.5, len(df) - 0.5)
    ax_dir.invert_yaxis()
    for i, (_, row) in enumerate(df.iterrows()):
        d = row.get('direction', row.get('DEG_Flag', 'Up'))
        if d == 'Up':
            ax_dir.annotate('', xy=(0.5, i - 0.25), xytext=(0.5, i + 0.25),
                           arrowprops=dict(arrowstyle='->', color=CASE, lw=2))
        else:
            ax_dir.annotate('', xy=(0.5, i + 0.25), xytext=(0.5, i - 0.25),
                           arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))
    ax_dir.set_xticks([]); ax_dir.set_yticks([])
    ax_dir.set_title('Dir.', fontsize=9, fontweight='bold')
    for sp in ax_dir.spines.values(): sp.set_visible(False)
    # FinalScore bars
    colors = [CAT_COLOR.get(FUNC_CAT.get(g, ''), MGRAY) for g in df['gene']]
    ax_bar.barh(range(len(df)), df['FinalScore'], color=colors, edgecolor='white', height=0.7)
    ax_bar.set_ylim(-0.5, len(df) - 0.5); ax_bar.invert_yaxis()
    ax_bar.set_yticks([]); ax_bar.set_xlabel('FinalScore', fontsize=9)
    for i, v in enumerate(df['FinalScore']):
        ax_bar.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=7, color=DGRAY)
    _despine(ax_bar)

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S2: STRING PPI Network Analysis
# ======================================================================
def make_supp_figure_s2(save_path=None):
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.35, width_ratios=[1.2, 0.8, 0.8])
    edges_df = pd.read_csv(S3V / 'SourceData_Fig3E_PPI_Edges.csv')
    deg_df = pd.read_csv(S3V / 'SourceData_Fig3F_Degrees.csv')
    cent_df = pd.read_csv(S3V / 'SourceData_Fig3G_Centrality.csv')
    deg_map = dict(zip(deg_df['node'], deg_df['degree']))
    # Panel A: Hub-centric subnetwork (hub genes + top neighbors)
    ax = fig.add_subplot(gs[0])
    G_full = nx.from_pandas_edgelist(edges_df, 'source', 'target')
    hub_nodes = [n for n in G_full.nodes() if n in FUNC_CAT]
    # Build subgraph: hub genes + top shared neighbors (progressively filter)
    neighbor_hub_count = {}
    for h in hub_nodes:
        if h in G_full:
            for nb in G_full.neighbors(h):
                if nb not in FUNC_CAT:
                    neighbor_hub_count[nb] = neighbor_hub_count.get(nb, 0) + 1
    # Progressively raise threshold until subgraph is manageable (40-80 nodes)
    TARGET_MAX = 70
    for min_hubs in range(max(neighbor_hub_count.values(), default=2), 0, -1):
        kept = {n for n, c in neighbor_hub_count.items() if c >= min_hubs}
        if len(kept) + len(hub_nodes) >= 30 or min_hubs == 1:
            break
    # If still too large, take top neighbors by hub-connectivity
    if len(kept) + len(hub_nodes) > TARGET_MAX:
        kept = set(sorted(kept, key=lambda n: neighbor_hub_count[n], reverse=True)[:TARGET_MAX - len(hub_nodes)])
    sub_nodes = set(hub_nodes) | kept
    G = G_full.subgraph(sub_nodes).copy()
    node_colors = [CAT_COLOR.get(FUNC_CAT.get(n, ''), '#DDDDDD') for n in G.nodes()]
    node_sizes = []
    for n in G.nodes():
        d = deg_map.get(n, 1)
        if n in FUNC_CAT:
            node_sizes.append(max(d, 30) * 2.5)
        else:
            node_sizes.append(max(d * 0.3, 15))
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color=MGRAY, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, edgecolors='white', linewidths=0.5)
    hub_labels = {n: n for n in G.nodes() if n in FUNC_CAT}
    other_labels = {n: n for n in G.nodes() if n not in FUNC_CAT and deg_map.get(n, 0) > 100}
    nx.draw_networkx_labels(G, pos, hub_labels, ax=ax, font_size=8, font_weight='bold', font_color=DGRAY)
    nx.draw_networkx_labels(G, pos, other_labels, ax=ax, font_size=5.5, font_color=MGRAY)
    ax.set_title(f'Hub-Centric PPI Subnetwork ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=8, label=cat) for cat, c in CAT_COLOR.items()]
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDDDDD',
                          markersize=6, label='Neighbor'))
    ax.legend(handles=handles, loc='lower left', fontsize=7)
    _label(ax, 'A', x=-0.05)
    # Panel B: Degree distribution (log scale)
    ax = fig.add_subplot(gs[1])
    degrees = list(deg_map.values())
    ax.hist(degrees, bins=50, color=BLUE, edgecolor='white', alpha=0.85, log=True)
    ax.set_xlabel('Degree'); ax.set_ylabel('Frequency (log)')
    ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    for h in hub_nodes:
        if h in deg_map:
            ax.axvline(deg_map[h], color=CASE, alpha=0.3, lw=0.8)
    _despine(ax); _label(ax, 'B')
    # Panel C: Centrality bars
    ax = fig.add_subplot(gs[2])
    ct = cent_df.nlargest(15, 'PPIEvidence').sort_values('PPIEvidence', ascending=True)
    colors_c = [CASE if f == 'Up' else BLUE for f in ct['DEG_Flag']]
    ax.barh(range(len(ct)), ct['PPIEvidence'], color=colors_c, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct['gene'], fontsize=8, fontweight='bold')
    ax.set_xlabel('PPI Evidence')
    ax.set_title('Top 15 by Centrality', fontsize=12, fontweight='bold')
    _despine(ax); _label(ax, 'C')

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S3: Functional Enrichment (TF, KEGG, GO)
# ======================================================================
def make_supp_figure_s3(save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
    datasets = [
        (S3V / 'SourceData_Fig3I_TF.csv', 'TF', 'Transcription Factor Enrichment'),
        (S3V / 'SourceData_Fig3J_KEGG.csv', 'Pathway', 'KEGG Pathway Enrichment'),
        (S3V / 'SourceData_Fig3K_GO.csv', 'Term', 'GO Biological Process'),
    ]
    for idx, (path, name_col, title) in enumerate(datasets):
        ax = axes[idx]
        df = pd.read_csv(path).nsmallest(10, 'pvalue').copy()
        df['-log10p'] = -np.log10(df['pvalue'].clip(lower=1e-300))
        df = df.sort_values('-log10p', ascending=True).reset_index(drop=True)
        df['label'] = df[name_col].str.replace(r'\s*\(GO:\d+\)', '', regex=True)
        df['label'] = df['label'].str.replace(r'\s*R-HSA-\d+$', '', regex=True)
        df['label'] = df['label'].apply(lambda s: _wrap(s, 35))
        norm_cs = df['combined_score'] / df['combined_score'].max()
        bar_colors = [plt.cm.YlOrRd(0.3 + 0.6 * v) for v in norm_cs]
        ax.barh(range(len(df)), df['-log10p'], color=bar_colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(df))); ax.set_yticklabels(df['label'], fontsize=7.5)
        ax.set_xlabel(r'$-\log_{10}$(P-value)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(-np.log10(0.05), ls='--', lw=0.8, color=MGRAY)
        _despine(ax); _label(ax, chr(65 + idx))

    fig.tight_layout()
    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S4: Druggability Lollipop Plot
# ======================================================================
def make_supp_figure_s4(save_path=None):
    df = pd.read_csv(S3V / 'SourceData_Fig3L_DrugTarget.csv')
    df = df.sort_values('druggability_score', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(df))
    colors = [CAT_COLOR.get(FUNC_CAT.get(g, ''), MGRAY) for g in df['gene']]
    ax.hlines(y, 0, df['druggability_score'], color=LGRAY, linewidth=1.0)
    ax.scatter(df['druggability_score'], y, c=colors, s=80, edgecolors='white', linewidths=0.6, zorder=3)
    approved = df['max_phase'] == 4
    if approved.any():
        ax.scatter(df.loc[approved, 'druggability_score'], y[approved],
                   marker='*', s=200, c='gold', edgecolors='#B8860B', linewidths=0.5, zorder=4)
    for i, (_, row) in enumerate(df.iterrows()):
        if row['max_phase'] == 4 and pd.notna(row.get('evidence', '')):
            drug_txt = str(row['evidence']).split(':')[-1].strip().split(',')[0].strip()
            if len(drug_txt) > 25: drug_txt = drug_txt[:22] + '...'
            ax.text(row['druggability_score'] + 0.02, i, drug_txt,
                    fontsize=6.5, fontstyle='italic', color=PURPLE, va='center')
    ax.set_yticks(y); ax.set_yticklabels(df['gene'], fontsize=8, fontweight='bold')
    ax.set_xlabel('Druggability Score', fontsize=10)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=7, label=cat) for cat, c in CAT_COLOR.items()]
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                          markersize=12, label='Approved Drug'))
    ax.legend(handles=handles, loc='lower right', fontsize=7)
    _despine(ax)
    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S5: Direction Concordance + External Signature Scoring
# ======================================================================
def make_supp_figure_s5(save_path=None):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1.2, 1.0])
    # Panel A: Concordance heatmap
    ax = fig.add_subplot(gs[0])
    conc = pd.read_csv(S3V / 'SourceData_Fig3M_Concordance.csv')
    genes = conc['gene'].unique()
    datasets = conc['dataset'].unique()
    mat = np.full((len(genes), len(datasets)), np.nan)
    fc_mat = np.full((len(genes), len(datasets)), np.nan)
    gene_idx = {g: i for i, g in enumerate(genes)}
    ds_idx = {d: i for i, d in enumerate(datasets)}
    for _, row in conc.iterrows():
        gi, di = gene_idx[row['gene']], ds_idx[row['dataset']]
        mat[gi, di] = 1.0 if row['concordant'] else 0.0
        fc_mat[gi, di] = row['external_fc']
    cmap_conc = mcolors.ListedColormap([CASE, '#DDDDDD', GREEN])
    display_mat = np.where(np.isnan(mat), 1.0, np.where(mat == 1, 2.0, 0.0))
    im = ax.imshow(display_mat, cmap=cmap_conc, aspect='auto', vmin=0, vmax=2)
    for i in range(len(genes)):
        for j in range(len(datasets)):
            if not np.isnan(fc_mat[i, j]):
                tc = 'white' if display_mat[i, j] != 1.0 else 'black'
                ax.text(j, i, f'{fc_mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color=tc, fontweight='bold')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=7, fontweight='bold')
    ax.set_title('Direction Concordance', fontsize=12, fontweight='bold')
    total_tested = np.sum(~np.isnan(mat))
    total_conc = np.sum(mat == 1)
    rate = total_conc / total_tested * 100 if total_tested > 0 else 0
    ax.text(0.5, -0.08, f'Concordance: {total_conc:.0f}/{total_tested:.0f} ({rate:.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold', color=GREEN)
    _label(ax, 'A')
    # Panel B: Signature scoring boxplots
    ax = fig.add_subplot(gs[1])
    sig = pd.read_csv(S3V / 'SourceData_Fig3N_Signature.csv')
    palette = {'case': CASE, 'control': CTRL}
    sns.boxplot(data=sig, x='dataset', y='score', hue='group', palette=palette, ax=ax,
                width=0.6, fliersize=3, linewidth=0.8)
    for ds in sig['dataset'].unique():
        case_vals = sig[(sig['dataset'] == ds) & (sig['group'] == 'case')]['score']
        ctrl_vals = sig[(sig['dataset'] == ds) & (sig['group'] == 'control')]['score']
        if len(case_vals) > 1 and len(ctrl_vals) > 1:
            _, pv = stats.mannwhitneyu(case_vals, ctrl_vals, alternative='two-sided')
            x_pos = list(sig['dataset'].unique()).index(ds)
            y_max = sig[sig['dataset'] == ds]['score'].max()
            star = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
            ax.text(x_pos, y_max + 0.15, star, ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Signature Score', fontsize=10)
    ax.set_title('External Signature Scoring', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title='Group', fontsize=7, title_fontsize=7)
    _despine(ax); _label(ax, 'B')

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S6: ROC Curves + Validation Scorecard
# ======================================================================
def make_supp_figure_s6(save_path=None):
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1.0, 1.0])
    # Panel A: ROC curves
    ax = fig.add_subplot(gs[0])
    roc = pd.read_csv(S3V / 'SourceData_Fig3O_ROC.csv')
    palette5 = [CASE, BLUE, GREEN, ORANGE, PURPLE]
    for i, ds in enumerate(roc['dataset'].unique()):
        sub = roc[roc['dataset'] == ds]
        auc_val = sub['auc'].iloc[0]
        c = palette5[i % len(palette5)]
        ax.plot(sub['fpr'], sub['tpr'], color=c, linewidth=2.0,
                label=f'{ds} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], ls='--', color=MGRAY, lw=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    _despine(ax); _label(ax, 'A')
    # Panel B: Validation scorecard
    ax = fig.add_subplot(gs[1])
    val = pd.read_csv(S3V / 'SourceData_Fig3P_Validation.csv')
    ax.axis('off')
    table_data = []
    for _, row in val.iterrows():
        ci_lo = f"{row.get('auc_ci_lower', 0):.3f}" if 'auc_ci_lower' in val.columns else '-'
        ci_hi = f"{row.get('auc_ci_upper', 0):.3f}" if 'auc_ci_upper' in val.columns else '-'
        validated = '\u2713' if row.get('validated', False) else '\u2717'
        table_data.append([row['dataset'], str(int(row['n_samples'])),
                          f"{row['auc']:.3f}", f"[{ci_lo}, {ci_hi}]", validated])
    col_labels = ['Dataset', 'N', 'AUC', '95% CI', 'Valid']
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                   cellLoc='center', colWidths=[0.28, 0.1, 0.14, 0.3, 0.1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(LGRAY)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            is_valid = table_data[r - 1][-1] == '\u2713'
            cell.set_facecolor('#E8F5E9' if is_valid else '#FFEBEE')
    ax.set_title('Validation Summary', fontsize=12, fontweight='bold', pad=20)
    _label(ax, 'B')

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S7: DEG Heatmap + Composite Scores + Radar Plots
# ======================================================================
def make_supp_figure_s7(save_path=None):
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30, height_ratios=[1.2, 1.0])
    # Panel A: Heatmap of top 30 DEGs
    ax = fig.add_subplot(gs[0, :])
    hm = pd.read_csv(S1D / 'SourceData_Fig1F_Heatmap.csv', index_col=0)
    hm = hm.head(30)
    hm_z = hm.apply(lambda r: (r - r.mean()) / (r.std() + 1e-10), axis=1)
    ctrl_cols = [c for c in hm_z.columns if 'control' in c.lower()]
    case_cols = [c for c in hm_z.columns if 'case' in c.lower()]
    ordered_cols = ctrl_cols + case_cols
    if len(ordered_cols) == len(hm_z.columns):
        hm_z = hm_z[ordered_cols]
    sns.heatmap(hm_z, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'shrink': 0.5, 'label': 'Z-score'},
                xticklabels=False, yticklabels=True)
    ax.set_ylabel(''); ax.set_xlabel('')
    for i, gene in enumerate(hm_z.index):
        if gene in FUNC_CAT:
            ax.text(-0.5, i + 0.5, '\u2605', fontsize=8,
                    color=CAT_COLOR.get(FUNC_CAT[gene], DGRAY),
                    ha='right', va='center', fontweight='bold')
    ax.tick_params(axis='y', labelsize=7)
    if len(ordered_cols) == len(hm_z.columns):
        n_ctrl, n_case = len(ctrl_cols), len(case_cols)
        cbar_ax = ax.inset_axes([0.0, 1.01, 1.0, 0.02])
        cbar_ax.barh(0, n_ctrl, color=CTRL, height=1)
        cbar_ax.barh(0, n_case, left=n_ctrl, color=CASE, height=1)
        cbar_ax.set_xlim(0, n_ctrl + n_case); cbar_ax.axis('off')
    ax.set_title('Top 30 DEGs (\u2605 = Hub Gene)', fontsize=12, fontweight='bold')
    _label(ax, 'A', x=-0.03)
    # Panel B: Composite score bars
    ax = fig.add_subplot(gs[1, 0])
    top20 = pd.read_csv(S3V / 'SourceData_Fig3D_Top20.csv')
    top20 = top20.sort_values('composite_score', ascending=True).reset_index(drop=True)
    colors_b = [CAT_COLOR.get(FUNC_CAT.get(g, ''), MGRAY) for g in top20['gene']]
    ax.barh(range(len(top20)), top20['composite_score'], color=colors_b,
            edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['gene'], fontsize=8, fontweight='bold')
    for i, (_, row) in enumerate(top20.iterrows()):
        arrow = '\u2191' if row['direction'] == 'Up' else '\u2193'
        c = CASE if row['direction'] == 'Up' else BLUE
        ax.text(row['composite_score'] + 0.005, i, arrow, fontsize=10, color=c, va='center')
    ax.set_xlabel('Composite Score', fontsize=10)
    ax.set_title('Hub Gene Composite Scores', fontsize=12, fontweight='bold')
    _despine(ax); _label(ax, 'B')
    # Panel C: Radar plots for top 5
    sc = pd.read_csv(S2D / 'SourceData_Fig2K_ScoreComponents.csv')
    sc = sc.sort_values('composite_score', ascending=False).head(5).reset_index(drop=True)
    categories_r = ['GS_norm', 'MM_norm', 'kWithin_norm']
    cat_labels = ['GS', 'MM', 'kWithin']
    N = len(categories_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    gs_radar = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 1], wspace=0.3)
    for i in range(len(sc)):
        ax = fig.add_subplot(gs_radar[0, i], polar=True)
        row = sc.iloc[i]
        gene = row['gene']
        values = [row[c] for c in categories_r]
        values += values[:1]
        cat = FUNC_CAT.get(gene, 'Metabolism')
        color = CAT_COLOR.get(cat, ORANGE)
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cat_labels, fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(gene, fontsize=9, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=5)
        if i == 0: _label(ax, 'C', x=-0.3, y=1.2)

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S8: Drug-Gene Targeting Map + Drug-Target-Ion Channel-Effect Axis
# ======================================================================
def make_supp_figure_s8(save_path=None):
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 1, hspace=0.40, height_ratios=[1.0, 1.2])
    dtev = pd.read_csv(S4D / 'SourceData_Fig4B_DrugTargetEvidence.csv')
    dtev = dtev[dtev['in_overlap'] == True].copy()
    bridge = pd.read_csv(S4D / 'SourceData_Fig4C_IonBridgePaths.csv')
    # Panel A: Drug-gene bipartite map
    ax = fig.add_subplot(gs[0])
    ax.axis('off')
    drugs = sorted(dtev['drug'].unique())
    genes_a = sorted(dtev['gene'].unique())
    max_y = max(len(drugs), len(genes_a))
    drug_y = {d: i * (max_y / max(len(drugs), 1)) for i, d in enumerate(drugs)}
    gene_y = {g: i * (max_y / max(len(genes_a), 1)) for i, g in enumerate(genes_a)}
    ax.set_xlim(0, 10); ax.set_ylim(-1, max_y + 0.5)
    for d, yp in drug_y.items():
        ax.add_patch(FancyBboxPatch((0.3, yp - 0.3), 2.5, 0.6, boxstyle='round,pad=0.1',
                     facecolor='#F0F0F8', edgecolor=PURPLE, linewidth=0.8))
        ax.text(1.55, yp, d.title()[:20], ha='center', va='center', fontsize=6.5,
                fontweight='bold', color=PURPLE)
    for g, yp in gene_y.items():
        cat = FUNC_CAT.get(g, 'Metabolism')
        c = CAT_COLOR.get(cat, ORANGE)
        ax.add_patch(FancyBboxPatch((7.0, yp - 0.3), 2.5, 0.6, boxstyle='round,pad=0.1',
                     facecolor=c, edgecolor='white', linewidth=0.8, alpha=0.3))
        ax.text(8.25, yp, g, ha='center', va='center', fontsize=7, fontweight='bold', color=DGRAY)
    for _, row in dtev.iterrows():
        dy = drug_y.get(row['drug'], 0)
        gy = gene_y.get(row['gene'], 0)
        ls = '-' if row['edge_type'] == 'DirectTarget' else '--'
        ax.plot([2.8, 7.0], [dy, gy], color=MGRAY, linewidth=0.6, linestyle=ls, alpha=0.5)
    ax.text(1.55, max_y + 0.3, 'Drugs', ha='center', fontsize=10, fontweight='bold', color=PURPLE)
    ax.text(8.25, max_y + 0.3, 'Hub Genes', ha='center', fontsize=10, fontweight='bold', color=DGRAY)
    _label(ax, 'A', x=-0.02, y=1.05)
    # Panel B: Bridge path flow
    ax = fig.add_subplot(gs[1])
    ax.axis('off')
    paths = []
    for _, row in bridge.iterrows():
        parts = [p.strip() for p in row['path_genes'].split('->')]
        paths.append({'hub': row['start_gene'],
                      'intermediates': parts[1:-1] if len(parts) > 2 else [],
                      'ion': row['end_ion_channel'],
                      'score': row['path_score'], 'grade': row['evidence_grade']})
    hubs = list(dict.fromkeys(p['hub'] for p in paths))
    intermediates = list(dict.fromkeys(im for p in paths for im in p['intermediates']))
    ions = list(dict.fromkeys(p['ion'] for p in paths))
    max_items = max(len(hubs), len(intermediates), len(ions), 1)
    def _ypos(items, max_n):
        return {item: i * (max_n / max(len(items), 1)) for i, item in enumerate(items)}
    hub_y = _ypos(hubs, max_items)
    int_y = _ypos(intermediates, max_items) if intermediates else {}
    ion_y = _ypos(ions, max_items)
    x_hub, x_int, x_ion = 1.5, 5.0, 8.5
    ax.set_xlim(0, 10); ax.set_ylim(-1, max_items + 0.5)
    for h, yp in hub_y.items():
        cat = FUNC_CAT.get(h, 'Metabolism')
        c = CAT_COLOR.get(cat, ORANGE)
        ax.text(x_hub, yp, h, ha='center', va='center', fontsize=7, fontweight='bold',
                color=DGRAY, bbox=dict(boxstyle='round,pad=0.15', facecolor=c, alpha=0.3, edgecolor='white'))
    for im_name, yp in int_y.items():
        ax.text(x_int, yp, im_name, ha='center', va='center', fontsize=6.5,
                color=DGRAY, bbox=dict(boxstyle='round,pad=0.1', facecolor=LGRAY, edgecolor=MGRAY, linewidth=0.5))
    for ion_name, yp in ion_y.items():
        ax.text(x_ion, yp, ion_name, ha='center', va='center', fontsize=7, fontweight='bold',
                fontstyle='italic', color=BLUE,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#E3F2FD', edgecolor=BLUE, linewidth=0.5))
    for p in paths:
        alpha = 0.7 if p['grade'] == 'strong' else 0.3
        lw = 1.2 if p['grade'] == 'strong' else 0.6
        hy = hub_y.get(p['hub'], 0)
        iy = ion_y.get(p['ion'], 0)
        if p['intermediates'] and p['intermediates'][0] in int_y:
            my = int_y[p['intermediates'][0]]
            ax.plot([x_hub + 0.5, x_int - 0.5], [hy, my], color=MGRAY, lw=lw, alpha=alpha)
            ax.plot([x_int + 0.5, x_ion - 0.5], [my, iy], color=MGRAY, lw=lw, alpha=alpha)
        else:
            ax.plot([x_hub + 0.5, x_ion - 0.5], [hy, iy], color=MGRAY, lw=lw, alpha=alpha)
    ax.text(x_hub, max_items + 0.3, 'Hub Gene', ha='center', fontsize=10, fontweight='bold')
    if intermediates:
        ax.text(x_int, max_items + 0.3, 'PPI Intermediate', ha='center', fontsize=10, fontweight='bold')
    ax.text(x_ion, max_items + 0.3, 'Ion Channel', ha='center', fontsize=10, fontweight='bold', color=BLUE)
    _label(ax, 'B', x=-0.02, y=1.05)

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# S9: Translational Summary
# ======================================================================
def make_supp_figure_s9(save_path=None):
    pr = pd.read_csv(S4D / 'SourceData_Fig4D_PriorityRanking.csv')
    pr_top = pr[pr['in_overlap'] == True].nlargest(5, 'PriorityScore').reset_index(drop=True)
    bridge = pd.read_csv(S4D / 'SourceData_Fig4C_IonBridgePaths.csv')
    dtev = pd.read_csv(S4D / 'SourceData_Fig4B_DrugTargetEvidence.csv')
    dtev = dtev[dtev['in_overlap'] == True]
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18); ax.set_ylim(-0.5, 5.5); ax.axis('off')
    headers = ['Rank', 'Hub Gene', 'Score', 'Drug(s)', 'Bridge Path', 'Ion Channel']
    x_pos = [0.5, 2.5, 4.5, 6.5, 10.0, 14.5]
    for xp, hdr in zip(x_pos, headers):
        ax.text(xp, 5.2, hdr, ha='center', va='center', fontsize=10, fontweight='bold', color=DGRAY)
    ax.axhline(5.0, color=DGRAY, linewidth=1.5, xmin=0.01, xmax=0.99)
    for i, (_, row) in enumerate(pr_top.iterrows()):
        y = 4.3 - i * 1.0
        gene = row['gene']
        cat = FUNC_CAT.get(gene, 'Metabolism')
        color = CAT_COLOR.get(cat, ORANGE)
        ax.text(x_pos[0], y, f'#{i+1}', ha='center', va='center', fontsize=12,
                fontweight='bold', color=DGRAY)
        ax.add_patch(FancyBboxPatch((x_pos[1] - 0.8, y - 0.3), 1.6, 0.6,
                     boxstyle='round,pad=0.1', facecolor=color, alpha=0.3,
                     edgecolor=color, linewidth=1))
        ax.text(x_pos[1], y, gene, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x_pos[2], y, f'{row["PriorityScore"]:.3f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=DGRAY)
        gene_drugs = dtev[dtev['gene'] == gene]['drug'].unique()
        drug_str = ', '.join(d.title()[:15] for d in gene_drugs[:2]) if len(gene_drugs) > 0 else '-'
        ax.text(x_pos[3], y, drug_str, ha='center', va='center', fontsize=7.5,
                fontstyle='italic', color=PURPLE)
        gene_paths = bridge[bridge['start_gene'] == gene]
        if len(gene_paths) > 0:
            path_str = gene_paths.iloc[0]['path_genes']
            if len(path_str) > 35: path_str = path_str[:32] + '...'
            ax.text(x_pos[4], y, path_str, ha='center', va='center', fontsize=7, color=DGRAY)
            ion = gene_paths.iloc[0]['end_ion_channel']
            ax.text(x_pos[5], y, ion, ha='center', va='center', fontsize=9,
                    fontweight='bold', fontstyle='italic', color=BLUE)
        if i < len(pr_top) - 1:
            ax.axhline(y - 0.5, color=LGRAY, linewidth=0.5, xmin=0.01, xmax=0.99)
    handles = [Patch(facecolor=c, alpha=0.3, edgecolor=c, label=cat)
               for cat, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=8, title='Functional Category',
              title_fontsize=8)

    if save_path: _save(fig, save_path)
    return fig


# ======================================================================
# COMPOSITE FIGURES (6 merged supplementary figures)
# ======================================================================


def make_composite_figure_s2(save_path=None):
    """Composite S2: Network & Enrichment Analysis (S2 + S3 merged)."""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                          height_ratios=[1.0, 1.0],
                          width_ratios=[1.2, 0.8, 0.8])
    edges_df = pd.read_csv(S3V / 'SourceData_Fig3E_PPI_Edges.csv')
    deg_df = pd.read_csv(S3V / 'SourceData_Fig3F_Degrees.csv')
    cent_df = pd.read_csv(S3V / 'SourceData_Fig3G_Centrality.csv')
    deg_map = dict(zip(deg_df['node'], deg_df['degree']))
    # --- Row 1: PPI Network panels (from S2) ---
    # Panel A: Hub-centric subnetwork
    ax = fig.add_subplot(gs[0, 0])
    G_full = nx.from_pandas_edgelist(edges_df, 'source', 'target')
    hub_nodes = [n for n in G_full.nodes() if n in FUNC_CAT]
    neighbor_hub_count = {}
    for h in hub_nodes:
        if h in G_full:
            for nb in G_full.neighbors(h):
                if nb not in FUNC_CAT:
                    neighbor_hub_count[nb] = neighbor_hub_count.get(nb, 0) + 1
    TARGET_MAX = 70
    for min_hubs in range(max(neighbor_hub_count.values(), default=2), 0, -1):
        kept = {n for n, c in neighbor_hub_count.items() if c >= min_hubs}
        if len(kept) + len(hub_nodes) >= 30 or min_hubs == 1:
            break
    if len(kept) + len(hub_nodes) > TARGET_MAX:
        kept = set(sorted(kept, key=lambda n: neighbor_hub_count[n], reverse=True)[:TARGET_MAX - len(hub_nodes)])
    sub_nodes = set(hub_nodes) | kept
    G = G_full.subgraph(sub_nodes).copy()
    node_colors = [CAT_COLOR.get(FUNC_CAT.get(n, ''), '#DDDDDD') for n in G.nodes()]
    node_sizes = []
    for n in G.nodes():
        d = deg_map.get(n, 1)
        if n in FUNC_CAT:
            node_sizes.append(max(d, 30) * 2.5)
        else:
            node_sizes.append(max(d * 0.3, 15))
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color=MGRAY, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, edgecolors='white', linewidths=0.5)
    hub_labels = {n: n for n in G.nodes() if n in FUNC_CAT}
    other_labels = {n: n for n in G.nodes() if n not in FUNC_CAT and deg_map.get(n, 0) > 100}
    nx.draw_networkx_labels(G, pos, hub_labels, ax=ax, font_size=7, font_weight='bold', font_color=DGRAY)
    nx.draw_networkx_labels(G, pos, other_labels, ax=ax, font_size=5, font_color=MGRAY)
    ax.set_title(f'Hub-Centric PPI Subnetwork ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=7, label=cat) for cat, c in CAT_COLOR.items()]
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDDDDD',
                          markersize=5, label='Neighbor'))
    ax.legend(handles=handles, loc='lower left', fontsize=6)
    _label(ax, 'A', x=-0.05)
    # Panel B: Degree distribution
    ax = fig.add_subplot(gs[0, 1])
    degrees = list(deg_map.values())
    ax.hist(degrees, bins=50, color=BLUE, edgecolor='white', alpha=0.85, log=True)
    ax.set_xlabel('Degree'); ax.set_ylabel('Frequency (log)')
    ax.set_title('Degree Distribution', fontsize=10, fontweight='bold')
    for h in hub_nodes:
        if h in deg_map:
            ax.axvline(deg_map[h], color=CASE, alpha=0.3, lw=0.8)
    _despine(ax); _label(ax, 'B')
    # Panel C: Centrality bars
    ax = fig.add_subplot(gs[0, 2])
    ct = cent_df.nlargest(15, 'PPIEvidence').sort_values('PPIEvidence', ascending=True)
    colors_c = [CASE if f == 'Up' else BLUE for f in ct['DEG_Flag']]
    ax.barh(range(len(ct)), ct['PPIEvidence'], color=colors_c, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct['gene'], fontsize=7, fontweight='bold')
    ax.set_xlabel('PPI Evidence')
    ax.set_title('Top 15 by Centrality', fontsize=10, fontweight='bold')
    _despine(ax); _label(ax, 'C')
    # --- Row 2: Functional Enrichment panels (from S3) ---
    datasets = [
        (S3V / 'SourceData_Fig3I_TF.csv', 'TF', 'TF Enrichment'),
        (S3V / 'SourceData_Fig3J_KEGG.csv', 'Pathway', 'KEGG Pathway'),
        (S3V / 'SourceData_Fig3K_GO.csv', 'Term', 'GO Biological Process'),
    ]
    for idx, (path, name_col, title) in enumerate(datasets):
        ax = fig.add_subplot(gs[1, idx])
        df = pd.read_csv(path).nsmallest(10, 'pvalue').copy()
        df['-log10p'] = -np.log10(df['pvalue'].clip(lower=1e-300))
        df = df.sort_values('-log10p', ascending=True).reset_index(drop=True)
        df['label'] = df[name_col].str.replace(r'\s*\(GO:\d+\)', '', regex=True)
        df['label'] = df['label'].str.replace(r'\s*R-HSA-\d+$', '', regex=True)
        df['label'] = df['label'].apply(lambda s: _wrap(s, 32))
        norm_cs = df['combined_score'] / df['combined_score'].max()
        bar_colors = [plt.cm.YlOrRd(0.3 + 0.6 * v) for v in norm_cs]
        ax.barh(range(len(df)), df['-log10p'], color=bar_colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(df))); ax.set_yticklabels(df['label'], fontsize=6.5)
        ax.set_xlabel(r'$-\log_{10}$(P-value)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axvline(-np.log10(0.05), ls='--', lw=0.8, color=MGRAY)
        _despine(ax); _label(ax, chr(68 + idx))  # D, E, F

    if save_path: _save(fig, save_path)
    return fig


def make_composite_figure_s4(save_path=None):
    """Composite S4: External Validation (S5 + S6 merged)."""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    # --- Panel A: Concordance heatmap (from S5) ---
    ax = fig.add_subplot(gs[0, 0])
    conc = pd.read_csv(S3V / 'SourceData_Fig3M_Concordance.csv')
    genes = conc['gene'].unique()
    datasets_c = conc['dataset'].unique()
    mat = np.full((len(genes), len(datasets_c)), np.nan)
    fc_mat = np.full((len(genes), len(datasets_c)), np.nan)
    gene_idx = {g: i for i, g in enumerate(genes)}
    ds_idx = {d: i for i, d in enumerate(datasets_c)}
    for _, row in conc.iterrows():
        gi, di = gene_idx[row['gene']], ds_idx[row['dataset']]
        mat[gi, di] = 1.0 if row['concordant'] else 0.0
        fc_mat[gi, di] = row['external_fc']
    cmap_conc = mcolors.ListedColormap([CASE, '#DDDDDD', GREEN])
    display_mat = np.where(np.isnan(mat), 1.0, np.where(mat == 1, 2.0, 0.0))
    im = ax.imshow(display_mat, cmap=cmap_conc, aspect='auto', vmin=0, vmax=2)
    for i in range(len(genes)):
        for j in range(len(datasets_c)):
            if not np.isnan(fc_mat[i, j]):
                tc = 'white' if display_mat[i, j] != 1.0 else 'black'
                ax.text(j, i, f'{fc_mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color=tc, fontweight='bold')
    ax.set_xticks(range(len(datasets_c)))
    ax.set_xticklabels(datasets_c, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=6.5, fontweight='bold')
    ax.set_title('Direction Concordance', fontsize=10, fontweight='bold')
    total_tested = np.sum(~np.isnan(mat))
    total_conc = np.sum(mat == 1)
    rate = total_conc / total_tested * 100 if total_tested > 0 else 0
    ax.text(0.5, -0.08, f'Concordance: {total_conc:.0f}/{total_tested:.0f} ({rate:.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=8, fontweight='bold', color=GREEN)
    _label(ax, 'A')
    # --- Panel B: Signature scoring boxplots (from S5) ---
    ax = fig.add_subplot(gs[0, 1])
    sig = pd.read_csv(S3V / 'SourceData_Fig3N_Signature.csv')
    palette = {'case': CASE, 'control': CTRL}
    sns.boxplot(data=sig, x='dataset', y='score', hue='group', palette=palette, ax=ax,
                width=0.6, fliersize=3, linewidth=0.8)
    for ds in sig['dataset'].unique():
        case_vals = sig[(sig['dataset'] == ds) & (sig['group'] == 'case')]['score']
        ctrl_vals = sig[(sig['dataset'] == ds) & (sig['group'] == 'control')]['score']
        if len(case_vals) > 1 and len(ctrl_vals) > 1:
            _, pv = stats.mannwhitneyu(case_vals, ctrl_vals, alternative='two-sided')
            x_pos = list(sig['dataset'].unique()).index(ds)
            y_max = sig[sig['dataset'] == ds]['score'].max()
            star = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
            ax.text(x_pos, y_max + 0.15, star, ha='center', fontsize=7, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=9); ax.set_ylabel('Signature Score', fontsize=9)
    ax.set_title('External Signature Scoring', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title='Group', fontsize=6, title_fontsize=6)
    _despine(ax); _label(ax, 'B')
    # --- Panel C: ROC curves (from S6) ---
    ax = fig.add_subplot(gs[1, 0])
    roc = pd.read_csv(S3V / 'SourceData_Fig3O_ROC.csv')
    palette5 = [CASE, BLUE, GREEN, ORANGE, PURPLE]
    for i, ds in enumerate(roc['dataset'].unique()):
        sub = roc[roc['dataset'] == ds]
        auc_val = sub['auc'].iloc[0]
        c = palette5[i % len(palette5)]
        ax.plot(sub['fpr'], sub['tpr'], color=c, linewidth=2.0,
                label=f'{ds} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], ls='--', color=MGRAY, lw=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title('ROC Curves', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    _despine(ax); _label(ax, 'C')
    # --- Panel D: Validation scorecard (from S6) ---
    ax = fig.add_subplot(gs[1, 1])
    val = pd.read_csv(S3V / 'SourceData_Fig3P_Validation.csv')
    ax.axis('off')
    table_data = []
    for _, row in val.iterrows():
        ci_lo = f"{row.get('auc_ci_lower', 0):.3f}" if 'auc_ci_lower' in val.columns else '-'
        ci_hi = f"{row.get('auc_ci_upper', 0):.3f}" if 'auc_ci_upper' in val.columns else '-'
        validated = '\u2713' if row.get('validated', False) else '\u2717'
        table_data.append([row['dataset'], str(int(row['n_samples'])),
                          f"{row['auc']:.3f}", f"[{ci_lo}, {ci_hi}]", validated])
    col_labels = ['Dataset', 'N', 'AUC', '95% CI', 'Valid']
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                   cellLoc='center', colWidths=[0.28, 0.1, 0.14, 0.3, 0.1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(LGRAY)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            is_valid = table_data[r - 1][-1] == '\u2713'
            cell.set_facecolor('#E8F5E9' if is_valid else '#FFEBEE')
    ax.set_title('Validation Summary', fontsize=10, fontweight='bold', pad=20)
    _label(ax, 'D')

    if save_path: _save(fig, save_path)
    return fig


def make_composite_figure_s5(save_path=None):
    """Composite S5: Hub Gene Characterization (S7 + S8 merged)."""
    fig = plt.figure(figsize=(24, 19))
    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.30,
                          height_ratios=[1.0, 0.8, 1.1])
    # --- Panel A: Heatmap of top 30 DEGs (from S7, spans full width) ---
    ax = fig.add_subplot(gs[0, :])
    hm = pd.read_csv(S1D / 'SourceData_Fig1F_Heatmap.csv', index_col=0)
    hm = hm.head(30)
    hm_z = hm.apply(lambda r: (r - r.mean()) / (r.std() + 1e-10), axis=1)
    ctrl_cols = [c for c in hm_z.columns if 'control' in c.lower()]
    case_cols = [c for c in hm_z.columns if 'case' in c.lower()]
    ordered_cols = ctrl_cols + case_cols
    if len(ordered_cols) == len(hm_z.columns):
        hm_z = hm_z[ordered_cols]
    sns.heatmap(hm_z, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'shrink': 0.5, 'label': 'Z-score'},
                xticklabels=False, yticklabels=True)
    ax.set_ylabel(''); ax.set_xlabel('')
    for i, gene in enumerate(hm_z.index):
        if gene in FUNC_CAT:
            ax.text(-0.5, i + 0.5, '\u2605', fontsize=7,
                    color=CAT_COLOR.get(FUNC_CAT[gene], DGRAY),
                    ha='right', va='center', fontweight='bold')
    ax.tick_params(axis='y', labelsize=6)
    if len(ordered_cols) == len(hm_z.columns):
        n_ctrl, n_case = len(ctrl_cols), len(case_cols)
        cbar_ax = ax.inset_axes([0.0, 1.01, 1.0, 0.02])
        cbar_ax.barh(0, n_ctrl, color=CTRL, height=1)
        cbar_ax.barh(0, n_case, left=n_ctrl, color=CASE, height=1)
        cbar_ax.set_xlim(0, n_ctrl + n_case); cbar_ax.axis('off')
    ax.set_title('Top 30 DEGs (\u2605 = Hub Gene)', fontsize=10, fontweight='bold')
    _label(ax, 'A', x=-0.03)
    # --- Panel B: Composite score bars (from S7) ---
    ax = fig.add_subplot(gs[1, 0])
    top20 = pd.read_csv(S3V / 'SourceData_Fig3D_Top20.csv')
    top20 = top20.sort_values('composite_score', ascending=True).reset_index(drop=True)
    colors_b = [CAT_COLOR.get(FUNC_CAT.get(g, ''), MGRAY) for g in top20['gene']]
    ax.barh(range(len(top20)), top20['composite_score'], color=colors_b,
            edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['gene'], fontsize=7, fontweight='bold')
    for i, (_, row) in enumerate(top20.iterrows()):
        arrow = '\u2191' if row['direction'] == 'Up' else '\u2193'
        c = CASE if row['direction'] == 'Up' else BLUE
        ax.text(row['composite_score'] + 0.005, i, arrow, fontsize=9, color=c, va='center')
    ax.set_xlabel('Composite Score', fontsize=9)
    ax.set_title('Hub Gene Composite Scores', fontsize=10, fontweight='bold')
    handles_b = [Patch(facecolor=c, edgecolor='white', label=cat) for cat, c in CAT_COLOR.items()]
    ax.legend(handles=handles_b, loc='lower right', fontsize=6, framealpha=0.9)
    _despine(ax); _label(ax, 'B')
    # --- Panel C: Radar plots for top 2 per category (2x4 grid) ---
    sc = pd.read_csv(S2D / 'SourceData_Fig2K_ScoreComponents.csv')
    sc['_cat'] = sc['gene'].map(FUNC_CAT)
    sc = sc.dropna(subset=['_cat'])
    radar_genes = sc.groupby('_cat').apply(
        lambda g: g.nlargest(2, 'composite_score')).reset_index(drop=True)
    cat_order = list(CAT_COLOR.keys())
    radar_genes['_cat_rank'] = radar_genes['_cat'].map({c: i for i, c in enumerate(cat_order)})
    radar_genes = radar_genes.sort_values(['_cat_rank', 'composite_score'],
                                          ascending=[True, False]).reset_index(drop=True)
    categories_r = ['GS_norm', 'MM_norm', 'kWithin_norm']
    cat_labels = ['GS', 'MM', 'kWithin']
    N = len(categories_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    gs_radar = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, 1], wspace=0.45, hspace=0.55)
    for i in range(len(radar_genes)):
        r_row, r_col = divmod(i, 4)
        ax = fig.add_subplot(gs_radar[r_row, r_col], polar=True)
        row = radar_genes.iloc[i]
        gene = row['gene']
        values = [row[c] for c in categories_r]
        values += values[:1]
        cat = FUNC_CAT.get(gene, 'Metabolism')
        color = CAT_COLOR.get(cat, ORANGE)
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cat_labels, fontsize=5)
        ax.set_ylim(0, 1)
        ax.set_title(gene, fontsize=7, fontweight='bold', pad=8)
        ax.tick_params(axis='y', labelsize=4)
        if i == 0: _label(ax, 'C', x=-0.3, y=1.2)
    # --- Panel D: Drug-Target Sankey Diagram (bottom left) ---
    dtev = pd.read_csv(S4D / 'SourceData_Fig4B_DrugTargetEvidence.csv')
    dtev = dtev[dtev['in_overlap'] == True].copy()
    bridge = pd.read_csv(S4D / 'SourceData_Fig4C_IonBridgePaths.csv')
    ax = fig.add_subplot(gs[2, 0])
    ax.set_xlim(-0.5, 10.5); ax.axis('off')
    drugs = sorted(dtev['drug'].unique())
    genes_d = sorted(dtev['gene'].unique())
    n_drugs, n_genes = len(drugs), len(genes_d)
    # Evenly space nodes vertically
    drug_y = {d: i for i, d in enumerate(drugs)}
    gene_y = {g: i * ((n_drugs - 1) / max(n_genes - 1, 1)) for i, g in enumerate(genes_d)}
    ax.set_ylim(-1, n_drugs + 0.5)
    ax.invert_yaxis()
    # Drug palette: muted tones per drug
    drug_palette = [
        '#8491B4', '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
        '#F39B7F', '#B09C85', '#7E6148', '#91D1C2', '#DC0000',
    ]
    drug_color = {d: drug_palette[i % len(drug_palette)] for i, d in enumerate(drugs)}
    # Draw Sankey ribbons (smooth Bezier curves with translucent fill)
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches
    x_left, x_right = 2.8, 7.2
    ribbon_hw = 0.18  # half-width of ribbon
    for _, row in dtev.iterrows():
        dy = drug_y[row['drug']]
        gy = gene_y[row['gene']]
        dc = drug_color[row['drug']]
        # Cubic Bezier control points for smooth S-curve
        mid_x = (x_left + x_right) / 2
        verts = [
            (x_left, dy - ribbon_hw),
            (mid_x, dy - ribbon_hw),
            (mid_x, gy - ribbon_hw),
            (x_right, gy - ribbon_hw),
            (x_right, gy + ribbon_hw),
            (mid_x, gy + ribbon_hw),
            (mid_x, dy + ribbon_hw),
            (x_left, dy + ribbon_hw),
            (x_left, dy - ribbon_hw),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=dc, edgecolor=dc, alpha=0.45, lw=0.3)
        ax.add_patch(patch)
    # Draw drug nodes (left)
    for d, yp in drug_y.items():
        dc = drug_color[d]
        ax.add_patch(FancyBboxPatch((0.0, yp - 0.32), 2.7, 0.64, boxstyle='round,pad=0.12',
                     facecolor=dc, edgecolor='white', linewidth=0.8, alpha=0.85))
        label = d.title()
        if len(label) > 18: label = label[:16] + '...'
        ax.text(1.35, yp, label, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='white')
    # Draw gene nodes (right)
    for g, yp in gene_y.items():
        cat = FUNC_CAT.get(g, 'Metabolism')
        c = CAT_COLOR.get(cat, ORANGE)
        ax.add_patch(FancyBboxPatch((7.3, yp - 0.32), 2.7, 0.64, boxstyle='round,pad=0.12',
                     facecolor=c, edgecolor='white', linewidth=0.8, alpha=0.85))
        ax.text(8.65, yp, g, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='white')
    ax.text(1.35, -0.7, 'Drugs', ha='center', fontsize=10, fontweight='bold', color=PURPLE)
    ax.text(8.65, -0.7, 'Target Genes', ha='center', fontsize=10, fontweight='bold', color=DGRAY)
    ax.set_title('Drug\u2013Target Sankey Diagram', fontsize=10, fontweight='bold', pad=12)
    _label(ax, 'D', x=-0.05, y=1.08)
    # --- Panel E: PPI & Ion Channel Network (bottom right) ---
    ax = fig.add_subplot(gs[2, 1])
    G_net = nx.Graph()
    hub_set = set()
    int_set = set()
    ion_set = set()
    for _, row in bridge.iterrows():
        parts = [p.strip() for p in row['path_genes'].split('->')]
        hub_gene = row['start_gene']
        ion_ch = row['end_ion_channel']
        hub_set.add(hub_gene)
        ion_set.add(ion_ch)
        score = row['path_score']
        grade = row['evidence_grade']
        for k in range(len(parts) - 1):
            G_net.add_edge(parts[k], parts[k+1], weight=score, grade=grade)
        if len(parts) > 2:
            for mid in parts[1:-1]:
                int_set.add(mid)
    # Track direct hub->ion pairs (no intermediate)
    direct_pairs = set()
    for _, row in bridge.iterrows():
        parts = [p.strip() for p in row['path_genes'].split('->')]
        if len(parts) == 2:
            direct_pairs.add((parts[0], parts[1]))
    # Node classification
    node_type = {}
    for n in G_net.nodes():
        if n in ion_set:
            node_type[n] = 'Ion Channel'
        elif n in hub_set:
            node_type[n] = 'Hub Gene'
        else:
            node_type[n] = 'PPI Intermediate'
    # Colors and sizes per type
    type_color = {'Hub Gene': CASE, 'PPI Intermediate': PURPLE, 'Ion Channel': BLUE}
    type_marker = {'Hub Gene': 'o', 'PPI Intermediate': 's', 'Ion Channel': 'D'}
    type_size = {'Hub Gene': 280, 'PPI Intermediate': 120, 'Ion Channel': 200}
    # Structured 3-column layout: Hub (left) -> Intermediate (center) -> Ion Channel (right)
    pos = {}
    hub_list = sorted(hub_set & set(G_net.nodes()))
    int_list = sorted(int_set & set(G_net.nodes()))
    ion_list = sorted(ion_set & set(G_net.nodes()))
    for i, n in enumerate(hub_list):
        pos[n] = (0.0, -i * 1.5 / max(len(hub_list) - 1, 1))
    for i, n in enumerate(int_list):
        pos[n] = (1.0, -i * 1.5 / max(len(int_list) - 1, 1))
    for i, n in enumerate(ion_list):
        pos[n] = (2.0, -i * 1.5 / max(len(ion_list) - 1, 1))
    # Handle any nodes not in the three sets
    for n in G_net.nodes():
        if n not in pos:
            pos[n] = (1.0, -1.2)
    # Edge endpoint offsets for box sides
    int_box_hw = 0.15  # half-width of intermediate boxes in data coords
    def _edge_x(node):
        """Return (x_left_attach, x_right_attach) for a node."""
        px = pos[node][0]
        if node_type.get(node) == 'PPI Intermediate':
            return (px - int_box_hw, px + int_box_hw)
        return (px, px)  # circles/diamonds: connect at center
    # Draw edges with curved connections, attaching to box sides
    for u, v, data in G_net.edges(data=True):
        is_direct = (u in hub_set and v in ion_set and (u, v) in direct_pairs) or \
                    (v in hub_set and u in ion_set and (v, u) in direct_pairs)
        ux, vx = pos[u][0], pos[v][0]
        uy, vy = pos[u][1], pos[v][1]
        u_left, u_right = _edge_x(u)
        v_left, v_right = _edge_x(v)
        x0 = u_right if vx > ux else u_left
        x1 = v_left if vx > ux else v_right
        if is_direct:
            # Wide arc that clearly bypasses the PPI intermediate column
            rad = -0.35 if uy <= vy else 0.35
            # Glow: wider translucent line underneath
            ax.annotate('', xy=(x1, vy), xytext=(x0, uy),
                        arrowprops=dict(arrowstyle='-', color=GREEN, lw=5.0,
                                       alpha=0.15,
                                       connectionstyle=f'arc3,rad={rad}'),
                        zorder=1)
            # Main direct edge: thick dashed green arrow
            ax.annotate('', xy=(x1, vy), xytext=(x0, uy),
                        arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.2,
                                       alpha=0.85, linestyle='dashed',
                                       connectionstyle=f'arc3,rad={rad}'),
                        zorder=2)
        else:
            lw = 1.8 if data.get('grade') == 'strong' else 0.9
            alpha = 0.55 if data.get('grade') == 'strong' else 0.3
            ec = '#888888' if data.get('grade') == 'strong' else '#BBBBBB'
            ax.annotate('', xy=(x1, vy), xytext=(x0, uy),
                        arrowprops=dict(arrowstyle='-', color=ec, lw=lw, alpha=alpha,
                                       connectionstyle='arc3,rad=0.08'),
                        zorder=1)
    # Draw Hub Gene and Ion Channel nodes (scatter)
    for ntype in ['Ion Channel', 'Hub Gene']:
        nodes = [n for n in G_net.nodes() if node_type.get(n) == ntype]
        if not nodes: continue
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        ax.scatter(xs, ys, c=type_color[ntype], s=type_size[ntype],
                   marker=type_marker[ntype], edgecolors='white', linewidths=1.0,
                   alpha=0.92, zorder=3 if ntype == 'Hub Gene' else 2)
    # Draw PPI Intermediate nodes as small rounded rects with white labels inside
    int_box_hh = 0.025  # half-height
    for n in G_net.nodes():
        if node_type.get(n) != 'PPI Intermediate': continue
        px, py = pos[n]
        ax.add_patch(FancyBboxPatch((px - int_box_hw, py - int_box_hh),
                     int_box_hw * 2, int_box_hh * 2,
                     boxstyle='round,pad=0.015', facecolor=PURPLE, edgecolor='white',
                     linewidth=0.6, alpha=0.88, zorder=3))
        ax.text(px, py, n, fontsize=5, fontweight='bold', color='white',
                ha='center', va='center', zorder=4)
    # Labels for Hub Gene and Ion Channel only
    for n in G_net.nodes():
        if node_type[n] == 'PPI Intermediate': continue
        fs = 7.5 if node_type[n] == 'Hub Gene' else 7
        fw = 'bold'
        fc = DGRAY if node_type[n] == 'Hub Gene' else BLUE
        if node_type[n] == 'Hub Gene':
            ax.text(pos[n][0] - 0.08, pos[n][1], n, fontsize=fs, fontweight=fw,
                    color=fc, ha='right', va='center', zorder=4)
        else:
            ax.text(pos[n][0] + 0.08, pos[n][1], n, fontsize=fs, fontweight=fw,
                    color=fc, ha='left', va='center', zorder=4)
    # Column headers
    y_top = 0.18
    ax.text(0.0, y_top, 'Hub Genes', ha='center', fontsize=8, fontweight='bold', color=CASE)
    ax.text(1.0, y_top, 'PPI Intermediates', ha='center', fontsize=8, fontweight='bold', color=PURPLE)
    ax.text(2.0, y_top, 'Ion Channels', ha='center', fontsize=8, fontweight='bold', color=BLUE)
    # Legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CASE,
               markersize=9, markeredgecolor='white', label='Hub Gene'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=PURPLE,
               markersize=7, markeredgecolor='white', label='PPI Intermediate'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=BLUE,
               markersize=7, markeredgecolor='white', label='Ion Channel'),
        Line2D([0], [0], color='#888888', lw=1.8, label='Strong (via PPI)'),
        Line2D([0], [0], color='#BBBBBB', lw=0.9, alpha=0.5, label='Moderate (via PPI)'),
        Line2D([0], [0], color=GREEN, lw=1.8, linestyle='dashed', label='Direct (Hub \u2192 Ion)'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=6,
              framealpha=0.9, edgecolor=LGRAY, borderpad=0.6, columnspacing=1.0,
              bbox_to_anchor=(0.5, -0.06))
    ax.set_title('PPI & Ion Channel Network', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.7, 0.35)
    ax.axis('off')
    _label(ax, 'E', x=-0.05, y=1.08)

    if save_path: _save(fig, save_path)
    return fig

# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    COUT = ROOT / 'figures_IJMS' / 'output' / 'supplementary' / 'composite'
    COUT.mkdir(parents=True, exist_ok=True)
    print(f'Output directory (individual): {OUT}')
    print(f'Output directory (composite):  {COUT}')
    print()
    # --- Individual figures (original S1-S9) ---
    figures = [
        (make_supp_figure_s1, 'FigureS1_EvidenceHeatmap'),
        (make_supp_figure_s2, 'FigureS2_PPINetwork'),
        (make_supp_figure_s3, 'FigureS3_FunctionalEnrichment'),
        (make_supp_figure_s4, 'FigureS4_DruggabilityLollipop'),
        (make_supp_figure_s5, 'FigureS5_DirectionConcordance'),
        (make_supp_figure_s6, 'FigureS6_ROCValidation'),
        (make_supp_figure_s7, 'FigureS7_HeatmapRadar'),
        (make_supp_figure_s8, 'FigureS8_DrugTargetAxis'),
        (make_supp_figure_s9, 'FigureS9_TranslationalSummary'),
    ]
    for i, (func, name) in enumerate(figures, 1):
        print(f'Generating individual S{i} ...')
        try:
            func(save_path=str(OUT / f'{name}.pdf'))
        except Exception as e:
            print(f'  ERROR in S{i}: {e}')
            import traceback; traceback.print_exc()
    print()
    # --- Composite figures (merged 6-figure set) ---
    composite_figures = [
        (make_supp_figure_s1,       'CompositeS1_MultiEvidence'),       # S1 as-is
        (make_composite_figure_s2,  'CompositeS2_NetworkEnrichment'),   # S2+S3
        (make_supp_figure_s4,       'CompositeS3_Druggability'),        # S4 -> S3
        (make_composite_figure_s4,  'CompositeS4_ExternalValidation'),  # S5+S6
        (make_composite_figure_s5,  'CompositeS5_HubGeneCharacterization'),  # S7+S8
        (make_supp_figure_s9,       'CompositeS6_TranslationalSummary'),    # S9 -> S6
    ]
    for i, (func, name) in enumerate(composite_figures, 1):
        print(f'Generating composite S{i} ...')
        try:
            func(save_path=str(COUT / f'{name}.pdf'))
        except Exception as e:
            print(f'  ERROR in composite S{i}: {e}')
            import traceback; traceback.print_exc()
    print()
    print(f'All figures saved.')
    print(f'  Individual: {OUT}')
    print(f'  Composite:  {COUT}')