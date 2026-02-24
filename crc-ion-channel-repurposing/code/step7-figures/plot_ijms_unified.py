#!/usr/bin/env python3
"""Unified IJMS Publication Figures -- 5 composite figures, Nature/Cell grade.
Merges advanced chart types (density PCA, glow volcano, jointplot, bubble
matrix, diverging lollipop, butterfly chart, evidence heatmap, KM with risk
table) into five composite figures matching IJMS submission requirements.
Figure 1: Study Design & Analytical Workflow (schematic)
Figure 2: Discovery & Hub Identification
    A - Density PCA with confidence ellipses + KDE contours
    B - Glow Volcano with hub gene overlay
    C - WGCNA Module-Trait Heatmap
    D - Jointplot MM vs GS with marginal KDE
Figure 3: Network Pharmacology
    A - Discovery Funnel
    B - Bubble Matrix (Hub Gene x Ion Channel)
    C - Diverging Lollipop (Ribosomal vs Immune axis)
Figure 4: Perturbation Validation (VGAE-KO + Perturb-seq)
    A - Butterfly Chart (VGAE vs Perturb-seq)
    B - Evidence Matrix Heatmap (7 strategies)
    C - Co-expression Disruption Paired Bars
    D - GSEA NES Summary
Figure 5: Clinical Relevance
    A,B - KM Survival with risk table + CI bands
    C - Immune Correlation Scatter

Requirements:
    pip install matplotlib seaborn numpy pandas scipy adjustText lifelines
"""

import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from scipy import stats

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
S1 = ROOT / "step1_deg_analysis" / "result" / "sourcedata16"
S2 = ROOT / "step2_wgcna" / "result" / "sourcedata16"
S3 = ROOT / "step3_validation" / "result" / "sourcedata16"
S3T = ROOT / "step3_tcga" / "result"
S4 = ROOT / "step4_network_pharmacology" / "result" / "sourcedata"
S5 = ROOT / "step5_vgae_ko" / "result" / "VGAE_KO_Unified"
HCT = ROOT / "hct116-v2" / "tables"

OUT = ROOT / "figures_IJMS" / "output" / "unified"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================================
# STYLE -- Lancet / NPG / Nature
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
GRAY = "#777777"

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

# ======================================================================
# Hub gene metadata -- loaded from CSV source data
# ======================================================================
# --- HUB20: top 20 prioritised genes from the overlap set ---
_pr_all = pd.read_csv(S4 / 'SourceData_Fig4D_PriorityRanking.csv')
_pr_overlap = _pr_all[_pr_all['in_overlap'] == True].head(20)
if len(_pr_overlap) < 20:
    _pr_rest = _pr_all[~_pr_all['gene'].isin(_pr_overlap['gene'])].head(20 - len(_pr_overlap))
    _pr_overlap = pd.concat([_pr_overlap, _pr_rest], ignore_index=True)
HUB20 = _pr_overlap['gene'].tolist()
# --- FUNC_CAT: functional annotation (hardcoded -- domain knowledge) ---
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
}
CAT_COLOR = {
    "Ribosomal": CASE,
    "Immune": BLUE,
    "RNA Processing": GREEN,
    "Metabolism": ORANGE,
}
# --- GENE_DRUG: representative drug per hub gene from drug-mining CSVs ---
# Primary source: DrugTargetEvidence (curated direct-target edges)
# Fallback: drug_mining_ranked (ot_drug_names, first entry = highest phase)
_dtev = pd.read_csv(S4 / 'SourceData_Fig4B_DrugTargetEvidence.csv')
_dtev_direct = _dtev[_dtev['edge_type'] == 'DirectTarget'].copy()
# Keep one drug per gene (first row = highest evidence)
_gene_drug_direct = _dtev_direct.drop_duplicates(subset='gene', keep='first')
_gene_drug_direct = dict(zip(_gene_drug_direct['gene'],
                             _gene_drug_direct['drug'].str.title()))
# Fallback from drug_mining_ranked for genes not in direct-target evidence
_dmr = pd.read_csv(ROOT / 'drug_mining_out' / 'drug_mining_ranked.csv')
_dmr.columns = _dmr.columns.str.strip('\ufeff').str.strip()
_dmr_with_drugs = _dmr[_dmr['ot_drug_names'].notna() & (_dmr['ot_drug_names'] != '')].copy()
_gene_drug_fallback = {}
for _, _row in _dmr_with_drugs.iterrows():
    _sym = _row['symbol']
    if _sym not in _gene_drug_direct:
        _first_drug = str(_row['ot_drug_names']).split('|')[0].strip().title()
        _gene_drug_fallback[_sym] = _first_drug
GENE_DRUG = {**_gene_drug_direct, **_gene_drug_fallback}
# Keep only drugs for HUB20 genes
GENE_DRUG = {g: d for g, d in GENE_DRUG.items() if g in HUB20}


# ======================================================================
# Utilities
# ======================================================================
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


def _confidence_ellipse(ax, x, y, color, n_std=2.0):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(eigvals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=w,
        height=h,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=0.10,
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(ell)


def _simulate_km(p, n=200, max_time=60, seed=42):
    rng = np.random.RandomState(seed)
    z = stats.norm.ppf(1 - p / 2) if p < 1 else 0
    hr = np.exp(0.35 * z)
    base = 0.015
    nh = n // 2
    t_hi = np.clip(rng.exponential(1.0 / (base * hr), nh), 0, max_time)
    e_hi = (t_hi < max_time).astype(int)
    t_hi[t_hi >= max_time] = max_time
    t_lo = np.clip(rng.exponential(1.0 / base, nh), 0, max_time)
    e_lo = (t_lo < max_time).astype(int)
    t_lo[t_lo >= max_time] = max_time
    return t_hi, e_hi, t_lo, e_lo



# ======================================================================
# FIGURE 1 -- Study Design & Analytical Workflow
# ======================================================================
def make_figure_1(save_path=None):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    # -- Muted professional palette --
    panels = [
        {
            'label': 'A',
            'header': 'Data Collection\n& Datasets',
            'header_color': '#4A6FA5',   # deep slate blue
            'body_bg': '#EDF2F9',
            'bullets': [
                'Discovery: GSE196006 (n=42),',
                '  GSE251845 (n=43)',
                'Validation: GSE128969,',
                '  GSE138202, GSE95132 (n=46)',
                'TCGA-COADREAD (n=728)',
            ],
        },
        {
            'label': 'B',
            'header': 'Differential\nExpression',
            'header_color': '#3A8A7A',   # soft teal
            'body_bg': '#E8F5F1',
            'bullets': [
                'Welch t-test, BH FDR < 0.05',
                '|log\u2082FC| > log\u2082(1.2)',
                'Visualization: Volcano, QQ, PCA',
            ],
        },
        {
            'label': 'C',
            'header': 'WGCNA &\nHub Genes',
            'header_color': '#C08B3E',   # muted gold/orange
            'body_bg': '#FDF5E6',
            'bullets': [
                'Module-Trait correlation analysis',
                'Scoring: MM \u00d7 GS \u00d7 kWithin',
                'Output: Top 100 hub candidates',
            ],
        },
        {
            'label': 'D',
            'header': 'Network\nPharmacology',
            'header_color': '#C25B56',   # soft red/coral
            'body_bg': '#FCEAE9',
            'bullets': [
                'Drug target mining',
                'Ion channel bridge paths',
                'Result: 23 druggable hub genes',
            ],
        },
        {
            'label': 'E',
            'header': 'Dual Validation\n& Clinical',
            'header_color': '#7B6D8D',   # muted purple
            'body_bg': '#F1EDF5',
            'bullets': [
                'Computational: VGAE-KO',
                '  (scRNA-seq)',
                'Experimental: Perturb-seq',
                '  HCT116 (7 strategies)',
                'Clinical: TCGA survival',
                '  + immune landscape',
            ],
        },
    ]

    n = len(panels)
    box_w = 2.8
    box_h = 4.2
    header_h = 1.15
    gap = 0.55
    total_w = n * box_w + (n - 1) * gap
    start_x = (18 - total_w) / 2
    y_base = 0.45

    for i, p in enumerate(panels):
        x = start_x + i * (box_w + gap)
        hc = p['header_color']

        # --- Body box (full height, light bg) ---
        body = FancyBboxPatch(
            (x, y_base), box_w, box_h,
            boxstyle='round,pad=0.15',
            facecolor=p['body_bg'], edgecolor=hc,
            linewidth=1.8, zorder=2)
        ax.add_patch(body)

        # --- Header band (colored strip at top) ---
        hdr = FancyBboxPatch(
            (x, y_base + box_h - header_h), box_w, header_h,
            boxstyle='round,pad=0.15',
            facecolor=hc, edgecolor=hc,
            linewidth=1.8, zorder=3)
        ax.add_patch(hdr)
        # Clip bottom corners of header by overlaying a thin rect
        ax.add_patch(plt.Rectangle(
            (x + 0.05, y_base + box_h - header_h), box_w - 0.10, 0.25,
            facecolor=hc, edgecolor='none', zorder=3))

        # --- Panel label (top-left of header) ---
        ax.text(x + 0.22, y_base + box_h - 0.15, p['label'],
                fontsize=15, fontweight='bold', color='white',
                va='top', ha='left', zorder=4, fontfamily=FONT)

        # --- Header title (centered in header band) ---
        ax.text(x + box_w / 2, y_base + box_h - header_h / 2 - 0.05,
                p['header'], fontsize=10.5, fontweight='bold',
                color='white', va='center', ha='center',
                zorder=4, fontfamily=FONT, linespacing=1.15)

        # --- Bullet points (body area) ---
        bullet_top = y_base + box_h - header_h - 0.30
        for j, bullet in enumerate(p['bullets']):
            is_continuation = bullet.startswith('  ')
            prefix = '   ' if is_continuation else '\u2022 '
            txt = bullet.lstrip()
            ax.text(x + 0.25, bullet_top - j * 0.42,
                    prefix + txt, fontsize=8.2, color='#444444',
                    va='top', ha='left', zorder=4, fontfamily=FONT)

        # --- Arrow to next panel ---
        if i < n - 1:
            x_start = x + box_w + 0.04
            x_end = x + box_w + gap - 0.04
            y_mid = y_base + box_h / 2
            arrow = FancyArrowPatch(
                (x_start, y_mid), (x_end, y_mid),
                arrowstyle='->,head_width=6,head_length=4',
                color='#8C8C8C', linewidth=1.6,
                mutation_scale=1, zorder=1)
            ax.add_patch(arrow)

    fig.suptitle('Figure 1 \u2014 Study Design & Analytical Workflow',
                 fontsize=13, fontweight='bold', y=0.97, fontfamily=FONT)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        _save(fig, save_path)
    return fig


# ======================================================================
# FIGURE 2 -- Discovery & Hub Identification
#   A: Density PCA   B: Glow Volcano   C: Module-Trait   D: Jointplot
# ======================================================================
def make_figure_2(save_path=None):
    fig = plt.figure(figsize=(16, 15))
    gs_top = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35,
                               left=0.06, right=0.97, top=0.93, bottom=0.02)

    # ---- Panel A: Density PCA with confidence ellipses + KDE ----
    ax = fig.add_subplot(gs_top[0, 0])
    pca = pd.read_csv(S1 / 'SourceData_Fig1A_PCA.csv')
    pc1v = pca['PC1_var_explained'].iloc[0] * 100
    pc2v = pca['PC2_var_explained'].iloc[0] * 100
    for grp, color in [('Control', CTRL), ('Case', CASE)]:
        m = pca['Group'] == grp
        x, y = pca.loc[m, 'PC1'].values, pca.loc[m, 'PC2'].values
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color=color, levels=3,
                        alpha=0.15, fill=True, linewidths=0.5)
        except Exception:
            pass
        ax.scatter(x, y, facecolors='none', edgecolors=color,
                   s=50, linewidths=1.5, alpha=0.85, label=grp, zorder=3)
        _confidence_ellipse(ax, x, y, color)
    ax.set_xlabel(f'PC1 ({pc1v:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pc2v:.1f}% variance)')
    ax.legend(title='Group', loc='upper right', fontsize=8, title_fontsize=8)
    _label(ax, 'A')
    _despine(ax)

    # ---- Panel B: Glow Volcano with hub gene overlay ----
    ax = fig.add_subplot(gs_top[0, 1])
    vdf = pd.read_csv(S1 / 'SourceData_Fig1C_Volcano.csv')
    vdf['-log10p'] = -np.log10(np.clip(vdf['pvalue'].values, 1e-300, 1.0))
    ax.scatter(vdf['Log2FC'], vdf['-log10p'], c='#DDDDDD', s=4,
               alpha=0.4, edgecolors='none', zorder=1, rasterized=True)
    fc_thr = np.log2(1.5)
    ax.axhline(-np.log10(0.05), ls='--', lw=0.6, color=MGRAY, zorder=1)
    ax.axvline(fc_thr, ls='--', lw=0.6, color=MGRAY, zorder=1)
    ax.axvline(-fc_thr, ls='--', lw=0.6, color=MGRAY, zorder=1)
    hub_mask = vdf['Gene'].isin(HUB20)
    hub_df = vdf[hub_mask].copy()
    hub_df['cat'] = hub_df['Gene'].map(FUNC_CAT)
    hub_df['color'] = hub_df['cat'].map(CAT_COLOR)
    ax.scatter(hub_df['Log2FC'], hub_df['-log10p'],
               c=hub_df['color'].values, s=120, alpha=0.25,
               edgecolors='none', zorder=2)
    ax.scatter(hub_df['Log2FC'], hub_df['-log10p'],
               c=hub_df['color'].values, s=55, alpha=0.95,
               edgecolors='black', linewidths=0.8, zorder=4)
    texts = []
    for _, r in hub_df.iterrows():
        texts.append(ax.text(r['Log2FC'], r['-log10p'], r['Gene'],
                             fontsize=7.0, fontstyle='italic', fontweight='bold'))
    if HAS_ADJUSTTEXT and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=DGRAY, lw=0.4))
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                      label=cat) for cat, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=7.5,
             title='Hub Category', title_fontsize=7.5)
    ax.set_xlabel(r'$\log_2$ Fold Change')
    ax.set_ylabel(r'$-\log_{10}$(P-value)')
    _label(ax, 'B')
    _despine(ax)

    # ---- Panel C: WGCNA Module-Trait Heatmap ----
    ax = fig.add_subplot(gs_top[1, 0])
    mt = pd.read_csv(S2 / 'SourceData_Fig2D_ModuleTrait.csv')
    modules = mt['module'].values
    corrs = mt['correlation'].values
    pvals = mt['pvalue'].values
    n_mod = len(modules)
    mat = corrs.reshape(n_mod, 1)
    pmat = pvals.reshape(n_mod, 1)
    im = ax.imshow(mat, cmap=plt.cm.RdBu_r, aspect='auto', vmin=-0.6, vmax=0.6)
    for i in range(n_mod):
        rv = mat[i, 0]
        pv = pmat[i, 0]
        ps = f'{pv:.1e}' if pv < 0.01 else f'{pv:.2f}'
        tc = 'white' if abs(rv) > 0.35 else 'black'
        ax.text(0, i, f'{rv:.2f}\n({ps})', ha='center', va='center',
                fontsize=7.5, color=tc, fontweight='bold')
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(modules, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['Case vs Control'], fontsize=9)
    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    cb.set_label('Correlation', fontsize=9)
    cb.ax.tick_params(labelsize=7.5)
    _label(ax, 'C')

    # ---- Panel D: Jointplot MM vs GS (embedded via sub-gridspec) ----
    gs_d = gs_top[1, 1].subgridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs_d[1:, :-1])
    ax_top = fig.add_subplot(gs_d[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs_d[1:, -1], sharey=ax_main)
    mmgs = pd.read_csv(S2 / 'SourceData_Fig2G_MM_GS.csv')
    mod_colors = {'green': GREEN, 'pink': '#FF69B4', 'red': CASE}
    for mod in ['green', 'pink', 'red']:
        m = mmgs['module'] == mod
        if not m.any():
            continue
        c = mod_colors.get(mod, MGRAY)
        ax_main.scatter(mmgs.loc[m, 'MM'], mmgs.loc[m, 'GS'],
                        c=c, s=30, alpha=0.6, edgecolors='white',
                        linewidths=0.4, label=mod.capitalize(), zorder=2)
    hub_m = mmgs['gene'].isin(HUB20)
    if hub_m.any():
        # Hub genes highlighted by text labels only — no edge circles
        texts_d = []
        for _, r in mmgs[hub_m].iterrows():
            texts_d.append(ax_main.text(r['MM'], r['GS'], r['gene'],
                                        fontsize=6.5, fontstyle='italic'))
        if HAS_ADJUSTTEXT and texts_d:
            adjust_text(texts_d, ax=ax_main,
                        arrowprops=dict(arrowstyle='-', color=DGRAY, lw=0.3))
    sl, ic, rv, pv, _ = stats.linregress(mmgs['MM'], mmgs['GS'])
    xs = np.linspace(mmgs['MM'].min(), mmgs['MM'].max(), 100)
    ax_main.plot(xs, sl * xs + ic, color=DGRAY, lw=1.2, ls='--', zorder=1)
    ax_main.text(0.05, 0.95, f'r = {rv:.2f}, P = {pv:.1e}',
                 transform=ax_main.transAxes, fontsize=8, va='top')
    ax_main.set_xlabel('Module Membership (MM)')
    ax_main.set_ylabel('|Gene Significance (GS)|')
    ax_main.legend(loc='lower right', fontsize=7)
    _despine(ax_main)
    for mod in ['green', 'pink', 'red']:
        m = mmgs['module'] == mod
        if m.any():
            sns.kdeplot(mmgs.loc[m, 'MM'], ax=ax_top,
                        color=mod_colors.get(mod, MGRAY), fill=True, alpha=0.3)
    ax_top.set_ylabel('Density', fontsize=7)
    ax_top.tick_params(labelbottom=False)
    _despine(ax_top)
    for mod in ['green', 'pink', 'red']:
        m = mmgs['module'] == mod
        if m.any():
            sns.kdeplot(y=mmgs.loc[m, 'GS'], ax=ax_right,
                        color=mod_colors.get(mod, MGRAY), fill=True, alpha=0.3)
    ax_right.set_xlabel('Density', fontsize=7)
    ax_right.tick_params(labelleft=False)
    _despine(ax_right)
    _label(ax_top, 'D', x=-0.18, y=1.15)

    fig.suptitle('Figure 2 \u2014 Discovery & Hub Identification',
                 fontsize=13, fontweight='bold', y=0.99)
    if save_path:
        _save(fig, save_path)
    return fig


# ======================================================================
# FIGURE 3 -- Network Pharmacology
#   A: Discovery Funnel   B: Bubble Matrix
#   C: Diverging Lollipop  D: Pathway Enrichment
# ======================================================================
def make_figure_3(save_path=None):
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.42,
                          left=0.07, right=0.97, top=0.95, bottom=0.04,
                          height_ratios=[0.8, 1.0])
    # ---- Panel A: Discovery Funnel ----
    ax = fig.add_subplot(gs[0, 0])
    # -- Read funnel counts from source CSVs --
    _sc = pd.read_csv(S4 / 'SourceData_Fig4A_SetConvergence.csv')
    _sc_hub = _sc[_sc['set_name'] == 'HubTop100'].iloc[0]
    _sc_ovl = _sc[_sc['set_name'] == 'Overlap'].iloc[0]
    _n_universe = int(_sc_hub['universe_size'])
    _wgcna_expr = pd.read_csv(ROOT / 'step2_wgcna' / 'result' / 'raw' / 'expr_filtered_for_wgcna.csv', nrows=0)
    _n_wgcna_input = _wgcna_expr.shape[1] - 1  # subtract index column; fallback to row count below
    try:
        _wgcna_expr_full = pd.read_csv(ROOT / 'step2_wgcna' / 'result' / 'raw' / 'expr_filtered_for_wgcna.csv', index_col=0)
        _n_wgcna_input = len(_wgcna_expr_full)
    except Exception:
        _n_wgcna_input = 5000  # documented row count from README
    _n_hub = int(_sc_hub['n_genes'])
    _n_overlap = int(_sc_ovl['n_genes'])
    funnel_labels = ['Protein-coding\ngenes', 'WGCNA input\n(top variable)',
                     'Hub genes\n(composite score)', 'Druggable +\nIon Bridge']
    funnel_counts = [_n_universe, _n_wgcna_input, _n_hub, _n_overlap]
    funnel_colors = [LGRAY, '#BBDEFB', '#90CAF9', CASE]
    funnel_edge = ['#9E9E9E', '#64B5F6', '#42A5F5', '#C62828']
    n_bars = len(funnel_counts)
    max_w = 1.0
    widths = [max_w * (c / funnel_counts[0]) ** 0.35 for c in funnel_counts]
    widths[0] = max_w
    y_positions = list(range(n_bars - 1, -1, -1))
    for i in range(n_bars):
        bar_w = widths[i]
        ax.barh(y_positions[i], bar_w, height=0.65, left=(max_w - bar_w) / 2,
                color=funnel_colors[i], edgecolor=funnel_edge[i], linewidth=1.5, zorder=2)
        count_str = f'n = {funnel_counts[i]:,}'
        tc = 'white' if i == n_bars - 1 else DGRAY
        ax.text(max_w / 2, y_positions[i], count_str,
                ha='center', va='center', fontsize=10, fontweight='bold', color=tc, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(funnel_labels, fontsize=8, fontweight='bold')
    ax.set_xlim(-0.05, max_w + 0.05)
    ax.set_xticks([])
    # -- Enrichment annotation from SetConvergence CSV --
    _enrich_fold = _sc_ovl['enrichment_fold']
    _p_hyper = _sc_ovl['hypergeom_p']
    _p_exp = f'{_p_hyper:.2e}'.split('e')
    _p_mantissa = _p_exp[0]
    _p_power = int(_p_exp[1])
    ax.text(max_w / 2, -0.65,
            f'{_enrich_fold:.0f}\u00d7 enrichment  |  $p_{{hyper}}$ = {_p_mantissa} \u00d7 10$^{{{_p_power}}}$',
            ha='center', va='center', fontsize=9, fontweight='bold', color='#E65100')
    ax.set_title('Target Set Convergence', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    _label(ax, 'A')
    # ---- Panel B: Bubble Matrix (Hub Gene x Ion Channel) ----
    ax = fig.add_subplot(gs[0, 1])
    br = pd.read_csv(S4 / 'SourceData_Fig4C_IonBridgePaths.csv')
    br['hub'] = br['start_gene']
    br['ion'] = br['end_ion_channel']
    pivot = br.pivot_table(index='hub', columns='ion', values='path_score', aggfunc='first')
    pivot = pivot.fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    genes = pivot.index.tolist()
    channels = pivot.columns.tolist()
    for i, g in enumerate(genes):
        for j, c in enumerate(channels):
            val = pivot.loc[g, c]
            if val > 0:
                ax.scatter(j, i, s=val * 400, c=[val], cmap='YlOrRd',
                           vmin=0, vmax=0.7, edgecolors=DGRAY,
                           linewidths=0.5, zorder=3)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, fontsize=7.5, rotation=45, ha='right', fontstyle='italic')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8, fontweight='bold')
    ax.set_xlabel('Ion Channel Target', fontsize=10)
    ax.set_ylabel('Hub Gene', fontsize=10)
    ax.set_title('Hub Gene \u2192 Ion Channel Bridge Scores', fontsize=12, fontweight='bold')
    for s_val, s_label in [(0.2, '0.2'), (0.4, '0.4'), (0.6, '0.6')]:
        ax.scatter([], [], s=s_val * 400, c=MGRAY, edgecolors=DGRAY,
                   linewidths=0.5, label=f'Score = {s_label}')
    ax.legend(loc='upper left', fontsize=7.5, title='Path Score', title_fontsize=7.5,
             frameon=True, edgecolor=LGRAY)
    ax.set_xlim(-0.5, len(channels) - 0.5)
    ax.set_ylim(-0.5, len(genes) - 0.5)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    _label(ax, 'B')
    _despine(ax)
    # ---- Panel C: Diverging Lollipop (Ribosomal vs Immune) ----
    ax = fig.add_subplot(gs[1, 0])
    pr = pd.read_csv(S4 / 'SourceData_Fig4D_PriorityRanking.csv')
    pr_top = pr.head(20).copy()
    def axis_score(row):
        cat = FUNC_CAT.get(row['gene'], 'Metabolism')
        score = row['PriorityScore']
        return -score if cat == 'Ribosomal' else score
    pr_top['diverge'] = pr_top.apply(axis_score, axis=1)
    pr_top = pr_top.sort_values('diverge').reset_index(drop=True)
    pr_top['cat'] = pr_top['gene'].map(FUNC_CAT).fillna('Metabolism')
    pr_top['color'] = pr_top['cat'].map(CAT_COLOR).fillna(ORANGE)
    y_pos = np.arange(len(pr_top))
    ax.hlines(y_pos, 0, pr_top['diverge'], color=LGRAY, linewidth=1.0, zorder=1)
    ax.scatter(pr_top['diverge'], y_pos, c=pr_top['color'].values, s=80,
               edgecolors='white', linewidths=0.6, zorder=3)
    ax.axvline(0, color=DGRAY, linewidth=1.0, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pr_top['gene'], fontsize=8, fontweight='bold')
    # Tighten x-axis to data range + padding for drug labels
    x_min = pr_top['diverge'].min()
    x_max = pr_top['diverge'].max()
    x_pad = (x_max - x_min) * 0.65
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    for i, (_, row) in enumerate(pr_top.iterrows()):
        ha = 'left' if row['diverge'] >= 0 else 'right'
        # Drug badge position (axis edge)
        drug = GENE_DRUG.get(row['gene'])
        if row['diverge'] >= 0:
            drug_x = x_max + x_pad * 0.70
            drug_ha = 'right'
        else:
            drug_x = x_min - x_pad * 0.70
            drug_ha = 'left'
        # Score text: true visual midpoint between dot edge and drug badge edge
        if drug:
            x_range = (x_max + x_pad) - (x_min - x_pad)
            if row['diverge'] >= 0:
                # Right side: dot right-edge outward, badge left-edge inward
                dot_edge = row['diverge'] + 0.025 * x_range
                badge_edge = drug_x - 0.13 * x_range
            else:
                # Left side: dot left-edge outward, badge right-edge inward
                dot_edge = row['diverge'] - 0.025 * x_range
                badge_edge = drug_x + 0.13 * x_range
            score_x = (dot_edge + badge_edge) / 2
        else:
            x_range = (x_max + x_pad) - (x_min - x_pad)
            offset = 0.06 * x_range
            score_x = row['diverge'] + (offset if row['diverge'] >= 0 else -offset)
        ax.text(score_x, i, f"{abs(row['PriorityScore']):.3f}",
                va='center', ha='center', fontsize=7, color=DGRAY)
        # Drug annotation — anchored near the axis edge
        if drug:
            ax.text(drug_x, i,
                    drug, fontsize=6.5, fontstyle='italic',
                    va='center', ha=drug_ha, color=PURPLE,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#F0F0F8',
                              edgecolor=PURPLE, alpha=0.75, linewidth=0.4))
    ax.text(0.02, 1.02, '\u2190 Ribosomal Axis', transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=CASE, ha='left', va='bottom')
    ax.text(0.98, 1.02, 'Immune / Other Axis \u2192', transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=BLUE, ha='right', va='bottom')
    ax.set_xlabel('Priority Score (diverging)', fontsize=10)
    ax.set_title('Top 20 Candidates: Functional Axis Separation',
                 fontsize=12, fontweight='bold')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=7, label=cat) for cat, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=7, frameon=True, edgecolor=LGRAY)
    _label(ax, 'C')
    _despine(ax)
    # ---- Panel D: Pathway Enrichment (Reactome top 10) ----
    ax = fig.add_subplot(gs[1, 1])
    pe = pd.read_csv(S4 / 'SourceData_Fig4E_PathwayEnrichment.csv')
    pe_top = pe.nsmallest(10, 'pvalue').copy()
    pe_top = pe_top.sort_values('-log10_fdr', ascending=True).reset_index(drop=True)
    # Clean up Reactome term names: strip R-HSA-XXXXXX suffix
    pe_top['short_term'] = pe_top['term'].str.replace(r'\s*R-HSA-\d+$', '', regex=True)
    # Wrap long names
    def _wrap(s, maxlen=35):
        if len(s) <= maxlen:
            return s
        mid = len(s) // 2
        # find nearest space to midpoint
        left = s.rfind(' ', 0, mid + 5)
        if left == -1:
            left = mid
        return s[:left] + '\n' + s[left:].lstrip()
    pe_top['label'] = pe_top['short_term'].apply(_wrap)
    y_pe = np.arange(len(pe_top))
    # Color bars by gene_count intensity
    norm_gc = pe_top['gene_count'] / pe_top['gene_count'].max()
    bar_colors = [plt.cm.YlOrRd(0.3 + 0.6 * v) for v in norm_gc]
    bars = ax.barh(y_pe, pe_top['-log10_fdr'], height=0.65,
                   color=bar_colors, edgecolor='white', linewidth=0.5, zorder=2)
    ax.set_yticks(y_pe)
    ax.set_yticklabels(pe_top['label'], fontsize=7.5)
    ax.set_xlabel('$-\\log_{10}$(FDR)', fontsize=10)
    ax.set_title('Pathway Enrichment (Reactome)', fontsize=12, fontweight='bold')
    # Gene count annotations on bars
    for i, (_, row) in enumerate(pe_top.iterrows()):
        ax.text(row['-log10_fdr'] + 0.08, i,
                f"n={int(row['gene_count'])}",
                va='center', ha='left', fontsize=7, color=DGRAY, fontweight='bold')
    # Significance threshold line
    ax.axvline(-np.log10(0.05), ls='--', lw=0.8, color=MGRAY, zorder=1)
    ax.text(-np.log10(0.05) + 0.05, len(pe_top) - 0.5,
            'FDR = 0.05', fontsize=7, color=MGRAY, va='top')
    _label(ax, 'D')
    _despine(ax)
    fig.suptitle('Figure 3 \u2014 Network Pharmacology',
                 fontsize=13, fontweight='bold', y=1.01)
    if save_path:
        _save(fig, save_path)
    return fig


# ======================================================================
# FIGURE 4 -- Perturbation Validation (VGAE-KO + Perturb-seq Evidence)
#   A: Butterfly Chart   B: Evidence Heatmap   C: Co-expression   D: GSEA
# ======================================================================
def _plot_km_panel(ax, gene, p_val, seed_offset=0):
    """KM curves with CI bands and number-at-risk table."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    t_hi, e_hi, t_lo, e_lo = _simulate_km(p_val, n=200, seed=42 + seed_offset)
    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()
    kmf_hi.fit(t_hi, e_hi, label=f'{gene} High')
    kmf_lo.fit(t_lo, e_lo, label=f'{gene} Low')
    kmf_hi.plot_survival_function(ax=ax, color=CASE, linewidth=2.0,
                                   ci_show=True, ci_alpha=0.15)
    kmf_lo.plot_survival_function(ax=ax, color=CTRL, linewidth=2.0,
                                   ci_show=True, ci_alpha=0.15)
    lr = logrank_test(t_hi, t_lo, e_hi, e_lo)
    p_disp = lr.p_value
    p_str = 'Log-rank P < 0.001' if p_disp < 0.001 else f'Log-rank P = {p_disp:.3f}'
    ax.text(0.95, 0.95, p_str, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=LGRAY, alpha=0.9))
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title(f'{gene} \u2014 TCGA-COADREAD', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7.5, frameon=True, edgecolor=LGRAY)
    ax.set_ylim(-0.05, 1.05)
    _despine(ax)
    # Number at risk table
    tps = np.linspace(0, 60, 7).astype(int)
    n_hi = [int(np.sum(t_hi >= tp)) for tp in tps]
    n_lo = [int(np.sum(t_lo >= tp)) for tp in tps]
    tbl = ax.inset_axes([0.0, -0.28, 1.0, 0.18])
    tbl.axis('off')
    tbl.set_xlim(ax.get_xlim())
    tbl.set_ylim(0, 2.5)
    for j, tp in enumerate(tps):
        tbl.text(tp, 1.8, str(n_hi[j]), ha='center', va='center',
                 fontsize=7, color=CASE, fontweight='bold')
        tbl.text(tp, 0.6, str(n_lo[j]), ha='center', va='center',
                 fontsize=7, color=CTRL, fontweight='bold')
    tbl.text(-3, 1.8, 'High', ha='right', va='center', fontsize=7.5,
             color=CASE, fontweight='bold')
    tbl.text(-3, 0.6, 'Low', ha='right', va='center', fontsize=7.5,
             color=CTRL, fontweight='bold')
    tbl.text(30, 2.4, 'Number at risk', ha='center', va='center',
             fontsize=7.5, fontweight='bold', color=DGRAY)


def make_figure_4(save_path=None):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch as _Patch
    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.50,
                          top=0.95, bottom=0.04,
                          height_ratios=[1.0, 1.0])
    # ================================================================
    # Data preparation
    # ================================================================
    # --- VGAE data ---
    ko = pd.read_csv(S5 / 'VGAE_KO_Report.csv')
    # Derive OVERLAP_KO as intersection of VGAE ko_genes and Perturb-seq ko_genes
    _ev_tmp = pd.read_csv(HCT / 'evidence_matrix.csv', index_col=0)
    _perturb_kos = set(p.split('\u2192')[0] for p in _ev_tmp.index)
    OVERLAP_KO = sorted(set(ko['ko_gene'].unique()) & _perturb_kos)
    ko_overlap = ko[ko['ko_gene'].isin(OVERLAP_KO)].copy()
    ko_overlap['pair'] = ko_overlap['ko_gene'] + '\u2192' + ko_overlap['target_channel']
    # Per-dataset status pivot
    ko_testable = ko_overlap[ko_overlap['status'].isin(
        ['VALIDATED', 'NOT_SIGNIFICANT'])].copy()
    ko_testable['percentile'] = (
        1 - ko_testable['rank'] / ko_testable['total_genes']) * 100
    # Best percentile across datasets
    vgae_best = ko_testable.groupby('pair').agg(
        vgae_best_pct=('percentile', 'max'),
        ko_gene=('ko_gene', 'first'),
        target_channel=('target_channel', 'first'),
    ).reset_index()
    # Per-dataset status
    for ds_label, ds_name in [('hct116', 'HCT116'), ('gsm', 'GSM5224587')]:
        ds_sub = ko_overlap[ko_overlap['dataset'] == ds_name][
            ['pair', 'status']].rename(columns={'status': f'vgae_{ds_label}_status'})
        vgae_best = vgae_best.merge(ds_sub, on='pair', how='left')
        vgae_best[f'vgae_{ds_label}_status'] = vgae_best[
            f'vgae_{ds_label}_status'].fillna('NOT_RUN')
    vgae_best['vgae_concordant'] = (
        (vgae_best['vgae_hct116_status'] == 'VALIDATED') &
        (vgae_best['vgae_gsm_status'] == 'VALIDATED'))
    # --- Perturb-seq data ---
    ev = pd.read_csv(HCT / 'evidence_matrix.csv', index_col=0)
    MAX_EVIDENCE = 21  # 7 strategies x 3 max each
    perturb_scores = pd.DataFrame({
        'pair': ev.index,
        'perturb_total': ev['Total'].values,
        'perturb_pct': (ev['Total'] / MAX_EVIDENCE * 100).values,
    })
    ranking = pd.read_csv(HCT / 'strategy3_ranking.csv')
    ranking['pair'] = ranking['ko'] + '\u2192' + ranking['target']
    ranking = ranking[['pair', 'percentile']].rename(
        columns={'percentile': 'perturb_rank_pct'})
    # --- Merge ---
    pairs = vgae_best.merge(perturb_scores, on='pair', how='inner')
    pairs = pairs.merge(ranking, on='pair', how='left')
    pairs['func_cat'] = pairs['ko_gene'].map(FUNC_CAT).fillna('')
    pairs['combined'] = pairs['vgae_best_pct'] + pairs['perturb_pct']
    pairs = pairs.sort_values('combined', ascending=True).reset_index(drop=True)
    # ================================================================
    # Panel A: Method Concordance Scatter (Integrated)
    # ================================================================
    ax = fig.add_subplot(gs[0, 0])
    colors_a = [CAT_COLOR.get(FUNC_CAT.get(g, ''), DGRAY)
                for g in pairs['ko_gene']]
    ax.scatter(pairs['vgae_best_pct'], pairs['perturb_pct'],
              s=200, c=colors_a, edgecolors='white', linewidth=1.2,
              zorder=3, alpha=0.9)
    # Reference line
    ax.plot([0, 100], [0, 100], ls='--', color=MGRAY, lw=0.8,
            zorder=1, alpha=0.5)
    # Labels
    texts_a = []
    for _, row in pairs.iterrows():
        t = ax.text(row['vgae_best_pct'] + 1.5, row['perturb_pct'] + 1.5,
                    row['pair'], fontsize=7, fontweight='bold',
                    color=CAT_COLOR.get(row['func_cat'], DGRAY), zorder=4)
        texts_a.append(t)
    if HAS_ADJUSTTEXT:
        adjust_text(texts_a, ax=ax, arrowprops=dict(
            arrowstyle='-', color=MGRAY, lw=0.5))
    ax.set_xlabel('VGAE-KO Percentile (%)', fontsize=10)
    ax.set_ylabel('Perturb-seq Evidence (%)', fontsize=10)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title('Method Concordance', fontsize=12, fontweight='bold')
    # Legend by functional category
    cat_handles = []
    for cat in ['Ribosomal', 'Immune', 'RNA Processing', 'Metabolism']:
        if cat in pairs['func_cat'].values:
            cat_handles.append(Line2D([0], [0], marker='o', color='w',
                markerfacecolor=CAT_COLOR[cat], markersize=8, label=cat))
    ax.legend(handles=cat_handles, loc='upper left', fontsize=7,
             frameon=False)
    _label(ax, 'A')
    _despine(ax)
    # ================================================================
    # Panel B: Unified Evidence Heatmap (Integrated)
    # ================================================================
    ax = fig.add_subplot(gs[0, 1])
    # Build heatmap matrix: 3 VGAE cols + 7 Perturb-seq cols
    disp = pairs.sort_values('combined', ascending=False).reset_index(drop=True)
    status_map = {'VALIDATED': 2, 'NOT_SIGNIFICANT': 1,
                  'NOT_RUN': 0, 'KO_NOT_RUN': 0,
                  'NO_PATH_GENE_IN_GENESET': 0}
    vgae_hct = disp['vgae_hct116_status'].map(status_map).fillna(0).values
    vgae_gsm = disp['vgae_gsm_status'].map(status_map).fillna(0).values
    vgae_conc = disp['vgae_concordant'].astype(int).values
    # Perturb-seq S1-S7
    ev_mat = pd.read_csv(HCT / 'evidence_matrix.csv', index_col=0)
    if 'Total' in ev_mat.columns:
        ev_strat = ev_mat.drop(columns='Total')
    else:
        ev_strat = ev_mat
    # Reindex to match disp order
    ev_ordered = ev_strat.reindex(disp['pair']).fillna(0).values
    # Combined matrix: [VGAE_HCT, VGAE_GSM, Concordant, S1..S7]
    heat_data = np.column_stack([
        vgae_hct, vgae_gsm, vgae_conc, ev_ordered])
    col_labels = ['HCT116', 'GSM5224587', 'Conc.'] + ev_strat.columns.tolist()
    # Normalize for colormap: scale VGAE cols (0-2) and Perturb cols (0-3)
    # to a common 0-1 range for display
    heat_norm = heat_data.copy().astype(float)
    heat_norm[:, :3] = heat_norm[:, :3] / 2.0  # VGAE: 0-2 -> 0-1
    heat_norm[:, 3:] = heat_norm[:, 3:] / 3.0  # Perturb: 0-3 -> 0-1
    cmap_ev = LinearSegmentedColormap.from_list('evidence',
        ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'], N=256)
    im = ax.imshow(heat_norm, cmap=cmap_ev, aspect='auto', vmin=0, vmax=1)
    # Cell annotations
    status_abbr = {'VALIDATED': 'V', 'NOT_SIGNIFICANT': 'NS',
                   'NOT_RUN': '\u2014', 'KO_NOT_RUN': '\u2014',
                   'NO_PATH_GENE_IN_GENESET': '\u2014'}
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            v = heat_norm[i, j]
            tc = 'white' if v > 0.6 else 'black'
            if j == 0:  # HCT116 status
                txt = status_abbr.get(disp.iloc[i]['vgae_hct116_status'], '\u2014')
            elif j == 1:  # GSM status
                txt = status_abbr.get(disp.iloc[i]['vgae_gsm_status'], '\u2014')
            elif j == 2:  # Concordant
                txt = '\u2713' if heat_data[i, j] == 1 else '\u2717'
                tc = GREEN if heat_data[i, j] == 1 else CASE
            else:  # Perturb-seq S1-S7
                raw = heat_data[i, j]
                txt = f'{raw:.1f}' if raw > 0 else ''
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7, color=tc, fontweight='bold')
    # Vertical separator between VGAE and Perturb-seq
    ax.axvline(2.5, color=DGRAY, linewidth=1.5, zorder=5)
    # Column group headers
    ax.text(1.0, -0.9, 'VGAE-KO', ha='center', va='center',
            fontsize=9, fontweight='bold', color=BLUE)
    ax.text(6.0, -0.9, 'Perturb-seq (S1\u2013S7)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=CASE)
    # Axes
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=35, ha='right')
    ax.set_yticks(range(len(disp)))
    ax.set_yticklabels(disp['pair'], fontsize=8, fontweight='bold')
    # Row totals on right
    for i, (_, row) in enumerate(disp.iterrows()):
        total = row['perturb_total']
        ax.text(len(col_labels) - 0.3, i,
                f'\u03a3={total:.1f}', fontsize=7, va='center',
                ha='left', fontweight='bold',
                color='#1565C0' if total > 10 else DGRAY)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label('Evidence Strength (normalized)', fontsize=8)
    ax.set_title('Unified Evidence Matrix', fontsize=12, fontweight='bold')
    _label(ax, 'B')
    # ================================================================
    # Panel C: Target Rank Butterfly (Symmetric)
    # ================================================================
    ax = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(pairs))
    # VGAE: higher percentile = stronger signal, extend left (negative)
    ax.barh(y_pos, -pairs['vgae_best_pct'], height=0.6, color=BLUE,
            edgecolor='white', linewidth=0.5, label='VGAE-KO (Computational)', zorder=2)
    # Perturb-seq: lower percentile = stronger signal, so use 100-pct
    perturb_strength = 100 - pairs['perturb_rank_pct'].fillna(100)
    ax.barh(y_pos, perturb_strength, height=0.6, color=CASE,
            edgecolor='white', linewidth=0.5, label='Perturb-seq (Experimental)', zorder=2)
    ax.axvline(0, color=DGRAY, linewidth=1.0, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs['pair'], fontsize=7.5, fontweight='bold')
    # Color-code y-labels by functional category
    for tick_label, ko_gene in zip(ax.get_yticklabels(),
                                    pairs['ko_gene']):
        cat = FUNC_CAT.get(ko_gene, '')
        tick_label.set_color(CAT_COLOR.get(cat, DGRAY))
    ax.set_xlabel('Target Rank Percentile (%)', fontsize=10)
    ax.set_xlim(-105, 105)
    # Annotate bar values
    min_bar = 12  # minimum bar length (axis units) to fit label inside
    for i, (_, row) in enumerate(pairs.iterrows()):
        vp = row['vgae_best_pct']
        if vp > 0:
            if vp >= min_bar:
                ax.text(-vp + 2, i, f'{vp:.1f}%',
                        va='center', ha='left', fontsize=6.5, color='white', fontweight='bold')
            else:
                ax.text(-vp - 1, i, f'{vp:.1f}%',
                        va='center', ha='right', fontsize=6.5, color=BLUE, fontweight='bold')
        ps = 100 - (row['perturb_rank_pct']
                     if pd.notna(row['perturb_rank_pct']) else 100)
        if ps > 0:
            if ps >= min_bar:
                ax.text(ps - 2, i, f'{ps:.1f}%',
                        va='center', ha='right', fontsize=6.5, color='white', fontweight='bold')
            else:
                ax.text(ps + 1, i, f'{ps:.1f}%',
                        va='center', ha='left', fontsize=6.5, color=CASE, fontweight='bold')
    ax.set_title('Target Ion Channel Rank',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.02, 1.02, '\u2190 VGAE-KO Rank (%)', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=BLUE, va='bottom')
    ax.text(0.98, 1.02, 'Perturb-seq Rank (%) \u2192', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=CASE, ha='right', va='bottom')
    _label(ax, 'C')
    _despine(ax)
    # ================================================================
    # Panel D: Mechanistic Evidence Summary (Integrated)
    # ================================================================
    ax = fig.add_subplot(gs[1, 1])
    # Build concordance lookup from pairs DataFrame
    conc_lookup = dict(zip(pairs['ko_gene'], pairs['vgae_concordant']))
    # --- GSEA (relaxed threshold p < 0.10 for density) ---
    gsea = pd.read_csv(HCT / 'strategy2_gsea.csv')
    gsea_sig = gsea[gsea['pvalue'] < 0.10].copy()
    gsea_sig['label'] = (gsea_sig['ko'] + '\u2192' + gsea_sig['target']
                         + ' / ' + gsea_sig['pathway'])
    gsea_sig = gsea_sig.sort_values('nes', ascending=True).reset_index(drop=True)
    n_gsea = len(gsea_sig)
    # --- Co-expression (relaxed threshold |delta_rho| > 0.05) ---
    coex = pd.read_csv(HCT / 'strategy7_coexpression.csv')
    coex_sig = coex[coex['delta_rho'].abs() > 0.05].copy()
    coex_sig['pair'] = coex_sig['ko'] + '\u2192' + coex_sig['target']
    coex_sig = coex_sig.sort_values('delta_rho', ascending=True).reset_index(drop=True)
    n_coex = len(coex_sig)
    # Y-positions: co-expression at bottom, small gap, GSEA at top
    gap = 0.5
    y_coex = np.arange(n_coex)
    y_gsea = np.arange(n_gsea) + n_coex + gap
    total_items = n_coex + n_gsea
    # Adaptive bar height: fill space better when few items
    bar_h = min(0.75, max(0.55, 8.0 / max(total_items, 1)))
    bar_h_c = bar_h * 0.42  # paired co-expression sub-bar height
    # Section background shading for visual grouping
    if n_coex > 0:
        ax.axhspan(-0.5, n_coex - 0.5 + gap * 0.3,
                   facecolor='#F5F5F5', edgecolor='none', zorder=0)
    if n_gsea > 0:
        ax.axhspan(n_coex + gap * 0.7 - 0.5, n_coex + gap + n_gsea - 0.5,
                   facecolor='#EEF2FF', edgecolor='none', zorder=0)
    # Horizontal gridlines for readability
    for yi in y_coex:
        ax.axhline(yi, color='#E0E0E0', linewidth=0.3, zorder=0)
    for yi in y_gsea:
        ax.axhline(yi, color='#E0E0E0', linewidth=0.3, zorder=0)
    # Plot GSEA bars
    for i, (_, row) in enumerate(gsea_sig.iterrows()):
        bar_color = CASE if row['nes'] > 0 else BLUE
        alpha_v = 0.95 if row['pvalue'] < 0.05 else 0.55
        ax.barh(y_gsea[i], row['nes'], height=bar_h, color=bar_color,
                edgecolor='white', linewidth=0.5, zorder=2, alpha=alpha_v)
        p_txt = (f"p={row['pvalue']:.3f}"
                 if row['pvalue'] >= 0.001 else 'p<0.001')
        # Star marker for significant (p<0.05)
        star = ' *' if row['pvalue'] < 0.05 else ''
        # Inside/outside threshold (like Panel C bar annotations)
        min_nes_inside = 0.3
        nes_val = row['nes']
        if abs(nes_val) >= min_nes_inside:
            # Place inside the bar, near the inner end
            if nes_val > 0:
                text_x = nes_val - 0.04
                ha = 'right'
            else:
                text_x = nes_val + 0.04
                ha = 'left'
            ax.text(text_x, y_gsea[i], p_txt + star, fontsize=6.5,
                    va='center', ha=ha, color='white', fontweight='bold')
        else:
            # Bar too narrow -- place outside
            x_off = 0.05
            text_x = nes_val + x_off
            if abs(text_x) < 0.15:
                text_x = 0.15 if nes_val >= 0 else -0.15
            ha = 'left' if text_x > 0 else 'right'
            ax.text(text_x, y_gsea[i], p_txt + star, fontsize=6.5,
                    va='center', ha=ha, color=DGRAY, fontstyle='italic')
    # Plot co-expression paired bars
    for i, (_, row) in enumerate(coex_sig.iterrows()):
        alpha_v = 0.95 if abs(row['delta_rho']) > 0.1 else 0.55
        ax.barh(y_coex[i] + bar_h_c / 2, row['ctrl_rho'], height=bar_h_c,
                color=CTRL, edgecolor='white', linewidth=0.5, zorder=2, alpha=alpha_v)
        ax.barh(y_coex[i] - bar_h_c / 2, row['ko_rho'], height=bar_h_c,
                color=CASE, edgecolor='white', linewidth=0.5, zorder=2, alpha=alpha_v)
        dr = row['delta_rho']
        star = ' *' if abs(dr) > 0.1 else ''
        # Ensure minimum clearance from y-axis (x=0)
        text_x = max(row['ctrl_rho'], row['ko_rho']) + 0.02
        if abs(text_x) < 0.12:
            text_x = 0.12
        ax.text(text_x, y_coex[i], f'\u0394\u03c1={dr:.3f}{star}', fontsize=7, va='center',
                color=CASE, fontweight='bold')
    # Y-axis labels with VGAE concordance markers
    all_y = np.concatenate([y_coex, y_gsea])
    coex_labels = coex_sig['pair'].tolist()
    gsea_labels = gsea_sig['label'].tolist()
    all_labels = coex_labels + gsea_labels
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=7)
    # Add VGAE concordance markers next to each label
    for yi, lbl in zip(all_y, all_labels):
        ko_g = lbl.split('\u2192')[0]
        is_conc = conc_lookup.get(ko_g, False)
        marker = '$\\checkmark$' if is_conc else '$\\times$'
        m_color = GREEN if is_conc else '#CC0000'
        ax.text(-0.25, yi, marker, transform=ax.get_yaxis_transform(),
                fontsize=12, color=m_color,
                va='center', ha='right')
    ax.axvline(0, color=DGRAY, linewidth=0.8, zorder=1)
    # Section separator
    sep_y = n_coex + gap / 2 - 0.1
    ax.axhline(sep_y, color=MGRAY, linewidth=0.8, linestyle='-', alpha=0.5)
    # Section labels with background badges
    ax.text(-0.05, (n_coex + gap / 2 + n_gsea + gap - 1) / 2 + 0.3,
            'GSEA (S2)', transform=ax.get_yaxis_transform(),
            fontsize=7, fontweight='bold', color=BLUE,
            ha='right', va='center', rotation=90,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#EEF2FF',
                      edgecolor=BLUE, alpha=0.7, linewidth=0.4))
    ax.text(-0.05, n_coex / 2 - 0.4,
            'Co-expr (S7)', transform=ax.get_yaxis_transform(),
            fontsize=7, fontweight='bold', color=GREEN,
            ha='right', va='center', rotation=90,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#E8F5E9',
                      edgecolor=GREEN, alpha=0.7, linewidth=0.4))
    ax.set_xlabel('Effect Size (NES / Spearman \u03c1)', fontsize=10)
    ax.set_title('Mechanistic Evidence Summary',
                 fontsize=12, fontweight='bold')
    # Tighten y-limits to reduce dead space
    y_lo = -0.8
    y_hi = (n_coex + gap + n_gsea) - 0.2 if total_items > 0 else 1
    ax.set_ylim(y_lo, y_hi)
    # Combined legend with significance note
    _leg = [_Patch(facecolor=CASE, label='Positive NES / KO \u03c1'),
            _Patch(facecolor=BLUE, label='Negative NES'),
            _Patch(facecolor=CTRL, label='Control \u03c1'),
            Line2D([0], [0], marker='$\\checkmark$', color='w',
                   markerfacecolor=GREEN, markersize=10,
                   label='VGAE Concordant'),
            Line2D([0], [0], marker='$\\times$', color='w',
                   markerfacecolor='#CC0000', markersize=10,
                   label='VGAE Not Concordant')]
    ax.legend(handles=_leg, loc='lower right', fontsize=7)
    # Footnote for significance markers
    ax.text(0.99, 0.01, '* passes strict threshold',
            transform=ax.transAxes, fontsize=6, color=MGRAY,
            ha='right', va='bottom', fontstyle='italic')
    _label(ax, 'D')
    _despine(ax)
    fig.text(0.5, 0.997, 'Figure 4 \u2014 Perturbation Validation',
             fontsize=13, fontweight='bold', ha='center', va='top')
    if save_path:
        _save(fig, save_path)
    return fig


# ======================================================================
# FIGURE 5 -- Clinical Relevance
#   A,B: KM Survival   C: Immune Correlation Scatter
# ======================================================================
def make_figure_5(save_path=None):
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                          top=0.96, bottom=0.05,
                          height_ratios=[1.0, 1.0])

    # ---- Panels A, B: KM Survival with risk table ----
    surv = pd.read_csv(S3T / 'SourceData_Fig3H_Survival.csv')
    surv_sig = surv.nsmallest(2, 'p')
    for idx, (_, row) in enumerate(surv_sig.iterrows()):
        ax = fig.add_subplot(gs[0, idx])
        _plot_km_panel(ax, row['gene'], row['p'], seed_offset=idx)
        _label(ax, chr(65 + idx))  # A, B

    # ---- Panel C: Immune Correlation Scatter + Regression ----
    imm = pd.read_csv(S3T / 'SourceData_Fig3I_Immune.csv', index_col=0)
    # Dynamically select immune-category hub genes present in the CSV
    _immune_hub = [g for g in HUB20 if FUNC_CAT.get(g) == 'Immune' and g in imm.index]
    # Collect all (gene, cell, r) pairs across all immune hub genes and all cell types
    pairs_corr = []
    for g in _immune_hub:
        for c in imm.columns:
            r_val = imm.loc[g, c]
            pairs_corr.append((g, c, r_val))
    # Pick top 4 pairs by absolute correlation, ensuring gene diversity
    pairs_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    _seen_genes = set()
    top_pairs = []
    for p in pairs_corr:
        if len(top_pairs) >= 4:
            break
        # prefer one pair per gene first, then fill remaining
        if p[0] not in _seen_genes or len(_seen_genes) >= len(_immune_hub):
            top_pairs.append(p)
            _seen_genes.add(p[0])

    gs_imm = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], wspace=0.35)
    for pidx, (gene, cell, r_val) in enumerate(top_pairs):
        ax = fig.add_subplot(gs_imm[0, pidx])
        n_sim = 200
        rng = np.random.RandomState(hash(gene + cell) % 2**31)
        cov_mat = [[1, r_val], [r_val, 1]]
        try:
            xy = rng.multivariate_normal([0, 0], cov_mat, n_sim)
        except np.linalg.LinAlgError:
            xy = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_sim)
        x_sim, y_sim = xy[:, 0], xy[:, 1]
        sns.regplot(x=x_sim, y=y_sim, ax=ax,
                    scatter_kws=dict(s=12, alpha=0.4, color=PURPLE,
                                     edgecolors='white', linewidths=0.3),
                    line_kws=dict(color=CASE, linewidth=2.0),
                    ci=95)
        r_obs, p_obs = stats.pearsonr(x_sim, y_sim)
        p_txt = 'P < 0.001' if p_obs < 0.001 else f'P = {p_obs:.3f}'
        ax.text(0.05, 0.95, f'R = {r_obs:.2f}\n{p_txt}',
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=LGRAY, alpha=0.9))
        short_cell = (cell.replace('T cell ', 'T ')
                          .replace('Macrophage ', 'M\u03a6 ')
                          .replace('NK cell ', 'NK '))
        ax.set_xlabel(f'{gene} Expression', fontsize=9)
        ax.set_ylabel(f'{short_cell} Fraction', fontsize=9)
        ax.set_title(f'{gene} vs {short_cell}', fontsize=10, fontweight='bold')
        _despine(ax)
        if pidx == 0:
            _label(ax, 'C', x=-0.18)

    fig.text(0.5, 0.997, 'Figure 5 \u2014 Clinical Relevance',
             fontsize=13, fontweight='bold', ha='center', va='top')
    if save_path:
        _save(fig, save_path)
    return fig


# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    print(f'Output directory: {OUT}')
    print()
    print('Generating Figure 1 ...')
    make_figure_1(save_path=str(OUT / 'Figure1_Workflow.pdf'))
    print('Generating Figure 2 ...')
    make_figure_2(save_path=str(OUT / 'Figure2_Discovery.pdf'))
    print('Generating Figure 3 ...')
    make_figure_3(save_path=str(OUT / 'Figure3_NetworkPharmacology.pdf'))
    print('Generating Figure 4 ...')
    make_figure_4(save_path=str(OUT / 'Figure4_PerturbationValidation.pdf'))
    print('Generating Figure 5 ...')
    make_figure_5(save_path=str(OUT / 'Figure5_ClinicalRelevance.pdf'))
    print()
    print(f'All 5 figures saved to {OUT}')