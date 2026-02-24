#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate TCGA survival + immune infiltration panels for Figure 3."""

import gzip
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

NPG = {
    "red": "#E64B35",
    "blue": "#3C5488",
    "green": "#00A087",
    "purple": "#8491B4",
    "orange": "#F39B7F",
    "cyan": "#4DBBD5",
    "gray": "#7E6148",
    "yellow": "#B09C85",
}
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(_SCRIPT_DIR)  # step3_tcga
ROOT = os.path.dirname(_STEP_DIR)  # PROJECT_ROOT
TCGA_DIR = os.path.join(_STEP_DIR, "data")
RESULT_DIR = os.path.join(_STEP_DIR, "result")
OUT = os.path.join(ROOT, "step6_figures_demo", "panels")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Load TCGA data
# ============================================================
print("Loading TCGA expression...")
expr = pd.read_csv(
    f"{TCGA_DIR}/TCGA.COADREAD.sampleMap_HiSeqV2.gz",
    sep="\t",
    index_col=0,
)
print(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} samples")

print("Loading survival data...")
surv = pd.read_csv(f"{TCGA_DIR}/survival_COADREAD_survival.txt", sep="\t")
surv = surv[surv["OS.time"].notna() & surv["OS"].notna()].copy()
surv["OS.time"] = surv["OS.time"].astype(float)
surv["OS"] = surv["OS"].astype(int)
surv = surv.set_index("sample")
print(f"  Survival: {len(surv)} patients")

print("Loading immune infiltration...")
immune = pd.read_csv(f"{TCGA_DIR}/infiltration_estimation_for_tcga.csv.gz")
immune = immune.rename(columns={"cell_type": "sample"})
# Filter to COAD/READ samples using the expression matrix sample IDs
coadread_samples = set(expr.columns)
immune = immune[immune["sample"].isin(coadread_samples)].copy()
immune = immune.set_index("sample")
print(f"  Immune: {immune.shape[0]} samples x {immune.shape[1]} cell types")

# Common samples
common = sorted(set(surv.index) & set(expr.columns))
print(f"  Common samples (expr ∩ surv): {len(common)}")

# Key genes: the 23 overlap targets + 6 validated ion channels
TARGETS = [
    "RPS19",
    "RPS21",
    "RPS2",
    "RPL12",
    "RPL39",
    "ITGAL",
    "CD27",
    "LAG3",
    "EXOSC5",
    "CD6",
    "LAGE3",
    "SNRPD2",
    "NAA10",
    "PDCD5",
    "LSM7",
    "TRMT112",
    "RIPK2",
    "GALK1",
    "PFDN4",
    "CCDC167",
    "FCRL5",
    "S100A2",
    "APBB1IP",
]
VALIDATED_CHANNELS = ["KCNA5", "AQP9", "CLIC1", "CFTR", "GRIN2B", "KCNQ2"]
ALL_GENES = TARGETS + VALIDATED_CHANNELS

# Filter to genes present in expression matrix
available = [g for g in ALL_GENES if g in expr.index]
print(f"  Genes available in TCGA: {len(available)}/{len(ALL_GENES)}")
missing = [g for g in ALL_GENES if g not in expr.index]
if missing:
    print(f"  Missing: {missing}")


# ============================================================
# Panel G: Kaplan-Meier Survival (top 6 genes by log-rank p)
# ============================================================
def panel_H():
    """KM survival curves for the most prognostic targets + ion channels."""
    # Compute log-rank p for all available genes
    results = []
    for gene in available:
        vals = expr.loc[gene, common].astype(float)
        median_val = vals.median()
        high = vals[vals >= median_val].index.tolist()
        low = vals[vals < median_val].index.tolist()

        s_high = surv.loc[[s for s in high if s in surv.index]]
        s_low = surv.loc[[s for s in low if s in surv.index]]

        if len(s_high) < 10 or len(s_low) < 10:
            continue

        lr = logrank_test(
            s_high["OS.time"],
            s_low["OS.time"],
            event_observed_A=s_high["OS"],
            event_observed_B=s_low["OS"],
        )
        results.append(
            {
                "gene": gene,
                "p": lr.p_value,
                "is_channel": gene in VALIDATED_CHANNELS,
            }
        )

    res_df = pd.DataFrame(results).sort_values("p")
    print(f"\n  Survival analysis: {len(res_df)} genes tested")
    print(res_df.head(10).to_string(index=False))

    # Pick top 6: prioritize mix of targets + channels
    top_channels = res_df[res_df["is_channel"]].head(3)
    top_targets = res_df[~res_df["is_channel"]].head(3)
    top6 = pd.concat([top_targets, top_channels]).sort_values("p").head(6)
    genes_to_plot = top6["gene"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, gene in enumerate(genes_to_plot):
        ax = axes[idx]
        vals = expr.loc[gene, common].astype(float)
        median_val = vals.median()

        high_samples = [s for s in common if vals[s] >= median_val and s in surv.index]
        low_samples = [s for s in common if vals[s] < median_val and s in surv.index]

        s_high = surv.loc[high_samples]
        s_low = surv.loc[low_samples]

        lr = logrank_test(
            s_high["OS.time"],
            s_low["OS.time"],
            event_observed_A=s_high["OS"],
            event_observed_B=s_low["OS"],
        )

        kmf_high = KaplanMeierFitter()
        kmf_high.fit(s_high["OS.time"], s_high["OS"], label=f"High (n={len(s_high)})")
        kmf_high.plot_survival_function(ax=ax, color=NPG["red"], lw=2)

        kmf_low = KaplanMeierFitter()
        kmf_low.fit(s_low["OS.time"], s_low["OS"], label=f"Low (n={len(s_low)})")
        kmf_low.plot_survival_function(ax=ax, color=NPG["blue"], lw=2)

        p_str = (
            f"p = {lr.p_value:.2e}" if lr.p_value < 0.01 else f"p = {lr.p_value:.3f}"
        )
        tag = " (ion channel)" if gene in VALIDATED_CHANNELS else ""
        ax.set_title(f"{gene}{tag}\n{p_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Days")
        ax.set_ylabel("Overall Survival")
        ax.legend(fontsize=8, loc="lower left", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(f"{OUT}/Fig3_H.png")
    plt.close()
    print("Panel H (KM Survival) done")

    # Save source data
    res_df.to_csv(f"{RESULT_DIR}/SourceData_Fig3H_Survival.csv", index=False)
    return res_df


# ============================================================
# Panel H: Immune Infiltration Correlation Heatmap
# ============================================================
def panel_I():
    """Correlation heatmap: target gene expression vs immune cell fractions."""
    # Use CIBERSORT columns (22 immune cell types)
    cibersort_cols = [c for c in immune.columns if "_CIBERSORT" in c]
    if not cibersort_cols:
        print("  WARNING: No CIBERSORT columns found, using TIMER")
        cibersort_cols = [c for c in immune.columns if "_TIMER" in c]

    # Clean column names for display
    clean_names = {
        c: c.replace("_CIBERSORT", "").replace("_TIMER", "") for c in cibersort_cols
    }

    # Common samples between expression, survival, and immune
    immune_common = sorted(set(common) & set(immune.index))
    print(f"\n  Immune correlation: {len(immune_common)} common samples")

    if len(immune_common) < 30:
        print("  WARNING: Too few common samples for immune correlation")
        return

    # Select genes: focus on immune-relevant targets + validated channels
    # Prioritize: LAG3, CD6, CD27, ITGAL (immune), + top validated channels
    immune_genes = [
        "LAG3",
        "CD6",
        "CD27",
        "ITGAL",
        "RIPK2",
        "GALK1",
        "EXOSC5",
        "LSM7",
        "RPS21",
        "RPL39",
    ]
    channel_genes = ["CLIC1", "CFTR", "GRIN2B", "KCNQ2", "AQP9", "KCNA5"]
    plot_genes = [g for g in immune_genes + channel_genes if g in expr.index]

    # Build correlation matrix
    corr_matrix = np.zeros((len(plot_genes), len(cibersort_cols)))
    pval_matrix = np.ones((len(plot_genes), len(cibersort_cols)))

    from scipy import stats as sp_stats

    for i, gene in enumerate(plot_genes):
        gene_expr = expr.loc[gene, immune_common].astype(float).values
        for j, col in enumerate(cibersort_cols):
            imm_vals = immune.loc[immune_common, col].astype(float).values
            # Remove NaN pairs
            mask = ~(np.isnan(gene_expr) | np.isnan(imm_vals))
            if mask.sum() < 20:
                continue
            r, p = sp_stats.spearmanr(gene_expr[mask], imm_vals[mask])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    # Filter to cell types with at least one significant correlation
    sig_mask = (pval_matrix < 0.05).any(axis=0)
    # Also keep cell types with decent variance
    keep_cols = [j for j in range(len(cibersort_cols)) if sig_mask[j]]
    if len(keep_cols) < 5:
        # Fallback: keep top 12 by max absolute correlation
        max_corr = np.abs(corr_matrix).max(axis=0)
        keep_cols = np.argsort(max_corr)[-12:][::-1].tolist()

    corr_sub = corr_matrix[:, keep_cols]
    pval_sub = pval_matrix[:, keep_cols]
    col_labels = [clean_names[cibersort_cols[j]] for j in keep_cols]

    # Separate targets and channels with a visual gap
    gene_labels = []
    for g in plot_genes:
        if g in VALIDATED_CHANNELS:
            gene_labels.append(f"★ {g}")
        else:
            gene_labels.append(g)

    fig, ax = plt.subplots(
        figsize=(max(10, len(keep_cols) * 0.6 + 3), len(plot_genes) * 0.45 + 2)
    )

    vmax = max(0.4, np.abs(corr_sub).max())
    im = ax.imshow(corr_sub, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    # Significance markers
    for i in range(corr_sub.shape[0]):
        for j in range(corr_sub.shape[1]):
            p = pval_sub[i, j]
            if p < 0.001:
                ax.text(
                    j, i, "***", ha="center", va="center", fontsize=7, fontweight="bold"
                )
            elif p < 0.01:
                ax.text(
                    j, i, "**", ha="center", va="center", fontsize=7, fontweight="bold"
                )
            elif p < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(gene_labels)))
    ax.set_yticklabels(gene_labels, fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="Spearman ρ")
    ax.set_title(
        "Target–Immune Cell Correlation\n(TCGA-COAD/READ, CIBERSORT)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(f"{OUT}/Fig3_I.png")
    plt.close()
    print("Panel I (Immune Correlation) done")

    # Save source data
    corr_df = pd.DataFrame(corr_sub, index=plot_genes, columns=col_labels)
    corr_df.to_csv(f"{RESULT_DIR}/SourceData_Fig3I_Immune.csv")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\nGenerating TCGA panels...")
    panel_H()
    panel_I()
    print("\nDone! Panels saved to:", OUT)
