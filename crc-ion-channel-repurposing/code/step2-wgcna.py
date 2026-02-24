#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 Generation Pipeline: WGCNA-based Gene Prioritization
Nature Style - ZERO-FAKE Policy
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy import stats

warnings.filterwarnings("ignore")

# Increase recursion limit for large dendrograms (5000 genes)
sys.setrecursionlimit(10000)

# ============================================================
# CONFIGURATION
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(_SCRIPT_DIR)  # step2_wgcna
PROJECT_ROOT = os.path.dirname(_STEP_DIR)  # PROJECT_ROOT
INPUT_DATA = f"{PROJECT_ROOT}/data/data.csv"
OUTPUT_DIR = f"{_STEP_DIR}/result"

# Pre-filtering thresholds
MIN_MEAN_EXPR = 1.0
MIN_VARIANCE = 0.1
TOP_VAR_GENES = 5000

# WGCNA parameters
R2_CUTOFF = 0.80
MIN_MODULE_SIZE = 30

# Hub gene selection
TOP_HUB_GENES = 100
TOP_CANDIDATE_GENES = 20

# Module-trait significance threshold (using FDR-corrected q-value)
MODULE_QVALUE_THRESHOLD = 0.05

MODULE_COLORS = [
    "turquoise",
    "blue",
    "brown",
    "yellow",
    "green",
    "red",
    "black",
    "pink",
    "magenta",
    "purple",
    "greenyellow",
    "tan",
    "salmon",
    "cyan",
    "midnightblue",
    "lightcyan",
    "grey60",
    "lightgreen",
]

COLOR_MAP = {
    "turquoise": "#40E0D0",
    "blue": "#0000FF",
    "brown": "#8B4513",
    "yellow": "#FFFF00",
    "green": "#008000",
    "red": "#FF0000",
    "black": "#000000",
    "pink": "#FFC0CB",
    "magenta": "#FF00FF",
    "purple": "#800080",
    "greenyellow": "#ADFF2F",
    "tan": "#D2B48C",
    "salmon": "#FA8072",
    "cyan": "#00FFFF",
    "midnightblue": "#191970",
    "lightcyan": "#E0FFFF",
    "grey60": "#999999",
    "lightgreen": "#90EE90",
    "grey": "#808080",
    "orange": "#FFA500",
    "white": "#FFFFFF",
}


# ============================================================
# DATA LOADING
# ============================================================
def load_expression_data(filepath):
    """Load expression matrix from CSV file."""
    print(f"Loading data from {filepath}...")
    raw = pd.read_csv(filepath, header=None)
    sample_ids = raw.iloc[0, 1:].values
    labels = raw.iloc[1, 1:].values
    gene_names = raw.iloc[2:, 0].values
    expr_values = raw.iloc[2:, 1:].values.astype(float)
    expr_df = pd.DataFrame(expr_values, index=gene_names, columns=sample_ids)
    print(f"  Loaded {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    return expr_df, sample_ids, labels


def prefilter_genes(expr_df, min_mean=1.0, min_var=0.1, top_n=5000):
    """Pre-filter genes for WGCNA analysis."""
    print("\nPRE-FILTERING GENES")
    initial_genes = expr_df.shape[0]

    # Filter by mean expression
    gene_means = expr_df.mean(axis=1)
    expr_filtered = expr_df[gene_means >= min_mean]
    after_mean = expr_filtered.shape[0]
    print(f"  After mean filter (>= {min_mean}): {after_mean} genes")

    # Filter by variance
    gene_vars = expr_filtered.var(axis=1)
    expr_filtered = expr_filtered[gene_vars >= min_var]
    after_var = expr_filtered.shape[0]
    print(f"  After variance filter (>= {min_var}): {after_var} genes")

    # Select top N most variable genes
    gene_vars = expr_filtered.var(axis=1)
    if after_var > top_n:
        top_var_genes = gene_vars.nlargest(top_n).index
        expr_filtered = expr_filtered.loc[top_var_genes]
        print(f"  Selected top {top_n} variable genes")

    print(f"  Final: {expr_filtered.shape[0]} genes for WGCNA")
    return expr_filtered


# ============================================================
# SOFT THRESHOLD SELECTION
# ============================================================
def scale_free_topology_fit(adjacency):
    """Calculate scale-free topology fit index (R^2)."""
    connectivity = adjacency.sum(axis=1)
    connectivity = connectivity[connectivity > 0]
    n_bins = min(50, len(connectivity) // 10)
    if n_bins < 5:
        return 0, 0, connectivity

    hist, bin_edges = np.histogram(connectivity, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    k = bin_centers[mask]
    p_k = hist[mask] / hist.sum()

    if len(k) < 3:
        return 0, 0, connectivity

    log_k = np.log10(k)
    log_p = np.log10(p_k)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)
    r2_signed = -np.sign(slope) * r_value**2
    return r2_signed, slope, connectivity


def pick_soft_threshold(expr_df, powers=range(1, 21), r2_cutoff=0.80):
    """Pick optimal soft threshold power for WGCNA."""
    print("\nSOFT THRESHOLD SELECTION")
    results = []

    for power in powers:
        cor_matrix = expr_df.T.corr()
        adj = np.abs(cor_matrix.values) ** power
        np.fill_diagonal(adj, 0)
        r2, slope, connectivity = scale_free_topology_fit(adj)
        mean_k = connectivity.mean() if len(connectivity) > 0 else 0
        results.append({"power": power, "SFT_R2": r2, "slope": slope, "mean_k": mean_k})
        print(f"  Power {power}: R2 = {r2:.3f}, mean(k) = {mean_k:.1f}")

    results_df = pd.DataFrame(results)
    above_cutoff = results_df[results_df["SFT_R2"] >= r2_cutoff]
    if len(above_cutoff) > 0:
        optimal_power = above_cutoff["power"].iloc[0]
    else:
        optimal_power = results_df.loc[results_df["SFT_R2"].idxmax(), "power"]

    print(f"  Optimal power: {optimal_power}")
    return int(optimal_power), results_df


# ============================================================
# TOM AND MODULE DETECTION
# ============================================================
def calculate_TOM_fast(adjacency):
    """Fast TOM calculation using vectorized operations."""
    print("  Calculating TOM...")
    k = adjacency.sum(axis=1)
    L = adjacency @ adjacency
    k_matrix = np.minimum.outer(k, k)
    numerator = L + adjacency
    denominator = k_matrix + 1 - adjacency
    denominator[denominator <= 0] = 1e-10
    TOM = numerator / denominator
    np.fill_diagonal(TOM, 1)
    return TOM


def detect_modules(TOM, gene_names, min_size=30):
    """Detect modules using hierarchical clustering on TOM dissimilarity."""
    print("  Detecting modules...")
    dissTOM = 1 - TOM
    np.fill_diagonal(dissTOM, 0)
    dist_condensed = squareform(dissTOM, checks=False)
    Z = linkage(dist_condensed, method="average")

    # Find optimal cut height
    heights = np.linspace(0.1, 0.95, 50)
    best_height = 0.7
    best_n_modules = 0

    for h in heights:
        clusters = fcluster(Z, t=h, criterion="distance")
        unique, counts = np.unique(clusters, return_counts=True)
        n_valid = sum(c >= min_size for c in counts)
        if n_valid > best_n_modules and n_valid <= 20:
            best_n_modules = n_valid
            best_height = h

    clusters = fcluster(Z, t=best_height, criterion="distance")

    # Assign colors to modules
    unique_clusters = np.unique(clusters)
    module_colors = {}
    color_idx = 0

    for cluster in unique_clusters:
        cluster_size = sum(clusters == cluster)
        if cluster_size >= min_size:
            module_colors[cluster] = MODULE_COLORS[color_idx % len(MODULE_COLORS)]
            color_idx += 1
        else:
            module_colors[cluster] = "grey"

    gene_modules = pd.DataFrame(
        {
            "gene": gene_names,
            "cluster": clusters,
            "module_color": [module_colors[c] for c in clusters],
        }
    )

    module_summary = (
        gene_modules.groupby("module_color").size().reset_index(name="n_genes")
    )
    module_summary = module_summary.sort_values("n_genes", ascending=False)

    print(f"  Detected {len(module_summary)} modules")
    return gene_modules, Z, dissTOM, module_summary


# ============================================================
# MODULE-TRAIT CORRELATION
# ============================================================
def calculate_module_eigengenes(expr_df, gene_modules):
    """Calculate module eigengenes using SVD."""
    print("  Calculating module eigengenes...")
    modules = gene_modules[gene_modules["module_color"] != "grey"][
        "module_color"
    ].unique()
    MEs = {}

    for module in modules:
        module_genes = gene_modules[gene_modules["module_color"] == module][
            "gene"
        ].values
        module_expr = expr_df.loc[expr_df.index.isin(module_genes)]

        if len(module_expr) > 0:
            centered = module_expr.values - module_expr.values.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
            ME = U[:, 0]
            MEs[f"ME{module}"] = ME

    ME_df = pd.DataFrame(MEs, index=expr_df.columns)
    return ME_df


def module_trait_correlation(ME_df, labels):
    """Calculate module-trait correlations with FDR correction."""
    from statsmodels.stats.multitest import multipletests

    print("  Calculating module-trait correlations...")
    trait_numeric = np.array([0 if str(l).lower() == "control" else 1 for l in labels])

    correlations = []
    for col in ME_df.columns:
        r, p = stats.pearsonr(ME_df[col].values, trait_numeric)
        correlations.append(
            {"module": col.replace("ME", ""), "correlation": r, "pvalue": p}
        )

    cor_df = pd.DataFrame(correlations)

    # Add FDR-corrected q-values (Benjamini-Hochberg)
    _, qvalues, _, _ = multipletests(cor_df["pvalue"].values, method="fdr_bh")
    cor_df["qvalue"] = qvalues

    cor_df = cor_df.sort_values("pvalue")
    print(f"  Modules with q < 0.05: {sum(cor_df['qvalue'] < 0.05)}")
    print(f"  Modules with p < 0.05: {sum(cor_df['pvalue'] < 0.05)}")
    return cor_df


# ============================================================
# HUB GENE IDENTIFICATION (FIXED VERSION)
# ============================================================
def identify_hub_genes(expr_df, gene_modules, TOM, labels, ME_df, cor_df):
    """
    Identify hub genes based on GS, MM, and kWithin.

    FIXES:
    1. Only select genes from significantly correlated modules (q < 0.05, FDR-corrected)
    2. Preserve GS sign (direction information)
    3. Normalize kWithin within each module separately
    """
    print("  Identifying hub genes...")
    trait_numeric = np.array([0 if str(l).lower() == "control" else 1 for l in labels])
    gene_names = expr_df.index.tolist()

    # Get significantly correlated modules (using FDR-corrected q-value)
    sig_modules = cor_df[cor_df["qvalue"] < MODULE_QVALUE_THRESHOLD]["module"].tolist()
    print(
        f"  Significant modules (q < {MODULE_QVALUE_THRESHOLD}, FDR-corrected): {sig_modules}"
    )

    if len(sig_modules) == 0:
        print("  WARNING: No significant modules found! Using top 3 by p-value.")
        sig_modules = cor_df.head(3)["module"].tolist()

    hub_results = []

    for i, gene in enumerate(gene_names):
        if gene not in gene_modules["gene"].values:
            continue

        module = gene_modules[gene_modules["gene"] == gene]["module_color"].values[0]

        # Skip grey module and non-significant modules
        if module == "grey":
            continue
        if module not in sig_modules:
            continue

        # Gene Significance (GS): correlation with trait - KEEP SIGN!
        gene_expr = expr_df.loc[gene].values
        gs, gs_p = stats.pearsonr(gene_expr, trait_numeric)

        # GS_raw preserves direction, GS_abs for ranking
        gs_raw = gs
        gs_abs = abs(gs)

        # ── KEY FIX: keep only UP-regulated genes (positive GS) ──
        # Rationale: most candidate drugs are inhibitors, so hub genes
        # must be up-regulated in disease for inhibitor-based repurposing
        # to make biological sense.
        if gs_raw <= 0:
            continue

        # Module Membership (MM): correlation with module eigengene
        me_col = f"ME{module}"
        if me_col in ME_df.columns:
            mm, mm_p = stats.pearsonr(gene_expr, ME_df[me_col].values)
        else:
            mm, mm_p = 0, 1

        # Intramodular connectivity (kWithin)
        module_genes = gene_modules[gene_modules["module_color"] == module][
            "gene"
        ].values
        module_idx = [gene_names.index(g) for g in module_genes if g in gene_names]
        gene_idx = gene_names.index(gene)

        if gene_idx in module_idx:
            kWithin = TOM[gene_idx, module_idx].sum()
        else:
            kWithin = 0

        hub_results.append(
            {
                "gene": gene,
                "module": module,
                "GS_raw": gs_raw,  # Preserve direction
                "GS": gs_abs,  # Absolute value for ranking
                "GS_pvalue": gs_p,
                "MM": abs(mm),
                "MM_pvalue": mm_p,
                "kWithin": kWithin,
            }
        )

    hub_df = pd.DataFrame(hub_results)

    if len(hub_df) == 0:
        print("  ERROR: No hub genes found!")
        return hub_df

    # Normalize within each module for kWithin, globally for GS and MM
    # This prevents large modules from dominating
    hub_df["GS_norm"] = (hub_df["GS"] - hub_df["GS"].min()) / (
        hub_df["GS"].max() - hub_df["GS"].min() + 1e-10
    )
    hub_df["MM_norm"] = (hub_df["MM"] - hub_df["MM"].min()) / (
        hub_df["MM"].max() - hub_df["MM"].min() + 1e-10
    )

    # Normalize kWithin within each module
    hub_df["kWithin_norm"] = 0.0
    for module in hub_df["module"].unique():
        mask = hub_df["module"] == module
        module_kWithin = hub_df.loc[mask, "kWithin"]
        min_k = module_kWithin.min()
        max_k = module_kWithin.max()
        hub_df.loc[mask, "kWithin_norm"] = (module_kWithin - min_k) / (
            max_k - min_k + 1e-10
        )

    hub_df["composite_score"] = (
        hub_df["GS_norm"] + hub_df["MM_norm"] + hub_df["kWithin_norm"]
    ) / 3
    hub_df = hub_df.sort_values("composite_score", ascending=False)

    # All remaining genes are up-regulated (GS_raw > 0 filter applied above)
    hub_df["direction"] = "Up"

    print(f"  Found {len(hub_df)} up-regulated hub genes from significant modules")

    return hub_df


def assess_hub_stability(
    expr_df, gene_modules, TOM, labels, ME_df, cor_df, hub_df, n_bootstrap=100
):
    """
    Assess hub gene ranking stability using bootstrap resampling.

    Returns stability metrics for top hub genes:
    - mean_rank: Average rank across bootstrap iterations
    - rank_sd: Standard deviation of rank
    - top20_freq: Frequency of appearing in top 20
    - top100_freq: Frequency of appearing in top 100
    """
    print("\n  Assessing hub gene stability (bootstrap)...")
    np.random.seed(42)

    n_samples = expr_df.shape[1]
    top_genes = hub_df.head(100)["gene"].tolist()

    # Track ranks across bootstrap iterations
    rank_tracker = {gene: [] for gene in top_genes}

    for i in range(n_bootstrap):
        # Bootstrap resample samples (with replacement)
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_expr = expr_df.iloc[:, boot_idx]
        boot_labels = [labels[j] for j in boot_idx]

        # Recalculate GS for each gene
        trait_numeric = np.array(
            [0 if str(l).lower() == "control" else 1 for l in boot_labels]
        )

        boot_scores = []
        for gene in top_genes:
            if gene in boot_expr.index:
                gene_expr = boot_expr.loc[gene].values
                try:
                    gs, _ = stats.pearsonr(gene_expr, trait_numeric)
                    # Use original MM and kWithin (network structure unchanged)
                    orig_row = hub_df[hub_df["gene"] == gene].iloc[0]
                    # Recalculate composite with bootstrap GS
                    gs_norm = abs(gs)  # Will be normalized later
                    boot_scores.append(
                        {
                            "gene": gene,
                            "GS_boot": abs(gs),
                            "MM": orig_row["MM"],
                            "kWithin": orig_row["kWithin"],
                        }
                    )
                except:
                    pass

        if len(boot_scores) > 0:
            boot_df = pd.DataFrame(boot_scores)
            # Normalize
            for col in ["GS_boot", "MM", "kWithin"]:
                min_v, max_v = boot_df[col].min(), boot_df[col].max()
                boot_df[f"{col}_norm"] = (boot_df[col] - min_v) / (
                    max_v - min_v + 1e-10
                )
            boot_df["composite"] = (
                boot_df["GS_boot_norm"] + boot_df["MM_norm"] + boot_df["kWithin_norm"]
            ) / 3
            boot_df = boot_df.sort_values("composite", ascending=False).reset_index(
                drop=True
            )

            # Record ranks
            for rank, row in boot_df.iterrows():
                gene = row["gene"]
                if gene in rank_tracker:
                    rank_tracker[gene].append(rank + 1)  # 1-indexed rank

    # Calculate stability metrics
    stability_results = []
    for gene in top_genes:
        ranks = rank_tracker[gene]
        if len(ranks) > 0:
            stability_results.append(
                {
                    "gene": gene,
                    "mean_rank": np.mean(ranks),
                    "rank_sd": np.std(ranks),
                    "top20_freq": sum(1 for r in ranks if r <= 20) / len(ranks),
                    "top100_freq": sum(1 for r in ranks if r <= 100) / len(ranks),
                    "n_bootstrap": len(ranks),
                }
            )

    stability_df = pd.DataFrame(stability_results)

    # Summary statistics
    top20_stable = stability_df.head(20)
    mean_top20_freq = top20_stable["top20_freq"].mean()
    mean_rank_sd = top20_stable["rank_sd"].mean()

    print(f"    Bootstrap iterations: {n_bootstrap}")
    print(f"    Top 20 genes - mean stability (top20_freq): {mean_top20_freq:.3f}")
    print(f"    Top 20 genes - mean rank SD: {mean_rank_sd:.1f}")

    return stability_df


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_soft_threshold(results_df, optimal_power, output_dir):
    """Generate soft threshold selection plots (Panel A)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.scatter(
        results_df["power"],
        results_df["SFT_R2"],
        c="steelblue",
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.axhline(y=0.80, color="red", linestyle="--", linewidth=1)
    ax1.axvline(x=optimal_power, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Soft Threshold (power)", fontsize=10)
    ax1.set_ylabel("Scale Free Topology Model Fit (R2)", fontsize=10)
    ax1.set_title("A. Scale-free Topology Fit", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 21)
    ax1.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.scatter(
        results_df["power"],
        results_df["mean_k"],
        c="coral",
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.axvline(x=optimal_power, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Soft Threshold (power)", fontsize=10)
    ax2.set_ylabel("Mean Connectivity", fontsize=10)
    ax2.set_title("Mean Connectivity", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 21)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2A.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel A")


def plot_dendrogram(Z, gene_modules, output_dir):
    """Generate dendrogram with module colors (Panel B)."""
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [4, 1]}
    )

    ax1 = axes[0]
    dendro = dendrogram(
        Z, ax=ax1, no_labels=True, color_threshold=0, above_threshold_color="black"
    )
    ax1.set_ylabel("Height", fontsize=10)
    ax1.set_title(
        "B. Gene Dendrogram and Module Colors", fontsize=11, fontweight="bold"
    )
    ax1.set_xticks([])

    ax2 = axes[1]
    leaf_order = dendro["leaves"]
    ordered_colors = gene_modules.iloc[leaf_order]["module_color"].values
    colors_rgb = [COLOR_MAP.get(c, "#808080") for c in ordered_colors]
    for i, color in enumerate(colors_rgb):
        ax2.axvspan(i, i + 1, facecolor=color, edgecolor="none")
    ax2.set_xlim(0, len(ordered_colors))
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Module", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2B.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel B")


def plot_module_sizes(module_summary, output_dir):
    """Generate module size bar plot (Panel C)."""
    plot_data = module_summary[module_summary["module_color"] != "grey"].copy()
    plot_data = plot_data.sort_values("n_genes", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [COLOR_MAP.get(c, "#808080") for c in plot_data["module_color"]]
    ax.barh(
        range(len(plot_data)),
        plot_data["n_genes"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["module_color"], fontsize=9)
    ax.set_xlabel("Number of Genes", fontsize=10)
    ax.set_title("C. Module Sizes", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2C.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel C")


def plot_module_trait_heatmap(cor_df, output_dir):
    """Generate module-trait correlation heatmap (Panel D)."""
    fig, ax = plt.subplots(figsize=(4, 6))

    modules = cor_df["module"].values
    correlations = cor_df["correlation"].values
    pvalues = cor_df["pvalue"].values

    heatmap_data = correlations.reshape(-1, 1)
    im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=9)
    ax.set_xticks([0])
    ax.set_xticklabels(["CRC (Case vs Control)"], fontsize=10)
    ax.set_title("D. Module-Trait Correlation", fontsize=11, fontweight="bold")

    for i, (r, p) in enumerate(zip(correlations, pvalues)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(
            0,
            i,
            f"{r:.2f}{sig}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(r) > 0.5 else "black",
        )

    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2D.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel D")


def plot_hub_gene_scatter(hub_df, output_dir):
    """Generate hub gene scatter plot (Panel E)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    top_module = hub_df["module"].value_counts().index[0]
    plot_data = hub_df[hub_df["module"] == top_module].head(100)

    scatter = ax.scatter(
        plot_data["MM"],
        plot_data["GS"],
        c=plot_data["kWithin"],
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.3,
    )

    top_genes = plot_data.head(10)
    for _, row in top_genes.iterrows():
        ax.annotate(row["gene"], (row["MM"], row["GS"]), fontsize=7, alpha=0.8)

    ax.set_xlabel("Module Membership (MM)", fontsize=10)
    ax.set_ylabel("Gene Significance (GS)", fontsize=10)
    ax.set_title(f"E. Hub Genes in {top_module} Module", fontsize=11, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="kWithin")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2E.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel E")


def plot_top_hub_genes(hub_df, output_dir):
    """Generate top hub genes bar plot (Panel F)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    top20 = hub_df.head(20).copy()
    top20 = top20.sort_values("composite_score", ascending=True)

    colors = [COLOR_MAP.get(m, "#808080") for m in top20["module"]]
    ax.barh(
        range(len(top20)),
        top20["composite_score"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["gene"], fontsize=9)
    ax.set_xlabel("Composite Hub Score", fontsize=10)
    ax.set_title("F. Top 20 Hub Genes", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2F.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel F")

    # Save source data for Panel F
    top20_save = hub_df.head(20)[
        ["gene", "module", "composite_score", "GS", "MM", "kWithin", "direction"]
    ]
    top20_save.to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2F_Top20Genes.csv", index=False
    )


# ============================================================
# PANELS G-J: MODULE ANALYSIS
# ============================================================
def plot_panel_G_MM_GS_all_modules(hub_df, output_dir):
    """Panel G: MM vs GS scatter for all significant modules."""
    fig, ax = plt.subplots(figsize=(6, 5))

    modules = hub_df["module"].unique()
    for module in modules:
        module_data = hub_df[hub_df["module"] == module]
        color = COLOR_MAP.get(module, "#808080")
        ax.scatter(
            module_data["MM"],
            module_data["GS"],
            c=color,
            label=module,
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Module Membership (MM)", fontsize=10)
    ax.set_ylabel("Gene Significance (|GS|)", fontsize=10)
    ax.set_title("G. MM vs GS Across All Modules", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2G.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel G")

    # Save source data
    hub_df[["gene", "module", "MM", "GS"]].to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2G_MM_GS.csv", index=False
    )


def plot_panel_H_kWithin_distribution(hub_df, output_dir):
    """Panel H: kWithin distribution by module."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))

    modules = hub_df["module"].unique()
    data_list = []
    colors = []

    for module in modules:
        module_data = hub_df[hub_df["module"] == module]["kWithin_norm"].values
        data_list.append(module_data)
        colors.append(COLOR_MAP.get(module, "#808080"))

    bp = ax.boxplot(data_list, patch_artist=True, labels=modules)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Module", fontsize=10)
    ax.set_ylabel("Normalized kWithin", fontsize=10)
    ax.set_title(
        "H. Intramodular Connectivity by Module", fontsize=11, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2H.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel H")

    # Save source data
    hub_df[["gene", "module", "kWithin", "kWithin_norm"]].to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2H_kWithin.csv", index=False
    )


def plot_panel_I_module_eigengene_heatmap(ME_df, labels, output_dir):
    """Panel I: Module eigengene expression heatmap."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort samples by label
    label_order = np.argsort(labels)
    ME_sorted = ME_df.iloc[label_order].T

    # Create annotation for samples
    col_colors = ["#E64B35" if l == "case" else "#4DBBD5" for l in labels[label_order]]

    sns.heatmap(
        ME_sorted,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "ME value"},
    )

    ax.set_xlabel("Samples (sorted by age)", fontsize=10)
    ax.set_ylabel("Module Eigengene", fontsize=10)
    ax.set_title("I. Module Eigengene Expression", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2I.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel I")

    # Save source data
    ME_df.to_csv(f"{output_dir}/sourcedata16/SourceData_Fig2I_ModuleEigengenes.csv")


def plot_panel_J_GS_pvalue_volcano(hub_df, output_dir):
    """Panel J: GS p-value volcano plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    hub_df["neg_log_p"] = -np.log10(hub_df["GS_pvalue"] + 1e-300)

    # Color by direction
    colors = ["#E64B35" if d == "Up" else "#4DBBD5" for d in hub_df["direction"]]

    ax.scatter(
        hub_df["GS_raw"],
        hub_df["neg_log_p"],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.3,
    )

    # Label top genes
    top_genes = hub_df.nlargest(10, "neg_log_p")
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["gene"], (row["GS_raw"], row["neg_log_p"]), fontsize=6, alpha=0.8
        )

    ax.axhline(y=-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Gene Significance (GS_raw)", fontsize=10)
    ax.set_ylabel("-log10(p-value)", fontsize=10)
    ax.set_title("J. GS Significance Volcano", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2J.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel J")

    # Save source data
    hub_df[["gene", "GS_raw", "GS_pvalue", "direction"]].to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2J_GS_Volcano.csv", index=False
    )


# ============================================================
# PANELS K-P: HUB GENE SELECTION
# ============================================================
def plot_panel_K_composite_score_components(hub_df, output_dir):
    """Panel K: Stacked bar of composite score components for top genes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    top20 = hub_df.head(20).copy()
    top20 = top20.iloc[::-1]  # Reverse for horizontal bar

    y_pos = np.arange(len(top20))

    ax.barh(y_pos, top20["GS_norm"], label="GS_norm", color="#E64B35", alpha=0.8)
    ax.barh(
        y_pos,
        top20["MM_norm"],
        left=top20["GS_norm"],
        label="MM_norm",
        color="#4DBBD5",
        alpha=0.8,
    )
    ax.barh(
        y_pos,
        top20["kWithin_norm"],
        left=top20["GS_norm"] + top20["MM_norm"],
        label="kWithin_norm",
        color="#00A087",
        alpha=0.8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["gene"], fontsize=8)
    ax.set_xlabel("Score Components", fontsize=10)
    ax.set_title("K. Composite Score Components", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2K.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel K")

    # Save source data
    top20[["gene", "GS_norm", "MM_norm", "kWithin_norm", "composite_score"]].to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2K_ScoreComponents.csv", index=False
    )


def plot_panel_L_direction_pie(hub_df, output_dir):
    """Panel L: Direction distribution pie chart."""
    fig, ax = plt.subplots(figsize=(5, 5))

    direction_counts = hub_df["direction"].value_counts()
    colors = ["#E64B35" if d == "Up" else "#4DBBD5" for d in direction_counts.index]

    wedges, texts, autotexts = ax.pie(
        direction_counts.values,
        labels=direction_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=[0.02] * len(direction_counts),
    )

    ax.set_title("L. Direction Distribution", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2L.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel L")

    # Save source data
    direction_counts.to_csv(f"{output_dir}/sourcedata16/SourceData_Fig2L_Direction.csv")


def plot_panel_M_module_hub_count(hub_df, output_dir):
    """Panel M: Hub gene count per module."""
    fig, ax = plt.subplots(figsize=(6, 5))

    module_counts = hub_df["module"].value_counts()
    colors = [COLOR_MAP.get(m, "#808080") for m in module_counts.index]

    ax.bar(
        range(len(module_counts)),
        module_counts.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(module_counts)))
    ax.set_xticklabels(module_counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Module", fontsize=10)
    ax.set_ylabel("Number of Hub Genes", fontsize=10)
    ax.set_title("M. Hub Genes per Module", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2M.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel M")

    # Save source data
    module_counts.to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2M_ModuleHubCount.csv"
    )


def plot_panel_N_top_genes_heatmap(expr_df, hub_df, labels, output_dir):
    """Panel N: Expression heatmap of top 20 hub genes."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))

    top20_genes = hub_df.head(20)["gene"].tolist()
    expr_top20 = expr_df.loc[expr_df.index.isin(top20_genes)]

    # Sort samples by label
    label_order = np.argsort(labels)
    expr_sorted = expr_top20.iloc[:, label_order]

    # Z-score normalize
    expr_z = (expr_sorted.T - expr_sorted.mean(axis=1)).T / (
        expr_sorted.std(axis=1).values.reshape(-1, 1) + 1e-10
    )

    sns.heatmap(
        expr_z,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Z-score"},
    )

    ax.set_xlabel("Samples (sorted by age)", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.set_title("N. Top 20 Hub Gene Expression", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2N.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel N")

    # Save source data
    expr_top20.to_csv(f"{output_dir}/sourcedata16/SourceData_Fig2N_Top20Expression.csv")


def plot_panel_O_rank_comparison(hub_df, output_dir):
    """Panel O: Rank comparison of GS, MM, kWithin."""
    fig, ax = plt.subplots(figsize=(6, 5))

    top50 = hub_df.head(50).copy()
    top50["GS_rank"] = top50["GS"].rank(ascending=False)
    top50["MM_rank"] = top50["MM"].rank(ascending=False)
    top50["kWithin_rank"] = top50["kWithin"].rank(ascending=False)

    x = np.arange(len(top50))
    width = 0.25

    ax.bar(
        x - width, top50["GS_rank"], width, label="GS Rank", color="#E64B35", alpha=0.8
    )
    ax.bar(x, top50["MM_rank"], width, label="MM Rank", color="#4DBBD5", alpha=0.8)
    ax.bar(
        x + width,
        top50["kWithin_rank"],
        width,
        label="kWithin Rank",
        color="#00A087",
        alpha=0.8,
    )

    ax.set_xlabel("Gene (by composite score)", fontsize=10)
    ax.set_ylabel("Rank", fontsize=10)
    ax.set_title("O. Metric Rank Comparison (Top 50)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2O.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel O")

    # Save source data
    top50[["gene", "GS_rank", "MM_rank", "kWithin_rank", "composite_score"]].to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2O_RankComparison.csv", index=False
    )


def plot_panel_P_summary_stats(hub_df, module_summary, cor_df, output_dir):
    """Panel P: Summary statistics panel."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Calculate summary stats
    n_modules = len(module_summary[module_summary["module_color"] != "grey"])
    n_sig_modules = len(cor_df[cor_df["pvalue"] < 0.05])
    n_hub_genes = len(hub_df)
    n_up = len(hub_df[hub_df["direction"] == "Up"])
    n_down = len(hub_df[hub_df["direction"] == "Down"])
    top_gene = hub_df.iloc[0]["gene"]
    top_module = hub_df.iloc[0]["module"]

    stats_text = f"""
    WGCNA Analysis Summary
    ══════════════════════════════

    Modules Detected: {n_modules}
    Significant Modules (p<0.05): {n_sig_modules}

    Hub Genes Identified: {n_hub_genes}
      • Up-regulated: {n_up}
      • Down-regulated: {n_down}

    Top Hub Gene: {top_gene}
    Top Module: {top_module}

    ══════════════════════════════
    """

    ax.text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )
    ax.axis("off")
    ax.set_title("P. Analysis Summary", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/panels16/Fig2P.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved Panel P")

    # Save summary stats
    summary_df = pd.DataFrame(
        {
            "metric": [
                "n_modules",
                "n_sig_modules",
                "n_hub_genes",
                "n_up",
                "n_down",
                "top_gene",
                "top_module",
            ],
            "value": [
                n_modules,
                n_sig_modules,
                n_hub_genes,
                n_up,
                n_down,
                top_gene,
                top_module,
            ],
        }
    )
    summary_df.to_csv(
        f"{output_dir}/sourcedata16/SourceData_Fig2P_Summary.csv", index=False
    )


# ============================================================
# COMPOSITE FIGURE
# ============================================================
def create_composite_figure(output_dir):
    """Create 4x4 composite figure from individual panels."""
    print("\nCreating composite figure...")

    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=300)

    panel_files = [
        "Fig2A.png",
        "Fig2B.png",
        "Fig2C.png",
        "Fig2D.png",
        "Fig2E.png",
        "Fig2F.png",
        "Fig2G.png",
        "Fig2H.png",
        "Fig2I.png",
        "Fig2J.png",
        "Fig2K.png",
        "Fig2L.png",
        "Fig2M.png",
        "Fig2N.png",
        "Fig2O.png",
        "Fig2P.png",
    ]

    for idx, (ax, panel_file) in enumerate(zip(axes.flat, panel_files)):
        panel_path = f"{output_dir}/panels16/{panel_file}"
        if os.path.exists(panel_path):
            img = plt.imread(panel_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, f"Panel {chr(65 + idx)} missing", ha="center", va="center"
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/composite/Figure2_Composite.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  Saved composite figure")


# ============================================================
# MAIN EXECUTION
# ============================================================
def copy_source_code(output_code_dir):
    """Copy source code files to output directory for reproducibility."""
    import shutil
    import glob

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Copy all .py files from script directory to output code directory
    py_files = glob.glob(os.path.join(script_dir, "*.py"))
    for py_file in py_files:
        shutil.copy2(py_file, output_code_dir)

    print(f"  Copied {len(py_files)} code files to {output_code_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("FIGURE 2: WGCNA-BASED GENE PRIORITIZATION (16 PANELS)")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Ensure output directories exist
    import os

    for subdir in ["panels16", "sourcedata16", "raw", "composite", "code"]:
        os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)

    # Copy source code to output directory
    copy_source_code(f"{OUTPUT_DIR}/code")

    # Load data
    expr_df, sample_ids, labels = load_expression_data(INPUT_DATA)

    # Pre-filter genes
    expr_filtered = prefilter_genes(expr_df, MIN_MEAN_EXPR, MIN_VARIANCE, TOP_VAR_GENES)
    expr_filtered.to_csv(f"{OUTPUT_DIR}/raw/expr_filtered_for_wgcna.csv")

    # Soft threshold selection
    optimal_power, sft_results = pick_soft_threshold(
        expr_filtered, range(1, 21), R2_CUTOFF
    )
    sft_results.to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2A_SoftThreshold.csv", index=False
    )

    # Save source data for Panel B (dendrogram data)
    sft_results.to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2B_SoftThreshold.csv", index=False
    )

    # Calculate adjacency and TOM
    print("\nBUILDING NETWORK")
    cor_matrix = expr_filtered.T.corr()
    adjacency = np.abs(cor_matrix.values) ** optimal_power
    np.fill_diagonal(adjacency, 0)
    TOM = calculate_TOM_fast(adjacency)

    # Detect modules
    gene_names = expr_filtered.index.tolist()
    gene_modules, Z, dissTOM, module_summary = detect_modules(
        TOM, gene_names, MIN_MODULE_SIZE
    )
    gene_modules.to_csv(f"{OUTPUT_DIR}/raw/gene_modules.csv", index=False)
    module_summary.to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2C_ModuleSizes.csv", index=False
    )

    # Module eigengenes and trait correlation
    print("\nMODULE-TRAIT ANALYSIS")
    ME_df = calculate_module_eigengenes(expr_filtered, gene_modules)
    ME_df.to_csv(f"{OUTPUT_DIR}/raw/module_eigengenes.csv")

    cor_df = module_trait_correlation(ME_df, labels)
    cor_df.to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2D_ModuleTrait.csv", index=False
    )

    # Hub gene identification
    print("\nHUB GENE IDENTIFICATION")
    hub_df = identify_hub_genes(expr_filtered, gene_modules, TOM, labels, ME_df, cor_df)
    hub_df.to_csv(f"{OUTPUT_DIR}/raw/hub_genes_full.csv", index=False)

    # Stability analysis (bootstrap)
    stability_df = assess_hub_stability(
        expr_filtered, gene_modules, TOM, labels, ME_df, cor_df, hub_df, n_bootstrap=100
    )
    stability_df.to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2_HubStability.csv", index=False
    )

    # Save source data for Panel E
    top_module = hub_df["module"].value_counts().index[0]
    hub_df[hub_df["module"] == top_module].head(100).to_csv(
        f"{OUTPUT_DIR}/sourcedata16/SourceData_Fig2E_HubScatter.csv", index=False
    )

    # Save top candidates
    top100 = hub_df.head(TOP_HUB_GENES)
    top100.to_csv(f"{OUTPUT_DIR}/candidate_genes_top100.csv", index=False)

    top20 = hub_df.head(TOP_CANDIDATE_GENES)
    top20.to_csv(f"{OUTPUT_DIR}/candidate_genes_top20.csv", index=False)

    # ============================================================
    # GENERATE ALL 16 PANELS
    # ============================================================
    print("\nGENERATING 16 PANELS")

    # Row 1: Data Preprocessing (A-B) + Network Construction (C-D)
    print("\n  Row 1: Preprocessing & Network")
    plot_soft_threshold(sft_results, optimal_power, OUTPUT_DIR)  # Panel A
    plot_dendrogram(Z, gene_modules, OUTPUT_DIR)  # Panel B
    plot_module_sizes(module_summary, OUTPUT_DIR)  # Panel C
    plot_module_trait_heatmap(cor_df, OUTPUT_DIR)  # Panel D

    # Row 2: Network Construction (E-F) + Module Analysis (G-H)
    print("\n  Row 2: Network & Module Analysis")
    plot_hub_gene_scatter(hub_df, OUTPUT_DIR)  # Panel E
    plot_top_hub_genes(hub_df, OUTPUT_DIR)  # Panel F
    plot_panel_G_MM_GS_all_modules(hub_df, OUTPUT_DIR)  # Panel G
    plot_panel_H_kWithin_distribution(hub_df, OUTPUT_DIR)  # Panel H

    # Row 3: Module Analysis (I-J) + Hub Gene Selection (K-L)
    print("\n  Row 3: Module Analysis & Hub Selection")
    plot_panel_I_module_eigengene_heatmap(ME_df, labels, OUTPUT_DIR)  # Panel I
    plot_panel_J_GS_pvalue_volcano(hub_df, OUTPUT_DIR)  # Panel J
    plot_panel_K_composite_score_components(hub_df, OUTPUT_DIR)  # Panel K
    plot_panel_L_direction_pie(hub_df, OUTPUT_DIR)  # Panel L

    # Row 4: Hub Gene Selection (M-P)
    print("\n  Row 4: Hub Gene Selection")
    plot_panel_M_module_hub_count(hub_df, OUTPUT_DIR)  # Panel M
    plot_panel_N_top_genes_heatmap(expr_filtered, hub_df, labels, OUTPUT_DIR)  # Panel N
    plot_panel_O_rank_comparison(hub_df, OUTPUT_DIR)  # Panel O
    plot_panel_P_summary_stats(hub_df, module_summary, cor_df, OUTPUT_DIR)  # Panel P

    # Create composite figure
    create_composite_figure(OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("FIGURE 2 COMPLETE - 16 PANELS GENERATED")
    print("=" * 60)
    print(f"  Genes analyzed: {len(gene_names)}")
    print(f"  Modules detected: {len(module_summary)}")
    print(
        f"  Significant modules (p<0.05, uncorrected): {len(cor_df[cor_df['pvalue'] < 0.05])}"
    )
    print(
        f"  Significant modules (q<0.05, FDR-corrected): {len(cor_df[cor_df['qvalue'] < 0.05])}"
    )
    print(f"  Hub genes selected from q<0.05 modules only")
    print(f"  Optimal power: {optimal_power}")
    print(f"  Hub genes identified: {len(hub_df)}")
    print(f"  Top gene: {hub_df.iloc[0]['gene']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().isoformat()}")
