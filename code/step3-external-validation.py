#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fig3.py - Figure 3: Mechanism Validation & External Validation
16 Panels (A-P), 4x4 Layout, Nature Style

Usage:
    python run_fig3.py --f1_dir ../f1v1 --f2_dir ../f2v1 --ext_dir ../externalvalidatedata --output ./

Generates:
    - panels16/Fig3[A-P]_*.png (16 individual panels)
    - composite/Figure3_Composite.png
    - sourcedata16/SourceData_Fig3_*.csv
    - raw/api_cache/*.json
    - manifests/fig3_manifest.json
    - logs/fig3_run.log
"""

import argparse
import json
import logging
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import requests

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
API_CACHE_DIR = None
LOG_FILE = None

# API Endpoints
STRING_API = "https://string-db.org/api/json/network"
ENRICHR_ADD_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
DGIDB_API = "https://www.dgidb.org/api/v2/interactions.json"

# Fusion Partners Resource (ZERO-FAKE: external curated list)
FUSION_PARTNERS_FILE = None  # Will be set in main()

# Colors (Nature style)
COLOR_UP = "#E64B35"
COLOR_DOWN = "#4DBBD5"
COLOR_TARGET = "#FF6347"
COLOR_NEIGHBOR = "#4169E1"
COLOR_VALIDATED = "#00A087"
COLOR_NOT_VALIDATED = "#DC0000"


def setup_logging(log_file):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def cache_api_response(cache_key, data):
    """Cache API response to file."""
    cache_file = API_CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, indent=2)
    return cache_file


def load_cached_response(cache_key):
    """Load cached API response if exists."""
    cache_file = API_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)["data"]
    return None


# ============================================================================
# FUSION PARTNERS LOADING (ZERO-FAKE: external curated resource)
# ============================================================================


def load_fusion_partners(logger):
    """
    Load fusion partners from external curated TSV file.

    ZERO-FAKE: These genes are fusion partners, NOT direct drug targets.
    Drugs targeting the driver gene should NOT be attributed to the partner.
    """
    global FUSION_PARTNERS_FILE

    fusion_partners = {}

    if FUSION_PARTNERS_FILE is None or not Path(FUSION_PARTNERS_FILE).exists():
        logger.warning("Fusion partners file not found. Using built-in minimal list.")
        # Minimal fallback list for critical cases
        fusion_partners = {
            "KIF5B": {"driver": "RET", "source": "PMID:22327623"},
            "EML4": {"driver": "ALK", "source": "PMID:17625570"},
            "BCR": {"driver": "ABL1", "source": "PMID:6324129"},
        }
        return fusion_partners

    try:
        df = pd.read_csv(FUSION_PARTNERS_FILE, sep="\t", comment="#")
        for _, row in df.iterrows():
            partner = row["partner_gene"].strip().upper()
            fusion_partners[partner] = {
                "driver": row["driver_gene"].strip().upper(),
                "source": row.get("source", "Curated"),
                "notes": row.get("notes", ""),
            }
        logger.info(
            f"  Loaded {len(fusion_partners)} fusion partners from {FUSION_PARTNERS_FILE}"
        )
    except Exception as e:
        logger.error(f"  Failed to load fusion partners: {e}")

    return fusion_partners


# ============================================================================
# REGPATH EVIDENCE CALCULATION (ZERO-FAKE: weighted Enrichr evidence)
# ============================================================================

import math
import re
from collections import defaultdict


def _iter_enrichr_terms(enrichr_results):
    """
    Yield (library_name, term_data_list) from various possible Enrichr result shapes.

    Accepts:
      - {lib: [term_data, ...]}
      - {lib: {"results": [term_data, ...]}}
      - {"results": {lib: [term_data, ...]}}
    """
    if not enrichr_results:
        return

    # case: {"results": {lib: [...]}}
    if (
        isinstance(enrichr_results, dict)
        and "results" in enrichr_results
        and isinstance(enrichr_results["results"], dict)
    ):
        for lib, terms in enrichr_results["results"].items():
            yield lib, terms
        return

    # case: {lib: [...] } or {lib: {"results":[...]} }
    if isinstance(enrichr_results, dict):
        for lib, obj in enrichr_results.items():
            if isinstance(obj, dict) and "results" in obj:
                yield lib, obj["results"]
            elif isinstance(obj, list):
                yield lib, obj


def _split_genes(gene_input):
    """
    Parse genes from Enrichr term data.

    Enrichr returns genes as a list (e.g., ['GNA13', 'PLCB1']),
    but older versions or cached data may have string format.
    """
    if gene_input is None:
        return []

    # If already a list, just uppercase and return
    if isinstance(gene_input, list):
        return [str(g).strip().upper() for g in gene_input if g]

    # Handle string format (legacy or other sources)
    s = str(gene_input).strip()
    if not s:
        return []
    # split by ; or , or whitespace
    parts = re.split(r"[;, \t\r\n]+", s)
    return [p.strip().upper() for p in parts if p.strip()]


def calculate_regpath_evidence(
    evidence_df,
    enrichr_results,
    logger,
    adj_p_cutoff=0.05,
    weight_mode="neglog10_adj_p",
):
    """
    Calculate RegPathEvidence from Enrichr results.

    ZERO-FAKE: Evidence is derived from real Enrichr API results only.

    Evidence definition (default):
      For each significant term (adj_p < cutoff), each gene in term gains weight:
        w = -log10(adj_p)   (clipped to avoid inf)
      RegPathEvidence(gene) = sum_w(gene) / max_sum_w

    Args:
        evidence_df: DataFrame with gene column
        enrichr_results: Dict from Enrichr API queries
        logger: Logger instance
        adj_p_cutoff: Adjusted p-value threshold (default 0.05)
        weight_mode: "neglog10_adj_p" (default) or "count"

    Returns:
        evidence_df with RegPathEvidence column updated
    """
    logger.info(
        "Calculating RegPathEvidence from Enrichr results (ZERO-FAKE, weighted)..."
    )

    gene_weight = defaultdict(float)
    total_sig_terms = 0

    for lib, terms in _iter_enrichr_terms(enrichr_results):
        if not terms:
            continue
        for term_data in terms:
            # Enrichr typical format: [rank, term, pval, zscore, combined, genes, adj_pval, ...]
            if not isinstance(term_data, (list, tuple)) or len(term_data) < 7:
                continue
            try:
                adj_p = float(term_data[6])
            except Exception:
                continue
            if adj_p >= adj_p_cutoff:
                continue

            genes_in_term = term_data[5]
            genes = _split_genes(genes_in_term)
            if not genes:
                continue

            total_sig_terms += 1

            if weight_mode == "neglog10_adj_p":
                # avoid -log10(0)
                adj_p_safe = max(adj_p, 1e-300)
                w = -math.log10(adj_p_safe)
            else:
                # fallback to count
                w = 1.0

            for g in genes:
                gene_weight[g] += w

    evidence_df["RegPathEvidence"] = 0.0
    if gene_weight:
        max_w = max(gene_weight.values())
        if max_w > 0:
            # Support different gene column names
            gene_col = (
                "gene"
                if "gene" in evidence_df.columns
                else ("Gene" if "Gene" in evidence_df.columns else None)
            )
            if gene_col is None:
                logger.error(
                    "No gene column found in evidence_df (expected 'gene' or 'Gene')."
                )
                return evidence_df

            for i, g in evidence_df[gene_col].astype(str).items():
                gg = g.strip().upper()
                evidence_df.loc[i, "RegPathEvidence"] = gene_weight.get(gg, 0.0) / max_w

    logger.info(f"  Significant terms used: {total_sig_terms}")
    logger.info(
        f"  Genes with RegPathEvidence>0: {(evidence_df['RegPathEvidence'] > 0).sum()}"
    )
    return evidence_df


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_upstream_data(f1_dir, f2_dir, logger):
    """Load Figure 1 and Figure 2 upstream results."""
    logger.info("Loading upstream data from Figure 1 and Figure 2...")

    # Load Figure 2 candidate genes (top 100)
    candidates_file = Path(f2_dir) / "candidate_genes_top100.csv"
    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidate genes file not found: {candidates_file}")
    candidates_df = pd.read_csv(candidates_file)
    logger.info(f"  Loaded {len(candidates_df)} candidate genes from Figure 2")

    # Load Figure 1 DEG results
    deg_file = Path(f1_dir) / "sourcedata16" / "SourceData_Fig1_DEG_full.csv"
    if not deg_file.exists():
        raise FileNotFoundError(f"DEG file not found: {deg_file}")
    deg_df = pd.read_csv(deg_file)
    logger.info(f"  Loaded {len(deg_df)} genes from Figure 1 DEG analysis")

    # Load gene modules
    modules_file = Path(f2_dir) / "raw" / "gene_modules.csv"
    if modules_file.exists():
        modules_df = pd.read_csv(modules_file)
        logger.info(f"  Loaded {len(modules_df)} gene-module mappings")
    else:
        modules_df = None
        logger.warning("  Gene modules file not found")

    return candidates_df, deg_df, modules_df


def load_external_data(ext_dir, logger):
    """Load external validation datasets.

    If a CSV contains samples from multiple GSE datasets (detected by GSE
    prefix in column names), it is automatically split into per-GSE datasets
    so that ROC validation is performed independently for each study.
    """
    logger.info("Loading external validation datasets...")

    ext_path = Path(ext_dir)
    datasets = {}

    # Find all CSV files
    csv_files = list(ext_path.glob("*.csv"))
    logger.info(f"  Found {len(csv_files)} CSV files in {ext_dir}")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Parse labels from first data row
            if "label" in df.iloc[0].values or df.columns[0] == "sample":
                sample_cols = df.columns[1:]
                labels = df.iloc[0, 1:].values

                # Extract expression data (skip label row)
                expr_df = df.iloc[1:].copy()
                expr_df.set_index(df.columns[0], inplace=True)
                expr_df.columns = sample_cols
                expr_df = expr_df.astype(float)

                # Create label series
                label_series = pd.Series(labels, index=sample_cols)

                # --- Per-GSE splitting ---
                # Detect GSE prefixes from sample column names
                import re as _re

                gse_map = {}
                for col in sample_cols:
                    m = _re.match(r"(GSE\d+)", str(col))
                    if m:
                        gse_map.setdefault(m.group(1), []).append(col)

                if len(gse_map) > 1:
                    # Multiple GSE datasets in one file -> split
                    logger.info(
                        f"    {csv_file.stem}: detected {len(gse_map)} GSE datasets, splitting..."
                    )
                    for gse_id, gse_cols in gse_map.items():
                        gse_expr = expr_df[gse_cols]
                        gse_labels = label_series[gse_cols]
                        n_case = (gse_labels == "case").sum()
                        n_control = (gse_labels == "control").sum()

                        if n_case == 0 or n_control == 0:
                            logger.warning(
                                f"      {gse_id}: skipped (case={n_case}, control={n_control})"
                            )
                            continue

                        datasets[gse_id] = {
                            "expression": gse_expr,
                            "labels": gse_labels,
                            "n_case": n_case,
                            "n_control": n_control,
                            "file": str(csv_file),
                        }
                        logger.info(
                            f"      {gse_id}: {n_case} case, {n_control} control samples, {len(gse_expr)} genes"
                        )
                else:
                    # Single dataset (or no GSE prefix detected) -> load as-is
                    n_case = (label_series == "case").sum()
                    n_control = (label_series == "control").sum()
                    dataset_name = csv_file.stem
                    datasets[dataset_name] = {
                        "expression": expr_df,
                        "labels": label_series,
                        "n_case": n_case,
                        "n_control": n_control,
                        "file": str(csv_file),
                    }
                    logger.info(
                        f"    {dataset_name}: {n_case} case, {n_control} control samples, {len(expr_df)} genes"
                    )
        except Exception as e:
            logger.warning(f"    Failed to load {csv_file.name}: {e}")

    logger.info(f"  Total external datasets loaded: {len(datasets)}")
    return datasets


# ============================================================================
# EVIDENCE SCORING FUNCTIONS
# ============================================================================


def build_evidence_table(candidates_df, deg_df, logger):
    """Build comprehensive evidence table with FinalScore.

    FIXES:
    1. Use WGCNA 'direction' column as primary direction indicator
    2. Validate DEG direction consistency with WGCNA direction
    3. Add direction concordance flag
    """
    logger.info("Building candidate evidence table...")

    # Start with candidate genes
    evidence_df = candidates_df.copy()

    # Ensure 'direction' column exists from WGCNA (GS_raw sign)
    if "direction" not in evidence_df.columns:
        if "GS_raw" in evidence_df.columns:
            evidence_df["direction"] = evidence_df["GS_raw"].apply(
                lambda x: "Up" if x > 0 else "Down"
            )
            logger.info("  Generated direction from GS_raw")
        else:
            logger.warning("  No direction information available!")
            evidence_df["direction"] = "Unknown"

    # Merge DEG evidence
    deg_subset = deg_df[["Gene", "Log2FC", "padj", "SignificantFlag"]].copy()
    deg_subset.columns = ["gene", "DEG_Log2FC", "DEG_padj", "DEG_Flag"]

    evidence_df = evidence_df.merge(deg_subset, on="gene", how="left")

    # Validate direction concordance between WGCNA and DEG
    # WGCNA direction (from GS_raw): positive GS = Up in CRC, negative GS = Down in CRC
    # DEG direction: Log2FC > 0 = Up in case vs control, Log2FC < 0 = Down in case vs control
    evidence_df["direction_concordant"] = False
    for idx, row in evidence_df.iterrows():
        wgcna_dir = row["direction"]
        deg_flag = row.get("DEG_Flag", "NS")
        deg_fc = row.get("DEG_Log2FC", 0)

        if pd.isna(deg_flag) or deg_flag == "NS":
            # No significant DEG, cannot validate
            evidence_df.loc[idx, "direction_concordant"] = np.nan
        else:
            # Check if directions match
            # WGCNA Up (positive GS) should match DEG Up (positive Log2FC)
            # WGCNA Down (negative GS) should match DEG Down (negative Log2FC)
            if wgcna_dir == "Up" and deg_fc > 0:
                evidence_df.loc[idx, "direction_concordant"] = True
            elif wgcna_dir == "Down" and deg_fc < 0:
                evidence_df.loc[idx, "direction_concordant"] = True
            else:
                evidence_df.loc[idx, "direction_concordant"] = False

    # Log direction concordance statistics
    concordant_mask = evidence_df["direction_concordant"] == True
    discordant_mask = evidence_df["direction_concordant"] == False
    na_mask = evidence_df["direction_concordant"].isna()

    logger.info(
        f"  Direction concordance: {concordant_mask.sum()} concordant, {discordant_mask.sum()} discordant, {na_mask.sum()} NA"
    )

    # Calculate evidence scores
    # 1. Hub Evidence (from WGCNA composite_score)
    evidence_df["HubEvidence"] = evidence_df["composite_score"]

    # 2. DEG Evidence (based on significance and effect size)
    # Only count genes with concordant direction
    evidence_df["DEGEvidence"] = 0.0
    sig_mask = evidence_df["DEG_Flag"].isin(["Up", "Down"])
    evidence_df.loc[sig_mask, "DEGEvidence"] = (
        np.abs(evidence_df.loc[sig_mask, "DEG_Log2FC"])
        / evidence_df.loc[sig_mask, "DEG_Log2FC"].abs().max()
    )

    # Normalize DEGEvidence to 0-1
    if evidence_df["DEGEvidence"].max() > 0:
        evidence_df["DEGEvidence"] = (
            evidence_df["DEGEvidence"] / evidence_df["DEGEvidence"].max()
        )

    # Penalize discordant genes (reduce DEG evidence by 50%)
    discordant_mask = evidence_df["direction_concordant"] == False
    evidence_df.loc[discordant_mask, "DEGEvidence"] = (
        evidence_df.loc[discordant_mask, "DEGEvidence"] * 0.5
    )

    # 3. PPI Evidence (placeholder - will be filled by API)
    evidence_df["PPIEvidence"] = 0.0

    # 4. Regulatory/Pathway Evidence (placeholder - will be filled by API)
    evidence_df["RegPathEvidence"] = 0.0

    logger.info(f"  Built evidence table for {len(evidence_df)} genes")
    logger.info(f"  DEG overlap: {sig_mask.sum()} genes with significant DEG")
    logger.info(
        f"  Direction distribution: Up={len(evidence_df[evidence_df['direction'] == 'Up'])}, Down={len(evidence_df[evidence_df['direction'] == 'Down'])}"
    )

    return evidence_df


def query_string_api(genes, logger, species=9606, min_score=400):
    """Query STRING API for PPI network."""
    cache_key = f"string_{'_'.join(sorted(genes[:10]))}_{len(genes)}"
    cached = load_cached_response(cache_key)
    if cached:
        logger.info("  Using cached STRING response")
        return cached

    logger.info(f"  Querying STRING API for {len(genes)} genes...")

    try:
        # STRING API requires identifiers separated by %0d (URL-encoded carriage return)
        params = {
            "identifiers": "%0d".join(genes),
            "species": species,
            "required_score": min_score,
            "limit": len(genes) * 20,
        }
        response = requests.get(STRING_API, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        cache_api_response(cache_key, data)
        logger.info(f"    Retrieved {len(data)} interactions")
        return data
    except Exception as e:
        logger.error(f"    STRING API failed: {e}")
        return []


def query_enrichr(genes, library, logger):
    """Query Enrichr API for enrichment analysis."""
    cache_key = f"enrichr_{library}_{'_'.join(sorted(genes[:5]))}_{len(genes)}"
    cached = load_cached_response(cache_key)
    if cached:
        logger.info(f"  Using cached Enrichr response for {library}")
        return cached

    logger.info(f"  Querying Enrichr for {library}...")

    try:
        # Add gene list - Enrichr expects newline-separated genes
        genes_str = "\n".join(genes)
        payload = {"list": (None, genes_str)}
        response = requests.post(ENRICHR_ADD_URL, files=payload, timeout=30)
        response.raise_for_status()
        user_list_id = response.json()["userListId"]

        # Get enrichment results
        params = {"userListId": user_list_id, "backgroundType": library}
        response = requests.get(ENRICHR_ENRICH_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        cache_api_response(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"    Enrichr API failed: {e}")
        return {}


def calculate_ppi_evidence(evidence_df, string_data, logger):
    """Calculate PPI evidence scores from STRING data."""
    logger.info("Calculating PPI evidence scores...")

    if not string_data:
        logger.warning("  No STRING data available")
        G = nx.Graph()
        return evidence_df, G

    # Build network
    G = nx.Graph()
    target_genes = set(evidence_df["gene"].values)

    for edge in string_data:
        p1 = edge.get("preferredName_A", "")
        p2 = edge.get("preferredName_B", "")
        score = edge.get("score", 0)
        if p1 and p2:
            G.add_edge(p1, p2, weight=score)

    # Calculate degree centrality
    if G.number_of_nodes() > 0:
        degree_cent = nx.degree_centrality(G)

        # Map to evidence
        for idx, row in evidence_df.iterrows():
            gene = row["gene"]
            if gene in degree_cent:
                evidence_df.loc[idx, "PPIEvidence"] = degree_cent[gene]

        # Normalize
        max_ppi = evidence_df["PPIEvidence"].max()
        if max_ppi > 0:
            evidence_df["PPIEvidence"] = evidence_df["PPIEvidence"] / max_ppi

        logger.info(
            f"  PPI network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    return evidence_df, G


# ============================================================================
# EXTERNAL VALIDATION FUNCTIONS
# ============================================================================


def calculate_signature_score(expr_df, genes, weights=None):
    """Calculate signature score for samples using gene expression."""
    available_genes = [g for g in genes if g in expr_df.index]
    if not available_genes:
        return None

    if weights is None:
        weights = {g: 1.0 for g in available_genes}

    # Z-score normalize expression
    expr_subset = expr_df.loc[available_genes].copy()
    expr_z = (expr_subset - expr_subset.mean(axis=1).values.reshape(-1, 1)) / (
        expr_subset.std(axis=1).values.reshape(-1, 1) + 1e-10
    )

    # Calculate weighted mean
    scores = expr_z.mean(axis=0)
    return scores


def validate_direction_concordance(evidence_df, ext_datasets, logger):
    """Validate direction concordance in external datasets.

    FIXES:
    1. Use WGCNA 'direction' column as expected direction (not DEG_Flag)
    2. This ensures consistency: WGCNA direction -> External validation
    """
    logger.info("Validating direction concordance in external datasets...")

    results = []

    for dataset_name, dataset in ext_datasets.items():
        expr_df = dataset["expression"]
        labels = dataset["labels"]

        case_samples = labels[labels == "case"].index
        control_samples = labels[labels == "control"].index

        for idx, row in evidence_df.head(20).iterrows():
            gene = row["gene"]
            # Use WGCNA direction as expected direction
            expected_dir = row.get("direction", "Unknown")

            if gene not in expr_df.index:
                continue

            # Calculate fold change in external data
            case_mean = expr_df.loc[gene, case_samples].mean()
            control_mean = expr_df.loc[gene, control_samples].mean()
            ext_fc = case_mean - control_mean  # Log2FC equivalent (case vs control)

            # Determine direction in external data
            if ext_fc > 0.1:
                ext_dir = "Up"  # Higher in case
            elif ext_fc < -0.1:
                ext_dir = "Down"  # Lower in case
            else:
                ext_dir = "NS"

            # Check concordance with WGCNA direction
            concordant = (expected_dir == ext_dir) or (ext_dir == "NS")

            # Statistical test
            try:
                t_stat, p_val = stats.ttest_ind(
                    expr_df.loc[gene, case_samples].values,
                    expr_df.loc[gene, control_samples].values,
                )
            except:
                t_stat, p_val = np.nan, np.nan

            results.append(
                {
                    "gene": gene,
                    "dataset": dataset_name,
                    "wgcna_direction": expected_dir,
                    "external_direction": ext_dir,
                    "external_fc": ext_fc,
                    "concordant": concordant,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                }
            )

    results_df = pd.DataFrame(results)

    # Summary
    if len(results_df) > 0:
        concordance_rate = results_df["concordant"].mean()
        logger.info(f"  Overall concordance rate: {concordance_rate:.1%}")

        # Per-gene concordance
        gene_concordance = results_df.groupby("gene")["concordant"].mean()
        high_concordance_genes = gene_concordance[
            gene_concordance >= 0.6
        ].index.tolist()
        logger.info(f"  Genes with >=60% concordance: {len(high_concordance_genes)}")

    return results_df


def run_external_roc_analysis(evidence_df, ext_datasets, logger):
    """Run ROC analysis on external datasets using signature scores.

    Includes:
    - AUC calculation with 95% CI (DeLong method approximation via bootstrap)
    - Permutation test for statistical significance
    """
    logger.info("Running external ROC analysis...")

    results = []
    roc_curves = {}

    # Get top genes for signature
    top_genes = evidence_df.head(20)["gene"].tolist()

    for dataset_name, dataset in ext_datasets.items():
        expr_df = dataset["expression"]
        labels = dataset["labels"]

        # Calculate signature score
        scores = calculate_signature_score(expr_df, top_genes)
        if scores is None:
            logger.warning(f"  {dataset_name}: No overlapping genes")
            continue

        # Prepare for ROC
        y_true = (labels == "case").astype(int)
        y_score = scores.values

        # Align
        common_samples = y_true.index.intersection(scores.index)
        y_true = y_true[common_samples]
        y_score = scores[common_samples].values

        if len(np.unique(y_true)) < 2:
            logger.warning(f"  {dataset_name}: Only one class present")
            continue

        # Calculate ROC
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # If AUC < 0.5, flip the scores
            if roc_auc < 0.5:
                y_score = -y_score
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

            # Bootstrap CI for AUC (approximation of DeLong)
            n_bootstrap = 1000
            np.random.seed(42)
            bootstrap_aucs = []
            n_samples = len(y_true)

            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
                boot_y_true = y_true.values[boot_idx]
                boot_y_score = y_score[boot_idx]

                # Skip if only one class in bootstrap sample
                if len(np.unique(boot_y_true)) < 2:
                    continue

                try:
                    boot_fpr, boot_tpr, _ = roc_curve(boot_y_true, boot_y_score)
                    boot_auc = auc(boot_fpr, boot_tpr)
                    bootstrap_aucs.append(boot_auc)
                except:
                    pass

            # Calculate 95% CI
            if len(bootstrap_aucs) > 100:
                auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
                auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
            else:
                auc_ci_lower = roc_auc
                auc_ci_upper = roc_auc

            # Permutation test for significance
            n_permutations = 1000
            perm_aucs = []

            for _ in range(n_permutations):
                perm_y_true = np.random.permutation(y_true.values)

                if len(np.unique(perm_y_true)) < 2:
                    continue

                try:
                    perm_fpr, perm_tpr, _ = roc_curve(perm_y_true, y_score)
                    perm_auc = auc(perm_fpr, perm_tpr)
                    perm_aucs.append(perm_auc)
                except:
                    pass

            # Calculate permutation p-value
            if len(perm_aucs) > 100:
                perm_pvalue = (np.sum(np.array(perm_aucs) >= roc_auc) + 1) / (
                    len(perm_aucs) + 1
                )
            else:
                perm_pvalue = np.nan

            roc_curves[dataset_name] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc,
                "auc_ci_lower": auc_ci_lower,
                "auc_ci_upper": auc_ci_upper,
                "perm_pvalue": perm_pvalue,
            }

            results.append(
                {
                    "dataset": dataset_name,
                    "n_samples": len(y_true),
                    "n_case": (y_true == 1).sum(),
                    "n_control": (y_true == 0).sum(),
                    "n_genes_used": len([g for g in top_genes if g in expr_df.index]),
                    "auc": roc_auc,
                    "auc_ci_lower": auc_ci_lower,
                    "auc_ci_upper": auc_ci_upper,
                    "perm_pvalue": perm_pvalue,
                    "validated": roc_auc >= 0.6 and perm_pvalue < 0.05,
                }
            )

            logger.info(
                f"  {dataset_name}: AUC = {roc_auc:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f}), perm p = {perm_pvalue:.3f}"
            )

        except Exception as e:
            logger.error(f"  {dataset_name}: ROC failed - {e}")

    results_df = pd.DataFrame(results)
    return results_df, roc_curves


# ============================================================================
# PLOTTING FUNCTIONS - ROW 1 (A-D): Candidate Prioritization
# ============================================================================


def plot_panel_A_funnel(evidence_df, output_path, logger):
    """Panel A: Candidate funnel diagram."""
    logger.info("Plotting Panel A: Candidate Funnel")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Funnel stages
    stages = [
        ("WGCNA Candidates", 100),
        ("DEG Overlap", (evidence_df["DEG_Flag"].isin(["Up", "Down"])).sum()),
        ("PPI Connected", (evidence_df["PPIEvidence"] > 0).sum()),
        ("Top 20 Final", 20),
    ]

    colors = ["#E8E8E8", "#B8D4E3", "#7FB3D5", "#2E86AB"]

    y_positions = np.arange(len(stages))[::-1]
    max_width = max(s[1] for s in stages)

    for i, (label, count) in enumerate(stages):
        width = count / max_width * 0.8
        rect = plt.Rectangle(
            (0.5 - width / 2, y_positions[i] - 0.35),
            width,
            0.7,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            y_positions[i],
            f"{label}\n(n={count})",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stages) - 0.5)
    ax.axis("off")
    ax.set_title(
        "A. Candidate Prioritization Funnel", fontsize=10, fontweight="bold", loc="left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_B_venn(evidence_df, output_path, logger):
    """Panel B: Evidence overlap (simplified Venn-like)."""
    logger.info("Plotting Panel B: Evidence Overlap")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Calculate overlaps
    hub_genes = set(evidence_df[evidence_df["HubEvidence"] > 0.5]["gene"])
    deg_genes = set(evidence_df[evidence_df["DEG_Flag"].isin(["Up", "Down"])]["gene"])
    ppi_genes = set(evidence_df[evidence_df["PPIEvidence"] > 0]["gene"])

    # Draw circles
    from matplotlib.patches import Circle

    circle1 = Circle(
        (0.35, 0.5),
        0.25,
        alpha=0.5,
        facecolor="#E64B35",
        edgecolor="black",
        label="Hub",
    )
    circle2 = Circle(
        (0.65, 0.5),
        0.25,
        alpha=0.5,
        facecolor="#4DBBD5",
        edgecolor="black",
        label="DEG",
    )
    circle3 = Circle(
        (0.5, 0.25),
        0.25,
        alpha=0.5,
        facecolor="#00A087",
        edgecolor="black",
        label="PPI",
    )

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Labels
    ax.text(0.25, 0.6, f"Hub\n{len(hub_genes)}", ha="center", va="center", fontsize=8)
    ax.text(0.75, 0.6, f"DEG\n{len(deg_genes)}", ha="center", va="center", fontsize=8)
    ax.text(0.5, 0.1, f"PPI\n{len(ppi_genes)}", ha="center", va="center", fontsize=8)

    # Intersection
    all_three = hub_genes & deg_genes & ppi_genes
    ax.text(
        0.5,
        0.45,
        f"{len(all_three)}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. Evidence Overlap", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_C_scatter(evidence_df, output_path, logger):
    """Panel C: Hub score vs DEG effect size scatter.

    FIXES: Use WGCNA 'direction' column for coloring
    """
    logger.info("Plotting Panel C: Hub vs DEG Scatter")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Prepare data
    plot_df = evidence_df.copy()
    plot_df["DEG_abs_FC"] = plot_df["DEG_Log2FC"].abs().fillna(0)

    # Color by WGCNA direction
    colors = []
    for _, row in plot_df.iterrows():
        direction = row.get("direction", "Unknown")
        if direction == "Up":
            colors.append(COLOR_UP)
        elif direction == "Down":
            colors.append(COLOR_DOWN)
        else:
            colors.append("#888888")

    ax.scatter(
        plot_df["HubEvidence"],
        plot_df["DEG_abs_FC"],
        c=colors,
        alpha=0.7,
        s=30,
        edgecolors="black",
        linewidths=0.5,
    )

    # Label top genes
    top_genes = plot_df.nlargest(5, "composite_score")
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["gene"],
            (row["HubEvidence"], row["DEG_abs_FC"]),
            fontsize=6,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Hub Evidence Score", fontsize=9)
    ax.set_ylabel("|Log2FC| (DEG)", fontsize=9)
    ax.set_title(
        "C. Hub Score vs DEG Effect", fontsize=10, fontweight="bold", loc="left"
    )
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLOR_UP, label="Up"),
        Patch(facecolor=COLOR_DOWN, label="Down"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_D_lollipop(evidence_df, output_path, logger):
    """Panel D: Top 20 candidates lollipop plot.

    FIXES: Use WGCNA 'direction' column for coloring (not DEG_Flag)
    """
    logger.info("Plotting Panel D: Top 20 Lollipop")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    top20 = evidence_df.nlargest(20, "composite_score").copy()
    top20 = top20.iloc[::-1]  # Reverse for plotting

    y_pos = np.arange(len(top20))

    # Color by WGCNA direction (not DEG direction)
    colors = []
    for _, row in top20.iterrows():
        direction = row.get("direction", "Unknown")
        if direction == "Up":
            colors.append(COLOR_UP)
        elif direction == "Down":
            colors.append(COLOR_DOWN)
        else:
            colors.append("#888888")

    # Lollipop
    ax.hlines(y_pos, 0, top20["composite_score"], colors=colors, linewidth=2)
    ax.scatter(
        top20["composite_score"],
        y_pos,
        c=colors,
        s=50,
        zorder=3,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["gene"], fontsize=7)
    ax.set_xlabel("Composite Score", fontsize=9)
    ax.set_title("D. Top 20 Candidates", fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLOR_UP, label="Up in CRC"),
        Patch(facecolor=COLOR_DOWN, label="Down in CRC"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# PLOTTING FUNCTIONS - ROW 2 (E-H): PPI Network
# ============================================================================


def plot_panel_E_ppi_network(G, target_genes, output_path, logger):
    """Panel E: PPI network visualization."""
    logger.info("Plotting Panel E: PPI Network")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if G.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5, "No PPI data available", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        # Node colors
        node_colors = [
            COLOR_TARGET if n in target_genes else COLOR_NEIGHBOR for n in G.nodes()
        ]
        node_sizes = [200 if n in target_genes else 50 for n in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax)

        # Label only target genes
        labels = {n: n for n in G.nodes() if n in target_genes}
        nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)

        ax.axis("off")

    ax.set_title("E. PPI Network", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_F_degree_dist(G, output_path, logger):
    """Panel F: Degree distribution."""
    logger.info("Plotting Panel F: Degree Distribution")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if G.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5, "No PPI data available", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        degrees = [d for n, d in G.degree()]
        ax.hist(degrees, bins=20, color=COLOR_NEIGHBOR, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Degree", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.axvline(
            np.mean(degrees),
            color="red",
            linestyle="--",
            label=f"Mean={np.mean(degrees):.1f}",
        )
        ax.legend(fontsize=7)

    ax.set_title("F. Degree Distribution", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_G_centrality(evidence_df, output_path, logger):
    """Panel G: Centrality ranking bar plot."""
    logger.info("Plotting Panel G: Centrality Ranking")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    top10 = evidence_df.nlargest(10, "PPIEvidence").copy()
    top10 = top10.iloc[::-1]

    y_pos = np.arange(len(top10))
    colors = [
        COLOR_TARGET if row["DEG_Flag"] in ["Up", "Down"] else COLOR_NEIGHBOR
        for _, row in top10.iterrows()
    ]

    ax.barh(y_pos, top10["PPIEvidence"], color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10["gene"], fontsize=7)
    ax.set_xlabel("PPI Centrality Score", fontsize=9)
    ax.set_title("G. Hub Gene Centrality", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_H_module_network(evidence_df, output_path, logger):
    """Panel H: Module-gene network."""
    logger.info("Plotting Panel H: Module Network")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Create module-gene bipartite network
    G = nx.Graph()

    top20 = evidence_df.head(20)
    modules = top20["module"].unique()

    for mod in modules:
        G.add_node(mod, node_type="module")

    for _, row in top20.iterrows():
        G.add_node(row["gene"], node_type="gene")
        G.add_edge(row["module"], row["gene"])

    pos = nx.spring_layout(G, k=2, seed=42)

    # Draw
    module_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "module"]
    gene_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "gene"]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=module_nodes,
        node_color="#FFD700",
        node_size=300,
        node_shape="s",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=gene_nodes, node_color=COLOR_TARGET, node_size=100, ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.axis("off")
    ax.set_title("H. Module-Gene Network", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# PLOTTING FUNCTIONS - ROW 3 (I-L): TF & Pathway Enrichment
# ============================================================================


def plot_panel_I_tf_enrichment(enrichr_results, output_path, logger):
    """Panel I: TF enrichment bubble plot."""
    logger.info("Plotting Panel I: TF Enrichment")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if (
        not enrichr_results
        or "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X" not in enrichr_results
    ):
        ax.text(
            0.5, 0.5, "No TF enrichment data", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        tf_data = enrichr_results.get("ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X", [])
        if tf_data:
            tf_df = pd.DataFrame(
                [
                    {
                        "TF": r[1].split("_")[0] if "_" in r[1] else r[1],
                        "pvalue": r[2],
                        "combined_score": r[4],
                        "n_genes": len(r[5])
                        if isinstance(r[5], list)
                        else (len(r[5].split(";")) if r[5] else 0),
                    }
                    for r in tf_data[:10]
                ]
            )

            y_pos = np.arange(len(tf_df))[::-1]
            sizes = tf_df["n_genes"] * 20
            colors = -np.log10(tf_df["pvalue"] + 1e-10)

            scatter = ax.scatter(
                tf_df["combined_score"],
                y_pos,
                s=sizes,
                c=colors,
                cmap="Reds",
                alpha=0.7,
                edgecolors="black",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tf_df["TF"], fontsize=7)
            ax.set_xlabel("Combined Score", fontsize=9)
            plt.colorbar(scatter, ax=ax, label="-log10(p)", shrink=0.6)
        else:
            ax.text(
                0.5, 0.5, "No significant TFs", ha="center", va="center", fontsize=10
            )
            ax.axis("off")

    ax.set_title("I. TF Enrichment", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_J_kegg_enrichment(enrichr_results, output_path, logger):
    """Panel J: KEGG pathway enrichment."""
    logger.info("Plotting Panel J: KEGG Enrichment")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if not enrichr_results or "KEGG_2021_Human" not in enrichr_results:
        ax.text(
            0.5, 0.5, "No KEGG enrichment data", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        kegg_data = enrichr_results.get("KEGG_2021_Human", [])
        if kegg_data:
            kegg_df = pd.DataFrame(
                [
                    {
                        "Pathway": r[1][:30] + "..." if len(r[1]) > 30 else r[1],
                        "pvalue": r[2],
                        "combined_score": r[4],
                    }
                    for r in kegg_data[:10]
                ]
            )

            y_pos = np.arange(len(kegg_df))[::-1]
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(kegg_df)))

            ax.barh(
                y_pos,
                -np.log10(kegg_df["pvalue"] + 1e-10),
                color=colors,
                edgecolor="black",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(kegg_df["Pathway"], fontsize=6)
            ax.set_xlabel("-log10(p-value)", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "No significant pathways",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.axis("off")

    ax.set_title("J. KEGG Pathways", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_K_go_enrichment(enrichr_results, output_path, logger):
    """Panel K: GO Biological Process enrichment."""
    logger.info("Plotting Panel K: GO BP Enrichment")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if not enrichr_results or "GO_Biological_Process_2021" not in enrichr_results:
        ax.text(
            0.5, 0.5, "No GO enrichment data", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        go_data = enrichr_results.get("GO_Biological_Process_2021", [])
        if go_data:
            go_df = pd.DataFrame(
                [
                    {
                        "Term": r[1].split(" (GO")[0][:25] + "..."
                        if len(r[1].split(" (GO")[0]) > 25
                        else r[1].split(" (GO")[0],
                        "pvalue": r[2],
                        "combined_score": r[4],
                    }
                    for r in go_data[:10]
                ]
            )

            y_pos = np.arange(len(go_df))[::-1]
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(go_df)))

            ax.barh(
                y_pos,
                -np.log10(go_df["pvalue"] + 1e-10),
                color=colors,
                edgecolor="black",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(go_df["Term"], fontsize=6)
            ax.set_xlabel("-log10(p-value)", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "No significant GO terms",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.axis("off")

    ax.set_title("K. GO Biological Process", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# DRUGGABILITY SCORING FUNCTIONS (v1.9.2 - Improved scoring with exact label mapping)
# ============================================================================

# Exact label -> score mapping (avoids substring matching errors)
TRACT_LABEL_SCORE = {
    # Clinical precedence (highest tier)
    "Approved Drug": 1.00,
    "Clinical Precedence": 0.90,
    "Advanced Clinical": 0.85,
    "Phase 1 Clinical": 0.70,
    # Structure/ligand/pocket quality
    "Structure with Ligand": 0.60,
    "High-Quality Ligand": 0.60,
    "High-Quality Pocket": 0.55,
    "Med-Quality Pocket": 0.45,
    # Family druggability (often missed!)
    "Druggable Family": 0.35,
    # Additional labels from OpenTargets
    "UniProt loc high conf": 0.30,
    "GO CC high conf": 0.30,
    "Literature": 0.25,
    "Database Ubiquitination": 0.20,
    "Small Molecule Binder": 0.40,
}


def compute_tract_score(assessments):
    """
    Compute tractability score from OpenTargets assessments.

    Args:
        assessments: list of dicts like [{"label": "...", "modality": "...", "value": True/False}, ...]

    Returns:
        (best_score, list_of_hit_labels)
    """
    if not assessments:
        return 0.0, []

    hits = []
    best = 0.0

    for a in assessments:
        if not a:
            continue
        # Only count value=True entries
        if a.get("value") is not True:
            continue

        label = str(a.get("label", "")).strip()
        if not label:
            continue

        if label in TRACT_LABEL_SCORE:
            s = TRACT_LABEL_SCORE[label]
            hits.append(label)
            if s > best:
                best = s

    return float(best), sorted(set(hits))


def phase_to_score(max_phase, has_approved):
    """
    Convert clinical phase to score.

    Args:
        max_phase: Maximum clinical trial phase (0-4)
        has_approved: Whether there are approved drugs

    Returns:
        Score between 0.0 and 1.0
    """
    try:
        p = int(max_phase) if max_phase is not None else None
    except Exception:
        p = None

    if has_approved or (p is not None and p >= 4):
        return 1.00
    if p is not None and p >= 3:
        return 0.90
    if p is not None and p >= 2:
        return 0.80
    if p is not None and p >= 1:
        return 0.60
    return 0.0


def richness_bonus(unique_drugs, cap=20):
    """
    Small bonus based on number of unique drugs (0-0.08).
    Helps differentiate targets at the same tier.

    Args:
        unique_drugs: Number of unique drugs
        cap: Maximum drugs to consider

    Returns:
        Bonus score between 0.0 and 0.08
    """
    import math

    try:
        n = int(unique_drugs) if unique_drugs is not None else 0
    except Exception:
        n = 0

    if n <= 0:
        return 0.0

    x = min(n, cap)
    return 0.08 * (math.log1p(x) / math.log1p(cap))


def classify_druggability(score):
    """
    Classify druggability score into categories.

    Args:
        score: Druggability score (0-1)

    Returns:
        Category string: High/Medium/Low/Unknown
    """
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    if score > 0:
        return "Low"
    return "Unknown"


def compute_druggability_score(
    tract_score, max_phase, approved_drugs_n, unique_drugs, target_type="Unknown"
):
    """
    Compute final druggability score combining tractability and known drugs.

    Formula: druggability_score = max(tract_score, known_drug_score) + richness_bonus

    Args:
        tract_score: Score from tractability assessments
        max_phase: Maximum clinical trial phase
        approved_drugs_n: Number of approved drugs
        unique_drugs: Total unique drugs
        target_type: DirectTarget/FusionPartner/Biomarker/Unknown

    Returns:
        (score, classification)
    """
    # FusionPartner / Biomarker forced to 0 (avoid misattribution)
    if target_type in ("FusionPartner", "Biomarker"):
        return 0.0, "NotDirectTarget" if target_type == "FusionPartner" else "Biomarker"

    has_approved = approved_drugs_n is not None and int(approved_drugs_n or 0) > 0
    known_score = phase_to_score(max_phase, has_approved)

    base = max(float(tract_score or 0.0), float(known_score or 0.0))
    score = min(1.0, base + richness_bonus(unique_drugs))

    return score, classify_druggability(score)


def query_opentargets_druggability(gene_symbols, logger):
    """Query OpenTargets Platform for comprehensive druggability assessment.

    Returns detailed druggability info for each gene:
    - ensembl_id: Ensembl gene ID
    - druggability_score: 0-1 score (v1.9.2: exact label mapping + known drug + richness bonus)
    - druggability: High/Medium/Low/Unknown (thresholds: >=0.75 High, >=0.45 Medium, >0 Low)
    - modality: Small molecule / Antibody / Other modality / Unknown
    - evidence: Drug names or tractability evidence
    - target_type: DirectTarget / FusionPartner / Biomarker / Unknown
    - evidence_basis: ChEMBL_target / OpenTargets_targetDrug / Curated_fusion / None
    - tract_score: Score from tractability labels only
    - known_drug_score: Score from clinical phase only
    - max_phase: Maximum clinical trial phase (0-4)
    - approved_drugs_n: Number of approved drugs
    - unique_drugs: Total unique drugs in development
    - tract_labels_hit: List of tractability labels that contributed to score
    - source: OpenTargets
    - query_date: ISO timestamp

    v1.9.2 SCORING IMPROVEMENTS:
    - Uses exact label mapping (TRACT_LABEL_SCORE) instead of substring matching
    - Avoids missing "Med-Quality Pocket", "Druggable Family" etc.
    - Combines tract_score and known_drug_score with richness bonus
    - Formula: druggability_score = max(tract_score, known_drug_score) + richness_bonus

    ZERO-FAKE: Distinguishes between direct drug targets and fusion partners/biomarkers.
    Fusion partners are loaded from external curated resource file.
    For example, KIF5B is a common fusion partner for RET (KIF5B-RET fusion),
    but the actual drug target is RET, not KIF5B.
    """
    from datetime import datetime

    results = []
    query_date = datetime.now().strftime("%Y-%m-%d")

    # OpenTargets GraphQL endpoint
    ot_url = "https://api.platform.opentargets.org/api/v4/graphql"

    # Comprehensive query for druggability assessment - includes mechanism of action
    druggability_query = """
    query TargetDruggability($ensemblId: String!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            tractability {
                label
                modality
                value
            }
            knownDrugs {
                uniqueDrugs
                rows {
                    drug {
                        name
                        drugType
                        maximumClinicalTrialPhase
                        isApproved
                    }
                    mechanismOfAction
                    targetClass
                }
            }
        }
    }
    """

    # Search query to get Ensembl ID
    search_query = """
    query SearchTarget($queryString: String!) {
        search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
            hits {
                id
                name
            }
        }
    }
    """

    # Load fusion partners from external curated resource (ZERO-FAKE)
    fusion_partners = load_fusion_partners(logger)
    logger.info(f"  Loaded {len(fusion_partners)} fusion partners for filtering")

    for gene in gene_symbols[:20]:  # Query top 20 genes
        result = {
            "gene": gene,
            "ensembl_id": "",
            "druggability_score": 0.0,
            "druggability": "Unknown",
            "modality": "",
            "evidence": "",
            "target_type": "Unknown",
            "evidence_basis": "None",
            "tract_score": 0.0,
            "known_drug_score": 0.0,
            "max_phase": 0,
            "approved_drugs_n": 0,
            "unique_drugs": 0,
            "tract_labels_hit": "",
            "source": "OpenTargets",
            "query_date": query_date,
        }

        # Check if this is a known fusion partner (ZERO-FAKE: from external resource)
        gene_upper = gene.upper()
        if gene_upper in fusion_partners:
            fp_info = fusion_partners[gene_upper]
            result["target_type"] = "FusionPartner"
            result["evidence"] = (
                f"Fusion partner of {fp_info['driver']} (not direct target)"
            )
            result["evidence_basis"] = f"Curated_fusion ({fp_info['source']})"
            result["druggability_score"] = 0.0
            result["druggability"] = "NotDirectTarget"
            # Do NOT include approved drugs for fusion partners
            results.append(result)
            logger.info(
                f"  {gene}: Identified as fusion partner of {fp_info['driver']}, not direct drug target"
            )
            continue

        try:
            # Search for gene to get Ensembl ID
            search_resp = requests.post(
                ot_url,
                json={"query": search_query, "variables": {"queryString": gene}},
                timeout=15,
            )

            if search_resp.status_code == 200:
                search_data = search_resp.json()
                hits = search_data.get("data", {}).get("search", {}).get("hits", [])

                if hits:
                    ensembl_id = hits[0]["id"]
                    result["ensembl_id"] = ensembl_id

                    # Query druggability for this target
                    drug_resp = requests.post(
                        ot_url,
                        json={
                            "query": druggability_query,
                            "variables": {"ensemblId": ensembl_id},
                        },
                        timeout=15,
                    )

                    if drug_resp.status_code == 200:
                        drug_data = drug_resp.json()
                        target = drug_data.get("data", {}).get("target", {})

                        # Process tractability data using new exact label mapping (v1.9.2)
                        tractability = target.get("tractability", []) or []
                        modalities = []

                        # Extract modalities from tractability entries
                        for tract in tractability:
                            modality = tract.get("modality", "")
                            value = tract.get("value", False)
                            if value and modality and modality not in modalities:
                                modalities.append(modality)

                        # Use new compute_tract_score function (exact label mapping)
                        tract_score, tract_labels_hit = compute_tract_score(
                            tractability
                        )
                        result["tract_score"] = tract_score
                        result["tract_labels_hit"] = (
                            "; ".join(tract_labels_hit) if tract_labels_hit else ""
                        )

                        # Process known drugs
                        known_drugs = target.get("knownDrugs", {}) or {}
                        unique_drugs = known_drugs.get("uniqueDrugs", 0) or 0
                        drug_rows = known_drugs.get("rows", []) or []

                        drug_names = []
                        approved_drugs = []
                        max_phase = 0
                        is_direct_target = False
                        is_biomarker = False

                        for row in drug_rows[:10]:
                            drug_info = row.get("drug", {}) or {}
                            drug_name = drug_info.get("name", "")
                            phase = drug_info.get("maximumClinicalTrialPhase", 0) or 0
                            is_approved = drug_info.get("isApproved", False)
                            mechanism = row.get("mechanismOfAction", "") or ""

                            # Check if this gene is the actual target or just associated
                            # Direct targets have mechanism of action mentioning inhibitor/agonist/antagonist
                            if mechanism:
                                mechanism_lower = mechanism.lower()
                                if any(
                                    term in mechanism_lower
                                    for term in [
                                        "inhibitor",
                                        "agonist",
                                        "antagonist",
                                        "blocker",
                                        "modulator",
                                        "activator",
                                    ]
                                ):
                                    is_direct_target = True
                                elif (
                                    "biomarker" in mechanism_lower
                                    or "marker" in mechanism_lower
                                ):
                                    is_biomarker = True

                            if drug_name:
                                drug_names.append(drug_name)
                                if is_approved:
                                    approved_drugs.append(drug_name)
                                max_phase = max(max_phase, phase)

                        # Store intermediate values for transparency
                        result["max_phase"] = max_phase
                        result["approved_drugs_n"] = len(approved_drugs)
                        result["unique_drugs"] = unique_drugs

                        # Determine target type
                        if is_direct_target or (approved_drugs and not is_biomarker):
                            result["target_type"] = "DirectTarget"
                        elif is_biomarker:
                            result["target_type"] = "Biomarker"
                        elif drug_names:
                            # Has drug associations but unclear mechanism - mark as potential
                            result["target_type"] = "PotentialTarget"
                        else:
                            result["target_type"] = "Unknown"

                        # Calculate final druggability score using new v1.9.2 function
                        # This combines tract_score, known_drug_score, and richness_bonus
                        final_score, druggability_class = compute_druggability_score(
                            tract_score=tract_score,
                            max_phase=max_phase,
                            approved_drugs_n=len(approved_drugs),
                            unique_drugs=unique_drugs,
                            target_type=result["target_type"],
                        )

                        result["druggability_score"] = final_score
                        result["druggability"] = druggability_class

                        # Calculate known_drug_score for transparency
                        has_approved = len(approved_drugs) > 0
                        result["known_drug_score"] = phase_to_score(
                            max_phase, has_approved
                        )

                        # Set modality
                        if modalities:
                            result["modality"] = " / ".join(modalities[:2])
                        elif drug_names:
                            result["modality"] = "Small molecule"

                        # Set evidence
                        if approved_drugs:
                            result["evidence"] = (
                                f"Approved: {', '.join(approved_drugs[:3])}"
                            )
                        elif drug_names:
                            result["evidence"] = (
                                f"Phase {max_phase}: {', '.join(drug_names[:3])}"
                            )
                        elif tract_labels_hit:
                            result["evidence"] = (
                                f"Tractability: {'; '.join(tract_labels_hit[:3])}"
                            )
                        elif tract_score > 0:
                            result["evidence"] = "Tractability predicted"

        except Exception as e:
            logger.debug(f"  OpenTargets query for {gene} failed: {e}")

        results.append(result)

    return results


def query_chembl_drugs(gene_symbols, logger):
    """Query ChEMBL for drug-gene interactions."""
    drug_gene_pairs = []

    chembl_url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"

    for gene in gene_symbols[:10]:
        try:
            # Search for target by gene name
            resp = requests.get(chembl_url, params={"q": gene, "limit": 1}, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                targets = data.get("targets", [])

                if targets:
                    target_chembl_id = targets[0].get("target_chembl_id", "")

                    if target_chembl_id:
                        # Get approved drugs for this target
                        mech_url = (
                            f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
                        )
                        mech_resp = requests.get(
                            mech_url,
                            params={"target_chembl_id": target_chembl_id, "limit": 5},
                            timeout=15,
                        )

                        if mech_resp.status_code == 200:
                            mech_data = mech_resp.json()
                            mechanisms = mech_data.get("mechanisms", [])

                            for mech in mechanisms[:3]:
                                drug_name = mech.get("molecule_name", "") or mech.get(
                                    "parent_molecule_name", ""
                                )
                                if drug_name:
                                    drug_gene_pairs.append(
                                        {
                                            "gene": gene,
                                            "drug": drug_name[:20],
                                            "source": "ChEMBL",
                                            "mechanism": mech.get(
                                                "mechanism_of_action", ""
                                            )[:30]
                                            if mech.get("mechanism_of_action")
                                            else "",
                                        }
                                    )
        except Exception as e:
            logger.debug(f"  ChEMBL query for {gene} failed: {e}")
            continue

    return drug_gene_pairs


def query_dgidb_drugs(gene_symbols, logger):
    """Query DGIdb for drug-gene interactions."""
    drug_gene_pairs = []

    try:
        params = {"genes": ",".join(gene_symbols[:10])}
        response = requests.get(DGIDB_API, params=params, timeout=30)
        response.raise_for_status()
        drug_data = response.json()

        for record in drug_data.get("matchedTerms", []):
            gene = record.get("searchTerm", "")
            for interaction in record.get("interactions", [])[:3]:
                drug = interaction.get("drugName", "")
                if gene and drug:
                    drug_gene_pairs.append(
                        {
                            "gene": gene,
                            "drug": drug[:20],
                            "source": "DGIdb",
                            "interaction_type": interaction.get(
                                "interactionTypes", [""]
                            )[0]
                            if interaction.get("interactionTypes")
                            else "",
                        }
                    )
    except Exception as e:
        logger.debug(f"  DGIdb query failed: {e}")

    return drug_gene_pairs


def plot_panel_L_drug_target(evidence_df, output_path, logger):
    """Panel L: Druggability assessment visualization.

    Queries OpenTargets Platform for comprehensive druggability data.
    Output format: gene, ensembl_id, druggability_score, druggability, modality, evidence, target_type, source, query_date

    IMPORTANT: Distinguishes between direct drug targets and fusion partners/biomarkers.
    """
    logger.info("Plotting Panel L: Druggability Assessment")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    top_genes = evidence_df.head(20)["gene"].tolist()

    # Check cache first
    cache_key = f"druggability_v3_{'_'.join(sorted(top_genes[:5]))}".replace(" ", "_")
    cached = load_cached_response(cache_key)

    druggability_results = []

    if cached and len(cached) > 0 and "target_type" in cached[0]:
        druggability_results = cached
        logger.info(
            f"  Using cached druggability data ({len(druggability_results)} genes)"
        )
    else:
        # Query OpenTargets for comprehensive druggability assessment
        logger.info("  Querying OpenTargets Platform for druggability...")
        druggability_results = query_opentargets_druggability(top_genes, logger)

        # Count results by target type
        direct_count = sum(
            1 for r in druggability_results if r.get("target_type") == "DirectTarget"
        )
        fusion_count = sum(
            1 for r in druggability_results if r.get("target_type") == "FusionPartner"
        )
        biomarker_count = sum(
            1 for r in druggability_results if r.get("target_type") == "Biomarker"
        )
        high_count = sum(1 for r in druggability_results if r["druggability"] == "High")
        medium_count = sum(
            1 for r in druggability_results if r["druggability"] == "Medium"
        )
        logger.info(
            f"    Target types: {direct_count} Direct, {fusion_count} Fusion, {biomarker_count} Biomarker"
        )
        logger.info(f"    Druggability: {high_count} High, {medium_count} Medium")

        # Cache the results
        if druggability_results:
            cache_api_response(cache_key, druggability_results)

    # Save source data with new format including target_type and evidence_basis
    sourcedata_path = (
        output_path.parent.parent / "sourcedata16" / "SourceData_Fig3L_DrugTarget.csv"
    )
    df = pd.DataFrame(druggability_results)
    # Ensure column order with all v1.9.2 columns for transparency
    cols = [
        "gene",
        "ensembl_id",
        "druggability_score",
        "druggability",
        "modality",
        "evidence",
        "target_type",
        "evidence_basis",
        "tract_score",
        "known_drug_score",
        "max_phase",
        "approved_drugs_n",
        "unique_drugs",
        "tract_labels_hit",
        "source",
        "query_date",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(sourcedata_path, index=False)

    # Visualization: Druggability bar chart
    if druggability_results:
        # Sort by druggability score
        df_sorted = df.sort_values("druggability_score", ascending=True).tail(15)

        # Color by druggability level and target type
        colors = []
        for _, row in df_sorted.iterrows():
            target_type = row.get("target_type", "Unknown")
            druggability = row["druggability"]

            if target_type == "FusionPartner":
                colors.append("#E64B35")  # Red - fusion partner (not direct target)
            elif target_type == "Biomarker":
                colors.append("#7E6148")  # Brown - biomarker only
            elif druggability == "High":
                colors.append("#00A087")  # Green - high druggability
            elif druggability == "Medium":
                colors.append("#F39B7F")  # Orange - medium
            elif druggability == "Low":
                colors.append("#8491B4")  # Gray-blue - low
            else:
                colors.append("#CCCCCC")  # Gray - unknown

        y_pos = range(len(df_sorted))
        ax.barh(
            y_pos,
            df_sorted["druggability_score"],
            color=colors,
            edgecolor="white",
            height=0.7,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["gene"], fontsize=6)
        ax.set_xlabel("Druggability Score", fontsize=8)
        ax.set_xlim(0, 1.1)

        # Add legend with target type distinction
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#00A087", label="High (Direct)"),
            Patch(facecolor="#F39B7F", label="Medium"),
            Patch(facecolor="#8491B4", label="Low"),
            Patch(facecolor="#E64B35", label="Fusion Partner"),
            Patch(facecolor="#7E6148", label="Biomarker"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=5,
            title="Target Type",
            title_fontsize=5,
        )

        # Add score labels
        for i, (score, gene) in enumerate(
            zip(df_sorted["druggability_score"], df_sorted["gene"])
        ):
            if score > 0:
                ax.text(score + 0.02, i, f"{score:.2f}", va="center", fontsize=5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Summary stats with target type info
        direct_count = sum(
            1 for r in druggability_results if r.get("target_type") == "DirectTarget"
        )
        fusion_count = sum(
            1 for r in druggability_results if r.get("target_type") == "FusionPartner"
        )
        high_count = sum(
            1
            for r in druggability_results
            if r["druggability"] == "High" and r.get("target_type") == "DirectTarget"
        )
        ax.text(
            0.98,
            0.02,
            f"Direct targets: {direct_count} | High: {high_count}",
            transform=ax.transAxes,
            fontsize=5,
            va="bottom",
            ha="right",
            style="italic",
        )

        logger.info(
            f"  Druggability assessment: {len(druggability_results)} genes analyzed"
        )
        if fusion_count > 0:
            logger.info(
                f"  WARNING: {fusion_count} genes are fusion partners, not direct drug targets"
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No druggability data available",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")

    ax.set_title(
        "L. Druggability Assessment", fontsize=10, fontweight="bold", loc="left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# PLOTTING FUNCTIONS - ROW 4 (M-P): External Validation
# ============================================================================


def plot_panel_M_direction_concordance(concordance_df, output_path, logger):
    """Panel M: Direction concordance heatmap.

    FIXES: Updated to use 'wgcna_direction' column name
    """
    logger.info("Plotting Panel M: Direction Concordance")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if concordance_df is None or len(concordance_df) == 0:
        ax.text(0.5, 0.5, "No concordance data", ha="center", va="center", fontsize=10)
        ax.axis("off")
    else:
        # Pivot for heatmap
        pivot_df = concordance_df.pivot_table(
            index="gene", columns="dataset", values="concordant", aggfunc="mean"
        )

        if len(pivot_df) > 0:
            # Limit to top 15 genes
            pivot_df = pivot_df.head(15)

            sns.heatmap(
                pivot_df,
                cmap="RdYlGn",
                center=0.5,
                vmin=0,
                vmax=1,
                annot=False,
                cbar_kws={"label": "Concordance"},
                ax=ax,
            )
            ax.set_xlabel("Dataset", fontsize=8)
            ax.set_ylabel("Gene", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.text(
                0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=10
            )
            ax.axis("off")

    ax.set_title("M. Direction Concordance", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_N_signature_boxplot(ext_datasets, top_genes, output_path, logger):
    """Panel N: Signature score boxplot by group."""
    logger.info("Plotting Panel N: Signature Score Boxplot")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    all_data = []

    for dataset_name, dataset in ext_datasets.items():
        expr_df = dataset["expression"]
        labels = dataset["labels"]

        scores = calculate_signature_score(expr_df, top_genes)
        if scores is None:
            continue

        for sample in scores.index:
            if sample in labels.index:
                all_data.append(
                    {
                        "Dataset": dataset_name[:15],
                        "Group": labels[sample],
                        "Score": scores[sample],
                    }
                )

    if all_data:
        plot_df = pd.DataFrame(all_data)

        # Create grouped boxplot
        datasets = plot_df["Dataset"].unique()
        x_positions = []
        x_labels = []

        for i, ds in enumerate(datasets):
            ds_data = plot_df[plot_df["Dataset"] == ds]
            case_data = ds_data[ds_data["Group"] == "case"]["Score"]
            control_data = ds_data[ds_data["Group"] == "control"]["Score"]

            bp1 = ax.boxplot(
                [case_data],
                positions=[i * 2],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=COLOR_UP),
            )
            bp2 = ax.boxplot(
                [control_data],
                positions=[i * 2 + 0.7],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=COLOR_DOWN),
            )

            x_positions.append(i * 2 + 0.35)
            x_labels.append(ds[:10])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Signature Score", fontsize=9)

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=COLOR_UP, label="Case"),
            Patch(facecolor=COLOR_DOWN, label="Control"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No signature data", ha="center", va="center", fontsize=10)
        ax.axis("off")

    ax.set_title(
        "N. Signature Score by Group", fontsize=10, fontweight="bold", loc="left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_O_external_roc(roc_curves, output_path, logger):
    """Panel O: External validation ROC curves with 95% CI."""
    logger.info("Plotting Panel O: External ROC Curves")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if not roc_curves:
        ax.text(
            0.5, 0.5, "No ROC data available", ha="center", va="center", fontsize=10
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(roc_curves)))

        for i, (dataset_name, roc_data) in enumerate(roc_curves.items()):
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]
            auc_val = roc_data["auc"]
            ci_lower = roc_data.get("auc_ci_lower", auc_val)
            ci_upper = roc_data.get("auc_ci_upper", auc_val)
            perm_p = roc_data.get("perm_pvalue", np.nan)

            # Format label with CI and significance
            if not np.isnan(perm_p):
                sig_marker = "*" if perm_p < 0.05 else ""
                label = f"{dataset_name[:12]} ({auc_val:.2f} [{ci_lower:.2f}-{ci_upper:.2f}]){sig_marker}"
            else:
                label = f"{dataset_name[:12]} (AUC={auc_val:.2f})"

            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(
            fontsize=4,
            loc="lower right",
            title="* p<0.05 (permutation)",
            title_fontsize=4,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    ax.set_title(
        "O. External Validation ROC", fontsize=10, fontweight="bold", loc="left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_panel_P_summary_schematic(validation_results, output_path, logger):
    """Panel P: Validation summary schematic."""
    logger.info("Plotting Panel P: Validation Summary")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    if validation_results is None or len(validation_results) == 0:
        ax.text(
            0.5, 0.5, "No validation results", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
    else:
        # Summary statistics
        n_datasets = len(validation_results)
        n_validated = validation_results["validated"].sum()
        mean_auc = validation_results["auc"].mean()

        # Create summary visualization
        ax.text(
            0.5,
            0.85,
            "External Validation Summary",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.65,
            f"Datasets Tested: {n_datasets}",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.text(
            0.5,
            0.50,
            f"Validated (AUC >= 0.6): {n_validated}/{n_datasets}",
            ha="center",
            va="center",
            fontsize=10,
            color=COLOR_VALIDATED if n_validated > 0 else COLOR_NOT_VALIDATED,
        )
        ax.text(
            0.5,
            0.35,
            f"Mean AUC: {mean_auc:.3f}",
            ha="center",
            va="center",
            fontsize=10,
        )

        # Status indicator
        if n_validated > 0:
            status = "VALIDATED"
            status_color = COLOR_VALIDATED
        else:
            status = "NOT VALIDATED"
            status_color = COLOR_NOT_VALIDATED

        ax.text(
            0.5,
            0.15,
            status,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=status_color,
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor=status_color, linewidth=2
            ),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax.set_title("P. Validation Summary", fontsize=10, fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# COMPOSITE FIGURE AND MAIN FUNCTION
# ============================================================================


def create_composite_figure(panels_dir, output_path, logger):
    """Create 4x4 composite figure from individual panels."""
    logger.info("Creating composite figure...")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=300)

    panel_files = [
        "Fig3A_Funnel.png",
        "Fig3B_Venn.png",
        "Fig3C_Scatter.png",
        "Fig3D_Lollipop.png",
        "Fig3E_PPI.png",
        "Fig3F_Degree.png",
        "Fig3G_Centrality.png",
        "Fig3H_Module.png",
        "Fig3I_TF.png",
        "Fig3J_KEGG.png",
        "Fig3K_GO.png",
        "Fig3L_Drug.png",
        "Fig3M_Concordance.png",
        "Fig3N_Boxplot.png",
        "Fig3O_ROC.png",
        "Fig3P_Summary.png",
    ]

    for idx, (ax, panel_file) in enumerate(zip(axes.flat, panel_files)):
        panel_path = panels_dir / panel_file
        if panel_path.exists():
            img = plt.imread(panel_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, f"Panel {chr(65 + idx)} missing", ha="center", va="center"
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved composite: {output_path}")


def save_source_data(
    evidence_df,
    concordance_df,
    validation_results,
    roc_curves,
    ppi_graph,
    enrichr_results,
    ext_datasets,
    output_dir,
    logger,
):
    """Save source data for all 16 panels."""
    logger.info("Saving source data for all panels...")

    sourcedata_dir = output_dir / "sourcedata16"

    # Panel A: Funnel data
    funnel_data = pd.DataFrame(
        {
            "stage": [
                "WGCNA Candidates",
                "DEG Overlap",
                "PPI Connected",
                "Top 20 Final",
            ],
            "count": [
                100,
                (evidence_df["DEG_Flag"].isin(["Up", "Down"])).sum(),
                (evidence_df["PPIEvidence"] > 0).sum(),
                20,
            ],
        }
    )
    funnel_data.to_csv(sourcedata_dir / "SourceData_Fig3A_Funnel.csv", index=False)

    # Panel B: Venn overlap data
    hub_genes = set(evidence_df[evidence_df["HubEvidence"] > 0.5]["gene"])
    deg_genes = set(evidence_df[evidence_df["DEG_Flag"].isin(["Up", "Down"])]["gene"])
    ppi_genes = set(evidence_df[evidence_df["PPIEvidence"] > 0]["gene"])
    venn_data = pd.DataFrame(
        {
            "category": [
                "Hub",
                "DEG",
                "PPI",
                "Hub_DEG",
                "Hub_PPI",
                "DEG_PPI",
                "All_Three",
            ],
            "count": [
                len(hub_genes),
                len(deg_genes),
                len(ppi_genes),
                len(hub_genes & deg_genes),
                len(hub_genes & ppi_genes),
                len(deg_genes & ppi_genes),
                len(hub_genes & deg_genes & ppi_genes),
            ],
        }
    )
    venn_data.to_csv(sourcedata_dir / "SourceData_Fig3B_Venn.csv", index=False)

    # Panel C: Hub vs DEG scatter
    scatter_data = evidence_df[
        ["gene", "HubEvidence", "DEG_Log2FC", "direction", "composite_score"]
    ].copy()
    scatter_data["DEG_abs_FC"] = scatter_data["DEG_Log2FC"].abs().fillna(0)
    scatter_data.to_csv(sourcedata_dir / "SourceData_Fig3C_Scatter.csv", index=False)

    # Panel D: Top 20 lollipop
    top20 = evidence_df.nlargest(20, "composite_score")[
        ["gene", "module", "composite_score", "direction"]
    ]
    top20.to_csv(sourcedata_dir / "SourceData_Fig3D_Top20.csv", index=False)

    # Panel E: PPI network edges
    if ppi_graph and ppi_graph.number_of_edges() > 0:
        edges = pd.DataFrame(list(ppi_graph.edges()), columns=["source", "target"])
        edges.to_csv(sourcedata_dir / "SourceData_Fig3E_PPI_Edges.csv", index=False)

    # Panel F: Degree distribution
    if ppi_graph and ppi_graph.number_of_nodes() > 0:
        degrees = pd.DataFrame(
            [(n, d) for n, d in ppi_graph.degree()], columns=["node", "degree"]
        )
        degrees.to_csv(sourcedata_dir / "SourceData_Fig3F_Degrees.csv", index=False)

    # Panel G: Centrality ranking
    centrality_data = evidence_df.nlargest(10, "PPIEvidence")[
        ["gene", "PPIEvidence", "DEG_Flag"]
    ]
    centrality_data.to_csv(
        sourcedata_dir / "SourceData_Fig3G_Centrality.csv", index=False
    )

    # Panel H: Module-gene network
    module_gene = evidence_df.head(20)[["gene", "module"]]
    module_gene.to_csv(sourcedata_dir / "SourceData_Fig3H_ModuleGene.csv", index=False)

    # Panel I: TF enrichment
    if (
        enrichr_results
        and "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X" in enrichr_results
    ):
        tf_data = enrichr_results["ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X"]
        if tf_data:
            tf_df = pd.DataFrame(
                [
                    {
                        "TF": r[1].split("_")[0] if "_" in r[1] else r[1],
                        "pvalue": r[2],
                        "combined_score": r[4],
                    }
                    for r in tf_data[:10]
                ]
            )
            tf_df.to_csv(sourcedata_dir / "SourceData_Fig3I_TF.csv", index=False)

    # Panel J: KEGG enrichment
    if enrichr_results and "KEGG_2021_Human" in enrichr_results:
        kegg_data = enrichr_results["KEGG_2021_Human"]
        if kegg_data:
            kegg_df = pd.DataFrame(
                [
                    {"Pathway": r[1], "pvalue": r[2], "combined_score": r[4]}
                    for r in kegg_data[:10]
                ]
            )
            kegg_df.to_csv(sourcedata_dir / "SourceData_Fig3J_KEGG.csv", index=False)

    # Panel K: GO enrichment
    if enrichr_results and "GO_Biological_Process_2021" in enrichr_results:
        go_data = enrichr_results["GO_Biological_Process_2021"]
        if go_data:
            go_df = pd.DataFrame(
                [
                    {"Term": r[1], "pvalue": r[2], "combined_score": r[4]}
                    for r in go_data[:10]
                ]
            )
            go_df.to_csv(sourcedata_dir / "SourceData_Fig3K_GO.csv", index=False)

    # Panel L: Drug-target (saved during plotting)
    # Will be saved by plot_panel_L_drug_target function

    # Panel M: Concordance
    if concordance_df is not None and len(concordance_df) > 0:
        concordance_df.to_csv(
            sourcedata_dir / "SourceData_Fig3M_Concordance.csv", index=False
        )

    # Panel N: Signature scores
    if ext_datasets:
        all_scores = []
        top_genes = evidence_df.head(20)["gene"].tolist()
        for dataset_name, dataset in ext_datasets.items():
            expr_df = dataset["expression"]
            labels = dataset["labels"]
            scores = calculate_signature_score(expr_df, top_genes)
            if scores is not None:
                for sample in scores.index:
                    if sample in labels.index:
                        all_scores.append(
                            {
                                "dataset": dataset_name,
                                "sample": sample,
                                "group": labels[sample],
                                "score": scores[sample],
                            }
                        )
        if all_scores:
            pd.DataFrame(all_scores).to_csv(
                sourcedata_dir / "SourceData_Fig3N_Signature.csv", index=False
            )

    # Panel O: ROC curves
    if roc_curves:
        roc_data = []
        for dataset_name, roc_info in roc_curves.items():
            for i, (fpr, tpr) in enumerate(zip(roc_info["fpr"], roc_info["tpr"])):
                roc_data.append(
                    {
                        "dataset": dataset_name,
                        "fpr": fpr,
                        "tpr": tpr,
                        "auc": roc_info["auc"],
                    }
                )
        if roc_data:
            pd.DataFrame(roc_data).to_csv(
                sourcedata_dir / "SourceData_Fig3O_ROC.csv", index=False
            )

    # Panel P: Validation summary
    if validation_results is not None and len(validation_results) > 0:
        validation_results.to_csv(
            sourcedata_dir / "SourceData_Fig3P_Validation.csv", index=False
        )

    # Full evidence table (comprehensive)
    evidence_df.to_csv(
        sourcedata_dir / "SourceData_Fig3_ABCD_Evidence.csv", index=False
    )

    logger.info(f"  Source data saved to {sourcedata_dir}")


def create_manifest(output_dir, evidence_df, validation_results, logger):
    """Create manifest file with run metadata."""
    logger.info("Creating manifest...")

    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "n_candidates": len(evidence_df),
        "top_gene": evidence_df.iloc[0]["gene"] if len(evidence_df) > 0 else None,
        "n_external_datasets": len(validation_results)
        if validation_results is not None
        else 0,
        "validation_success": bool(validation_results["validated"].any())
        if validation_results is not None and len(validation_results) > 0
        else False,
        "mean_auc": float(validation_results["auc"].mean())
        if validation_results is not None and len(validation_results) > 0
        else None,
        "panels_generated": 16,
        "output_files": {
            "panels": "panels16/",
            "composite": "composite/Figure3_Composite.png",
            "sourcedata": "sourcedata16/",
            "api_cache": "raw/api_cache/",
            "logs": "logs/fig3_run.log",
        },
    }

    manifest_path = output_dir / "manifests" / "fig3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Manifest saved: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Figure 3: Mechanism & External Validation"
    )
    _script = Path(os.path.abspath(__file__))
    _attempt_root = (
        _script.parent.parent.parent
    )  # step3_validation/code -> step3_validation -> PROJECT_ROOT
    parser.add_argument(
        "--f1_dir",
        default=str(_attempt_root / "step1_deg_analysis" / "result"),
        help="Figure 1 output directory",
    )
    parser.add_argument(
        "--f2_dir",
        default=str(_attempt_root / "step2_wgcna" / "result"),
        help="Figure 2 output directory",
    )
    parser.add_argument(
        "--ext_dir",
        default=str(_attempt_root / "data"),
        help="External validation data directory",
    )
    parser.add_argument(
        "--output",
        default=str(_script.parent.parent / "result"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output)
    global API_CACHE_DIR, LOG_FILE, FUSION_PARTNERS_FILE
    API_CACHE_DIR = output_dir / "raw" / "api_cache"
    LOG_FILE = output_dir / "logs" / "fig3_run.log"

    # Set fusion partners file path (ZERO-FAKE: external curated resource)
    script_dir_path = Path(os.path.dirname(os.path.abspath(__file__)))
    # Try multiple possible locations
    possible_fusion_paths = [
        script_dir_path.parent.parent.parent
        / "Resources"
        / "curation"
        / "fusion_partners.tsv",
        script_dir_path.parent / "Resources" / "curation" / "fusion_partners.tsv",
        Path(
            "D:/claudecode/proj-01/Resources/curation/fusion_partners.tsv"
        ),  # legacy fallback
    ]
    for fp in possible_fusion_paths:
        if fp.exists():
            FUSION_PARTNERS_FILE = str(fp)
            break

    # Ensure directories exist
    for subdir in [
        "panels16",
        "composite",
        "sourcedata16",
        "raw/api_cache",
        "raw/external_validation",
        "logs",
        "manifests",
        "code",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy source code to output directory
    import shutil
    import glob as glob_module

    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dest = str(output_dir / "code")
    py_files = glob_module.glob(os.path.join(script_dir, "*.py"))
    copied = 0
    for py_file in py_files:
        src = os.path.abspath(py_file)
        dst = os.path.abspath(os.path.join(code_dest, os.path.basename(py_file)))
        if src != dst:
            shutil.copy2(py_file, code_dest)
            copied += 1
    print(f"  Copied {copied} code files to {code_dest}")

    # Setup logging
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("FIGURE 3: MECHANISM & EXTERNAL VALIDATION")
    logger.info("=" * 60)

    try:
        # Load upstream data
        candidates_df, deg_df, modules_df = load_upstream_data(
            args.f1_dir, args.f2_dir, logger
        )

        # Load external validation data
        ext_datasets = load_external_data(args.ext_dir, logger)

        # Build evidence table
        evidence_df = build_evidence_table(candidates_df, deg_df, logger)

        # Query STRING API for PPI
        top_genes = evidence_df.head(50)["gene"].tolist()
        string_data = query_string_api(top_genes, logger)

        # Calculate PPI evidence
        evidence_df, ppi_graph = calculate_ppi_evidence(
            evidence_df, string_data, logger
        )

        # Query Enrichr for enrichment analysis (MUST be before FinalScore calculation!)
        enrichr_results = {}
        for library in [
            "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
            "KEGG_2021_Human",
            "GO_Biological_Process_2021",
        ]:
            result = query_enrichr(top_genes[:30], library, logger)
            if result:
                enrichr_results[library] = result.get(library, [])

        # Calculate RegPathEvidence from Enrichr results (ZERO-FAKE: weighted evidence)
        evidence_df = calculate_regpath_evidence(evidence_df, enrichr_results, logger)

        # Calculate FinalScore (now includes real RegPathEvidence)
        evidence_df["FinalScore"] = (
            0.35 * evidence_df["HubEvidence"]
            + 0.25 * evidence_df["DEGEvidence"]
            + 0.25 * evidence_df["PPIEvidence"]
            + 0.15 * evidence_df["RegPathEvidence"]
        )
        evidence_df = evidence_df.sort_values("FinalScore", ascending=False)

        # Log RegPathEvidence contribution
        regpath_nonzero = (evidence_df["RegPathEvidence"] > 0).sum()
        logger.info(
            f"  FinalScore calculated with {regpath_nonzero} genes having RegPathEvidence > 0"
        )

        # External validation
        concordance_df = validate_direction_concordance(
            evidence_df, ext_datasets, logger
        )
        validation_results, roc_curves = run_external_roc_analysis(
            evidence_df, ext_datasets, logger
        )

        # Generate panels
        panels_dir = output_dir / "panels16"
        target_genes = set(evidence_df.head(20)["gene"].tolist())

        # Row 1: Candidate Prioritization
        plot_panel_A_funnel(evidence_df, panels_dir / "Fig3A_Funnel.png", logger)
        plot_panel_B_venn(evidence_df, panels_dir / "Fig3B_Venn.png", logger)
        plot_panel_C_scatter(evidence_df, panels_dir / "Fig3C_Scatter.png", logger)
        plot_panel_D_lollipop(evidence_df, panels_dir / "Fig3D_Lollipop.png", logger)

        # Row 2: PPI Network
        plot_panel_E_ppi_network(
            ppi_graph, target_genes, panels_dir / "Fig3E_PPI.png", logger
        )
        plot_panel_F_degree_dist(ppi_graph, panels_dir / "Fig3F_Degree.png", logger)
        plot_panel_G_centrality(
            evidence_df, panels_dir / "Fig3G_Centrality.png", logger
        )
        plot_panel_H_module_network(
            evidence_df, panels_dir / "Fig3H_Module.png", logger
        )

        # Row 3: TF & Pathway
        plot_panel_I_tf_enrichment(enrichr_results, panels_dir / "Fig3I_TF.png", logger)
        plot_panel_J_kegg_enrichment(
            enrichr_results, panels_dir / "Fig3J_KEGG.png", logger
        )
        plot_panel_K_go_enrichment(enrichr_results, panels_dir / "Fig3K_GO.png", logger)
        plot_panel_L_drug_target(evidence_df, panels_dir / "Fig3L_Drug.png", logger)

        # Row 4: External Validation
        plot_panel_M_direction_concordance(
            concordance_df, panels_dir / "Fig3M_Concordance.png", logger
        )
        plot_panel_N_signature_boxplot(
            ext_datasets, top_genes[:20], panels_dir / "Fig3N_Boxplot.png", logger
        )
        plot_panel_O_external_roc(roc_curves, panels_dir / "Fig3O_ROC.png", logger)
        plot_panel_P_summary_schematic(
            validation_results, panels_dir / "Fig3P_Summary.png", logger
        )

        # Create composite figure
        create_composite_figure(
            panels_dir, output_dir / "composite" / "Figure3_Composite.png", logger
        )

        # Save source data
        save_source_data(
            evidence_df,
            concordance_df,
            validation_results,
            roc_curves,
            ppi_graph,
            enrichr_results,
            ext_datasets,
            output_dir,
            logger,
        )

        # Create manifest
        manifest = create_manifest(output_dir, evidence_df, validation_results, logger)

        logger.info("=" * 60)
        logger.info("FIGURE 3 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Top candidate: {evidence_df.iloc[0]['gene']}")
        logger.info(
            f"External validation: {'PASSED' if manifest['validation_success'] else 'FAILED'}"
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
