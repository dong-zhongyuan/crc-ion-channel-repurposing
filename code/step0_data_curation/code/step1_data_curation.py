#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: CRC Data Curation — Discovery & External Validation Dataset Construction

Discovery: GSE196006 + GSE251845 (clinical CRC Tumor vs Normal)
Validation: GSE227315 + GSE253699 + GSE272456 + GSE308900 + GSE236896 (HCT116 cell line perturbation)

Output (in data/ folder):
    data.csv                   — Discovery expression (normalized, batch-corrected)
    metadata.csv               — Discovery metadata
    external_validate_data.csv — Validation expression (normalized, batch-corrected)
    external_metadata.csv      — Validation metadata
"""

import os
import re
import gzip
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
WORK_DIR = (
    _SCRIPT_DIR.parent.parent
)  # step0_data_curation/code -> step0_data_curation -> PROJECT_ROOT
RAW_DATA_DIR = WORK_DIR / "raw_data"
MART_EXPORT_FILE = WORK_DIR / "mart_export.csv"
OUTPUT_DIR = WORK_DIR / "data"

RANDOM_SEED = 42
MAX_MISSING_PERCENT = 20.0

np.random.seed(RANDOM_SEED)

# =============================================================================
# LOGGING
# =============================================================================

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            WORK_DIR / "step1_data_curation.log", encoding="utf-8", mode="w"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================

DISCOVERY_DATASETS = {
    "GSE196006": {
        "file": "GSE196006_raw_counts.csv.gz",
        "description": "Paired CRC tumor vs adjacent normal (early-onset CRC)",
        "loader": "csv_gz_ensembl",
        "id_type": "ensembl",
        "group_logic": "suffix",
        "tumor_pattern": "_L7_|_B7_|_E7_|_A7_|_D7_|_H7_",
        "normal_pattern": "_L0_|_B0_|_E0_|_A0_|_D0_|_H0_",
    },
    "GSE251845": {
        "file": "GSE251845_htseq_raw_counts.csv.gz",
        "description": "Paired early-onset CRC tumor vs normal",
        "loader": "csv_gz_ensembl",
        "id_type": "ensembl",
        "group_logic": "suffix",
        "tumor_pattern": "C_",
        "normal_pattern": "N_",
    },
}

VALIDATION_DATASETS = {
    "GSE227315": {
        "file": "GSE227315_HCT116_Oxa24h_Raw_Norm_counts.txt.gz",
        "description": "HCT116 Oxaliplatin treatment (NT vs OXA, raw counts only)",
        "loader": "tsv_gz_ensembl_symbol",
        "id_type": "ensembl_symbol",
        "group_logic": "select_columns",
        "case_columns": ["OXA1", "OXA2", "OXA3"],
        "control_columns": ["NT1", "NT2", "NT3"],
    },
    "GSE272456": {
        "file": "GSE272456_PN0129B_HCT116_raw_counts.txt.gz",
        "description": "HCT116 5-FU treatment (0h ctrl vs 24h+48h 5FU)",
        "loader": "tsv_gz_symbol",
        "id_type": "symbol",
        "group_logic": "contains",
        "tumor_pattern": "5FU",
        "normal_pattern": "Ctrl",
    },
    "GSE308900": {
        "file": "GSE308900_readcount.txt.gz",
        "description": "HCT116 Ganoderma triterpenoid treatment (Control vs GTEA1)",
        "loader": "tsv_gz_ensembl",
        "id_type": "ensembl",
        "group_logic": "prefix",
        "tumor_pattern": "A1_",
        "normal_pattern": "Control_",
    },
    "GSE236896": {
        "file": "GSE236896_All_samples_raw_counts.txt.gz",
        "description": "HCT116 EZH2 inhibitor (Vehicle vs Combo)",
        "loader": "tsv_gz_ensembl",
        "id_type": "ensembl",
        "group_logic": "select_columns",
        "case_columns": [
            "Combo_D6_rep1",
            "Combo_D6_rep2",
            "Combo_D6_rep3",
            "Combo_D3_rep1",
            "Combo_D3_rep2",
            "Combo_D3_rep3",
        ],
        "control_columns": [
            "Veh_D6_rep1",
            "Veh_D6_rep2",
            "Veh_D6_rep3",
            "Veh_D3_B1_rep1",
            "Veh_D3_B1_rep2",
            "Veh_D3_B1_rep3",
            "Veh_D3_B2_rep1",
            "Veh_D3_B2_rep2",
            "Veh_D3_B2_rep3",
        ],
    },
    "GSE253699": {
        "file": "GSE253699_raw_counts.txt.gz",
        "description": "HCT116 FSTL3 treatment (NC vs FSTL3)",
        "loader": "tsv_gz_ensembl_extra",
        "id_type": "ensembl",
        "group_logic": "prefix",
        "tumor_pattern": "FSTL3_",
        "normal_pattern": "HCT_NC_",
        "count_columns": [
            "HCT_NC_1",
            "HCT_NC_2",
            "HCT_NC_3",
            "FSTL3_1",
            "FSTL3_2",
            "FSTL3_3",
        ],
        "gene_id_col": "gene_id",
    },
}


# =============================================================================
# ID MAPPING
# =============================================================================


def load_id_mapping(mart_path):
    df = pd.read_csv(mart_path)
    df = df[df["HGNC symbol"].notna() & (df["HGNC symbol"] != "")]
    mapping = dict(zip(df["Gene stable ID"], df["HGNC symbol"]))
    logger.info(f"Loaded {len(mapping)} Ensembl->Symbol mappings")
    return mapping


def parse_ensembl_id(eid):
    if not isinstance(eid, str):
        return str(eid)
    return eid.split(".")[0].split("_")[0]


def convert_ensembl_to_symbol(df, id_map):
    parsed = df.index.map(parse_ensembl_id)
    symbols = parsed.map(lambda x: id_map.get(x, None))
    mapped = symbols.notna().sum()
    logger.info(f"  Mapped {mapped}/{len(df)} Ensembl IDs to gene symbols")
    df = df.copy()
    df["_symbol"] = symbols
    df = df[df["_symbol"].notna()]
    sample_cols = [c for c in df.columns if c != "_symbol"]
    df = df.groupby("_symbol")[sample_cols].mean()
    logger.info(f"  After aggregation: {len(df)} unique genes")
    return df


# =============================================================================
# DATASET LOADERS
# =============================================================================


def _assign_groups(df, config, gse):
    logic = config["group_logic"]
    cols = df.columns.tolist()

    if logic == "suffix":
        tp, np_ = config["tumor_pattern"], config["normal_pattern"]
        case_cols = [c for c in cols if any(p in c for p in tp.split("|"))]
        ctrl_cols = [c for c in cols if any(p in c for p in np_.split("|"))]
    elif logic == "prefix":
        case_cols = [c for c in cols if c.startswith(config["tumor_pattern"])]
        ctrl_cols = [c for c in cols if c.startswith(config["normal_pattern"])]
    elif logic == "suffix_simple":
        case_cols = [c for c in cols if c.endswith(config["tumor_pattern"])]
        ctrl_cols = [c for c in cols if c.endswith(config["normal_pattern"])]
    elif logic == "contains":
        case_cols = [c for c in cols if config["tumor_pattern"] in c]
        ctrl_cols = [c for c in cols if config["normal_pattern"] in c]
    elif logic == "regex":
        case_cols = [c for c in cols if re.search(config["tumor_regex"], c)]
        ctrl_cols = [c for c in cols if re.search(config["normal_regex"], c)]
    elif logic == "select_columns":
        case_cols = [c for c in config["case_columns"] if c in cols]
        ctrl_cols = [c for c in config["control_columns"] if c in cols]
    else:
        raise ValueError(f"Unknown group_logic: {logic}")

    overlap = set(case_cols) & set(ctrl_cols)
    if overlap:
        logger.warning(f"  {gse}: {len(overlap)} overlap columns, removing from case")
        case_cols = [c for c in case_cols if c not in overlap]

    logger.info(f"  {gse}: {len(ctrl_cols)} control, {len(case_cols)} case")
    if len(ctrl_cols) == 0 or len(case_cols) == 0:
        raise ValueError(f"{gse}: No control or case columns found")

    ctrl_data = df[ctrl_cols].copy()
    case_data = df[case_cols].copy()
    ctrl_data.columns = [f"{gse}_{c}_control" for c in ctrl_cols]
    case_data.columns = [f"{gse}_{c}_case" for c in case_cols]

    combined = pd.concat([ctrl_data, case_data], axis=1)
    labels = ["control"] * len(ctrl_cols) + ["case"] * len(case_cols)
    return combined, labels


def load_dataset(gse, config, id_map):
    file_path = RAW_DATA_DIR / config["file"]
    loader = config["loader"]
    logger.info(f"Loading {gse}: {file_path.name} (loader={loader})")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if loader == "csv_gz_ensembl":
        df = pd.read_csv(file_path, compression="gzip", index_col=0, quotechar='"')
        df.index = df.index.astype(str).str.strip('"')
    elif loader in ("tsv_gz_symbol", "tsv_gz_ensembl"):
        df = pd.read_csv(file_path, sep="\t", compression="gzip", index_col=0)
    elif loader == "tsv_gz_ensembl_extra":
        raw = pd.read_csv(file_path, sep="\t", compression="gzip")
        df = raw.set_index(config["gene_id_col"])[config["count_columns"]]
    elif loader == "tsv_gz_ensembl_symbol":
        # NEW: For GSE227315 — index is ENSG_SYMBOL format, file has raw + norm columns
        df = pd.read_csv(file_path, sep="\t", compression="gzip", index_col=0)
        # Keep only the columns we need (raw counts, not NORM_ columns)
        keep_cols = config.get("case_columns", []) + config.get("control_columns", [])
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]
    elif loader == "xls_gz_symbol":
        with gzip.open(file_path, "rb") as f_in:
            data = f_in.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".xls", delete=False)
        tmp.write(data)
        tmp.close()
        try:
            df = pd.read_excel(tmp.name, engine="xlrd", index_col=0)
        finally:
            os.unlink(tmp.name)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    df = df.apply(pd.to_numeric, errors="coerce")
    combined, labels = _assign_groups(df, config, gse)

    if config["id_type"] == "ensembl":
        combined = convert_ensembl_to_symbol(combined, id_map)
    elif config["id_type"] == "ensembl_symbol":
        # NEW: Parse ENSG00000000003_TSPAN6 -> TSPAN6
        new_index = []
        for idx in combined.index:
            idx_str = str(idx)
            if "_" in idx_str:
                symbol = idx_str.split("_", 1)[1]
                new_index.append(symbol)
            else:
                # Try Ensembl mapping
                parsed = parse_ensembl_id(idx_str)
                mapped = id_map.get(parsed, None)
                new_index.append(mapped if mapped else idx_str)
        combined.index = new_index
        # Remove unmapped (None or empty)
        combined = combined[combined.index.notna() & (combined.index != "")]
        # Aggregate duplicates
        if combined.index.duplicated().any():
            n_dup = combined.index.duplicated().sum()
            logger.info(f"  Aggregating {n_dup} duplicate gene symbols (mean)")
            combined = combined.groupby(combined.index).mean()
        logger.info(f"  After symbol extraction: {len(combined)} unique genes")
    else:
        combined.index = combined.index.astype(str)
        if combined.index.duplicated().any():
            n_dup = combined.index.duplicated().sum()
            logger.info(f"  Aggregating {n_dup} duplicate gene symbols (mean)")
            combined = combined.groupby(combined.index).mean()

    logger.info(f"  Final: {combined.shape[0]} genes x {combined.shape[1]} samples")
    return combined, labels


# =============================================================================
# STEP 1: DATA INGESTION (raw counts)
# =============================================================================


def ingest_raw_data(datasets, id_map, cohort_name):
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[STEP 1] Ingesting {cohort_name} ({len(datasets)} datasets)")
    logger.info(f"{'=' * 80}")

    all_dfs, all_labels, all_dataset_labels = [], [], []

    for gse, config in datasets.items():
        try:
            df, labels = load_dataset(gse, config, id_map)
            all_dfs.append(df)
            all_labels.extend(labels)
            all_dataset_labels.extend([gse] * len(labels))
        except Exception as e:
            logger.error(f"  FAILED to load {gse}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_dfs:
        raise ValueError(f"No datasets loaded for {cohort_name}")

    merged = pd.concat(all_dfs, axis=1, join="outer")
    sample_ids = merged.columns.tolist()
    logger.info(
        f"\nMerged {cohort_name}: {merged.shape[0]} genes x {merged.shape[1]} samples"
    )

    # Handle missing values
    n_samples = merged.shape[1]
    missing_per_gene = merged.isna().sum(axis=1)
    total_missing = missing_per_gene.sum()
    total_cells = merged.shape[0] * n_samples
    logger.info(
        f"  Missing: {total_missing}/{total_cells} ({100 * total_missing / total_cells:.2f}%)"
    )

    max_allowed = int(MAX_MISSING_PERCENT / 100 * n_samples)
    to_drop = (missing_per_gene > max_allowed).sum()
    if to_drop > 0:
        merged = merged[missing_per_gene <= max_allowed]
        logger.info(f"  Dropped {to_drop} genes with >{MAX_MISSING_PERCENT}% missing")

    remaining = merged.isna().sum().sum()
    if remaining > 0:
        merged = merged.fillna(0)
        logger.info(f"  Filled {remaining} remaining NaN with 0")

    logger.info(
        f"  Final raw counts: {merged.shape[0]} genes x {merged.shape[1]} samples"
    )
    return merged, all_labels, sample_ids, all_dataset_labels


# =============================================================================
# STEP 2: NORMALIZATION & BATCH CORRECTION
# =============================================================================


def normalize_and_correct(count_df, sample_ids, labels, dataset_labels, cohort_name):
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[STEP 2] Normalizing {cohort_name}")
    logger.info(f"{'=' * 80}")

    count_matrix = count_df.values.astype(float)
    gene_names = count_df.index.values
    labels_arr = np.array(labels)
    ds_arr = np.array(dataset_labels)

    unique_datasets = list(dict.fromkeys(dataset_labels))
    control_cols = np.where(labels_arr == "control")[0]
    case_cols = np.where(labels_arr == "case")[0]
    logger.info(
        f"  Input: {count_matrix.shape[0]} genes x {count_matrix.shape[1]} samples"
    )
    logger.info(f"  Datasets: {unique_datasets}")
    logger.info(f"  Control: {len(control_cols)}, Case: {len(case_cols)}")

    # Step 2.1: CPM Normalization
    logger.info("\n  [Step 2.1] CPM Normalization")
    library_sizes = np.sum(count_matrix, axis=0)
    library_sizes[library_sizes == 0] = 1
    cpm_matrix = (count_matrix / library_sizes) * 1_000_000
    logger.info("  CPM normalization completed")

    # Step 2.2: Log2 Transformation
    logger.info("  [Step 2.2] Log2 Transformation")
    log_matrix = np.log2(cpm_matrix + 1)
    logger.info("  Log2(CPM+1) transformation completed")

    # Step 2.3: ComBat Batch Correction (with biological covariate protection)
    logger.info(
        "  [Step 2.3] ComBat Batch Correction (preserving case/control biology)"
    )

    if len(unique_datasets) <= 1:
        logger.info("  Only 1 dataset - no batch correction needed")
    else:
        from pycombat import Combat

        # Batch labels — one per sample
        batch = list(ds_arr)

        # Biological covariate to PRESERVE (case/control difference)
        # X tells ComBat: "this variation is REAL biology, don't remove it"
        # ComBat fits: expression ~ X*beta + batch_effect, then removes only batch_effect
        bio_covariate = np.array([1 if l == "case" else 0 for l in labels_arr]).reshape(
            -1, 1
        )

        logger.info(f"  Batches: {unique_datasets}")
        logger.info(
            f"  Biological covariate (X): {bio_covariate.sum()} case, {len(bio_covariate) - bio_covariate.sum()} control"
        )
        logger.info(f"  ComBat mode: parametric prior (robust for small batches)")

        # Filter out zero/near-zero variance genes BEFORE ComBat.
        # pycombat divides by per-gene variance internally; zero-variance genes
        # produce NaN that propagates to the entire output matrix.
        # These genes (constant across all samples) have no batch effect to correct anyway.
        gene_var = log_matrix.var(axis=1)
        good_mask = gene_var > 1e-10
        n_filtered = (~good_mask).sum()
        logger.info(
            f"  Filtering {n_filtered} zero-variance genes before ComBat "
            f"({good_mask.sum()} genes retained)"
        )

        # Run ComBat only on variable genes
        # pycombat expects Y = (n_samples, n_features), so transpose
        log_good = log_matrix[good_mask]
        combat = Combat(mode="p")  # parametric prior — stable with small batches
        Y_corrected = combat.fit_transform(Y=log_good.T, b=batch, X=bio_covariate)

        # Reassemble: corrected variable genes + unchanged constant genes
        log_matrix[good_mask] = Y_corrected.T

        n_nan = np.isnan(log_matrix).sum()
        if n_nan > 0:
            logger.warning(f"  {n_nan} NaN values after ComBat — replacing with 0")
            log_matrix = np.nan_to_num(log_matrix, nan=0.0)

        logger.info(f"  ComBat correction completed")

    logger.info(
        f"\n  Done: {log_matrix.shape[0]} genes x {log_matrix.shape[1]} samples"
    )
    return pd.DataFrame(log_matrix, index=gene_names, columns=count_df.columns)


# =============================================================================
# OUTPUT
# =============================================================================


def save_expression_csv(df, labels, sample_ids, output_path):
    lines = ["sample," + ",".join(sample_ids), "label," + ",".join(labels)]
    for gene in df.index:
        lines.append(f"{gene}," + ",".join(str(v) for v in df.loc[gene].values))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {output_path} ({df.shape[0]} genes x {df.shape[1]} samples)")


def save_metadata_csv(sample_ids, labels, dataset_labels, output_path):
    pd.DataFrame(
        {"SampleID": sample_ids, "Group": labels, "Dataset": dataset_labels}
    ).to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(sample_ids)} samples)")


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("Step 1: CRC Data Curation (Two-Step: Ingest -> Normalize)")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    id_map = load_id_mapping(MART_EXPORT_FILE)

    # === DISCOVERY ===
    logger.info("\n\n>>> DISCOVERY COHORT <<<")
    disc_raw, disc_labels, disc_samples, disc_ds = ingest_raw_data(
        DISCOVERY_DATASETS, id_map, "Discovery"
    )
    disc_norm = normalize_and_correct(
        disc_raw, disc_samples, disc_labels, disc_ds, "Discovery"
    )
    save_expression_csv(disc_norm, disc_labels, disc_samples, OUTPUT_DIR / "data.csv")
    save_metadata_csv(disc_samples, disc_labels, disc_ds, OUTPUT_DIR / "metadata.csv")

    # === VALIDATION ===
    logger.info("\n\n>>> VALIDATION COHORT <<<")
    val_raw, val_labels, val_samples, val_ds = ingest_raw_data(
        VALIDATION_DATASETS, id_map, "Validation"
    )
    val_norm = normalize_and_correct(
        val_raw, val_samples, val_labels, val_ds, "Validation"
    )
    save_expression_csv(
        val_norm, val_labels, val_samples, OUTPUT_DIR / "external_validate_data.csv"
    )
    save_metadata_csv(
        val_samples, val_labels, val_ds, OUTPUT_DIR / "external_metadata.csv"
    )

    # === SUMMARY ===
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info(f"Discovery: {len(disc_samples)} samples, {disc_norm.shape[0]} genes")
    logger.info(f"Validation: {len(val_samples)} samples, {val_norm.shape[0]} genes")
    logger.info(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
