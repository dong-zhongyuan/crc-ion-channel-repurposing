"""
Convert GSM5225487_HCT116-mock_anno.csv.gz into scTenifoldKnk-compatible CSV.

Input format (from annotation gz):
  - Rows: ENSEMBL IDs (e.g. ENSG00000243485), last column = Gene_name (symbol)
  - Columns: cell barcodes
  - Values: raw integer counts

Output format (scTenifoldKnk):
  - Row 0: header = empty + cell barcodes
  - Col 0: gene symbols (not ENSEMBL)
  - Values: raw integer counts
  - Genes with no valid symbol or duplicates are handled
"""

import pandas as pd
import numpy as np
import os
import time

# ── Paths ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(_SCRIPT_DIR)  # step5_vgae_ko
_DATA_DIR = os.path.join(_STEP_DIR, "data")
INPUT_GZ = os.path.join(_DATA_DIR, "GSM5224587", "GSM5225487_HCT116-mock_anno.csv.gz")
MART_CSV = os.path.join(_DATA_DIR, "mart_export.csv")
OUTPUT_DIR = _DATA_DIR
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scTenifoldKnk_GSM5224587_count.csv")

print(f"[{time.strftime('%H:%M:%S')}] Loading annotation matrix from gz...")
df = pd.read_csv(INPUT_GZ, index_col=0)
print(f"  Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")

# ── Extract Gene_name column (last col) as primary symbol source ──
if "Gene_name" in df.columns:
    gene_name_col = df["Gene_name"].copy()
    df = df.drop(columns=["Gene_name"])
    print(f"  Found Gene_name column with {gene_name_col.notna().sum()} entries")
else:
    gene_name_col = pd.Series([np.nan] * len(df), index=df.index)
    print("  No Gene_name column found, will rely on mart_export.csv")

print(f"  Count matrix: {df.shape[0]} genes x {df.shape[1]} cells")

# ── Build ENSEMBL → symbol mapping from mart_export.csv ──
print(
    f"[{time.strftime('%H:%M:%S')}] Loading mart_export.csv for ENSEMBL→symbol mapping..."
)
mart = pd.read_csv(MART_CSV)
mart.columns = ["ensembl_id", "gene_symbol"]
mart = mart.dropna(subset=["gene_symbol"])
mart = mart[mart["gene_symbol"].str.strip() != ""]
# Keep first occurrence (most canonical)
mart_map = mart.drop_duplicates(subset="ensembl_id", keep="first")
ensembl_to_symbol = dict(zip(mart_map["ensembl_id"], mart_map["gene_symbol"]))
print(f"  Loaded {len(ensembl_to_symbol)} ENSEMBL→symbol mappings")

# ── Resolve gene symbols ──
# Priority: Gene_name column > mart_export > keep ENSEMBL as-is
print(f"[{time.strftime('%H:%M:%S')}] Resolving gene symbols...")

resolved_symbols = []
source_stats = {"gene_name_col": 0, "mart_export": 0, "kept_ensembl": 0}

for ensembl_id, gname in zip(df.index, gene_name_col):
    # 1) Use Gene_name column if valid
    if pd.notna(gname) and str(gname).strip() != "":
        resolved_symbols.append(str(gname).strip())
        source_stats["gene_name_col"] += 1
    # 2) Fall back to mart_export
    elif ensembl_id in ensembl_to_symbol:
        resolved_symbols.append(ensembl_to_symbol[ensembl_id])
        source_stats["mart_export"] += 1
    # 3) Keep ENSEMBL ID as last resort
    else:
        resolved_symbols.append(str(ensembl_id))
        source_stats["kept_ensembl"] += 1

print(
    f"  Symbol sources: Gene_name={source_stats['gene_name_col']}, "
    f"mart_export={source_stats['mart_export']}, "
    f"kept_ensembl={source_stats['kept_ensembl']}"
)

# ── Handle duplicate gene symbols (sum counts) ──
df.index = resolved_symbols

n_before = len(df)
n_unique = df.index.nunique()
if n_unique < n_before:
    print(
        f"[{time.strftime('%H:%M:%S')}] Merging {n_before - n_unique} duplicate gene symbols by summing counts..."
    )
    df = df.groupby(df.index).sum()

print(f"  Final matrix: {df.shape[0]} genes x {df.shape[1]} cells")

# ── Filter out genes with zero total counts ──
gene_sums = df.sum(axis=1)
nonzero_mask = gene_sums > 0
n_zero = (~nonzero_mask).sum()
if n_zero > 0:
    df = df[nonzero_mask]
    print(f"  Removed {n_zero} zero-count genes -> {df.shape[0]} genes remaining")

# ── Save ──
print(f"[{time.strftime('%H:%M:%S')}] Saving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE)

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"  Done! File size: {file_size_mb:.1f} MB")
print(f"  Final dimensions: {df.shape[0]} genes x {df.shape[1]} cells")
print(f"  Sample genes: {list(df.index[:10])}")
