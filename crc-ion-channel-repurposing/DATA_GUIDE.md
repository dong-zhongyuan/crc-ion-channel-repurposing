# Data Directory Structure

This document describes the expected data directory structure for running the analysis pipeline.

## Overview

Due to file size limitations, raw data files are **not included** in this repository. You must download them from the public sources listed below.

## Required Directory Structure

```
crc-ion-channel-repurposing/
├── code/                    # Analysis scripts (included in repo)
├── data/                    # Raw data (NOT in repo - download separately)
│   ├── discovery/
│   │   ├── GSE196006/
│   │   │   └── GSE196006_counts.csv
│   │   └── GSE251845/
│   │       └── GSE251845_counts.csv
│   ├── validation/
│   │   ├── GSE128969/
│   │   │   └── GSE128969_counts.csv
│   │   ├── GSE138202/
│   │   │   └── GSE138202_counts.csv
│   │   └── GSE95132/
│   │       └── GSE95132_counts.csv
│   ├── tcga/
│   │   ├── TCGA_COADREAD_expression.tsv
│   │   ├── TCGA_COADREAD_survival.tsv
│   │   └── TCGA_COADREAD_clinical.tsv
│   └── single_cell/
│       ├── hct116_scrnaseq/
│       │   └── SCDS0000040_HCT116.h5ad
│       └── hct116_perturbseq/
│           └── GSM5224587_perturbseq.h5ad
└── output/                  # Generated results (created by scripts)
    ├── step1_deg/
    ├── step2_wgcna/
    ├── step3_validation/
    ├── step4_network/
    ├── step5_vgae/
    ├── step6_perturbseq/
    └── figures/
```

## Data Download Instructions

### 1. Discovery Cohorts (GEO)

**GSE196006** (n=42, early-onset CRC):
```bash
# Download from GEO
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE196006&format=file
# Extract and place in data/discovery/GSE196006/
```

**GSE251845** (n=43, early-onset CRC):
```bash
# Download from GEO
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE251845&format=file
# Extract and place in data/discovery/GSE251845/
```

### 2. Validation Cohorts (GEO)

**GSE128969** (n=6):
```bash
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE128969&format=file
```

**GSE138202** (n=16):
```bash
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE138202&format=file
```

**GSE95132** (n=24):
```bash
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE95132&format=file
```

### 3. TCGA-COADREAD (n=728)

Download from UCSC Xena:
- Visit: https://xenabrowser.net
- Select: TCGA Colon and Rectal Cancer (COADREAD)
- Download:
  - Gene expression (HiSeq V2, log2-normalized)
  - Survival data
  - Clinical data

### 4. HCT116 scRNA-seq

Download from Cell-omics Data Coordinate Platform:
- Visit: https://ngdc.cncb.ac.cn/cdcp/
- Dataset: SCDS0000040
- Download the HCT116 single-cell RNA-seq data

### 5. HCT116 Perturb-seq

Download from GEO:
```bash
# GSE171429, sample GSM5224587
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM5224587&format=file
```

## Data Format Requirements

### Gene Expression Data
- Format: CSV or TSV
- Rows: Genes (HGNC symbols)
- Columns: Samples
- Values: Raw counts or normalized expression

### Single-Cell Data
- Format: AnnData (.h5ad) or Scanpy-compatible format
- Required fields:
  - `X`: Expression matrix
  - `obs`: Cell metadata (including perturbation labels for Perturb-seq)
  - `var`: Gene metadata

### TCGA Data
- Expression: Tab-separated, log2-normalized
- Survival: Columns must include `OS`, `OS.time`, `sample_id`
- Clinical: Standard TCGA clinical data format

## Creating the Data Directory

```bash
# From the repository root
mkdir -p data/discovery/GSE196006
mkdir -p data/discovery/GSE251845
mkdir -p data/validation/GSE128969
mkdir -p data/validation/GSE138202
mkdir -p data/validation/GSE95132
mkdir -p data/tcga
mkdir -p data/single_cell/hct116_scrnaseq
mkdir -p data/single_cell/hct116_perturbseq
mkdir -p output
```

## Troubleshooting

### File Not Found Errors
- Verify the data directory structure matches the expected layout
- Check that file names match exactly (case-sensitive)
- Ensure files are in the correct format (CSV/TSV/H5AD)

### Memory Issues
- Large datasets may require 16GB+ RAM
- Consider using a machine with sufficient memory
- Process datasets in batches if needed

## Questions?

If you encounter issues downloading or organizing the data, please open an issue on GitHub.
