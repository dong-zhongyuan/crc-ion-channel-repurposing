# Ion Channel-Mediated Drug Repurposing in Colorectal Cancer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Computational pipeline for identifying druggable CRC hub genes connected to ion channel targets through PPI bridge paths, validated by dual single-cell perturbation approaches (VGAE-KO and Perturb-seq).

## Repository Structure

```
code/
│
├── step0_data_curation/
│   └── code/
│       ├── step1_data_curation.py           # Download & merge raw counts into data.csv
│       ├── step1_data_curation_with_GSE138202.py
│       └── fix_ext_val.py                   # External validation data preprocessing
│
├── step1_deg_analysis/
│   ├── code/
│   │   ├── step1-deg.py                     # DEG analysis (Welch t-test, BH FDR)
│   │   └── run_fig1.py                      # Figure 1 generation
│   └── result/                              # (generated at runtime)
│
├── step2_wgcna/
│   ├── code/
│   │   └── run_fig2.py                      # WGCNA + hub gene identification
│   └── result/                              # (generated at runtime)
│
├── step3_validation/
│   ├── code/
│   │   └── run_fig3.py                      # External cohort validation (ROC)
│   └── result/                              # (generated at runtime)
│
├── step4_network_pharmacology/
│   ├── code/
│   │   ├── run_fig4.py                      # PPI bridge paths + drug mining
│   │   └── config.json                      # Input path configuration
│   └── result/                              # (generated at runtime)
│
├── step5_vgae_ko/
│   ├── code/
│   │   ├── step5-vgae-ko.py                 # VGAE-based virtual knockout
│   │   ├── convert_gsm5224587.py            # GSM5224587 format conversion
│   │   └── fig_genki_validation.py
│   ├── data/
│   │   ├── scTenifoldKnk_HCT116_count.csv        # SCDS0000040 (pre-formatted)
│   │   ├── scTenifoldKnk_GSM5224587_count.csv    # GSM5224587 (pre-formatted)
│   │   ├── HCT116.count_mtx.tsv.txt               # Raw count matrix
│   │   ├── mart_export.csv                         # Ensembl→Symbol mapping
│   │   └── GSM5224587/
│   │       └── GSM5225487_HCT116-mock_anno.csv.gz
│   └── result/                              # (generated at runtime)
│       └── VGAE_KO_Unified/
│           └── weights/
│               ├── WT-HCT116_weights.pt     # Trained VGAE weights
│               ├── WT-GSM5224587_weights.pt
│               ├── negctrl_seed_HCT116.json # Random seeds for negative controls
│               └── negctrl_seed_GSM5224587.json
│
├── HCT116-preturb-seq/                      # Perturb-seq evidence matrix
│   ├── code/
│   │   ├── run_strategy1_pseudobulk.py      # S1: Pseudobulk DE (run first!)
│   │   ├── run_strategy2_gsea.py            # S2: GSEA pathway enrichment
│   │   ├── run_strategy3_ranking.py         # S3: Transcriptome-wide ranking
│   │   ├── run_strategy4_mast.py            # S4: Zero-inflated DE (MAST)
│   │   ├── run_strategy5_network.py         # S5: Indirect mediator network
│   │   ├── run_strategy6_perturbation.py    # S6: Global perturbation score
│   │   ├── run_strategy7_coexpr.py          # S7: Co-expression disruption
│   │   └── run_figures_summary.py           # Evidence matrix + figures
│   ├── data/
│   │   └── subset_6pairs_allgenes.h5ad      # Perturb-seq AnnData (335 MB)
│   └── result/                              # (generated at runtime)
│       └── tables/
│
└──drug_mining_cache/                       # API cache (STRING, OpenTargets, ChEMBL, DGIdb)

```
```bash
git clone https://github.com/dong-zhongyuan/crc-ion-channel-repurposing.git
cd crc-ion-channel-repurposing

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9. Key dependencies include `numpy`, `pandas`, `scipy`, `scanpy`, `anndata`, `torch`, `torch-geometric`, `gseapy`, `statsmodels`, `pydeseq2`, `pycombat`, and `networkx`. See `requirements.txt` for full details.

## Data Setup

### Automated (Steps 0–4)

Steps 0–4 download and process public bulk RNA-seq data automatically. Place raw GEO files in `raw_data/` or let step0 download them.

### Manual Downloads Required

Two single-cell datasets must be obtained manually:

1. **SCDS0000040** (VGAE-KO, Dataset 1):
   Download from https://ngdc.cncb.ac.cn/cdcp/dataset/SCDS0000040
   → Place count matrix as `step5_vgae_ko/data/scTenifoldKnk_HCT116_count.csv`

2. **GSM5224587** (VGAE-KO, Dataset 2):
   Download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171429
   → Place as `step5_vgae_ko/data/GSM5224587/GSM5225487_HCT116-mock_anno.csv.gz`
   → Run `convert_gsm5224587.py` to generate the formatted count matrix

3. **HCT116 CRISPRi Perturb-seq** (Replogle et al., Cell 2022):
   Download from https://plus.figshare.com/ndownloader/files/55021257
   → Subset to 6 KO genes (EXOSC5, LSM7, GALK1, RIPK2, TRMT112, RPS21) + non-targeting controls
   → Save as `HCT116-preturb-seq/data/subset_6pairs_allgenes.h5ad`

> **Note:** For reproducibility, the Perturb-seq h5ad file must be placed at `HCT116-preturb-seq/data/subset_6pairs_allgenes.h5ad` relative to the repository root. Step 6 scripts resolve this path automatically.

### VGAE-KO Weights

Pre-trained VGAE weights are stored in `step5_vgae_ko/result/VGAE_KO_Unified/` and are included in this repository for reproducibility. To retrain from scratch, delete the weights directory and run step5 with `--force`.

## Usage

Run the pipeline sequentially. Each step reads from the previous step's output:

```bash
# Step 0: Data curation
cd code/step0_data_curation/code
python step1_data_curation.py
python fix_ext_val.py

# Step 1: Differential expression analysis
cd code/step1_deg_analysis/code
python step1-deg.py
python run_fig1.py

# Step 2: WGCNA hub gene identification
cd code/step2_wgcna/code
python run_fig2.py

# Step 3: External validation
cd code/step3_validation/code
python run_fig3.py

# Step 4: Network pharmacology + drug mining
cd code/step4_network_pharmacology/code
python run_fig4.py --config config.json

# Step 5: VGAE-KO virtual knockout
cd code/step5_vgae_ko/code
python convert_gsm5224587.py
python step5-vgae-ko.py

# Step 6: Perturb-seq evidence matrix (7 strategies)
cd code/HCT116-preturb-seq/code
python run_strategy1_pseudobulk.py   # Run first! Generates DEG tables for S2/S3/S5
python run_strategy2_gsea.py
python run_strategy3_ranking.py
python run_strategy4_mast.py
python run_strategy5_network.py
python run_strategy6_perturbation.py
python run_strategy7_coexpr.py
python run_figures_summary.py        # Run last
```

## Data Availability

All datasets are publicly available:

| Dataset | Type | Samples | Source |
|---------|------|---------|--------|
| GSE196006 | Bulk RNA-seq (Discovery) | 42 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196006) |
| GSE251845 | Bulk RNA-seq (Discovery) | 43 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251845) |
| GSE128969 | Bulk RNA-seq (Validation) | 6 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128969) |
| GSE138202 | Bulk RNA-seq (Validation) | 16 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138202) |
| GSE95132 | Bulk RNA-seq (Validation) | 24 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95132) |
| SCDS0000040 | scRNA-seq (VGAE-KO) | — | [CDCP](https://ngdc.cncb.ac.cn/cdcp/dataset/SCDS0000040) |
| GSM5224587 | scRNA-seq (VGAE-KO) | — | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171429) |
| HCT116 Perturb-seq | CRISPRi scRNA-seq | 8,445 cells | [Figshare](https://plus.figshare.com/ndownloader/files/55021257) |

### External Databases
- [STRING v12.0](https://string-db.org/) · [OpenTargets](https://platform.opentargets.org/) · [DGIdb](https://dgidb.org/) · [ChEMBL](https://www.ebi.ac.uk/chembl/) · [HGNC](https://www.genenames.org/)

## Key Results

- **100 hub genes** across 3 functional programs (Ribosomal, RNA Processing, Immune)
- **23 ion channel bridge paths** spanning 7 channel families
- **LSM7→CLIC1**: Strongest validation (VGAE-KO 99.8th percentile; Perturb-seq 15.5/21)
- **RPS21→KCNQ2**: Most clinically actionable (ataluren, EMA-approved)
- **External validation**: AUC 0.93–1.00 across 3 independent cohorts

## License

MIT License. See [LICENSE](LICENSE) for details.
