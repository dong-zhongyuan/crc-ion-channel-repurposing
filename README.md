# Ion Channel-Mediated Drug Repurposing in Colorectal Cancer

This repository contains the analysis code for the manuscript:

**"Ion Channel-Mediated Drug Repurposing Opportunities Revealed by Network Pharmacology and Single-Cell Perturbation Validation in Colorectal Cancer"**

## Overview

This study presents an integrated computational pipeline that identifies druggable CRC hub genes connected to ion channel targets through protein-protein interaction bridge paths, validated by dual single-cell perturbation approaches (VGAE-KO and Perturb-seq).

## Repository Structure

```
code/
├── step0-preprocessing.py          # Data preprocessing and batch correction
├── step1-deg.py                    # Differential expression analysis
├── step2-wgcna.py                  # WGCNA hub gene identification
├── step3-external-validation.py   # External cohort validation
├── step3_tcga.py                   # TCGA-COADREAD survival and immune analysis
├── step4-network-pharmacology.py  # Drug-target mining and bridge path discovery
├── step5-vgae-ko.py               # VGAE-based virtual gene knockout
├── step6-perturbseq/              # HCT116 Perturb-seq analysis (7 strategies)
│   ├── run_all.py
│   ├── run_strategy2_gsea.py
│   ├── run_strategy3_ranking.py
│   ├── run_strategy4_mast.py
│   ├── run_strategy5_network.py
│   ├── run_strategy6_perturbation.py
│   ├── run_strategy7_coexpr.py
│   └── run_figures_summary.py
└── step7-figures/                 # Figure generation scripts
    ├── plot_ijms_unified.py       # Main figures 1-5
    └── plot_supplementary_figures.py  # Supplementary figures S1-S6
```

## Requirements

### Python Environment
- Python 3.9+
- See `requirements.txt` for complete package list

### Key Dependencies
- **Data analysis**: pandas, numpy, scipy
- **Machine learning**: scikit-learn, torch, torch-geometric
- **Single-cell analysis**: scanpy, anndata
- **Statistical analysis**: statsmodels, lifelines, pydeseq2
- **Network analysis**: networkx
- **Visualization**: matplotlib, seaborn
- **Pathway analysis**: gseapy

## Installation

```bash
# Clone the repository
git clone https://github.com/dong-zhongyuan/crc-ion-channel-repurposing.git
cd crc-ion-channel-repurposing

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Availability

### Public Datasets Used

All datasets are publicly available:

- **Discovery cohorts**: 
  - GSE196006 (n=42) - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196006)
  - GSE251845 (n=43) - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251845)

- **Validation cohorts**:
  - GSE128969 (n=6) - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128969)
  - GSE138202 (n=16) - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138202)
  - GSE95132 (n=24) - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95132)

- **TCGA-COADREAD** (n=728): [UCSC Xena](https://xenabrowser.net)

- **HCT116 scRNA-seq**:
- CDCP dataset SCDS0000040 - [Cell-omics Data Coordinate Platform](https://ngdc.cncb.ac.cn/cdcp/)
- GSE171429, sample GSM5224587 - [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171429)

- **HCT116 Perturb-seq**: [Figshare](https://plus.figshare.com/ndownloader/files/55021257)

## Usage

### Step-by-Step Analysis Pipeline

Run the analysis scripts in order:

```bash
# Step 0: Preprocessing
python code/step0-preprocessing.py

# Step 1: Differential expression analysis
python code/step1-deg.py

# Step 2: WGCNA hub gene identification
python code/step2-wgcna.py

# Step 3: External validation
python code/step3-external-validation.py
python code/step3_tcga.py

# Step 4: Network pharmacology
python code/step4-network-pharmacology.py

# Step 5: VGAE-KO validation
python code/step5-vgae-ko.py

# Step 6: Perturb-seq analysis (all strategies)
python code/step6-perturbseq/run_all.py

# Step 7: Generate figures
python code/step7-figures/plot_ijms_unified.py
python code/step7-figures/plot_supplementary_figures.py
```

### Configuration

Each script contains configurable parameters at the top of the file. Key parameters include:
- Input/output paths
- Statistical thresholds (FDR, fold-change)
- WGCNA parameters (soft-thresholding power, module size)
- VGAE hyperparameters (latent dimensions, training epochs)

## Key Results

- **23 druggable hub genes** connected to **18 ion channel genes** across 9 channel families
- **200-fold enrichment** over chance (hypergeometric p = 7.76 × 10⁻⁵⁵)
- **Two novel regulatory axes**:
  - Ribosomal protein–ion channel axis (targetable by ataluren)
  - Immune checkpoint–ion channel axis (targetable by clinical antibodies)
- **Dual validation**: VGAE-KO (13/28 validated pairs) + Perturb-seq (6 KO genes, 7 strategies)
- **Clinical relevance**: GALK1 and CFTR associated with overall survival in TCGA-COADREAD

## Citation

If you use this code or data, please cite:

```
[Citation to be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [Your email to be added]

## Acknowledgments

We thank the Gene Expression Omnibus (GEO), TCGA Research Network, and the Cell-omics Data Coordinate Platform for providing public access to the datasets used in this study. We acknowledge the developers of the GenKI methodology and the X-Atlas/Orion Perturb-seq platform for their foundational contributions to single-cell perturbation analysis.
