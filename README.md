# Ion Channel-Mediated Drug Repurposing in Colorectal Cancer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the computational pipeline for identifying druggable colorectal cancer (CRC) hub genes connected to ion channel targets through protein-protein interaction (PPI) bridge paths, validated by dual single-cell perturbation approaches (VGAE-KO and Perturb-seq).

## Repository Structure

```
code/
├── step0-pre-processing/
│   ├── preprocessing_unified.py    # Batch correction and normalization
│   ├── fix_ext_val.py              # External validation data preprocessing
├── step1-deg.py                    # Differential expression analysis
├── step2-wgcna.py                  # WGCNA hub gene identification
├── step3-external-validation.py   # External cohort validation
├── step4-network-pharmacology.py  # PPI bridge path + drug mining
├── step5-vgae-ko/
│   ├── vgae_ko_pipeline.py        # VGAE-based virtual knockout
│   └── convert_gsm5224587.py      # GSM5224587 data conversion
└── step6-perturb-seq/
    ├── run_all.py                 # Master script for all strategies
    ├── run_strategy2_gsea.py      # GSEA pathway enrichment
    ├── run_strategy3_ranking.py   # Transcriptome-wide ranking
    ├── run_strategy4_mast.py      # Zero-inflated differential expression
    ├── run_strategy5_network.py   # Indirect mediator network
    ├── run_strategy6_perturbation.py  # Global perturbation score
    ├── run_strategy7_coexpr.py    # Co-expression disruption
    └── run_figures_summary.py     # Evidence matrix visualization
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crc-ion-channel-repurposing.git
cd crc-ion-channel-repurposing

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Availability

All datasets are publicly available:

### Discovery Cohorts (Bulk RNA-seq)
- **GSE196006** (n=42): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196006
- **GSE251845** (n=43): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251845

### External Validation Cohorts (Bulk RNA-seq)
- **GSE128969** (n=6): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128969
- **GSE138202** (n=16): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138202
- **GSE95132** (n=24): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95132

### Single-Cell RNA-seq (VGAE-KO Validation)
- **SCDS0000040** : https://ngdc.cncb.ac.cn/cdcp/dataset/SCDS0000040
- **GSM5224587** : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171429

### Perturb-seq Data (Experimental Validation)
- **HCT116 CRISPRi Perturb-seq** (Replogle et al., Cell 2022): https://plus.figshare.com/ndownloader/files/55021257

### External Databases
- **STRING v12.0**: https://string-db.org/api/json/network
- **OpenTargets**: https://api.platform.opentargets.org/api/v4/graphql 
- **DGIdb**: https://dgidb.org/api/graphql
- **HGNC**: https://rest.genenames.org/fetch/symbol/%7Bsymbol%7D
- **ChEMBL Target Search**: https://www.ebi.ac.uk/chembl/api/data/target/search.json
- **ChEMBL Mechanism**: https://www.ebi.ac.uk/chembl/api/data/mechanism.json

## Usage

Run the analysis pipeline in order:

```bash
# Step 0: Preprocessing
python code/step0-pre-processing/preprocessing_unified.py
python code/step0-pre-processing/fix_ext_val.py

# Step 1: Differential expression analysis
python code/step1-deg.py

# Step 2: WGCNA hub gene identification
python code/step2-wgcna.py

# Step 3: External validation
python code/step3-external-validation.py

# Step 4: Network pharmacology
python code/step4-network-pharmacology.py

# Step 5: VGAE-KO validation
python code/convert_gsm5224587.py
python code/step5-vgae-ko/vgae_ko_pipeline.py

# Step 6: Perturb-seq analysis (all 7 strategies)
python code/step6-perturb-seq/run_all.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the Gene Expression Omnibus (GEO), TCGA Research Network, Cell-omics Data Coordinate Platform (CDCP), and the developers of the datasets used in this study.

## Contact

For questions or issues, please open an issue on GitHub.
