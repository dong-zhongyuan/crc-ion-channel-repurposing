# Quick Start Guide

This guide will help you get started with the CRC ion channel repurposing analysis pipeline.

## Prerequisites

- Python 3.9 or higher
- 16GB+ RAM recommended
- ~50GB disk space for data and outputs

## Installation (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/[USERNAME]/crc-ion-channel-repurposing.git
cd crc-ion-channel-repurposing
```

### 2. Set Up Python Environment

**Option A: Using venv (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n crc-analysis python=3.9
conda activate crc-analysis
pip install -r requirements.txt
```

### 3. Download Data

See [DATA_GUIDE.md](DATA_GUIDE.md) for detailed instructions.

**Quick setup:**
```bash
# Create data directories
mkdir -p data/{discovery,validation,tcga,single_cell}/{GSE196006,GSE251845,GSE128969,GSE138202,GSE95132,hct116_scrnaseq,hct116_perturbseq}

# Download datasets from GEO, TCGA, and CDCP
# (See DATA_GUIDE.md for download links)
```

## Running the Analysis

### Option 1: Run Complete Pipeline

```bash
# Run all steps sequentially (takes several hours)
bash run_pipeline.sh
```

### Option 2: Run Individual Steps

```bash
# Step 1: Differential expression
python code/step1-deg.py

# Step 2: WGCNA
python code/step2-wgcna.py

# Step 3: Validation
python code/step3-external-validation.py
python code/step3_tcga.py

# Step 4: Network pharmacology
python code/step4-network-pharmacology.py

# Step 5: VGAE-KO
python code/step5-vgae-ko.py

# Step 6: Perturb-seq
python code/step6-perturbseq/run_all.py

# Step 7: Generate figures
python code/step7-figures/plot_ijms_unified.py
python code/step7-figures/plot_supplementary_figures.py
```

## Expected Outputs

After running the pipeline, you should have:

```
output/
├── step1_deg/
│   ├── deg_results.csv
│   ├── volcano_plot.pdf
│   └── pca_plot.pdf
├── step2_wgcna/
│   ├── hub_genes.csv
│   ├── module_trait_correlation.csv
│   └── network_plots.pdf
├── step3_validation/
│   ├── external_validation_results.csv
│   └── roc_curves.pdf
├── step4_network/
│   ├── bridge_paths.csv
│   ├── drug_target_evidence.csv
│   └── network_plots.pdf
├── step5_vgae/
│   ├── vgae_ko_results.csv
│   └── validation_plots.pdf
├── step6_perturbseq/
│   ├── evidence_matrix.csv
│   ├── strategy_results/
│   └── summary_plots.pdf
└── figures/
    ├── Figure1.pdf
    ├── Figure2.pdf
    ├── Figure3.pdf
    ├── Figure4.pdf
    ├── Figure5.pdf
    └── supplementary/
        ├── FigureS1.pdf
        ├── FigureS2.pdf
        ├── FigureS3.pdf
        ├── FigureS4.pdf
        ├── FigureS5.pdf
        └── FigureS6.pdf
```

## Key Results

The pipeline identifies:
- **23 druggable hub genes** connected to **18 ion channels**
- **200-fold enrichment** (p = 7.76 × 10⁻⁵⁵)
- **Two therapeutic axes**:
  - Ribosomal proteins → K⁺/NMDA channels (ataluren)
  - Immune checkpoints → Ca²⁺/glutamate channels (antibodies)

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**2. Memory errors**
```python
# Reduce batch size in scripts
# Or use a machine with more RAM
```

**3. File not found**
```bash
# Check data directory structure
ls -R data/
# Compare with DATA_GUIDE.md
```

**4. CUDA errors (for VGAE)**
```python
# Use CPU instead
# Edit step5-vgae-ko.py: device = 'cpu'
```

## Next Steps

- Explore individual analysis steps in detail
- Modify parameters for your own datasets
- Generate custom figures
- Extend the analysis pipeline

## Getting Help

- Check [README.md](README.md) for detailed documentation
- See [DATA_GUIDE.md](DATA_GUIDE.md) for data setup
- Open an issue on GitHub for bugs or questions
- Read the manuscript for methodological details

## Citation

```
[Citation to be added upon publication]
```

## License

MIT License - see [LICENSE](LICENSE) for details.
