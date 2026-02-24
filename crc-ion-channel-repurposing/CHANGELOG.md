# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-24

### Added
- Initial release of analysis code for CRC ion channel repurposing manuscript
- Complete analysis pipeline (steps 0-7)
- VGAE-based virtual gene knockout implementation
- HCT116 Perturb-seq analysis with 7-strategy evidence matrix
- Figure generation scripts for main and supplementary figures
- Comprehensive README with installation and usage instructions
- Data download guide (DATA_GUIDE.md)
- Requirements file with all Python dependencies
- MIT License
- Contributing guidelines

### Features
- Step 0: Data preprocessing and batch correction
- Step 1: Differential expression analysis (Welch's t-test)
- Step 2: WGCNA hub gene identification
- Step 3: External validation in 3 independent cohorts
- Step 3 (TCGA): Survival and immune infiltration analysis
- Step 4: Network pharmacology and ion channel bridge path discovery
- Step 5: VGAE-KO validation on 2 independent scRNA-seq datasets
- Step 6: Perturb-seq analysis (7 complementary strategies)
- Step 7: Publication-quality figure generation

### Documentation
- Complete README with project overview
- Data directory structure guide
- Installation instructions
- Usage examples
- Citation information

## [Unreleased]

### Planned
- Docker container for reproducible environment
- Example notebooks for key analyses
- Pre-computed intermediate results for quick testing
- Additional validation datasets
