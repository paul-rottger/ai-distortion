# AI Distortion

Code and data for our paper:

> **Measuring and mitigating persona distortions from AI writing assistance**
>
> Paul Röttger, Kobi Hackenburg, Hannah Rose Kirk, and Christopher Summerfield

## Overview

This repository contains the analysis code and analysis-ready datasets for our paper on persona distortions from AI writing assistance.

| Folder | Description |
|--------|-------------|
| `analysis/` | Python and R scripts for data analysis and figure generation |
| `data/` | Analysis-ready datasets |
| `figures/` | Figures generated from the analyses in the paper |
| `results/` | Results derived from the analyses, including model outputs and statistical test results |
| `reranking/` | Code for the Reranking method used in the mitigation study |

## Requirements

- No special hardware is required for the statistical analyses and plotting workflows in this repository.
- Python dependencies are listed in `requirements_python.txt`.
- R package versions used for the analyses are listed in `requirements_R.csv`.

## Reproducing Results

To reproduce the statistical analyses and figures from our paper, please run the following commands from the root of the repository:

```bash
# Create and activate virtual environment
python -m venv .venv_analysis
source .venv_analysis/bin/activate

# Install Python packages
pip install -r requirements_python.txt

# Install R packages
Rscript requirements_r.R

# Run all analyses and plotting scripts
./run_all.sh 
```

`run_all.sh` supports a small set of optional flags:

- `--run_regressions`: also run the phase 2 distortion regression scripts, which are skipped by default.
- `--debug`: run supported R scripts in debug mode, using `n = 1000` samples instead of the full data. This is most useful together with `--run_regressions`.

Examples:

```bash
# Default pipeline without the phase 2 distortion regressions
./run_all.sh

# Full pipeline including the phase 2 distortion regressions
./run_all.sh --run_regressions

# Faster debug run for supported R scripts, including regression scripts
./run_all.sh --run_regressions --debug
```

Details on each analysis script are listed in `analysis/README.md`.

## Data Availability

All data used in the analyses for our paper is included in the `data/` directory of this repository.
Details are described in `data/README.md`.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License.