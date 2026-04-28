# Measuring and Mitigating Persona Distortions from AI Writing Assistance

## Overview

This repository contains all analysis code and data for our paper on "Measuring and Mitigating Persona Distortions from AI Writing Assistance".
This is joint work by Paul Röttger, Kobi Hackenburg, Hannah Rose Kirk, and Christopher Summerfield, supported by the UK's AI Security Institute.

| Folder | Description |
|--------|-------------|
| `analysis/` | Python and R scripts for data analysis and figure generation |
| `data/` | Analysis-ready datasets |
| `figures/` | Figures generated from the analyses in the paper |
| `results/` | Results derived from the analyses, including model outputs and statistical test results |
| `reranking/` | Code for the Reranking method used in the mitigation study |

## Requirements

- No special hardware is required for the statistical analyses and plotting workflows in this repository.
- Python dependencies and versions are listed in `requirements_python.txt`.
- R packages and versions are listed in `requirements_r.R`.

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

`run_all.sh` supports an optional flag:

- `--demo`: run demo-capable analyses in demo mode. Demo-capable R analyses use `n = 1000` samples instead of the full data, and Python analysis scripts write generated results and figures to `demo_results/` and `demo_figures/` where applicable so they do not overwrite the main outputs.

Examples:

```bash
# Full pipeline
./run_all.sh

# Faster demo run for demo-capable analyses
./run_all.sh --demo
```

Details on each analysis script are listed in `analysis/README.md`.

## Data Availability

All data used in the analyses for our paper is included in the `data/` directory of this repository.
Details are described in `data/README.md`.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License.
Please cite the paper below if you use any of the code or data in this repository.

## Citation

```bibtex
@misc{rottger2026measuring,
      title={Measuring and Mitigating Persona Distortions from AI Writing Assistance},
      author={Paul Röttger and Kobi Hackenburg and Hannah Rose Kirk and Christopher Summerfield},
      year={2026},
      eprint={2604.22503},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.22503},
}
```