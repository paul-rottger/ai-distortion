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
| `reranking/` | Code for the Reranking method used in the mitigation study (Study 3) |

## Requirements

- **Python:** Python 3.11. Package dependencies and pinned versions are listed in `requirements_python.txt`.
- **R:** R 4.5. Package dependencies and pinned versions are listed in `requirements_r.R`.
- **Java:** Java 17 or later, for the LanguageTool spelling and grammar check. LanguageTool 6.8 (approx. 260MB) is downloaded automatically on first run.
- **Internet access** on first run, to download LanguageTool and the `all-mpnet-base-v2` sentence embedding model (approx. 420MB) from Hugging Face.
- **Hardware:** No special or non-standard hardware is required.

## Reproducing Results

All results and figures from our paper and supplement are shown in the `results/` and `figures/` directories.
To reproduce these results and figures, please run the commands below from the root of the repository.
Installing the Python and R dependencies takes approximately 3 minutes on a normal desktop computer with a broadband connection.
**Please note** that the full analysis workflow takes several days to complete (tested on 2021 M1 MacBook Pro).
This is primarily due to the mixed-effects multinomial logistic regression analysis for nominal variables in our main study.
For a faster **demo run** (approx. 15-20 minutes), pass the `--demo` flag to `run_all.sh`.
This will run all regression analyses on smaller random subsets of the data and write outputs to `demo_results/` and `demo_figures/` instead of the main output directories.

```bash
# Create and activate virtual environment
python -m venv .venv_analysis
source .venv_analysis/bin/activate

# Install Python packages
pip install -r requirements_python.txt

# Install R packages
Rscript requirements_r.R

# Run all analyses and plotting scripts in DEMO MODE...
# Results and figures will be written to demo_results/ and demo_figures/
./run_all.sh --demo

# ... OR run all analyses and plotting scripts in full (takes several days)
# Results and figures will be written to results/ and figures/
# ./run_all.sh 

# On successful completion, the script prints a line of the form:
# ALL ANALYSES COMPLETED IN HH:MM:SS
```

**On Windows**, run the same workflow in Git Bash (included with [Git for Windows](https://gitforwindows.org/)).
Python 3.11 and R 4.5 must be installed.

```bash
# Create and activate virtual environment
py -3.11 -m venv .venv_analysis
source .venv_analysis/Scripts/activate

# Install Python and R packages
pip install -r requirements_python.txt
Rscript requirements_r.R

# Run all analyses and plotting scripts in DEMO MODE (or omit --demo for the full run)
./run_all.sh --demo
```

The R installer does not add `Rscript` to `PATH` by default.
If `Rscript` is not found, add R to your `PATH` in Git Bash before running the commands above, e.g. `export PATH="/c/Program Files/R/R-4.5.0/bin:$PATH"` (adjust to your installed R version).
Alternatively, you can use the macOS/Linux commands above unchanged in [WSL](https://learn.microsoft.com/en-us/windows/wsl/).

## Data Availability

All data used in the analyses for our paper is included in the `data/` directory of this repository.
Details on each data file are listed in `data/README.md`.

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