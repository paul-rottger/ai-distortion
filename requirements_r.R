# R package requirements
# Install with: Rscript requirements_r.R
#
# Tested with R 4.5. The package versions used for the analyses in the paper
# are listed alongside each package below.

packages <- c(
  "lme4",             # 1.1.37
  "mclogit",          # 0.9.15
  "Matrix",           # 1.7.4
  "ordinal",          # 2023.12.4.1
  "marginaleffects",  # 0.30.0
  "broom.mixed",      # 0.2.9.6
  "glmmTMB",          # 1.1.12
  "lubridate",        # 1.9.4
  "forcats",          # 1.0.0
  "stringr",          # 1.5.2
  "dplyr",            # 1.1.4
  "purrr",            # 1.1.0
  "readr",            # 2.1.5
  "tidyr",            # 1.3.1
  "tibble",           # 3.3.0
  "ggplot2",          # 4.0.0
  "tidyverse",        # 2.0.0
  "effsize",          # 0.8.1
  "scales"            # 1.4.0
)

install.packages(packages, repos = "https://cloud.r-project.org")