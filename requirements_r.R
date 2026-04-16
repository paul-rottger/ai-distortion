# R package requirements for 8-human-studies-analysis
# Install with: Rscript requirements_r.R

packages <- c(
  "lme4",
  "mclogit",
  "Matrix",
  "ordinal",
  "marginaleffects",
  "broom.mixed",
  "glmmTMB",
  "lubridate",
  "forcats",
  "stringr",
  "dplyr",
  "purrr",
  "readr",
  "tidyr",
  "tibble",
  "ggplot2",
  "tidyverse",
  "effsize",
  "scales"
)

install.packages(packages, repos = "https://cloud.r-project.org")