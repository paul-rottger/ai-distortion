parse_demo_mode <- function(args = commandArgs(trailingOnly = TRUE)) {
  "demo" %in% args
}

get_results_dir <- function(demo_mode, ...) {
  file.path(if (demo_mode) "./demo_results" else "./results", ...)
}

get_figures_dir <- function(demo_mode, ...) {
  file.path(if (demo_mode) "./demo_figures" else "./figures", ...)
}