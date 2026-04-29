parse_demo_mode <- function(args = commandArgs(trailingOnly = TRUE)) {
  "demo" %in% args
}

get_results_dir <- function(demo_mode, ...) {
  file.path(if (demo_mode) "./demo_results" else "./results", ...)
}

get_figures_dir <- function(demo_mode, ...) {
  file.path(if (demo_mode) "./demo_figures" else "./figures", ...)
}

get_results_input_dir <- function(demo_mode, ...) {
  if (demo_mode) {
    demo_path <- get_results_dir(TRUE, ...)
    if (dir.exists(demo_path)) return(demo_path)
  }
  get_results_dir(FALSE, ...)
}