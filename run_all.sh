#!/bin/bash
# =============================================================================
# Run All Analysis Scripts
# =============================================================================

# ==========================================
# CONFIGURATION
# ==========================================

set -euo pipefail # Exit on error, undefined variable, or pipeline failure

# Parse arguments
RUN_REGRESSIONS=false
DEBUG_ARGS=()

for arg in "$@"; do
    case $arg in
        --run_regressions)
			RUN_REGRESSIONS=true
            echo "Will run regressions..."
            ;;
		--debug)
			DEBUG_ARGS=("debug")
			echo "Will run supported R regression scripts in debug mode..."
			;;
		*)
			echo "Unknown argument: $arg" >&2
			exit 1
			;;
    esac
done

# Determine repository root and paths to executables
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$REPO_ROOT/env/bin/python"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
run_regressions=true

# Check for Python executable
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python executable not found at $PYTHON_BIN"
	exit 1
fi

# Check for Rscript executable
if ! command -v "$RSCRIPT_BIN" >/dev/null 2>&1; then
	echo "Rscript executable not found: $RSCRIPT_BIN"
	exit 1
fi

# Change to repository root
cd "$REPO_ROOT"

# Helper functions to run Python scripts
run_python() {
	local script_path="$1"
	echo ">>> Running $script_path"
	"$PYTHON_BIN" "$script_path"
	echo
}

# Helper functions to run R scripts
run_r() {
	local script_path="$1"
	shift || true
	echo ">>> Running $script_path"
	"$RSCRIPT_BIN" "$script_path" "$@"
	echo
}

# Helper function to print section headers
print_section() {
	local title="$1"
	echo "============================================================"
	echo "$title"
	echo "============================================================"
}


# ==========================================
# EXECUTE ANALYSES
# ==========================================

# ==========================
print_section "REPOSITORY-LEVEL DATA PREP AND OVERVIEWS"
# ==========================
run_python "analysis/data_aggregation.py"
run_python "analysis/participant_overview.py"
run_r "analysis/study_overview.R"

# ==========================
print_section "MAIN STUDY: PHASE 1 (WRITING)"
# ==========================
run_python "analysis/1_main_study/phase_1_paragraph_edits.py"
run_python "analysis/1_main_study/phase_1_writer_engagement.py"
run_python "analysis/1_main_study/phase_1_distortion_tolerance.py"
run_r "analysis/1_main_study/phase_1_paragraph_preference.R"

# ==========================
print_section "MAIN STUDY: PHASE 2 (RATING - DISTRIBUTIONS)"
# ==========================
run_r "analysis/1_main_study/phase_2_distribution_variables.R"
run_r "analysis/1_main_study/phase_2_homogenisation.R"
run_python "analysis/1_main_study/phase_2_distribution_plots.py"

# ==========================
print_section "MAIN STUDY: PHASE 2 (RATING - DISTORTIONS)"
# ==========================
if [[ "$RUN_REGRESSIONS" == true ]]; then
	run_r "analysis/1_main_study/phase_2_distortion_scale_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/1_main_study/phase_2_distortion_ordinal_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/1_main_study/phase_2_distortion_nominal_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/1_main_study/phase_2_distortion_by_proposition_leaning.R" "${DEBUG_ARGS[@]}"
fi
run_python "analysis/1_main_study/phase_2_distortion_by_model.py"
run_python "analysis/1_main_study/phase_2_distortion_by_input_condition.py"
run_python "analysis/1_main_study/phase_2_distortion_plots.py"

# ==========================
print_section "DISCLAIMER STUDY: PHASE 1 (WRITING)"
# ==========================
run_python "analysis/2_disclaimer_study/phase_1_distortion_tolerance.py"
run_r "analysis/2_disclaimer_study/phase_1_paragraph_preference.R"

# ==========================
print_section "MITIGATION STUDY: PHASE 1 (WRITING)"
# ==========================
run_r "analysis/3_mitigation_study/phase_1_paragraph_preference.R" "${DEBUG_ARGS[@]}"

# ==========================
print_section "MITIGATION STUDY: PHASE 2 (RATING - DISTORTIONS)"
# ==========================
if [[ "$RUN_REGRESSIONS" == true ]]; then
	run_r "analysis/3_mitigation_study/phase_2_distortion_scale_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/3_mitigation_study/phase_2_distortion_ordinal_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/3_mitigation_study/phase_2_distortion_nominal_variables.R" "${DEBUG_ARGS[@]}"
	run_r "analysis/3_mitigation_study/phase_2_distortion_side_effects.R"
fi
run_python "analysis/3_mitigation_study/phase_2_distortion_plots.py"
run_python "analysis/3_mitigation_study/phase_2_mitigation_side_effects_plots.py"

# ==========================
print_section "FOLLOWUP MITIGATION: PHASE 2 (RATING - DISTRIBUTIONS)"
# ==========================
run_r "analysis/3_mitigation_study/phase_2_distribution_variables.R"
run_python "analysis/3_mitigation_study/phase_2_distribution_plots.py"

print_section "ALL ANALYSES COMPLETED"