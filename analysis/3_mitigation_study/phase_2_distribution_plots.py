#!/usr/bin/env python3

# =============================================================================
# MITIGATION STUDY - PHASE 2 VISUALIZATION: DISTRIBUTIONS
#
# Builds distribution plots comparing writer and model annotations by mitigation split.
#
# - Loads followup mitigation phase-2 annotations and phase-1 preferences.
# - Applies preprocessing aligned with distribution analysis conventions.
# - Exports split-specific figures to `figures/followup_mitigation_phase_2_distributions/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import os
import numpy as np
import matplotlib.colors as mcolors
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
os.chdir(REPO_ROOT)

# Internal imports
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "utils_py"))
from variable_definitions import SCALE_ATTRIBUTES, CATEGORICAL_VARS, CATEGORICAL_LEVELS

# Plot configuration
ANNOTATIONS_PATH = os.path.join(REPO_ROOT, "data", "followup_mitigation_phase_2", "annotations.csv")
PREFERENCES_PATH = os.path.join(REPO_ROOT, "data", "followup_mitigation_phase_1", "proposition_responses.csv")
OUTPUT_BASE_DIR = os.path.join(REPO_ROOT, "figures", "followup_mitigation_phase_2_distributions")

# Set plotting parameters
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.figsize"] = [10, 6]

# Color palette for writer vs model
colors = {
    "writer": "#0070C0",  # Blue
    "model": "#7030A0",  # Purple
}
ALL_SPLITS = ["unedited", "edited", "preferred"]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_phase_2_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations_df = pd.read_csv(ANNOTATIONS_PATH)
    phase_1_preferences_df = pd.read_csv(PREFERENCES_PATH)
    return annotations_df, phase_1_preferences_df


# =============================================================================
# ANALYSIS
# =============================================================================


def prepare_annotations_data(df):
    """Apply shared preprocessing used by the distribution analysis scripts."""

    data = df.copy()

    data["paragraph_type_"] = pd.Categorical(data["paragraph_type"])

    for variable, levels in CATEGORICAL_LEVELS.items():
        if variable in data.columns:
            data[variable] = pd.Categorical(data[variable], categories=levels, ordered=True)

    # Match the R analysis scripts.
    data = data[data["writer_education"] != "Other"].copy()
    return data


def replace_model_with_edited(df):
    """R-equivalent preprocessing:
    - within each (writer_id, proposition_id), if any `edited` exists,
      drop `model` rows
    - relabel remaining `edited` rows to `model`
    - create ordered `paragraph_type_` with `writer` as reference
    """

    data_edited = df.copy()

    has_edited_in_group = data_edited.groupby(["writer_id", "proposition_id"])[
        "paragraph_type"
    ].transform(lambda s: (s == "edited").any())

    drop_model_when_edited_exists = (
        (data_edited["paragraph_type"] == "model") & has_edited_in_group
    )
    dropped_model_rows = int(drop_model_when_edited_exists.sum())

    data_edited = data_edited.loc[~drop_model_when_edited_exists].copy()
    data_edited.loc[data_edited["paragraph_type"] == "edited", "paragraph_type"] = (
        "model"
    )

    data_edited["paragraph_type_"] = pd.Categorical(
        data_edited["paragraph_type"],
        categories=["writer", "model"],
        ordered=True,
    )

    # Keep only writer/model rows used for final comparison
    data_edited = data_edited[data_edited["paragraph_type_"].notna()].copy()

    return data_edited, dropped_model_rows


def build_split_datasets(annotations, preferences):
    """Build unedited, edited, and preferred datasets to match the R analysis."""

    processed = prepare_annotations_data(annotations)

    data_unedited = processed[processed["paragraph_type"].isin(["writer", "model"])].copy()
    data_unedited["paragraph_type_"] = pd.Categorical(
        data_unedited["paragraph_type"], categories=["writer", "model"], ordered=True
    )

    data_edited, dropped_model_rows = replace_model_with_edited(processed)

    preferred_exclusions = preferences[
        preferences["writer_preference"] == "original"
    ][["writer_id", "proposition_id"]].drop_duplicates()

    data_preferred = data_edited.merge(
        preferred_exclusions.assign(_exclude=True),
        on=["writer_id", "proposition_id"],
        how="left",
    )
    data_preferred = data_preferred[data_preferred["_exclude"].isna()].drop(columns="_exclude")

    return {
        "unedited": data_unedited,
        "edited": data_edited,
        "preferred": data_preferred,
    }, dropped_model_rows


# =============================================================================
# OUTPUTS
# =============================================================================


def get_split_output_dir(split_name):
    split_output_dir = os.path.join(OUTPUT_BASE_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    return split_output_dir

# ===== KDE AXIS STANDARDIZATION =====
KDE_Y_MIN = 0.0
KDE_Y_MAX = 0.05

# ===== VIOLIN AXIS STANDARDIZATION =====
VIOLIN_DENSITY_MAX = 0.05


def draw_kde_with_overlap_fill(ax, writer_data, model_data, x_min=0, x_max=100):
    """Draw two KDE curves with:
    - white fill where densities overlap
    - transparent line-color fill where only one curve is above the other
    """

    if len(writer_data) < 2 or len(model_data) < 2:
        return False

    x_grid = np.linspace(x_min, x_max, 512)

    writer_kde = gaussian_kde(writer_data)
    model_kde = gaussian_kde(model_data)

    writer_y = writer_kde(x_grid)
    model_y = model_kde(x_grid)

    overlap_y = np.minimum(writer_y, model_y)

    # White overlap area (base)
    ax.fill_between(x_grid, 0, overlap_y, color="white", alpha=1.0, zorder=1)

    # Non-overlap regions in transparent line colors
    writer_above = writer_y > model_y
    model_above = model_y > writer_y

    ax.fill_between(
        x_grid,
        overlap_y,
        writer_y,
        where=writer_above,
        color=colors["writer"],
        alpha=1,
        interpolate=True,
        zorder=2,
    )
    ax.fill_between(
        x_grid,
        overlap_y,
        model_y,
        where=model_above,
        color=colors["model"],
        alpha=1,
        interpolate=True,
        zorder=2,
    )

    # Outline curves
    ax.plot(x_grid, writer_y, color=colors["writer"], linewidth=1.5, label="Writer", zorder=3)
    ax.plot(x_grid, model_y, color=colors["model"], linewidth=1.5, label="Model", zorder=3)

    return True


def prepare_categorical_data(df):
    """Prepare categorical variables for plotting consistency."""
    cat_df = df.copy()

    # Match analysis convention: drop 'Other' from writer_education
    cat_df = cat_df[cat_df["writer_education"] != "Other"].copy()

    # Set ordered categories for all categorical variables
    for var, levels in CATEGORICAL_LEVELS.items():
        if var in cat_df.columns:
            cat_df[var] = pd.Categorical(cat_df[var], categories=levels, ordered=True)

    return cat_df


def _categorical_order(plot_data, variable):
    if variable in CATEGORICAL_LEVELS:
        return CATEGORICAL_LEVELS[variable]
    # Sort categories by total frequency (descending) if no predefined order
    counts = plot_data[variable].value_counts(dropna=True)
    return counts.index.tolist()


def _categorical_percent_wide(plot_data, variable, order):
    """Return category-level writer/model percentages in wide format."""
    counts = (
        plot_data.groupby(["paragraph_type_", variable], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("paragraph_type_", observed=True)["count"].transform("sum")
    counts["percent"] = (counts["count"] / totals) * 100

    wide = (
        counts.pivot(index=variable, columns="paragraph_type_", values="percent")
        .reindex(order)
        .fillna(0)
    )

    if "writer" not in wide.columns:
        wide["writer"] = 0.0
    if "model" not in wide.columns:
        wide["model"] = 0.0

    wide = wide[["writer", "model"]].copy()
    return wide


def create_combined_categorical_grid(df, output_dir):
    """Create one combined subplot figure for all categorical variables."""

    n_vars = len(CATEGORICAL_VARS)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.8 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, variable in enumerate(CATEGORICAL_VARS):
        ax = axes[i]
        plot_data = df[["paragraph_type_", variable]].dropna().copy()

        if len(plot_data) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {variable}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(variable.replace("_", " ").title(), fontsize=11)
            continue

        order = _categorical_order(plot_data, variable)
        counts = (
            plot_data.groupby(["paragraph_type_", variable], observed=True)
            .size()
            .reset_index(name="count")
        )
        totals = counts.groupby("paragraph_type_", observed=True)["count"].transform("sum")
        counts["percent"] = (counts["count"] / totals) * 100

        sns.barplot(
            data=counts,
            x=variable,
            y="percent",
            hue="paragraph_type_",
            order=order,
            hue_order=["writer", "model"],
            palette=[colors["writer"], colors["model"]],
            dodge=True,
            ax=ax,
        )
        ax.set_axisbelow(True)

        ax.set_title(f"{variable.replace('_', ' ').title()}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.2, linestyle="-", linewidth=0.5)

        if len(order) > 4:
            ax.tick_params(axis="x", rotation=30)

        # Keep single legend, hide duplicates
        if i == 0:
            ax.legend(title="Paragraph type", frameon=True, fontsize=9)
        else:
            if ax.legend_ is not None:
                ax.legend_.remove()

    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    fig.text(0.5, 0.02, "Category", ha="center", fontsize=13)
    fig.text(0.02, 0.5, "Percentage within paragraph type (%)", va="center", rotation="vertical", fontsize=13)

    plt.tight_layout(rect=[0.03, 0.03, 0.99, 0.96])
    combined_path = os.path.join(output_dir, "barplot_all_categorical_variables_combined.pdf")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined categorical barplot: {combined_path}")


def _abbreviate_category_label(variable, label):
    """Abbreviate long categorical labels for compact heatmap display."""
    if pd.isna(label):
        return ""

    label = str(label)

    abbreviations = {
        "writer_income": {
            "Under £15,000": "<£15k",
            "£15,000-£24,999": "£15–£24k",
            "£25,000-£34,999": "£25–£34k",
            "£35,000-£49,999": "£35–£49k",
            "£50,000-£74,999": "£50–£74k",
            "£75,000-£99,999": "£75–£99k",
            "£100,000+": "£100k+",
        },
        "writer_age_binned": {
            "18-29": "18–29",
            "30-39": "30–39",
            "40-49": "40–49",
            "50-59": "50–59",
            "60-69": "60–69",
            "70+": "70+",
        },
        "writer_english_first": {
            "Yes": "Yes",
            "No": "No",
        },
        "writer_english_skills": {
            "Basic": "Basic",
            "Intermediate": "Intermediate",
            "Advanced": "Advanced",
            "Expert": "Expert",
        },
        "writer_education": {
            "GCSEs or equivalent": "GCSE",
            "A-levels or equivalent": "A-level",
            "Vocational qualification": "Vocational",
            "Undergraduate degree": "Undergrad",
            "Postgraduate degree (Master's)": "Master's",
            "Doctorate (PhD)": "PhD",
        },
        "writer_politicalIdeology": {
            "Very Left-Wing": "V. Left",
            "Moderately Left-Wing": "Mod. Left",
            "Centrist": "Center",
            "Moderately Right-Wing": "Mod. Right",
            "Very Right-Wing": "V. Right",
            "Other": "Other",
        },
        "writer_politicalParty": {
            "Conservative": "Consv.",
            "Labour": "Labour",
            "Liberal Democrats": "Lib Dems",
            "Greens": "Greens",
            "Scottish National Party": "SNP",
            "Did not vote": "Did not vote",
            "Not eligible to vote": "Cannot vote",
            "Other": "Other",
        },
    }

    return abbreviations.get(variable, {}).get(label, label)


def _two_line_label(label):
    """Split a label across two lines when possible to save horizontal space."""
    text = str(label)
    parts = text.split()

    if len(parts) <= 1:
        return text

    # Split near midpoint by word count
    split_idx = len(parts) // 2
    return " ".join(parts[:split_idx]) + "\n" + " ".join(parts[split_idx:])


def _blend_with_white(hex_color, intensity):
    """Blend a base color with white by intensity in [0,1]."""
    base = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    intensity = np.clip(float(intensity), 0.0, 1.0)
    mixed = white * (1.0 - intensity) + base * intensity
    return tuple(mixed)


def create_combined_categorical_heatmap(df, output_dir):
    """Create a combined heatmap-style plot for all categorical variables."""
    n_vars = len(CATEGORICAL_VARS)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 8))
    fig.patch.set_facecolor("#EBEBEB")
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, variable in enumerate(CATEGORICAL_VARS):
        ax = axes[i]
        ax.set_facecolor("#EBEBEB")
        plot_data = df[["paragraph_type_", variable]].dropna().copy()

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f"No data for {variable}", ha="center", va="center")
            ax.axis("off")
            continue

        order = _categorical_order(plot_data, variable)

        # Heatmap-only collapse for political party voting non-participation categories
        if variable == "writer_politicalParty":
            collapsed_label = "Did not vote"
            plot_data[variable] = (
                plot_data[variable]
                .astype(str)
                .replace(
                    {
                        "Did not vote": collapsed_label,
                        "Not eligible to vote": collapsed_label,
                    }
                )
            )

            order = [
                collapsed_label
                if cat in ["Did not vote", "Not eligible to vote"]
                else cat
                for cat in order
            ]
            order = list(dict.fromkeys(order))

        wide = _categorical_percent_wide(plot_data, variable, order)

        writer_vals = wide["writer"].to_numpy()
        model_vals = wide["model"].to_numpy()

        overall_max = max(writer_vals.max(), model_vals.max(), 1)

        for j, category in enumerate(order):
            w_pct = writer_vals[j]
            m_pct = model_vals[j]

            w_int = w_pct / overall_max
            m_int = m_pct / overall_max

            w_color = _blend_with_white(colors["writer"], w_int)
            m_color = _blend_with_white(colors["model"], m_int)

            # Model row (top)
            ax.add_patch(
                plt.Rectangle(
                    (j, 1),
                    1,
                    1,
                    facecolor=m_color,
                    edgecolor="#EBEBEB",
                    linewidth=2,
                )
            )
            # Writer row (bottom)
            ax.add_patch(
                plt.Rectangle(
                    (j, 0),
                    1,
                    1,
                    facecolor=w_color,
                    edgecolor="#EBEBEB",
                    linewidth=2,
                )
            )

        display_labels = [_abbreviate_category_label(variable, cat) for cat in order]
        ax.set_xlim(0, len(order))
        ax.set_ylim(0, 2)
        ax.set_xticks(np.arange(len(order)) + 0.5)
        ax.set_xticklabels(display_labels, fontsize=12, linespacing=0.9)
        ax.set_yticks([])

        ax.set_title(
            variable.replace("_", " ").title(),
            fontsize=18,
            fontweight="bold",
            pad=14,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="x", length=0)

    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0.04, 0.02, 0.99, 0.98], h_pad=2.8, w_pad=2.1)

    heatmap_path = os.path.join(output_dir, "heatmap_all_categorical_variables_combined.pdf")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined categorical heatmap: {heatmap_path}")


# ===== FUNCTION TO CREATE COMBINED GRID PLOT =====
def get_scale_attributes_sorted_by_cohens_d(data_split):
    """Return scale attributes sorted by descending Cohen's d from split-specific by_type results."""

    results_dir = os.path.join("results", "followup_mitigation_phase_2_distribution", data_split)
    d_rows = []

    for variable in SCALE_ATTRIBUTES:
        file_path = os.path.join(results_dir, f"{variable}_by_type.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if "cohens_d" not in df.columns:
            continue

        d_rows.append(
            {
                "variable": variable,
                "cohens_d": float(df.iloc[0]["cohens_d"]),
            }
        )

    if not d_rows:
        return SCALE_ATTRIBUTES, {}

    d_df = pd.DataFrame(d_rows).sort_values("cohens_d", ascending=False)
    ordered = d_df["variable"].tolist()
    d_lookup = dict(zip(d_df["variable"], d_df["cohens_d"]))

    remaining = [v for v in SCALE_ATTRIBUTES if v not in ordered]
    return ordered + remaining, d_lookup


def get_scale_attributes_sorted_by_ame(data_split):
    """Return scale attributes sorted by descending AME from split-specific distortion results."""

    results_dir = os.path.join("results", "followup_mitigation_phase_2_distortion", data_split)
    ame_rows = []

    for variable in SCALE_ATTRIBUTES:
        file_path = os.path.join(results_dir, f"{variable}_by_type.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if "ame" not in df.columns:
            continue

        ame_rows.append(
            {
                "variable": variable,
                "ame": float(df.iloc[0]["ame"]),
            }
        )

    if not ame_rows:
        return SCALE_ATTRIBUTES, {}

    ame_df = pd.DataFrame(ame_rows).sort_values("ame", ascending=False)
    ordered = ame_df["variable"].tolist()
    ame_lookup = dict(zip(ame_df["variable"], ame_df["ame"]))

    # Keep any missing variables at the end in original order
    remaining = [v for v in SCALE_ATTRIBUTES if v not in ordered]
    return ordered + remaining, ame_lookup


def create_combined_kde_grid(df, output_dir, data_split):
    """Create a grid of KDE plots for all scale variables."""

    ordered_scale_attributes, d_lookup = get_scale_attributes_sorted_by_cohens_d(data_split)

    # Calculate grid dimensions (try to make roughly square)
    n_vars = len(ordered_scale_attributes)
    n_cols = int(np.ceil(np.sqrt(n_vars)))
    n_rows = int(np.ceil(n_vars / n_cols))

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 14), sharey="row")
    axes = axes.flatten() if n_vars > 1 else [axes]

    # Create KDE plot for each variable
    for i, variable in enumerate(ordered_scale_attributes):
        ax = axes[i]

        # Filter data for the variable and remove missing values
        plot_data = df[["paragraph_type_", variable]].dropna()

        if len(plot_data) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {variable}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            d_value = d_lookup.get(variable)
            d_label = f"Cohen's d: {d_value:.2f}" if d_value is not None else "Cohen's d: N/A"
            ax.set_title(
                f"{variable.replace('_', ' ').title()} ({d_label})",
                fontsize=14,
                pad=10,
            )
            continue

        # Create KDE plots for each paragraph type
        writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][variable].to_numpy()
        model_data = plot_data[plot_data["paragraph_type_"] == "model"][variable].to_numpy()

        drew_custom = draw_kde_with_overlap_fill(ax, writer_data, model_data)
        if not drew_custom:
            for ptype in ["writer", "model"]:
                data = plot_data[plot_data["paragraph_type_"] == ptype][variable]
                if len(data) > 0:
                    sns.kdeplot(
                        data=data,
                        label=f"{ptype.capitalize()}",
                        color=colors[ptype],
                        linewidth=1,
                        ax=ax,
                    )

        # Customize subplot
        ax.set_xlabel("")  # Remove individual x-labels for cleaner look
        ax.set_ylabel("")  # Remove individual y-labels for cleaner look
        d_value = d_lookup.get(variable)
        d_label = f"Cohen's d: {d_value:.2f}" if d_value is not None else "Cohen's d: N/A"
        ax.set_title(
            f"{variable.replace('_', ' ').title()} ({d_label})",
            fontsize=18,
            pad=10,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(KDE_Y_MIN, KDE_Y_MAX)
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=12)

        # Show y-axis tick labels only on first column (shared within each row)
        col_idx = i % n_cols
        if col_idx != 0:
            ax.tick_params(axis="y", labelleft=False)

        # Remove all legends
        if ax.legend_ is not None:
            ax.legend_.remove()


    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95], h_pad=2.0)

    # Save combined plot
    combined_path = os.path.join(output_dir, "kde_all_scale_variables_combined.pdf")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined KDE plot: {combined_path}")


def draw_split_violin_with_means(ax, writer_data, model_data, x_min=0, x_max=100):
    """Draw split violins from KDE densities around y=0 with group mean lines."""

    writer_vals = np.asarray(writer_data, dtype=float)
    model_vals = np.asarray(model_data, dtype=float)

    if len(writer_vals) < 2 or len(model_vals) < 2:
        return False

    x_grid = np.linspace(x_min, x_max, 512)

    writer_kde = gaussian_kde(writer_vals)
    model_kde = gaussian_kde(model_vals)

    writer_density = writer_kde(x_grid)
    model_density = model_kde(x_grid)

    writer_y = np.minimum(writer_density, VIOLIN_DENSITY_MAX)
    model_y = np.minimum(model_density, VIOLIN_DENSITY_MAX)

    writer_fill = mcolors.to_rgba(colors["writer"], alpha=0.45)
    model_fill = mcolors.to_rgba(colors["model"], alpha=0.45)

    ax.fill_between(x_grid, -writer_y, 0, color=writer_fill, linewidth=0)
    ax.fill_between(x_grid, 0, model_y, color=model_fill, linewidth=0)

    writer_mean = float(np.mean(writer_vals))
    model_mean = float(np.mean(model_vals))

    ax.vlines(
        writer_mean,
        ymin=-VIOLIN_DENSITY_MAX,
        ymax=0,
        colors=colors["writer"],
        linewidth=2.2,
    )
    ax.vlines(
        model_mean,
        ymin=0,
        ymax=VIOLIN_DENSITY_MAX,
        colors=colors["model"],
        linewidth=2.2,
    )

    ax.axhline(0, color="#666666", linewidth=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-VIOLIN_DENSITY_MAX, VIOLIN_DENSITY_MAX)
    return True


def create_combined_violin_grid(df, output_dir, data_split):
    """Create a single-column stack of split violin plots for all scale variables."""

    ordered_scale_attributes, ame_lookup = get_scale_attributes_sorted_by_ame(data_split)

    n_cols = 1
    n_rows = len(ordered_scale_attributes)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5, max(2.2 * n_rows, 12)),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("#EBEBEB")
    axes = axes.flatten()

    for i, variable in enumerate(ordered_scale_attributes):
        ax = axes[i]
        ax.set_facecolor("#EBEBEB")

        plot_data = df[["paragraph_type_", variable]].dropna()

        if len(plot_data) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data\nfor {variable}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ame_value = ame_lookup.get(variable)
            ame_label = f"AME: {ame_value:.2f}" if ame_value is not None else "AME: N/A"
            ax.set_title(
                f"{variable.replace('_', ' ').title()} ({ame_label})",
                fontsize=12,
                pad=10,
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-VIOLIN_DENSITY_MAX, VIOLIN_DENSITY_MAX)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            continue

        writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][
            variable
        ].to_numpy()
        model_data = plot_data[plot_data["paragraph_type_"] == "model"][
            variable
        ].to_numpy()

        drew = draw_split_violin_with_means(
            ax, writer_data, model_data, x_min=0, x_max=100
        )
        if not drew:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(-VIOLIN_DENSITY_MAX, VIOLIN_DENSITY_MAX)

        ame_value = ame_lookup.get(variable)
        ame_label = f"AME: {ame_value:.2f}" if ame_value is not None else "AME: N/A"
        ax.set_title(
            f"{variable.replace('_', ' ').title()} ({ame_label})", fontsize=12, pad=10
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    for i in range(len(ordered_scale_attributes), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0.03, 0.03, 0.99, 0.98], h_pad=1.0)

    combined_path = os.path.join(output_dir, "violin_all_scale_variables_combined.pdf")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined violin plot: {combined_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("Loading phase 2 annotations data...")
    annotations_df, phase_1_preferences = load_phase_2_inputs()
    split_datasets, dropped_model_rows = build_split_datasets(
        annotations_df,
        phase_1_preferences,
    )

    print(f"Loaded {len(annotations_df):,} raw annotations")
    print(f"Dropped {dropped_model_rows:,} model rows where edited rows existed")
    for split_name, split_df in split_datasets.items():
        print(f"Prepared {split_name} split with {len(split_df):,} annotations")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("=== Distribution Plots for Phase 2 Variables ===")
    print(f"Output base directory: {OUTPUT_BASE_DIR}")
    print(f"Processing {len(SCALE_ATTRIBUTES)} scale variables across {len(ALL_SPLITS)} splits...")

    for data_split in ALL_SPLITS:
        split_df = split_datasets[data_split]
        output_dir = get_split_output_dir(data_split)

        print(f"\n=== Processing split: {data_split} ===")
        print(f"Output directory: {output_dir}")

        print("\nCreating combined KDE grid plot...")
        create_combined_kde_grid(split_df, output_dir, data_split)

        print("\nCreating combined violin grid plot...")
        create_combined_violin_grid(split_df, output_dir, data_split)

        print("\nPreparing categorical data...")
        categorical_df = prepare_categorical_data(split_df)

        print("\nCreating combined categorical barplot grid...")
        create_combined_categorical_grid(categorical_df, output_dir)

        print("\nCreating combined categorical heatmap...")
        create_combined_categorical_heatmap(categorical_df, output_dir)

        print("\nGenerating summary statistics...")
        summary_stats = []

        for variable in SCALE_ATTRIBUTES:
            writer_data = split_df[split_df["paragraph_type_"] == "writer"][variable].dropna()
            model_data = split_df[split_df["paragraph_type_"] == "model"][variable].dropna()

            if len(writer_data) > 0 and len(model_data) > 0:
                summary_stats.append(
                    {
                        "variable": variable,
                        "writer_n": len(writer_data),
                        "writer_mean": writer_data.mean(),
                        "writer_std": writer_data.std(),
                        "writer_median": writer_data.median(),
                        "model_n": len(model_data),
                        "model_mean": model_data.mean(),
                        "model_std": model_data.std(),
                        "model_median": model_data.median(),
                        "mean_diff": model_data.mean() - writer_data.mean(),
                    }
                )

        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(output_dir, "kde_summary_statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary statistics: {summary_path}")

        print("\n=== Summary ===")
        print(f"Generated KDE plots for {len(SCALE_ATTRIBUTES)} scale variables")
        print(f"Total annotations processed: {len(split_df):,}")
        print(
            f"Writer paragraphs: {len(split_df[split_df['paragraph_type_'] == 'writer']):,}"
        )
        print(
            f"Model paragraphs: {len(split_df[split_df['paragraph_type_'] == 'model']):,}"
        )
        print(f"Figures saved to: {output_dir}/")

        print("\n=== Top 5 Largest Mean Differences (Model - Writer) ===")
        top_diffs = summary_df.nlargest(5, "mean_diff")[["variable", "mean_diff"]]
        for _, row in top_diffs.iterrows():
            print(f"{row['variable']:.<30} {row['mean_diff']:+6.2f}")


if __name__ == "__main__":
    main()
