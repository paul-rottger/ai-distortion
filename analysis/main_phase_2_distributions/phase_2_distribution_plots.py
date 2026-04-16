#!/usr/bin/env python3

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
from variable_definitions import SCALE_ATTRIBUTES, CATEGORICAL_VARS, CATEGORICAL_LEVELS  # noqa: E402

# Plot configuration
ANNOTATIONS_PATH = os.path.join(REPO_ROOT, "data", "main_phase_2", "annotations.csv")
PREFERENCES_PATH = os.path.join(REPO_ROOT, "data", "main_phase_1", "proposition_responses.csv")
OUTPUT_BASE_DIR = os.path.join(REPO_ROOT, "figures", "main_phase_2_distributions")

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


# =============================================================================
# ANALYSIS
# =============================================================================

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
        data_edited["paragraph_type"] == "model"
    ) & has_edited_in_group
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
    """Draw two overlapping KDE curves with semi-transparent fills."""

    if len(writer_data) < 2 or len(model_data) < 2:
        return False

    x_grid = np.linspace(x_min, x_max, 512)

    writer_kde = gaussian_kde(writer_data)
    model_kde = gaussian_kde(model_data)

    writer_y = writer_kde(x_grid)
    model_y = model_kde(x_grid)

    ax.fill_between(
        x_grid,
        0,
        writer_y,
        color=colors["writer"],
        alpha=0.3,
        zorder=1,
    )
    ax.fill_between(
        x_grid,
        0,
        model_y,
        color=colors["model"],
        alpha=0.3,
        zorder=1,
    )

    ax.plot(
        x_grid,
        writer_y,
        color=colors["writer"],
        linewidth=1.5,
        label="Writer",
        zorder=2,
    )
    ax.plot(
        x_grid, model_y, color=colors["model"], linewidth=1.5, label="Model", zorder=2
    )

    return True


def add_kde_mean_annotations(ax, writer_values, model_values, y_max):
    """Add writer/model mean lines and labels at a common height."""

    if len(writer_values) == 0 or len(model_values) == 0:
        return

    writer_mean = float(np.mean(writer_values))
    model_mean = float(np.mean(model_values))
    label_y = y_max * 0.92
    x_offset = 1.2

    ax.axvline(writer_mean, color=colors["writer"], linewidth=1.4, linestyle="-")
    ax.axvline(model_mean, color=colors["model"], linewidth=1.4, linestyle="-")

    left_is_writer = writer_mean <= model_mean

    writer_x = max(writer_mean - x_offset, 0) if left_is_writer else min(writer_mean + x_offset, 100)
    writer_ha = "right" if left_is_writer else "left"
    model_x = max(model_mean - x_offset, 0) if not left_is_writer else min(model_mean + x_offset, 100)
    model_ha = "right" if not left_is_writer else "left"

    ax.text(
        writer_x,
        label_y,
        f"{writer_mean:.1f}",
        color=colors["writer"],
        fontsize=10,
        ha=writer_ha,
        va="center",
    )
    ax.text(
        model_x,
        label_y,
        f"{model_mean:.1f}",
        color=colors["model"],
        fontsize=10,
        ha=model_ha,
        va="center",
    )


def add_bar_count_labels(ax, plot_counts, variable, order, min_padding=0.8, fontsize=9):
    """Annotate grouped percentage bars with raw observation counts."""

    value_lookup = {}
    for paragraph_type in ["writer", "model"]:
        for category in order:
            match = plot_counts[
                (plot_counts["paragraph_type_"] == paragraph_type)
                & (plot_counts[variable] == category)
            ]
            count = int(match["count"].iloc[0]) if not match.empty else 0
            value_lookup[(paragraph_type, category)] = count

    containers = list(ax.containers)
    hue_order = ["writer", "model"]

    for paragraph_type, container in zip(hue_order, containers):
        labels = []
        for category, patch in zip(order, container.patches):
            if np.isnan(patch.get_height()):
                labels.append("")
                continue
            count = value_lookup[(paragraph_type, category)]
            labels.append(f"{count}")

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=fontsize,
        )

    ymax = ax.get_ylim()[1]
    max_percent = float(plot_counts["percent"].max()) if not plot_counts.empty else 0.0
    ax.set_ylim(0, max(ymax, max_percent + min_padding + 3.0))


def apply_shortened_category_tick_labels(ax, variable, order):
    """Apply abbreviated category labels to categorical barplots."""

    display_labels = [_abbreviate_category_label(variable, category) for category in order]
    ax.set_xticklabels(display_labels)


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


def create_categorical_barplot(df, variable, output_path):
    """Create side-by-side grouped percentage barplot by paragraph type."""

    plot_data = df[["paragraph_type_", variable]].dropna().copy()
    if len(plot_data) == 0:
        print(f"Warning: No data available for {variable}")
        return

    order = _categorical_order(plot_data, variable)
    if len(order) == 0:
        print(f"Warning: No valid categories for {variable}")
        return

    counts = (
        plot_data.groupby(["paragraph_type_", variable], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("paragraph_type_", observed=True)["count"].transform("sum")
    counts["percent"] = (counts["count"] / totals) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axisbelow(True)
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
    add_bar_count_labels(ax, counts, variable, order)
    apply_shortened_category_tick_labels(ax, variable, order)

    ax.set_title(
        f"Categorical Distribution: {variable.replace('_', ' ').title()}\nWriter vs AI Model Generated Paragraphs",
        fontsize=14,
        pad=16,
    )
    ax.set_xlabel(variable.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Percentage within paragraph type (%)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(title="Paragraph type", frameon=True)

    if len(order) > 4:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved categorical barplot for {variable}")


def create_combined_categorical_grid(df, output_dir):
    """Create one combined subplot figure for all categorical variables."""

    n_vars = len(CATEGORICAL_VARS)
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.8 * n_rows))
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
        totals = counts.groupby("paragraph_type_", observed=True)["count"].transform(
            "sum"
        )
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
        add_bar_count_labels(ax, counts, variable, order, min_padding=0.5, fontsize=8)
        apply_shortened_category_tick_labels(ax, variable, order)
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

    fig.text(
        0.02,
        0.5,
        "% within paragraph type",
        va="center",
        rotation="vertical",
        fontsize=13,
    )

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
    n_cols = 2
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
            # Deduplicate while preserving order
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


# ===== FUNCTION TO CREATE KDE PLOT =====
def create_kde_plot(df, variable, output_path):
    """Create KDE plot for a single variable comparing writer vs model paragraphs."""

    # Filter data for the variable and remove missing values
    plot_data = df[["paragraph_type_", variable]].dropna()

    if len(plot_data) == 0:
        print(f"Warning: No data available for {variable}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][
        variable
    ].to_numpy()
    model_data = plot_data[plot_data["paragraph_type_"] == "model"][variable].to_numpy()

    drew_custom = draw_kde_with_overlap_fill(ax, writer_data, model_data)
    if not drew_custom:
        # Fallback if a group has too few points
        for ptype in ["writer", "model"]:
            data = plot_data[plot_data["paragraph_type_"] == ptype][variable]
            if len(data) > 0:
                sns.kdeplot(
                    data=data,
                    label=f"{ptype.capitalize()}",
                    color=colors[ptype],
                    linewidth=1.5,
                    fill=True,
                    alpha=0.3,
                    ax=ax,
                )

    # Customize the plot
    ax.set_xlabel(f"{variable.replace('_', ' ').title()} (0-100 scale)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Distribution of {variable.replace('_', ' ').title()} Ratings\nWriter vs AI Model Generated Paragraphs",
        fontsize=14,
        pad=20,
    )

    # Add legend
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Set x-axis limits to 0-100 for scale variables
    ax.set_xlim(0, 100)
    ax.set_ylim(KDE_Y_MIN, KDE_Y_MAX)
    add_kde_mean_annotations(ax, writer_data, model_data, KDE_Y_MAX)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved KDE plot for {variable}")


# ===== FUNCTION TO CREATE COMBINED GRID PLOT =====
def get_scale_attributes_sorted_by_ame(data_split):
    """Return scale attributes sorted by descending AME from split-specific distortion results."""

    results_dir = os.path.join("results", "main_phase_2_distortion", data_split)
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

    ordered_scale_attributes = SCALE_ATTRIBUTES

    # Calculate grid dimensions (try to make roughly square)
    n_vars = len(ordered_scale_attributes)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 22), sharey="row")
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
            ax.set_title(
                f"{variable.replace('_', ' ').title()}",
                fontsize=14,
                pad=10,
            )
            continue

        # Create KDE plots for each paragraph type
        writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][
            variable
        ].to_numpy()
        model_data = plot_data[plot_data["paragraph_type_"] == "model"][
            variable
        ].to_numpy()

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
                        fill=True,
                        alpha=0.3,
                        ax=ax,
                    )

        # Customize subplot
        ax.set_xlabel("")  # Remove individual x-labels for cleaner look
        ax.set_ylabel("")  # Remove individual y-labels for cleaner look
        ax.set_title(
            f"{variable.replace('_', ' ').title()}",
            fontsize=18,
            pad=10,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(KDE_Y_MIN, KDE_Y_MAX)
        add_kde_mean_annotations(ax, writer_data, model_data, KDE_Y_MAX)
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

    # Keep raw KDE densities and use a common x-axis density limit across all variables.
    writer_y = np.minimum(writer_density, VIOLIN_DENSITY_MAX)
    model_y = np.minimum(model_density, VIOLIN_DENSITY_MAX)

    writer_fill = mcolors.to_rgba(colors["writer"], alpha=0.45)
    model_fill = mcolors.to_rgba(colors["model"], alpha=0.45)

    # Fill only, no outlines
    ax.fill_between(x_grid, -writer_y, 0, color=writer_fill, linewidth=0)
    ax.fill_between(x_grid, 0, model_y, color=model_fill, linewidth=0)

    writer_mean = float(np.mean(writer_vals))
    model_mean = float(np.mean(model_vals))

    # Solid vertical mean lines
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

    # Center divider
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

        # Force identical limits across all violin subplots after axis flip
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


def _significance_stars(p_value):
    """Return significance stars for a p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def report_bonferroni_corrected_scale_pvalues(output_dir, data_split):
    """Load split-specific test p-values for scale + categorical vars and print Bonferroni-corrected results."""

    results_dir = os.path.join("results", "main_phase_2_distribution", data_split)
    all_variables = SCALE_ATTRIBUTES + CATEGORICAL_VARS
    rows = []
    missing_files = []
    missing_terms = []

    for variable in all_variables:
        file_path = os.path.join(results_dir, f"{variable}_by_type.csv")

        if not os.path.exists(file_path):
            missing_files.append(variable)
            continue

        res_df = pd.read_csv(file_path)
        p_col = None
        if "p_value" in res_df.columns:
            p_col = "p_value"
        elif "p" in res_df.columns:
            p_col = "p"

        if p_col is None or res_df.empty:
            missing_terms.append(variable)
            continue

        rows.append(
            {
                "variable": variable,
                "variable_type": "scale" if variable in SCALE_ATTRIBUTES else "categorical",
                "p_raw": float(res_df.iloc[0][p_col]),
            }
        )

    print(
        f"\n=== Bonferroni-corrected p-values ({data_split} tests: scale + categorical) ==="
    )

    if not rows:
        print("No usable p-values found.")
        return

    m_tests = len(all_variables)  # full set correction
    pvals_df = pd.DataFrame(rows)
    pvals_df["p_bonferroni"] = np.minimum(pvals_df["p_raw"] * m_tests, 1.0)
    pvals_df["sig"] = pvals_df["p_bonferroni"].apply(_significance_stars)

    pvals_df["variable"] = pd.Categorical(
        pvals_df["variable"], categories=all_variables, ordered=True
    )
    pvals_df = pvals_df.sort_values("variable")

    print(f"Correction factor (m): {m_tests}")
    print("Significance: * < 0.05, ** < 0.01, *** < 0.001")
    print("")

    for _, row in pvals_df.iterrows():
        print(
            f"{row['variable']:.<30} corrected={row['p_bonferroni']:.4g} {row['sig']}"
        )

    if missing_files:
        print("\nMissing result files:")
        for variable in missing_files:
            print(f"- {variable}")

    if missing_terms:
        print("\nMissing t-test p-value column (`p_value`/`p`) in results file:")
        for variable in missing_terms:
            print(f"- {variable}")

    save_path = os.path.join(output_dir, f"{data_split}_all_bonferroni_ttest_pvalues.csv")
    pvals_df.to_csv(save_path, index=False)
    print(f"\nSaved Bonferroni p-value table: {save_path}")


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

        print("\nCreating individual KDE plots...")
        for variable in SCALE_ATTRIBUTES:
            output_path = os.path.join(output_dir, f"kde_{variable}.pdf")
            create_kde_plot(split_df, variable, output_path)

        print("\nCreating combined KDE grid plot...")
        create_combined_kde_grid(split_df, output_dir, data_split)

        print("\nCreating combined violin grid plot...")
        create_combined_violin_grid(split_df, output_dir, data_split)

        print("\nPreparing categorical data...")
        categorical_df = prepare_categorical_data(split_df)

        print("\nCreating categorical barplots...")
        for variable in CATEGORICAL_VARS:
            output_path = os.path.join(output_dir, f"barplot_{variable}.pdf")
            create_categorical_barplot(categorical_df, variable, output_path)

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

        report_bonferroni_corrected_scale_pvalues(output_dir, data_split)


if __name__ == "__main__":
    main()
