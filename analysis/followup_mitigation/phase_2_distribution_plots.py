#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings("ignore")

# ===== SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..")
os.chdir(REPO_ROOT)

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

# ===== DATA LOADING =====
print("Loading phase 2 annotations data...")
annotations_df = pd.read_csv("data/followup_mitigation_phase_2/annotations.csv")

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


annotations_df, dropped_model_rows = replace_model_with_edited(annotations_df)
print(f"Loaded and processed {len(annotations_df):,} annotations")
print(f"Dropped {dropped_model_rows:,} model rows where edited rows existed")

# ===== SCALE VARIABLES DEFINITION =====
SCALE_ATTRIBUTES = [
    # Text quality
    "paragraph_formality",
    "paragraph_clarity",
    "paragraph_informativeness",
    "paragraph_originality",
    "paragraph_relevance",
    # Writer perception
    "writer_knowledge",
    "writer_importance",
    "writer_confidence",
    "writer_stance_polarity",
    # Emotion scales
    "paragraph_hope",
    "paragraph_excitement",
    "paragraph_fear",
    "paragraph_disgust",
    "paragraph_anger",
    # Affect dimensions
    "writer_affect_x",
    "writer_affect_y",
    # Social perception
    "writer_optimism",
    "writer_community",
    "writer_friendliness",
    "writer_openness",
]

# ===== CATEGORICAL VARIABLES DEFINITION =====
CATEGORICAL_VARS = [
    "writer_age_binned",
    "writer_gender",
    "writer_race",
    "writer_education",
    "writer_income",
    "writer_english_first",
    "writer_english_skills",
    "writer_politicalParty",
    "writer_politicalIdeology",
]

CATEGORICAL_LEVELS = {
    "writer_age_binned": ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    "writer_english_first": ["No", "Yes"],
    "writer_english_skills": ["Basic", "Intermediate", "Advanced", "Expert"],
    "writer_education": [
        "GCSEs or equivalent",
        "A-levels or equivalent",
        "Vocational qualification",
        "Undergraduate degree",
        "Postgraduate degree (Master's)",
        "Doctorate (PhD)",
    ],
    "writer_income": [
        "Under £15,000",
        "£15,000-£24,999",
        "£25,000-£34,999",
        "£35,000-£49,999",
        "£50,000-£74,999",
        "£75,000-£99,999",
        "£100,000+",
    ],
    "writer_politicalIdeology": [
        "Very Left-Wing",
        "Moderately Left-Wing",
        "Centrist",
        "Moderately Right-Wing",
        "Very Right-Wing",
    ],
    "writer_politicalParty": [
        "Labour",
        "Conservative",
        "Liberal Democrats",
        "Greens",
        "Reform UK",
        "Scottish National Party",
        "Did not vote",
        "Not eligible to vote",
        "Other"
    ],
    "writer_gender": [
        "Male",
        "Female",
        "Other"
    ],
    "writer_race": [
        "White",
        "Black",
        "Asian",
        "Mixed",
        "Other",
    ],
}

# ===== FIGURE OUTPUT DIRECTORY =====
output_dir = 'figures/followup_mitigation_phase_2_distributions'
os.makedirs(output_dir, exist_ok=True)

categorical_output_dir = os.path.join(output_dir, "categorical")
os.makedirs(categorical_output_dir, exist_ok=True)

# ===== KDE AXIS STANDARDIZATION =====
KDE_Y_MIN = 0.0
KDE_Y_MAX = 0.05


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

    ax.set_title(
        f"Categorical Distribution: {variable.replace('_', ' ').title()}\nWriter vs AI Model Generated Paragraphs",
        fontsize=14,
        pad=16,
    )
    ax.set_xlabel(variable.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Percentage within paragraph type (%)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.5)
    ax.legend(title="Paragraph type", frameon=True)

    if len(order) > 4:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved categorical barplot for {variable}")


def create_combined_categorical_grid(df):
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
    combined_path = os.path.join(categorical_output_dir, "barplot_all_categorical_variables_combined.pdf")
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
            "Very Left-Wing": "Very Left",
            "Moderately Left-Wing": "Mod. Left",
            "Centrist": "Center",
            "Moderately Right-Wing": "Mod. Right",
            "Very Right-Wing": "Very Right",
            "Other": "Other",
        },
        "writer_politicalParty": {
            "Conservative": "Consv.",
            "Labour": "Labour",
            "Liberal Democrats": "Lib Dems",
            "Greens": "Greens",
            "Reform UK": "Reform UK",
            "Scottish National Party": "SNP",
            "Did not vote": "Did not vote",
            "Not eligible to vote": "Cannot vote",
            "Other": "Other",
        },
    }

    return abbreviations.get(variable, {}).get(label, label)


def _blend_with_white(hex_color, intensity):
    """Blend a base color with white by intensity in [0,1]."""
    base = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    intensity = np.clip(float(intensity), 0.0, 1.0)
    mixed = white * (1.0 - intensity) + base * intensity
    return tuple(mixed)


def create_combined_categorical_heatmap(df):
    """Create a combined heatmap-style plot for all categorical variables."""
    n_vars = len(CATEGORICAL_VARS)

    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 1.4 * n_vars + 1.5))
    axes = axes if n_vars > 1 else [axes]

    for i, variable in enumerate(CATEGORICAL_VARS):
        ax = axes[i]
        plot_data = df[["paragraph_type_", variable]].dropna().copy()

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f"No data for {variable}", ha="center", va="center")
            ax.axis("off")
            continue

        order = _categorical_order(plot_data, variable)
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

            # Writer row (top)
            ax.add_patch(
                plt.Rectangle(
                    (j, 1),
                    1,
                    1,
                    facecolor=w_color,
                    edgecolor="white",
                    linewidth=2,
                )
            )
            # Model row (bottom)
            ax.add_patch(
                plt.Rectangle(
                    (j, 0),
                    1,
                    1,
                    facecolor=m_color,
                    edgecolor="white",
                    linewidth=2,
                )
            )

            w_txt = "white" if w_int > 0.45 else "#4c5a70"
            m_txt = "white" if m_int > 0.45 else "#6f4b4b"

            ax.text(j + 0.5, 1.5, f"{w_pct:.0f}%", ha="center", va="center", fontsize=12, color=w_txt, fontweight="bold")
            ax.text(j + 0.5, 0.5, f"{m_pct:.0f}%", ha="center", va="center", fontsize=12, color=m_txt, fontweight="bold")

        display_labels = [_abbreviate_category_label(variable, cat) for cat in order]
        ax.set_xlim(0, len(order))
        ax.set_ylim(0, 2)
        ax.set_xticks(np.arange(len(order)) + 0.5)
        ax.set_xticklabels(display_labels, fontsize=12)
        ax.set_yticks([])

        pretty_name = variable.replace("_", " ").title()
        ax.text(-0.15, 1.0, pretty_name, ha="right", va="center", fontsize=12, fontweight="bold", transform=ax.transData)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="x", length=0)
    
    plt.tight_layout(rect=[0.10, 0.02, 0.99, 0.98], h_pad=1.1)

    heatmap_path = os.path.join(categorical_output_dir, "heatmap_all_categorical_variables_combined.pdf")
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

    writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][variable].to_numpy()
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

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Add summary statistics as text
    writer_data = plot_data[plot_data["paragraph_type_"] == "writer"][variable]
    model_data = plot_data[plot_data["paragraph_type_"] == "model"][variable]

    if len(writer_data) > 0 and len(model_data) > 0:
        writer_mean = writer_data.mean()
        model_mean = model_data.mean()
        writer_std = writer_data.std()
        model_std = model_data.std()

        # Add text box with summary stats
        stats_text = f"Writer: μ={writer_mean:.1f}, σ={writer_std:.1f} (n={len(writer_data):,})\n"
        stats_text += (
            f"Model: μ={model_mean:.1f}, σ={model_std:.1f} (n={len(model_data):,})"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved KDE plot for {variable}")


# ===== FUNCTION TO CREATE COMBINED GRID PLOT =====
def get_scale_attributes_sorted_by_ame():
    """Return scale attributes sorted by descending AME from edited by_type results."""

    results_dir = os.path.join("results", "followup_mitigation_phase_2_distortion", "edited")
    ame_rows = []

    for variable in SCALE_ATTRIBUTES:
        file_path = os.path.join(results_dir, f"{variable}_by_type.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        target = df[df["term"] == "paragraph_type_model"]
        if target.empty or "ame" not in target.columns:
            continue

        ame_rows.append({
            "variable": variable,
            "ame": float(target.iloc[0]["ame"]),
        })

    if not ame_rows:
        return SCALE_ATTRIBUTES, {}

    ame_df = pd.DataFrame(ame_rows).sort_values("ame", ascending=False)
    ordered = ame_df["variable"].tolist()
    ame_lookup = dict(zip(ame_df["variable"], ame_df["ame"]))

    # Keep any missing variables at the end in original order
    remaining = [v for v in SCALE_ATTRIBUTES if v not in ordered]
    return ordered + remaining, ame_lookup


def create_combined_kde_grid():
    """Create a grid of KDE plots for all scale variables."""

    ordered_scale_attributes, ame_lookup = get_scale_attributes_sorted_by_ame()

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
        plot_data = annotations_df[["paragraph_type_", variable]].dropna()

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
        ame_value = ame_lookup.get(variable)
        ame_label = f"AME: {ame_value:.2f}" if ame_value is not None else "AME: N/A"
        ax.set_title(
            f"{variable.replace('_', ' ').title()} ({ame_label})",
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


# ===== MAIN EXECUTION =====
def main():
    print("=== KDE Distribution Plots for Scale Rating Variables ===")
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(SCALE_ATTRIBUTES)} scale variables...")

    # Create individual KDE plots
    print("\nCreating individual KDE plots...")
    for variable in SCALE_ATTRIBUTES:
        output_path = os.path.join(output_dir, f"kde_{variable}.pdf")
        create_kde_plot(annotations_df, variable, output_path)

    # Create combined grid plot
    print("\nCreating combined KDE grid plot...")
    create_combined_kde_grid()

    # Prepare categorical data and create categorical barplots
    print("\nPreparing categorical data...")
    categorical_df = prepare_categorical_data(annotations_df)

    print("\nCreating categorical barplots...")
    for variable in CATEGORICAL_VARS:
        output_path = os.path.join(categorical_output_dir, f"barplot_{variable}.pdf")
        create_categorical_barplot(categorical_df, variable, output_path)

    print("\nCreating combined categorical barplot grid...")
    create_combined_categorical_grid(categorical_df)

    print("\nCreating combined categorical heatmap...")
    create_combined_categorical_heatmap(categorical_df)

    # Create summary statistics table
    print("\nGenerating summary statistics...")
    summary_stats = []

    for variable in SCALE_ATTRIBUTES:
        writer_data = annotations_df[annotations_df["paragraph_type_"] == "writer"][
            variable
        ].dropna()
        model_data = annotations_df[annotations_df["paragraph_type_"] == "model"][
            variable
        ].dropna()

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

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "kde_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics: {summary_path}")

    # Print brief summary
    print(f"\n=== Summary ===")
    print(f"Generated KDE plots for {len(SCALE_ATTRIBUTES)} scale variables")
    print(f"Total annotations processed: {len(annotations_df):,}")
    print(
        f"Writer paragraphs: {len(annotations_df[annotations_df['paragraph_type_'] == 'writer']):,}"
    )
    print(
        f"Model paragraphs: {len(annotations_df[annotations_df['paragraph_type_'] == 'model']):,}"
    )
    print(f"Figures saved to: {output_dir}/")

    print("\n=== Top 5 Largest Mean Differences (Model - Writer) ===")
    top_diffs = summary_df.nlargest(5, "mean_diff")[["variable", "mean_diff"]]
    for _, row in top_diffs.iterrows():
        print(f"{row['variable']:.<30} {row['mean_diff']:+6.2f}")


if __name__ == "__main__":
    main()
