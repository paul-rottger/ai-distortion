import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "../../results/main_phase_2_distortion/"

################################
# SCALE ATTRIBUTES
################################

SCALE_ATTRIBUTES = [
    "paragraph_formality",
    "paragraph_clarity",
    #
    "paragraph_informativeness",
    "paragraph_originality",
    "paragraph_relevance",
    #
    "writer_knowledge",
    "writer_importance",
    "writer_confidence",
    "writer_stance_polarity",
    #
    "paragraph_hope",
    "paragraph_excitement",
    "paragraph_fear",
    "paragraph_disgust",
    "paragraph_anger",
    #
    "writer_affect_x",
    "writer_affect_y",
    #
    "writer_optimism",
    "writer_community",
    "writer_friendliness",
    "writer_openness",
]

regression_dict = {}

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(SCALE_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in [
            "by_type",
            "by_model",
            "by_input",
        ]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


# Create horizontal AME plot for "by_type" subset
def create_horizontal_ame_plot(
    regression_dict,
    subset="by_type",
    include_unedited=True,
    figsize=(10, 7),
    xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Scale Attributes (Grouped by Type)",
    save_path=None
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each condition
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }  # Sea green for unedited, crimson for edited
    else:
        colors = {
            "edited": "#000000",
        }  # Black for edited only

    # Define y-position offsets to avoid overlapping points
    y_offset = {"unedited": -0.1, "edited": 0.1} if include_unedited else {"edited": 0}  # No offset if only edited

    # Process both unedited and edited data
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
        if para_type in regression_dict:
            # Select the appropriate dataframe based on subset
            regression_df = pd.concat(
                [
                    df.assign(outcome=attr)
                    for attr, df in (
                        (attr, regression_dict[para_type][attr][subset])
                        for attr in regression_dict[para_type]
                    )
                ],
                ignore_index=True,
            )

            # Create y-positions with offset
            y_positions = [
                list(regression_dict[para_type].keys()).index(outcome)
                + y_offset[para_type]
                for outcome in regression_df["outcome"]
            ]

            # Plot horizontal error bars
            ax.errorbar(
                regression_df["ame"],
                y_positions,
                xerr=[
                    regression_df["ame"] - regression_df["ame_low"],
                    regression_df["ame_high"] - regression_df["ame"],
                ],
                fmt="o",
                capsize=3,
                capthick=1,
                elinewidth=1,
                color=colors[para_type],
                label=para_type.capitalize(),
                zorder=4,
            )

    # Add reference line at x=0
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Add vertical lines to separate attribute groups
    group_boundaries = [3.5, 5.5, 10.5, 14.5]
    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(SCALE_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(SCALE_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, loc="lower right") if include_unedited else None

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Usage
create_horizontal_ame_plot(
    regression_dict,
    include_unedited=False, 
    ylabel = None,
    save_path="../../figures/main_phase_2_distortion/distortion_scale_variables_ame.pdf",
)


################################
# ORDINAL ATTRIBUTES
################################

ORDINAL_ATTRIBUTES = [
    "writer_english_first",
    "writer_age_binned",
    "writer_income",
    "writer_english_skills",
    "writer_education",
]

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(ORDINAL_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in ["by_type"]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


# Create horizontal cliffs_delta plot for "by_type" subset
def create_horizontal_cliffs_delta_plot(
    regression_dict,
    subset="by_type",
    include_unedited=True,
    figsize=(10, 2.5),
    xlabel="Cliff's Delta for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Ordinal Attributes",
    save_path=None,
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each condition
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }  # Sea green for unedited, crimson for edited
    else:
        colors = { "edited": "#000000", } # Black for editedÍ

    # Define y-position offsets to avoid overlapping points
    y_offset = {"unedited": -0.1, "edited": 0.1} if include_unedited else {"edited": 0}  # No offset if only edited

    # Process both unedited and edited data
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
        if para_type in regression_dict:
            # Select the appropriate dataframe based on subset
            regression_df = pd.concat(
                [
                    df.assign(outcome=attr)
                    for attr, df in (
                        (attr, regression_dict[para_type][attr][subset])
                        for attr in regression_dict[para_type]
                    )
                ],
                ignore_index=True,
            )

            # Create y-positions with offset
            y_positions = [
                list(regression_dict[para_type].keys()).index(outcome)
                + y_offset[para_type]
                for outcome in regression_df["outcome"]
            ]

            # Plot horizontal error bars
            ax.errorbar(
                regression_df["cliffs_delta"],
                y_positions,
                xerr=[
                    regression_df["cliffs_delta"]
                    - regression_df["cliffs_delta_ci_low"],
                    regression_df["cliffs_delta_ci_high"]
                    - regression_df["cliffs_delta"],
                ],
                fmt="o",
                capsize=3,
                capthick=1,
                elinewidth=1,
                color=colors[para_type],
                label=para_type.capitalize(),
                zorder=4,
            )

    # Add reference line at x=0
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(ORDINAL_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(ORDINAL_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, loc="lower right") if include_unedited else None

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Usage
create_horizontal_cliffs_delta_plot(
    regression_dict,
    include_unedited=False,
    ylabel = None,
    save_path="../../figures/main_phase_2_distortion/distortion_ordinal_variables_cliffs_delta.pdf",
)

plt.show()
