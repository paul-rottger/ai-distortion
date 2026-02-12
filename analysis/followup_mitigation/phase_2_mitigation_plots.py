import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = "../../results/followup_mitigation_phase_2_distortion/"

################################
# SCALE ATTRIBUTES
################################

SCALE_ATTRIBUTES = [
    "writer_stance_polarity",
]

regression_dict = {}

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(SCALE_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in [
            "by_mitigation",
        ]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


# Create horizontal AME plot for "by_mitigation" subset
def create_horizontal_ame_plot(
    regression_dict,
    subset="by_mitigation",
    include_unedited=True,
    figsize=(8, 5),
    xlabel="Average Marginal Effect (AME)\nHigher = More Distortion from AI.",
    ylabel="Mitigation Condition",
    save_path=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    # Mitigation order (controls row order)
    mitigation_order = ["none", "prompting", "reranking"]

    # Map mitigation names to regression term strings
    mitigation_terms = {
        "none": "mitigation_condition_none",
        "prompting": "mitigation_condition_prompting",
        "reranking": "mitigation_condition_reranking",
    }

    # Colors
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }
        y_offset = {"unedited": -0.1, "edited": 0.1}
    else:
        colors = {"edited": "#000000"}
        y_offset = {"edited": 0}

    para_types = ["unedited", "edited"] if include_unedited else ["edited"]

    for para_type in para_types:
        if para_type not in regression_dict:
            continue

        # Concatenate attribute dfs (you currently only have one, but this generalizes)
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

        # Filter only mitigation rows
        regression_df = regression_df[
            regression_df["term"].isin(mitigation_terms.values())
        ]

        # Compute y positions
        y_positions = []
        ame_values = []
        ame_low = []
        ame_high = []

        for mitigation in mitigation_order:
            term_name = mitigation_terms[mitigation]
            row = regression_df[regression_df["term"] == term_name]

            if not row.empty:
                y_positions.append(
                    mitigation_order.index(mitigation) + y_offset[para_type]
                )
                ame_values.append(row["ame"].values[0])
                ame_low.append(row["ame_low"].values[0])
                ame_high.append(row["ame_high"].values[0])

        # Plot
        ax.errorbar(
            ame_values,
            y_positions,
            xerr=[
                [ame - low for ame, low in zip(ame_values, ame_low)],
                [high - ame for ame, high in zip(ame_values, ame_high)],
            ],
            fmt="o",
            capsize=3,
            elinewidth=1,
            capthick=1,
            color=colors[para_type],
            label=para_type.capitalize(),
            zorder=4,
        )

    # Reference line at 0
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Y-axis labels
    ax.set_yticks(range(len(mitigation_order)))
    ax.set_yticklabels(mitigation_order)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if include_unedited:
        ax.legend(frameon=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

# Example usage:
fig, ax = create_horizontal_ame_plot(
    regression_dict,
    subset="by_mitigation",
    include_unedited=False,
    save_path="../../figures/followup_mitigation_phase_2/stance_polarity_ame_by_mitigation.pdf",
)
