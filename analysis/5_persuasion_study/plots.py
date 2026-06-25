#!/usr/bin/env python3

"""
Persuasion study plots

1) Whisker plot of AI-vs-human AMEs on stance shift:
   - Overall (RQ1 primary)
   - High distortion / Low distortion (RQ2 binned model)

2) Absolute-persuasion-vs-control panel:
   - Pooled human / AI (RQ1) and the four authorship x distortion cells (RQ2),
     each as an AME relative to the off-topic control baseline.

3) Exploratory 5x4 binned-means grid:
   - x: paragraph attribute rating (0-100), from the original study
   - y: stance shift (attitude points), treated cycles only
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT / "analysis" / "utils_py"))
from variable_definitions import SCALE_ATTRIBUTES  # noqa: E402
from demo_paths import (  # noqa: E402
    parse_demo_mode,
    get_results_input_dir,
    get_figures_dir,
)

DEMO_MODE = parse_demo_mode()
RESULTS_DIR = get_results_input_dir(REPO_ROOT, "followup_persuasion", demo_mode=DEMO_MODE)
FIGURES_DIR = get_figures_dir(REPO_ROOT, "followup_persuasion", demo_mode=DEMO_MODE)

ANNOTATIONS_PATH = REPO_ROOT / "data" / "followup_persuasion" / "annotations.csv"
PAIRS_PATH = REPO_ROOT / "data" / "followup_persuasion" / "paragraph_pairs.csv"

HUMAN_COLOR = "#0070C0"
AI_COLOR = "#7030A0"


# ---------------------------------------------------------------------------
# 1) AI-vs-human whisker
# ---------------------------------------------------------------------------

def load_ame_for_whisker() -> pd.DataFrame:
    rq1 = pd.read_csv(RESULTS_DIR / "rq1_ame.csv")
    rq2 = pd.read_csv(RESULTS_DIR / "rq2_ame.csv")

    overall = rq1[
        (rq1["model"] == "rq1_primary")
        & (rq1["term"] == "paragraph_type_ai_vs_human")
    ].copy()
    if overall.empty:
        raise ValueError("Could not find overall AI-vs-human AME in rq1_ame.csv")
    overall = overall.iloc[[0]].assign(label="Overall")

    by_bin = rq2[
        (rq2["model"] == "rq2_binned")
        & (rq2["term"] == "paragraph_type_ai_vs_human")
    ].copy()
    if by_bin.empty:
        raise ValueError("Could not find bin-specific AI-vs-human AMEs in rq2_ame.csv")

    label_map = {
        "low_distortion": "Low Persona Distortion",
        "high_distortion": "High Persona Distortion",
    }
    by_bin["label"] = by_bin["distortion_bin"].map(label_map).fillna(
        by_bin["distortion_bin"].astype(str)
    )

    out = pd.concat([overall, by_bin], ignore_index=True)
    order = ["Overall", "High Persona Distortion", "Low Persona Distortion"]
    out["label"] = pd.Categorical(out["label"], categories=order, ordered=True)
    return out.sort_values("label").reset_index(drop=True)


def plot_ame_whisker(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "persuasion_ame_ai_vs_human_whisker.png"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = list(range(len(df)))
    x = df["ame"].to_numpy()
    xerr_low = (df["ame"] - df["ame_low"]).to_numpy()
    xerr_high = (df["ame_high"] - df["ame"]).to_numpy()

    ax.errorbar(
        x, y_pos, xerr=[xerr_low, xerr_high],
        fmt="o", capsize=4, linewidth=2,
        color=AI_COLOR, ecolor=AI_COLOR, markersize=7,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel("∆ in stance shift for AI-assisted vs. human writing (AME, attitude points)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2) Absolute persuasion vs control
# ---------------------------------------------------------------------------

def load_absolute_persuasion() -> pd.DataFrame:
    rq1 = pd.read_csv(RESULTS_DIR / "rq1_ame.csv")
    rq2 = pd.read_csv(RESULTS_DIR / "rq2_ame.csv")

    pooled = rq1[rq1["term"].isin(
        ["paragraph_type3_human_vs_control", "paragraph_type3_ai_vs_control"]
    )].copy()
    pooled["label"] = pooled["term"].map({
        "paragraph_type3_human_vs_control": "Human (pooled)",
        "paragraph_type3_ai_vs_control": "AI (pooled)",
    })
    pooled["group"] = pooled["term"].str.contains("_ai_").map({True: "ai", False: "human"})

    cells = rq2[rq2["term"].str.endswith("_vs_control")].copy()
    cell_label = {
        "condition5_human-low_vs_control": "Human, low distortion",
        "condition5_human-high_vs_control": "Human, high distortion",
        "condition5_ai-low_vs_control": "AI, low distortion",
        "condition5_ai-high_vs_control": "AI, high distortion",
    }
    cells = cells[cells["term"].isin(cell_label)].copy()
    cells["label"] = cells["term"].map(cell_label)
    cells["group"] = cells["term"].str.contains("ai-").map({True: "ai", False: "human"})

    order = [
        "Human (pooled)", "Human, low distortion", "Human, high distortion",
        "AI (pooled)", "AI, low distortion", "AI, high distortion",
    ]
    out = pd.concat([pooled, cells], ignore_index=True)
    out["label"] = pd.Categorical(out["label"], categories=order, ordered=True)
    return out.sort_values("label").reset_index(drop=True)


def plot_absolute_persuasion(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "persuasion_vs_control.png"

    fig, ax = plt.subplots(figsize=(8, 5.0))
    y_pos = list(range(len(df)))
    x = df["ame"].to_numpy()
    xerr_low = (df["ame"] - df["ame_low"]).to_numpy()
    xerr_high = (df["ame_high"] - df["ame"]).to_numpy()
    colors = [AI_COLOR if g == "ai" else HUMAN_COLOR for g in df["group"]]

    for yi, xi, lo, hi, c in zip(y_pos, x, xerr_low, xerr_high, colors):
        ax.errorbar(xi, yi, xerr=[[lo], [hi]], fmt="o", capsize=4,
                    linewidth=2, color=c, ecolor=c, markersize=7)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Stance shift relative to off-topic control (AME, attitude points)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3) Exploratory binned-means grid (y = stance shift, treated only)
# ---------------------------------------------------------------------------

def build_exploratory_plot_data() -> pd.DataFrame:
    ann = pd.read_csv(ANNOTATIONS_PATH)
    pairs = pd.read_csv(PAIRS_PATH)

    ann = ann[ann["condition_type"] == "persuasion"].copy()
    ann["writer_id"] = ann["writer_id"].astype(str)
    ann["proposition_id"] = ann["proposition_id"].astype(str)
    ann["paragraph_type"] = ann["source"].map({"human": "human", "ai": "ai"})

    # Recompute policy attitude and the oriented stance shift (treated cycles).
    def policy_attitude(prefix: str) -> pd.Series:
        return (
            ann[f"{prefix}_support"]
            + (100 - ann[f"{prefix}_bad_idea"])
            + ann[f"{prefix}_good_consequences"]
        ) / 3

    ann["policy_attitude_pre"] = policy_attitude("pre").fillna(ann["policy_attitude_pre"])
    ann["policy_attitude_post"] = policy_attitude("post").fillna(ann["policy_attitude_post"])

    pairs = pairs.copy()
    pairs["writer_id"] = pairs["writer_id"].astype(str)
    pairs["proposition_id"] = pairs["proposition_id"].astype(str)

    human_cols = [f"{attr}_human" for attr in SCALE_ATTRIBUTES]
    ai_cols = [f"{attr}_ai" for attr in SCALE_ATTRIBUTES]
    stance_cols = ["writer_stance_human", "writer_stance_ai"]
    missing = [c for c in (human_cols + ai_cols + stance_cols) if c not in pairs.columns]
    if missing:
        raise ValueError(f"Missing expected columns in pairs file: {missing}")

    pairs_long = pairs[
        ["writer_id", "proposition_id", *human_cols, *ai_cols, *stance_cols]
    ].melt(id_vars=["writer_id", "proposition_id"], var_name="src_col", value_name="value")
    split = pairs_long["src_col"].str.rsplit("_", n=1, expand=True)
    pairs_long["base"] = split[0]
    pairs_long["paragraph_type"] = split[1]

    # Perceived paragraph stance per (writer, proposition, human/ai).
    stance = (
        pairs_long[pairs_long["base"] == "writer_stance"]
        .rename(columns={"value": "writer_stance"})
        [["writer_id", "proposition_id", "paragraph_type", "writer_stance"]]
    )
    # Attribute ratings (long).
    attrs = pairs_long[pairs_long["base"].isin(SCALE_ATTRIBUTES)].rename(
        columns={"base": "attribute", "value": "rating"}
    )[["writer_id", "proposition_id", "paragraph_type", "attribute", "rating"]]

    data = ann.merge(stance, on=["writer_id", "proposition_id", "paragraph_type"], how="left")
    direction = np.where(data["writer_stance"] < 50, -1.0, 1.0)
    data["stance_shift"] = direction * (
        data["policy_attitude_post"] - data["policy_attitude_pre"]
    )

    merged = data.merge(attrs, on=["writer_id", "proposition_id", "paragraph_type"], how="left")
    merged = merged[
        merged["attribute"].isin(SCALE_ATTRIBUTES)
        & merged["rating"].notna()
        & merged["stance_shift"].notna()
    ].copy()
    return merged


def load_exploratory_ame() -> pd.DataFrame:
    ame = pd.read_csv(RESULTS_DIR / "exploratory_attributes_ame.csv")
    if "attribute" not in ame.columns:
        raise ValueError("Expected 'attribute' column in exploratory_attributes_ame.csv")
    if "p_bonferroni" not in ame.columns:
        m = max(1, len(ame))
        ame["p_bonferroni"] = (ame["p"] * m).clip(upper=1.0)
    if "significant_bonferroni" not in ame.columns:
        ame["significant_bonferroni"] = ame["p_bonferroni"] < 0.05
    return ame


def plot_exploratory_binned_means_grid(df: pd.DataFrame, ame_df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "persuasion_exploratory_attribute_binned_means_5x4.png"

    ordered_attrs = (
        ame_df.drop_duplicates(subset=["attribute"])
        .sort_values("ame", ascending=False)["attribute"]
        .tolist()
    )
    ordered_attrs = [a for a in ordered_attrs if a in SCALE_ATTRIBUTES]
    ordered_attrs += [a for a in SCALE_ATTRIBUTES if a not in ordered_attrs]

    fig, axes = plt.subplots(5, 4, figsize=(16, 18), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    bins = list(range(0, 101, 10))
    bin_midpoints = [b + 5 for b in bins[:-1]]

    y_lo, y_hi = -15, 25
    for i, attr in enumerate(ordered_attrs):
        ax = axes_flat[i]
        d = df[df["attribute"] == attr].copy()
        ame_row = ame_df[ame_df["attribute"] == attr].head(1)
        if not ame_row.empty:
            ame_val = float(ame_row["ame"].iloc[0])
            p_bonf = float(ame_row["p_bonferroni"].iloc[0])
            sig_bonf = bool(ame_row["significant_bonferroni"].iloc[0])
        else:
            ame_val, p_bonf, sig_bonf = float("nan"), float("nan"), False

        d["bin"] = pd.cut(d["rating"], bins=bins, include_lowest=True, right=True)
        summary = (
            d.groupby("bin", observed=False)["stance_shift"]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        summary["x"] = bin_midpoints
        summary["sd"] = summary["sd"].fillna(0.0)
        summary["se"] = (summary["sd"] / summary["n"].pow(0.5)).fillna(0.0)
        summary["ci95"] = 1.96 * summary["se"]
        summary["lower"] = summary["mean"] - summary["ci95"]
        summary["upper"] = summary["mean"] + summary["ci95"]

        valid = summary["mean"].notna()
        x = summary.loc[valid, "x"]
        mean_y = summary.loc[valid, "mean"]
        lower_y = summary.loc[valid, "lower"]
        upper_y = summary.loc[valid, "upper"]

        if sig_bonf:
            fill_color, line_color, title_color, face_color = "#2a9d8f", "#264653", "#1b4332", "white"
        else:
            fill_color, line_color, title_color, face_color = "#c7c7c7", "#8d8d8d", "#7a7a7a", "#f2f2f2"

        ax.set_facecolor(face_color)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        if len(x) > 0:
            ax.fill_between(x, lower_y, upper_y, color=fill_color, alpha=0.22, linewidth=0)
            ax.plot(x, mean_y, color=line_color, linewidth=1.8, marker="o", markersize=3.5)

        ax.set_title(attr, fontsize=10, color=title_color)
        if pd.notna(ame_val) and pd.notna(p_bonf):
            p_label = "<1e-4" if p_bonf < 1e-4 else f"{p_bonf:.3f}"
            ax.text(
                0.02, 0.95, f"AME={ame_val:+.3f}\np_bonf={p_label}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                color=title_color,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )
        ax.set_xlim(0, 100)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(SCALE_ATTRIBUTES), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.supxlabel("Attribute rating (0-100)")
    fig.supylabel("Stance shift (attitude points, 0-100 scale)")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    whisker_path = plot_ame_whisker(load_ame_for_whisker())
    abs_path = plot_absolute_persuasion(load_absolute_persuasion())

    binned_df = build_exploratory_plot_data()
    binned_path = plot_exploratory_binned_means_grid(binned_df, load_exploratory_ame())

    print(f"Saved: {whisker_path}")
    print(f"Saved: {abs_path}")
    print(f"Saved: {binned_path}")


if __name__ == "__main__":
    main()
