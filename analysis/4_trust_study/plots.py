#!/usr/bin/env python3

"""
Trust study plots

1) Whisker plot of AMEs (AI vs human):
   - Overall (RQ1 primary)
   - Low distortion (RQ2 bin model)
   - High distortion (RQ2 bin model)

2) Exploratory 5x4 binned-means grid:
   - x: paragraph attribute rating (0-100)
   - y: trust allocation (0-20)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
RESULTS_DIR = get_results_input_dir(REPO_ROOT, "followup_trust", demo_mode=DEMO_MODE)
FIGURES_DIR = get_figures_dir(REPO_ROOT, "followup_trust", demo_mode=DEMO_MODE)
EXPLORATORY_AME_PATH = RESULTS_DIR / "exploratory_attributes_ame.csv"

TRUST_PATH = REPO_ROOT / "data" / "followup_trust" / "annotations.csv"
PAIRS_PATH = REPO_ROOT / "data" / "followup_trust" / "paragraph_pairs.csv"


def load_ame_for_whisker() -> pd.DataFrame:
	rq1_path = RESULTS_DIR / "rq1_ame.csv"
	rq2_path = RESULTS_DIR / "rq2_ame.csv"

	rq1 = pd.read_csv(rq1_path)
	rq2 = pd.read_csv(rq2_path)

	overall = rq1[
		(rq1["model"] == "rq1_primary")
		& (rq1["term"].astype(str).str.contains("paragraph_type", regex=False))
	].copy()
	if overall.empty:
		raise ValueError("Could not find overall paragraph_type AME in rq1_ame.csv")

	overall = overall.iloc[[0]].assign(label="Overall")

	by_bin = rq2[
		(rq2["model"] == "rq2_interaction_bin")
		& (rq2["term"] == "paragraph_type_ai_vs_human")
	].copy()
	if by_bin.empty:
		raise ValueError("Could not find bin-specific paragraph_type AMEs in rq2_ame.csv")

	label_map = {
		"low_distortion": "Low Persona Distortion",
		"high_distortion": "High Persona Distortion",
	}
	by_bin["label"] = by_bin["distortion_bin"].map(label_map).fillna(by_bin["distortion_bin"].astype(str))

	out = pd.concat([overall, by_bin], ignore_index=True)

	order = ["Overall", "High Persona Distortion", "Low Persona Distortion"]
	out["label"] = pd.Categorical(out["label"], categories=order, ordered=True)
	out = out.sort_values("label").reset_index(drop=True)
	return out


def plot_ame_whisker(df: pd.DataFrame) -> Path:
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)
	out_path = FIGURES_DIR / "trust_ame_ai_vs_human_whisker.png"

	fig, ax = plt.subplots(figsize=(8, 4.8))

	y_pos = list(range(len(df)))
	x = df["ame_pence"].to_numpy()
	xerr_low = (df["ame_pence"] - df["ame_low_pence"]).to_numpy()
	xerr_high = (df["ame_high_pence"] - df["ame_pence"]).to_numpy()

	ax.errorbar(
		x,
		y_pos,
		xerr=[xerr_low, xerr_high],
		fmt="o",
		capsize=4,
		linewidth=2,
		color="#1f77b4",
		ecolor="#1f77b4",
		markersize=7,
	)
	ax.axvline(0, color="black", linestyle="--", linewidth=1)

	ax.set_yticks(y_pos)
	ax.set_yticklabels(df["label"].astype(str).tolist())
	ax.invert_yaxis()
	ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
	ax.set_xlabel("∆ in trust game allocation for AI-assisted vs. human writing (AME, 0-20p)")
	ax.grid(axis="x", alpha=0.25)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	fig.tight_layout()
	fig.savefig(out_path, dpi=300)
	plt.close(fig)
	return out_path


def build_exploratory_plot_data() -> pd.DataFrame:
	trust = pd.read_csv(TRUST_PATH)
	pairs = pd.read_csv(PAIRS_PATH)

	trust = trust.copy()
	trust["writer_id"] = trust["writer_id"].astype(str)
	trust["proposition_id"] = trust["proposition_id"].astype(str)
	trust["paragraph_type"] = trust["source"].map({"human": "human", "ai": "ai"})

	pairs = pairs.copy()
	pairs["writer_id"] = pairs["writer_id"].astype(str)
	pairs["proposition_id"] = pairs["proposition_id"].astype(str)

	human_cols = [f"{attr}_human" for attr in SCALE_ATTRIBUTES]
	ai_cols = [f"{attr}_ai" for attr in SCALE_ATTRIBUTES]

	missing_cols = [c for c in (human_cols + ai_cols) if c not in pairs.columns]
	if missing_cols:
		raise ValueError(f"Missing expected attribute columns in pairs file: {missing_cols}")

	pairs_long = (
		pairs[["writer_id", "proposition_id", *human_cols, *ai_cols]]
		.melt(
			id_vars=["writer_id", "proposition_id"],
			var_name="attr_type",
			value_name="rating",
		)
	)

	split = pairs_long["attr_type"].str.rsplit("_", n=1, expand=True)
	pairs_long["attribute"] = split[0]
	pairs_long["paragraph_type"] = split[1]

	data = trust.merge(
		pairs_long,
		on=["writer_id", "proposition_id", "paragraph_type"],
		how="left",
	)

	data = data[
		data["attribute"].isin(SCALE_ATTRIBUTES)
		& data["rating"].notna()
		& data["trust_allocation_pence"].notna()
	].copy()

	return data


def load_exploratory_ame() -> pd.DataFrame:
	ame = pd.read_csv(EXPLORATORY_AME_PATH)
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
	out_path = FIGURES_DIR / "trust_exploratory_attribute_binned_means_5x4.png"

	ordered_attrs = (
		ame_df.drop_duplicates(subset=["attribute"])
		.sort_values("ame_pence", ascending=False)["attribute"]
		.tolist()
	)
	ordered_attrs = [a for a in ordered_attrs if a in SCALE_ATTRIBUTES]
	ordered_attrs += [a for a in SCALE_ATTRIBUTES if a not in ordered_attrs]

	fig, axes = plt.subplots(5, 4, figsize=(16, 18), sharex=True, sharey=True)
	axes_flat = axes.flatten()
	bins = list(range(0, 101, 10))
	bin_midpoints = [b + 5 for b in bins[:-1]]

	for i, attr in enumerate(ordered_attrs):
		ax = axes_flat[i]
		d = df[df["attribute"] == attr]
		ame_row = ame_df[ame_df["attribute"] == attr].head(1)
		if not ame_row.empty:
			ame_val = float(ame_row["ame_pence"].iloc[0])
			p_bonf = float(ame_row["p_bonferroni"].iloc[0])
			sig_bonf = bool(ame_row["significant_bonferroni"].iloc[0])
		else:
			ame_val = float("nan")
			p_bonf = float("nan")
			sig_bonf = False

		d = d.copy()
		d["bin"] = pd.cut(d["rating"], bins=bins, include_lowest=True, right=True)

		summary = (
			d.groupby("bin", observed=False)["trust_allocation_pence"]
			.agg(mean="mean", sd="std", n="count")
			.reset_index()
		)
		summary["x"] = bin_midpoints
		summary["sd"] = summary["sd"].fillna(0.0)
		summary["se"] = summary["sd"] / summary["n"].pow(0.5)
		summary["se"] = summary["se"].fillna(0.0)
		summary["ci95"] = 1.96 * summary["se"]
		summary["lower"] = (summary["mean"] - summary["ci95"]).clip(lower=0)
		summary["upper"] = (summary["mean"] + summary["ci95"]).clip(upper=20)

		valid = summary["mean"].notna()
		x = summary.loc[valid, "x"]
		mean_y = summary.loc[valid, "mean"]
		lower_y = summary.loc[valid, "lower"]
		upper_y = summary.loc[valid, "upper"]

		if sig_bonf:
			fill_color = "#2a9d8f"
			line_color = "#264653"
			title_color = "#1b4332"
			face_color = "white"
		else:
			fill_color = "#c7c7c7"
			line_color = "#8d8d8d"
			title_color = "#7a7a7a"
			face_color = "#f2f2f2"

		ax.set_facecolor(face_color)

		if len(x) > 0:
			ax.fill_between(
				x,
				lower_y,
				upper_y,
				color=fill_color,
				alpha=0.22,
				linewidth=0,
			)
			ax.plot(
				x,
				mean_y,
				color=line_color,
				linewidth=1.8,
				marker="o",
				markersize=3.5,
			)

		ax.set_title(attr, fontsize=10, color=title_color)
		if pd.notna(ame_val) and pd.notna(p_bonf):
			p_label = "<1e-4" if p_bonf < 1e-4 else f"{p_bonf:.3f}"
			ax.text(
				0.02,
				0.95,
				f"AME={ame_val:+.3f}p\np_bonf={p_label}",
				transform=ax.transAxes,
				fontsize=8,
				va="top",
				ha="left",
				color=title_color,
				bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
			)
		ax.set_xlim(0, 100)
		ax.set_ylim(0, 20)
		ax.grid(alpha=0.2)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)

	for j in range(len(SCALE_ATTRIBUTES), len(axes_flat)):
		axes_flat[j].axis("off")

	fig.supxlabel("Attribute rating (0-100)")
	fig.supylabel("Trust allocation (pence, 0-20)")
	fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
	fig.savefig(out_path, dpi=300)
	plt.close(fig)
	return out_path


def main() -> None:
	ame_df = load_ame_for_whisker()
	whisker_path = plot_ame_whisker(ame_df)

	binned_df = build_exploratory_plot_data()
	exploratory_ame_df = load_exploratory_ame()
	binned_path = plot_exploratory_binned_means_grid(binned_df, exploratory_ame_df)

	print(f"Saved: {whisker_path}")
	print(f"Saved: {binned_path}")


if __name__ == "__main__":
	main()
