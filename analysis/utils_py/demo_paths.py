from __future__ import annotations

import sys
from pathlib import Path


def parse_demo_mode(argv: list[str] | None = None) -> bool:
	args = sys.argv[1:] if argv is None else argv
	return "demo" in args


def get_results_dir(repo_root: Path, *parts: str, demo_mode: bool = False) -> Path:
	base_dir = repo_root / ("demo_results" if demo_mode else "results")
	return base_dir.joinpath(*parts)


def get_results_input_dir(repo_root: Path, *parts: str, demo_mode: bool = False) -> Path:
	if demo_mode:
		demo_path = get_results_dir(repo_root, *parts, demo_mode=True)
		if demo_path.exists():
			return demo_path

	return get_results_dir(repo_root, *parts, demo_mode=False)


def get_figures_dir(repo_root: Path, *parts: str, demo_mode: bool = False) -> Path:
	base_dir = repo_root / ("demo_figures" if demo_mode else "figures")
	return base_dir.joinpath(*parts)