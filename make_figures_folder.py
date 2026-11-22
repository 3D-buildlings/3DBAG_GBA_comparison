#!/usr/bin/env python3
"""Create the complete figures/ folder from an existing pipeline output.

The generated folder contains the paper figures (overall comparison, H3 density
boxplots, H3 spatial performance), the study-area map, and the density-stratified
low/medium/high figures.  The script runs the existing figure generators, so the
numbers in the figures always come from the current pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

PAPER_FIGURES = (
    "figure2_overall_comparison",
    "figure3_iou_by_h3_density",
    "figure3_iou_by_h3_density_gaussian",
    "figure4_height_error_by_h3_density",
    "figure4_height_error_by_h3_density_gaussian",
    "figure5_h3_spatial_performance",
)
DENSITY_FIGURES = (
    "density_map.png",
    "geometric_accuracy_by_density.png",
    "height_accuracy_by_density.png",
)
H3_REQUIRED = (
    Path("h3") / "h3_group_metrics.geojson",
    Path("h3") / "h3_cell_summary.csv",
    Path("h3") / "h3_metadata.json",
)


def _log(message: str = "") -> None:
    print(message, flush=True)


def _run(command) -> None:
    _log("Running: " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT, check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _number(value, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not pd.notna(number):
        return ""
    return f"{number:.{digits}f}"


def _integer(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return ""


def _ensure_h3(out: Path, resolution: int, force: bool, skip: bool) -> None:
    missing = [path for path in H3_REQUIRED if not (out / path).exists()]
    if skip:
        if missing:
            raise FileNotFoundError(
                "H3 inputs are required for the paper figures but are missing: "
                + ", ".join(str(path) for path in missing)
            )
        return
    if missing or force:
        _run(
            [
                sys.executable,
                str(ROOT / "h3_analysis.py"),
                "--out",
                str(out),
                "--resolution",
                str(resolution),
            ]
        )


def _copy_density_figures(out: Path, fig_dir: Path) -> list[Path]:
    copied = []
    for name in DENSITY_FIGURES:
        source = out / name
        target = fig_dir / name
        if source.exists():
            shutil.copy2(source, target)
            copied.append(target)
            _log(f"copied {name} -> {target}")
        else:
            _log(f"WARNING: density figure not found: {source}")
    return copied


def _write_figures_readme(
    out: Path,
    fig_dir: Path,
    paper_files: list[Path],
    density_files: list[Path],
    study_files: list[Path],
    matching_files: list[Path] | None = None,
) -> Path:
    meta = _read_json(out / "run_metadata.json")
    summary = _read_csv(out / "comparison_summary.csv")
    h3_summary = _read_csv(out / "h3" / "h3_density_summary.csv")
    density_summary = _read_csv(out / "density_accuracy_summary.csv")
    sensitivity = _read_csv(out / "grid_size_sensitivity.csv")

    lines = [
        "# Figures generated from the GBA vs 3DBAG comparison",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Pipeline output directory: `{out}`",
        "",
        "## Input provenance",
        "",
        f"- Input mode: **{meta.get('input_mode')}**",
        f"- GBA source: `{meta.get('gba_source')}`",
        f"- 3DBAG source: `{meta.get('tdbag_source')}`",
        f"- Remote dataset retrieval: **{meta.get('remote_dataset_retrieval')}**",
        "",
    ]

    if not summary.empty:
        row = summary.iloc[0]
        lines.extend(
            [
                "## Overall comparison numbers shown in figure 2",
                "",
                f"- 3DBAG reference buildings: {_integer(row.get('reference_count'))}",
                f"- GBA buildings used in comparison: {_integer(row.get('compared_count'))}",
                f"- Matched 3DBAG reference buildings: {_integer(row.get('matched_reference_count'))}",
                f"- Matched GBA buildings: {_integer(row.get('matched_compared_count'))}",
                f"- Matched groups: {_integer(row.get('matched_group_count'))}",
                f"- False negatives: {_integer(row.get('false_negative_count'))}",
                f"- False positives: {_integer(row.get('false_positive_count'))}",
                f"- Mean group IoU: {_number(row.get('mean_group_iou'))}",
                f"- Mean group F1: {_number(row.get('mean_group_f1'))}",
                f"- Mean group precision: {_number(row.get('mean_group_precision'))}",
                f"- Mean group recall: {_number(row.get('mean_group_recall'))}",
                f"- Height MAE: {_number(row.get('height_mae_m'))} m",
                f"- Height RMSE: {_number(row.get('height_rmse_m'))} m",
                f"- Height MAPE: {_number(row.get('height_mape_pct'), 2)}%",
                "",
            ]
        )

    if not h3_summary.empty:
        lines.extend(["## H3 density numbers shown in figures 3-5", ""])
        for _, row in h3_summary.iterrows():
            lines.append(
                f"- {row.get('density')}: {_integer(row.get('n_groups'))} groups, "
                f"mean IoU {_number(row.get('mean_iou'))}, "
                f"median IoU {_number(row.get('median_iou'))}, "
            f"mean signed height error {_number(row.get('mean_height_error_m'), 2)} m, "
            f"mean absolute height error {_number(row.get('mean_abs_height_error_m'), 2)} m"
            )
        lines.extend(["", f"H3 resolution: {meta.get('h3_resolution', 8)}", ""])

    if not density_summary.empty:
        thresholds = density_summary.iloc[0]
        grid_size = thresholds.get("grid_size_m")
        p33 = thresholds.get("p33_threshold_per_km2")
        p66 = thresholds.get("p66_threshold_per_km2")
        lines.extend(
            [
                "## Density-stratified numbers shown in the additional density figures",
                "",
                f"- Primary grid size: {_number(grid_size, 0)} m",
                f"- P33 threshold: {_number(p33, 2)} buildings/km2",
                f"- P66 threshold: {_number(p66, 2)} buildings/km2",
                "",
                "| Class | Cells | Reference buildings | TP | FP | FN | Precision | Recall | F1 | Height MAE (m) | Height RMSE (m) | Height bias (m) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in density_summary.iterrows():
            label = row.get("density_class")
            if label not in ("low", "medium", "high"):
                continue
            lines.append(
                f"| {label} | {_integer(row.get('n_grid_cells'))} | "
                f"{_integer(row.get('n_reference_buildings'))} | "
                f"{_integer(row.get('n_tp'))} | {_integer(row.get('n_fp'))} | "
                f"{_integer(row.get('n_fn'))} | {_number(row.get('precision'))} | "
                f"{_number(row.get('recall'))} | {_number(row.get('f1'))} | "
                f"{_number(row.get('height_mae_m'))} | "
                f"{_number(row.get('height_rmse_m'))} | "
                f"{_number(row.get('height_bias_m'))} |"
            )
        unclassified = density_summary[
            density_summary["density_class"] == "unclassified_empty_cell"
        ]
        if not unclassified.empty:
            fp_unclassified = int(unclassified.iloc[0].get("n_fp", 0))
            lines.extend(
                [
                    "",
                    f"False positives in empty cells (unclassified, not low density): {fp_unclassified:,}",
                ]
            )
        lines.append("")

    if not sensitivity.empty:
        lines.extend(
            [
                "## Grid-size sensitivity",
                "",
                "| Grid (m) | Total cells | Non-empty cells | Reference buildings | P33 | P66 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sensitivity.iterrows():
            lines.append(
                f"| {_number(row.get('grid_size_m'), 0)} | "
                f"{_integer(row.get('total_grid_cells'))} | "
                f"{_integer(row.get('nonempty_grid_cells'))} | "
                f"{_integer(row.get('n_reference_buildings'))} | "
                f"{_number(row.get('p33_density_per_km2'), 2)} | "
                f"{_number(row.get('p66_density_per_km2'), 2)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Files in this folder",
            "",
        ]
    )
    for path in sorted(paper_files + study_files + density_files):
        lines.append(f"- `{path.name}`")
    if matching_files:
        lines.extend(
            [
                "- `matching_examples_panel.png` / `.pdf` — six-panel matching-category overview.",
                "- `matching_examples/` — individual three-panel examples (GBA only, 3DBAG only, overlay), "
                "README, and `matching_examples_selection.csv`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Figures 2-5 are regenerated from `comparison_summary.csv`, `h3/h3_density_summary.csv`, "
            "`h3/h3_group_metrics.geojson`, `h3/h3_cell_summary.csv`, and `h3/h3_metadata.json`.",
            "- The study-area map uses the repository's Amsterdam + Utrecht defaults unless custom "
            "`--area` values were passed to `make_figures_folder.py`.",
            "- The additional density figures come from `density_accuracy_summary.csv`, "
            "`grid_size_sensitivity.csv`, and the primary density grid.",
            "- The additional Gaussian-fit figures overlay normal curves using the sample mean and "
            "standard deviation. The empirical IoU and height-error distributions are strongly skewed; "
            "the Gaussian curves are reference overlays, not distributional models.",
            "- Matching-category examples are generated by `make_matching_examples.py`; selected "
            "3DBAG polygons are blue, selected GBA polygons are orange, and grey polygons are context.",
            "- Basemap imagery is downloaded from EOX/Copernicus Sentinel-2 cloudless; if the network "
            "is unavailable, the scripts use grey fallback tiles.",
            "- Local input provenance is not verified from the file contents; see "
            "`run_metadata.json` and `density_analysis_report.md` for provenance and limitations.",
            "",
        ]
    )
    path = fig_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_figures(
    out: str | Path,
    resolution: int = 8,
    force_h3: bool = False,
    skip_h3: bool = False,
    skip_paper: bool = False,
    skip_study_area: bool = False,
    skip_density: bool = False,
    skip_matching_examples: bool = False,
    areas=None,
) -> dict:
    """Generate the complete figures folder for an existing pipeline output."""
    out = Path(out).resolve()
    if not (out / "comparison_summary.csv").exists():
        raise FileNotFoundError(
            f"comparison_summary.csv not found under {out}; run run_pipeline.py first."
        )
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Figures output directory: {fig_dir}")

    _ensure_h3(out, resolution, force=force_h3, skip=skip_h3)

    paper_files: list[Path] = []
    if not skip_paper:
        _run([sys.executable, str(ROOT / "make_paper_figures.py"), "--out", str(out)])
        for stem in PAPER_FIGURES:
            for suffix in (".png", ".pdf"):
                path = fig_dir / f"{stem}{suffix}"
                if path.exists():
                    paper_files.append(path)

    study_files: list[Path] = []
    if not skip_study_area:
        command = [sys.executable, str(ROOT / "study_area_map.py"), "--out", str(out)]
        if areas:
            for area in areas:
                command.extend(["--area", area])
        _run(command)
        for stem in ("study_areas_netherlands", "study_area_map"):
            for suffix in (".png", ".pdf"):
                path = fig_dir / f"{stem}{suffix}"
                if path.exists():
                    study_files.append(path)

    density_files: list[Path] = []
    if not skip_density:
        density_files = _copy_density_figures(out, fig_dir)

    matching_files: list[Path] = []
    if not skip_matching_examples:
        _run([sys.executable, str(ROOT / "make_matching_examples.py"), "--out", str(out)])
        matching_dir = fig_dir / "matching_examples"
        for path in (
            fig_dir / "matching_examples_panel.png",
            fig_dir / "matching_examples_panel.pdf",
            matching_dir / "README.md",
            matching_dir / "matching_examples_selection.csv",
        ):
            if path.exists():
                matching_files.append(path)
        for key in (
            "1to1",
            "many_to_one",
            "one_to_many",
            "many_to_many",
            "false_negative",
            "false_positive",
        ):
            for suffix in (".png", ".pdf"):
                path = matching_dir / f"{key}{suffix}"
                if path.exists():
                    matching_files.append(path)

    readme = _write_figures_readme(
        out, fig_dir, paper_files, density_files, study_files, matching_files
    )
    _log(f"wrote {readme}")
    return {
        "figures_dir": str(fig_dir),
        "paper_figures": [str(path) for path in paper_files],
        "study_area_figures": [str(path) for path in study_files],
        "density_figures": [str(path) for path in density_files],
        "matching_examples": [str(path) for path in matching_files],
        "readme": str(readme),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the complete figures folder from pipeline outputs"
    )
    parser.add_argument("--out", required=True, help="pipeline output directory")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution (default 8)")
    parser.add_argument("--force-h3", action="store_true", help="rerun h3_analysis.py even if present")
    parser.add_argument("--skip-h3", action="store_true", help="use existing H3 outputs; fail if missing")
    parser.add_argument("--skip-paper", action="store_true", help="skip figures 2-5")
    parser.add_argument("--skip-study-area", action="store_true", help="skip the study-area map")
    parser.add_argument("--skip-density", action="store_true", help="skip the density figures")
    parser.add_argument(
        "--skip-matching-examples",
        action="store_true",
        help="skip the matching-category example maps",
    )
    parser.add_argument(
        "--area",
        action="append",
        default=None,
        metavar="NAME:min_lon,min_lat,max_lon,max_lat",
        help="study area for the map (repeatable; default: Amsterdam + Utrecht)",
    )
    args = parser.parse_args()
    generate_figures(
        out=args.out,
        resolution=args.resolution,
        force_h3=args.force_h3,
        skip_h3=args.skip_h3,
        skip_paper=args.skip_paper,
        skip_study_area=args.skip_study_area,
        skip_density=args.skip_density,
        skip_matching_examples=args.skip_matching_examples,
        areas=args.area,
    )


if __name__ == "__main__":
    main()
