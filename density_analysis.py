#!/usr/bin/env python3
"""Density-stratified accuracy analysis for the GBA vs 3DBAG comparison.

Density is the number of 3DBAG reference buildings per square kilometre in
square grid cells built in the projected metric CRS used by the comparison
pipeline (EPSG:28992).  The analysis keeps the existing overall comparison
outputs unchanged and adds low/medium/high density breakdowns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from shapely.ops import unary_union

import pipeline_lib as p

GRID_SIZES_M = (100.0, 250.0, 500.0)
PRIMARY_GRID_SIZE_M = 250.0
DENSITY_CLASSES = ("low", "medium", "high")
DENSITY_LABELS = {
    "low": "Low density",
    "medium": "Medium density",
    "high": "High density",
    "unclassified_empty_cell": "Unclassified (empty cell)",
    "unclassified_outside_grid": "Unclassified (outside grid)",
}
DENSITY_COLORS = {
    "low": "#fee08b",
    "medium": "#fc8d59",
    "high": "#d73027",
    "unclassified_empty_cell": "#cccccc",
    "unclassified_outside_grid": "#999999",
}


@dataclass
class DensityGrid:
    cell_size_m: float
    x0: float
    y0: float
    nx: int
    ny: int
    counts: np.ndarray
    density_per_km2: np.ndarray

    @property
    def total_cells(self) -> int:
        return int(self.nx * self.ny)

    @property
    def nonempty_mask(self) -> np.ndarray:
        return self.counts > 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _centroid_xy(geoms) -> tuple[np.ndarray, np.ndarray]:
    series = gpd.GeoSeries(geoms)
    centroids = series.centroid
    return centroids.x.to_numpy(dtype=float), centroids.y.to_numpy(dtype=float)


def build_density_grid(
    ref_centroids,
    cell_size_m: float,
    extent_geoms=None,
) -> DensityGrid:
    """Build a square grid in the projected CRS and count reference centroids."""
    cell_size_m = float(cell_size_m)
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")
    ref_x, ref_y = _centroid_xy(ref_centroids)
    if len(ref_x) == 0:
        raise ValueError("Cannot build a density grid without reference buildings.")

    xs = [ref_x]
    ys = [ref_y]
    if extent_geoms is not None:
        ext_x, ext_y = _centroid_xy(extent_geoms)
        if len(ext_x):
            xs.append(ext_x)
            ys.append(ext_y)
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    if not bool(finite.any()):
        raise ValueError("All building centroids are non-finite; cannot build a grid.")
    all_x = all_x[finite]
    all_y = all_y[finite]

    min_x = float(all_x.min())
    max_x = float(all_x.max())
    min_y = float(all_y.min())
    max_y = float(all_y.max())
    x0 = math.floor(min_x / cell_size_m) * cell_size_m
    y0 = math.floor(min_y / cell_size_m) * cell_size_m
    x1 = math.floor(max_x / cell_size_m) * cell_size_m + cell_size_m
    y1 = math.floor(max_y / cell_size_m) * cell_size_m + cell_size_m
    nx = int(round((x1 - x0) / cell_size_m))
    ny = int(round((y1 - y0) / cell_size_m))
    if nx <= 0 or ny <= 0:
        raise ValueError("Density grid has zero extent.")

    ix = np.full(len(ref_x), -1, dtype=np.int64)
    iy = np.full(len(ref_x), -1, dtype=np.int64)
    finite_ref = np.isfinite(ref_x) & np.isfinite(ref_y)
    ix[finite_ref] = np.floor((ref_x[finite_ref] - x0) / cell_size_m).astype(np.int64)
    iy[finite_ref] = np.floor((ref_y[finite_ref] - y0) / cell_size_m).astype(np.int64)
    inside = finite_ref & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    counts = np.zeros(nx * ny, dtype=np.int64)
    if bool(inside.any()):
        flat = iy[inside] * nx + ix[inside]
        np.add.at(counts, flat, 1)
    area_km2 = (cell_size_m * cell_size_m) / 1_000_000.0
    density = counts.astype(float) / area_km2
    return DensityGrid(
        cell_size_m=cell_size_m,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
        counts=counts,
        density_per_km2=density,
    )


def _cell_indices(x: np.ndarray, y: np.ndarray, grid: DensityGrid):
    finite = np.isfinite(x) & np.isfinite(y)
    ix = np.full(len(x), -1, dtype=np.int64)
    iy = np.full(len(y), -1, dtype=np.int64)
    ix[finite] = np.floor((x[finite] - grid.x0) / grid.cell_size_m).astype(np.int64)
    iy[finite] = np.floor((y[finite] - grid.y0) / grid.cell_size_m).astype(np.int64)
    inside = finite & (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    flat = np.full(len(x), -1, dtype=np.int64)
    flat[inside] = iy[inside] * grid.nx + ix[inside]
    return flat, inside


def density_statistics(grid: DensityGrid) -> dict:
    nonempty = grid.nonempty_mask
    densities = grid.density_per_km2[nonempty]
    if len(densities) == 0:
        raise ValueError("Density grid has no occupied cells.")
    p33 = float(np.percentile(densities, 33, method="linear"))
    p66 = float(np.percentile(densities, 66, method="linear"))
    distribution_values, distribution_counts = np.unique(
        np.round(densities, 6), return_counts=True
    )
    ties_p33 = int(np.isclose(densities, p33, rtol=1e-12, atol=1e-12).sum())
    ties_p66 = int(np.isclose(densities, p66, rtol=1e-12, atol=1e-12).sum())
    return {
        "cell_size_m": grid.cell_size_m,
        "cell_area_km2": (grid.cell_size_m**2) / 1_000_000.0,
        "total_grid_cells": int(grid.total_cells),
        "nonempty_grid_cells": int(nonempty.sum()),
        "empty_grid_cells": int((~nonempty).sum()),
        "n_reference_buildings": int(grid.counts.sum()),
        "min_density_per_km2": float(densities.min()),
        "max_density_per_km2": float(densities.max()),
        "mean_density_per_km2": float(densities.mean()),
        "median_density_per_km2": float(np.median(densities)),
        "p33_density_per_km2": p33,
        "p66_density_per_km2": p66,
        "ties_at_p33": ties_p33,
        "ties_at_p66": ties_p66,
        "distribution_values": distribution_values,
        "distribution_counts": distribution_counts,
    }


def classify_cell_density(densities: np.ndarray, p33: float, p66: float) -> np.ndarray:
    densities = np.asarray(densities, dtype=float)
    classes = np.full(len(densities), "unclassified_empty_cell", dtype=object)
    valid = np.isfinite(densities) & (densities > 0)
    low = valid & (densities <= p33)
    high = valid & (densities > p66)
    medium = valid & (~low) & (~high)
    classes[low] = "low"
    classes[medium] = "medium"
    classes[high] = "high"
    return classes


def classify_centroids(centroids, grid: DensityGrid, stats: dict) -> np.ndarray:
    x, y = _centroid_xy(centroids)
    flat, inside = _cell_indices(x, y, grid)
    classes = np.full(len(x), "unclassified_outside_grid", dtype=object)
    counts = np.zeros(len(x), dtype=np.int64)
    if bool(inside.any()):
        counts[inside] = grid.counts[flat[inside]]
    valid = inside & (counts > 0)
    densities = np.zeros(len(x), dtype=float)
    densities[valid] = grid.density_per_km2[flat[valid]]
    classes[valid] = classify_cell_density(
        densities[valid], stats["p33_density_per_km2"], stats["p66_density_per_km2"]
    )
    classes[inside & (counts == 0)] = "unclassified_empty_cell"
    return classes


def grid_cell_classes(grid: DensityGrid, stats: dict) -> np.ndarray:
    return classify_cell_density(
        grid.density_per_km2,
        stats["p33_density_per_km2"],
        stats["p66_density_per_km2"],
    )


def assign_group_classes(
    groups: pd.DataFrame,
    main: gpd.GeoDataFrame,
    grid: DensityGrid,
    stats: dict,
) -> np.ndarray:
    """Assign each matched group to the cell of its 3DBAG reference centroid.

    One-to-one groups use the reference centroid.  Multi-feature groups use the
    centroid of the union of their reference buildings, following h3_analysis.py.
    """
    if groups is None or groups.empty:
        return np.array([], dtype=object)

    main = main.reset_index(drop=True)
    ids = main["ref_id"].astype(str).to_numpy()
    centroids = main.geometry.centroid
    cx = pd.Series(centroids.x.to_numpy(dtype=float), index=ids)
    cy = pd.Series(centroids.y.to_numpy(dtype=float), index=ids)

    multi_mask = pd.to_numeric(groups.get("n_3dbag_in_group"), errors="coerce").fillna(1) > 1
    need_geoms = bool(multi_mask.any())
    geom_lookup = (
        dict(zip(ids, main.geometry.values)) if need_geoms else {}
    )

    xs = np.full(len(groups), np.nan, dtype=float)
    ys = np.full(len(groups), np.nan, dtype=float)
    for position, row in enumerate(groups.itertuples(index=False)):
        refs = [r for r in str(getattr(row, "ref_ids", "")).split(";") if r]
        if not refs:
            continue
        if len(refs) == 1:
            xs[position] = cx.get(refs[0], np.nan)
            ys[position] = cy.get(refs[0], np.nan)
            continue
        geoms = [geom_lookup[ref] for ref in refs if ref in geom_lookup]
        if not geoms:
            continue
        try:
            centroid = unary_union(geoms).centroid
        except Exception:
            centroid = geoms[0].centroid
        xs[position] = centroid.x
        ys[position] = centroid.y

    points = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=p.WORK_CRS)
    return classify_centroids(points, grid, stats)


def pooled_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def height_metrics(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {
            "n_groups": 0,
            "n_valid_height_groups": 0,
            "n_excluded_height_groups": 0,
            "n_missing_3dbag_height": 0,
            "n_missing_gba_height": 0,
            "n_nonpositive_3dbag_height": 0,
            "height_mae_m": None,
            "height_rmse_m": None,
            "height_bias_m": None,
        }
    ref = pd.to_numeric(frame.get("height_3dbag_group_m"), errors="coerce")
    cmp = pd.to_numeric(frame.get("height_compared_group_m"), errors="coerce")
    ref_missing = ref.isna() | ~np.isfinite(ref)
    cmp_missing = cmp.isna() | ~np.isfinite(cmp)
    nonpositive_ref = (~ref_missing) & (ref <= 0)
    valid = (~ref_missing) & (~cmp_missing) & (ref > 0)
    n_valid = int(valid.sum())
    if n_valid:
        error = cmp[valid] - ref[valid]
        mae = float(error.abs().mean())
        rmse = float(np.sqrt((error * error).mean()))
        bias = float(error.mean())
    else:
        mae = rmse = bias = None
    return {
        "n_groups": int(len(frame)),
        "n_valid_height_groups": n_valid,
        "n_excluded_height_groups": int(len(frame) - n_valid),
        "n_missing_3dbag_height": int(ref_missing.sum()),
        "n_missing_gba_height": int(cmp_missing.sum()),
        "n_nonpositive_3dbag_height": int(nonpositive_ref.sum()),
        "height_mae_m": mae,
        "height_rmse_m": rmse,
        "height_bias_m": bias,
    }


def _mean_numeric(frame: pd.DataFrame, column: str):
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _median_numeric(frame: pd.DataFrame, column: str):
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _format_distribution(values: np.ndarray, counts: np.ndarray) -> str:
    return "|".join(f"{float(v):.6g}:{int(c)}" for v, c in zip(values, counts))


def _md_table(frame: pd.DataFrame, columns, floatfmt="{:.2f}") -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(floatfmt.format(float(value)))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _flat_to_gdf_order(grid: DensityGrid, values) -> np.ndarray:
    """Reorder flat iy*nx+ix arrays to the ix-outer/iy-inner GeoDataFrame order."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(grid.ny, grid.nx)
    flat_ix = np.repeat(np.arange(grid.nx), grid.ny)
    flat_iy = np.tile(np.arange(grid.ny), grid.nx)
    return arr[flat_iy, flat_ix]


def _build_primary_grid_geodataframe(grid: DensityGrid, stats: dict) -> gpd.GeoDataFrame:
    size = grid.cell_size_m
    xs = grid.x0 + np.arange(grid.nx) * size
    ys = grid.y0 + np.arange(grid.ny) * size
    minx = np.repeat(xs, grid.ny)
    miny = np.tile(ys, grid.nx)
    cells = shapely.box(minx, miny, minx + size, miny + size)
    counts = _flat_to_gdf_order(grid, grid.counts)
    density_all = _flat_to_gdf_order(grid, grid.density_per_km2)
    classes_all = _flat_to_gdf_order(grid, grid_cell_classes(grid, stats))
    classes = np.where(counts > 0, classes_all, None)
    density = np.where(counts > 0, density_all, np.nan)
    flat_ix = np.repeat(np.arange(grid.nx), grid.ny)
    flat_iy = np.tile(np.arange(grid.ny), grid.nx)
    return gpd.GeoDataFrame(
        {
            "cell_id": [f"x{ix}_y{iy}" for ix, iy in zip(flat_ix, flat_iy)],
            "grid_size_m": float(size),
            "ix": flat_ix,
            "iy": flat_iy,
            "n_ref_buildings": counts,
            "density_per_km2": density,
            "density_class": classes,
            "is_empty": counts == 0,
        },
        geometry=cells,
        crs=p.WORK_CRS,
    )


def _plot_density_map(grid: DensityGrid, stats: dict, path: Path) -> None:
    size = grid.cell_size_m
    xs = grid.x0 + np.arange(grid.nx) * size
    ys = grid.y0 + np.arange(grid.ny) * size
    minx = np.repeat(xs, grid.ny)
    miny = np.tile(ys, grid.nx)
    boxes = shapely.box(minx, miny, minx + size, miny + size)
    cell_classes = _flat_to_gdf_order(grid, grid_cell_classes(grid, stats))
    nonempty = _flat_to_gdf_order(grid, grid.counts) > 0

    polygons = []
    colors = []
    for geom, cls in zip(boxes[nonempty], cell_classes[nonempty]):
        polygons.append(np.asarray(geom.exterior.coords))
        colors.append(DENSITY_COLORS[cls])

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#f5f5f5")
    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection(collection)
    ax.set_xlim(grid.x0, grid.x0 + grid.nx * size)
    ax.set_ylim(grid.y0, grid.y0 + grid.ny * size)
    ax.set_aspect("equal")
    ax.set_xlabel("RD x (m, EPSG:28992)")
    ax.set_ylabel("RD y (m, EPSG:28992)")
    ax.set_title(
        f"3DBAG building density classes, {size:.0f} m grid\n"
        f"percentiles: P33={stats['p33_density_per_km2']:.2f}, "
        f"P66={stats['p66_density_per_km2']:.2f} buildings/km2"
    )
    class_counts = {
        cls: int(np.sum(nonempty & (cell_classes == cls))) for cls in DENSITY_CLASSES
    }
    legend = [
        Patch(
            facecolor=DENSITY_COLORS[cls],
            label=f"{DENSITY_LABELS[cls]} (n={class_counts[cls]} cells)",
        )
        for cls in DENSITY_CLASSES
    ]
    legend.append(Patch(facecolor="#f5f5f5", label="Empty cell (not classified)"))
    ax.legend(handles=legend, loc="upper right", frameon=True)
    ax.text(
        0.01,
        0.01,
        f"Occupied cells: {stats['nonempty_grid_cells']:,} / {stats['total_grid_cells']:,}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_geometric(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[summary["density_class"].isin(DENSITY_CLASSES)].copy()
    labels = [DENSITY_LABELS[c] for c in frame["density_class"]]
    x = np.arange(len(frame))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for offset, column, name in (
        (-width, "precision", "Precision"),
        (0.0, "recall", "Recall"),
        (width, "f1", "F1"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offset, values, width, label=name)
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score")
    ax.set_title("Pooled geometric accuracy by density class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    ax = axes[1]
    for offset, column, name in (
        (-width, "n_tp", "TP"),
        (0.0, "n_fp", "FP"),
        (width, "n_fn", "FN"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offset, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Building count")
    ax.set_title("TP / FP / FN counts by density class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_height(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[summary["density_class"].isin(DENSITY_CLASSES)].copy()
    labels = [DENSITY_LABELS[c] for c in frame["density_class"]]
    x = np.arange(len(frame))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for offset, column, name in ((-width / 2, "height_mae_m", "MAE"), (width / 2, "height_rmse_m", "RMSE")):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offset, values, width, label=name)
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xi, value + 0.04, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Height error (m)")
    ax.set_title("Height accuracy by density class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1]
    bias = pd.to_numeric(frame["height_bias_m"], errors="coerce").to_numpy(dtype=float)
    colors = ["#4575b4" if (np.isfinite(v) and v >= 0) else "#d73027" for v in bias]
    ax.bar(x, bias, 0.5, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for xi, value in zip(x, bias):
        if np.isfinite(value):
            va = "bottom" if value >= 0 else "top"
            ax.text(xi, value, f"{value:.2f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean signed error, GBA - 3DBAG (m)")
    ax.set_title("Height bias by density class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_row(
    density_class: str,
    grid: DensityGrid,
    stats: dict,
    cell_classes: np.ndarray,
    tp_classes: np.ndarray,
    fn_classes: np.ndarray,
    fp_classes: np.ndarray,
    group_classes: np.ndarray,
    groups: pd.DataFrame,
) -> dict:
    class_cells = cell_classes == density_class
    n_cells = int(class_cells.sum())
    n_ref_cells = int(grid.counts[class_cells].sum())
    densities = grid.density_per_km2[class_cells]
    if len(densities):
        min_d = float(densities.min())
        max_d = float(densities.max())
        mean_d = float(densities.mean())
    else:
        min_d = max_d = mean_d = None
    tp = int(np.sum(tp_classes == density_class))
    fp = int(np.sum(fp_classes == density_class))
    fn = int(np.sum(fn_classes == density_class))
    metrics = pooled_metrics(tp, fp, fn)

    group_class_mask = group_classes == density_class
    class_groups = groups.loc[group_class_mask] if len(group_classes) else groups.iloc[0:0]
    height = height_metrics(class_groups)
    iou_mean = _mean_numeric(class_groups, "group_iou")
    iou_median = _median_numeric(class_groups, "group_iou")
    if len(class_groups) and "group_iou" in class_groups.columns:
        n_iou = int(pd.to_numeric(class_groups["group_iou"], errors="coerce").notna().sum())
    else:
        n_iou = 0
    return {
        "density_class": density_class,
        "density_label": DENSITY_LABELS[density_class],
        "density_lower_bound_per_km2": min_d,
        "density_upper_bound_per_km2": max_d,
        "mean_cell_density_per_km2": mean_d,
        "n_grid_cells": n_cells,
        "n_reference_buildings": n_ref_cells,
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_matched_groups": height["n_groups"],
        "n_matched_pairs": height["n_groups"],
        "n_groups_with_iou": n_iou,
        "mean_iou": iou_mean,
        "median_iou": iou_median,
        "n_valid_height_groups": height["n_valid_height_groups"],
        "n_valid_height_pairs": height["n_valid_height_groups"],
        "n_excluded_height_groups": height["n_excluded_height_groups"],
        "n_excluded_height_pairs": height["n_excluded_height_groups"],
        "n_missing_3dbag_height": height["n_missing_3dbag_height"],
        "n_missing_gba_height": height["n_missing_gba_height"],
        "n_nonpositive_3dbag_height": height["n_nonpositive_3dbag_height"],
        "height_mae_m": height["height_mae_m"],
        "height_rmse_m": height["height_rmse_m"],
        "height_bias_m": height["height_bias_m"],
        "p33_threshold_per_km2": stats["p33_density_per_km2"],
        "p66_threshold_per_km2": stats["p66_density_per_km2"],
        "grid_size_m": grid.cell_size_m,
    }


def run_density_analysis(
    out_dir: Path,
    main: gpd.GeoDataFrame,
    false_positives: gpd.GeoDataFrame,
    groups: pd.DataFrame,
    counts: dict,
    provenance: dict,
    grid_sizes_m=GRID_SIZES_M,
    primary_grid_size_m: float = PRIMARY_GRID_SIZE_M,
    reference_height_definition: str | None = None,
    compared_height_definition: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p.log(
        f"Density analysis: building grids in {p.WORK_CRS} at "
        f"{', '.join(f'{s:.0f} m' for s in grid_sizes_m)} (primary={primary_grid_size_m:.0f} m)."
    )

    required_main = {
        "ref_id",
        "is_matched",
        "is_false_negative",
        "height_3dbag_m",
        "height_compared_m",
        "group_id",
    }
    missing_main = sorted(required_main - set(main.columns))
    if missing_main:
        raise ValueError(f"Density analysis input is missing main columns: {missing_main}")

    main = main.reset_index(drop=True)
    false_positives = false_positives.reset_index(drop=True)
    if groups is None:
        groups = pd.DataFrame()
    else:
        groups = groups.copy()

    required_group = {
        "group_id",
        "ref_ids",
        "n_3dbag_in_group",
        "height_3dbag_group_m",
        "height_compared_group_m",
        "group_iou",
    }
    if not groups.empty:
        missing_groups = sorted(required_group - set(groups.columns))
        if missing_groups:
            raise ValueError(f"Density analysis input is missing group columns: {missing_groups}")

    ref_centroids = main.geometry.centroid
    fp_centroids = false_positives.geometry.centroid if len(false_positives) else gpd.GeoSeries([], crs=p.WORK_CRS)

    grids: dict[float, DensityGrid] = {}
    stats_by_size: dict[float, dict] = {}
    for size in grid_sizes_m:
        grid = build_density_grid(ref_centroids, size, extent_geoms=fp_centroids)
        stats = density_statistics(grid)
        grids[float(size)] = grid
        stats_by_size[float(size)] = stats
        p.log(
            f"  grid {size:.0f} m: cells={stats['total_grid_cells']:,}, "
            f"occupied={stats['nonempty_grid_cells']:,}, "
            f"P33={stats['p33_density_per_km2']:.2f}, "
            f"P66={stats['p66_density_per_km2']:.2f} buildings/km2."
        )

    primary = float(primary_grid_size_m)
    if primary not in grids:
        raise ValueError(
            f"Primary grid size {primary} m is not one of the evaluated grid sizes {sorted(grids)}."
        )
    primary_grid = grids[primary]
    primary_stats = stats_by_size[primary]
    primary_cell_classes = grid_cell_classes(primary_grid, primary_stats)

    ref_classes = classify_centroids(ref_centroids, primary_grid, primary_stats)
    is_matched = main["is_matched"].fillna(False).astype(bool).to_numpy()
    is_fn = main["is_false_negative"].fillna(False).astype(bool).to_numpy()
    tp_classes = ref_classes[is_matched]
    fn_classes = ref_classes[is_fn]
    fp_classes = (
        classify_centroids(fp_centroids, primary_grid, primary_stats)
        if len(false_positives)
        else np.array([], dtype=object)
    )
    group_classes = assign_group_classes(groups, main, primary_grid, primary_stats)

    unclassified_tp = int(np.sum(~np.isin(tp_classes, DENSITY_CLASSES)))
    unclassified_fn = int(np.sum(~np.isin(fn_classes, DENSITY_CLASSES)))
    unclassified_fp_empty = int(np.sum(fp_classes == "unclassified_empty_cell"))
    unclassified_fp_outside = int(np.sum(fp_classes == "unclassified_outside_grid"))
    unclassified_groups = int(np.sum(~np.isin(group_classes, DENSITY_CLASSES))) if len(group_classes) else 0
    p.log(
        "  unclassified: "
        f"TP={unclassified_tp}, FN={unclassified_fn}, "
        f"FP in empty cells={unclassified_fp_empty}, FP outside grid={unclassified_fp_outside}, "
        f"matched groups={unclassified_groups}."
    )

    rows = [
        _summary_row(
            cls,
            primary_grid,
            primary_stats,
            primary_cell_classes,
            tp_classes,
            fn_classes,
            fp_classes,
            group_classes,
            groups,
        )
        for cls in DENSITY_CLASSES
    ]
    unclassified_row = {
        "density_class": "unclassified_empty_cell",
        "density_label": DENSITY_LABELS["unclassified_empty_cell"],
        "density_lower_bound_per_km2": None,
        "density_upper_bound_per_km2": None,
        "mean_cell_density_per_km2": None,
        "n_grid_cells": primary_stats["empty_grid_cells"],
        "n_reference_buildings": 0,
        "n_tp": 0,
        "n_fp": unclassified_fp_empty,
        "n_fn": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "n_matched_groups": unclassified_groups,
        "n_matched_pairs": unclassified_groups,
        "n_groups_with_iou": 0,
        "mean_iou": None,
        "median_iou": None,
        "n_valid_height_groups": 0,
        "n_valid_height_pairs": 0,
        "n_excluded_height_groups": unclassified_groups,
        "n_excluded_height_pairs": unclassified_groups,
        "n_missing_3dbag_height": 0,
        "n_missing_gba_height": 0,
        "n_nonpositive_3dbag_height": 0,
        "height_mae_m": None,
        "height_rmse_m": None,
        "height_bias_m": None,
        "p33_threshold_per_km2": primary_stats["p33_density_per_km2"],
        "p66_threshold_per_km2": primary_stats["p66_density_per_km2"],
        "grid_size_m": primary,
    }
    summary = pd.DataFrame(rows + [unclassified_row])
    summary.to_csv(out_dir / "density_accuracy_summary.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows = []
    distribution_rows = []
    for size, grid in grids.items():
        stats = stats_by_size[size]
        stats_cell_classes = grid_cell_classes(grid, stats)
        class_cell_counts = {
            cls: int(np.sum(stats_cell_classes == cls)) for cls in DENSITY_CLASSES
        }
        sensitivity_rows.append(
            {
                "grid_size_m": size,
                "grid_area_km2": (size * size) / 1_000_000.0,
                "total_grid_cells": stats["total_grid_cells"],
                "nonempty_grid_cells": stats["nonempty_grid_cells"],
                "empty_grid_cells": stats["empty_grid_cells"],
                "n_reference_buildings": stats["n_reference_buildings"],
                "min_density_per_km2": stats["min_density_per_km2"],
                "max_density_per_km2": stats["max_density_per_km2"],
                "mean_density_per_km2": stats["mean_density_per_km2"],
                "median_density_per_km2": stats["median_density_per_km2"],
                "p33_density_per_km2": stats["p33_density_per_km2"],
                "p66_density_per_km2": stats["p66_density_per_km2"],
                "low_density_cells": class_cell_counts["low"],
                "medium_density_cells": class_cell_counts["medium"],
                "high_density_cells": class_cell_counts["high"],
                "ties_at_p33": stats["ties_at_p33"],
                "ties_at_p66": stats["ties_at_p66"],
                "density_distribution": _format_distribution(
                    stats["distribution_values"], stats["distribution_counts"]
                ),
                "is_primary_grid": bool(size == primary),
            }
        )
        for value, count in zip(stats["distribution_values"], stats["distribution_counts"]):
            distribution_rows.append(
                {
                    "grid_size_m": size,
                    "density_per_km2": float(value),
                    "n_cells": int(count),
                    "share_of_nonempty_cells": float(count) / stats["nonempty_grid_cells"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out_dir / "grid_size_sensitivity.csv", index=False, encoding="utf-8-sig")
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(
        out_dir / "density_distribution_by_grid.csv", index=False, encoding="utf-8-sig"
    )

    primary_grid_gdf = _build_primary_grid_geodataframe(primary_grid, primary_stats)
    grid_path = out_dir / f"density_grid_{int(round(primary))}m.geojson"
    pyogrio.write_dataframe(
        primary_grid_gdf.to_crs(p.WGS84),
        grid_path,
        driver="GeoJSON",
        layer_options={"COORDINATE_PRECISION": 6},
    )

    _plot_density_map(primary_grid, primary_stats, out_dir / "density_map.png")
    _plot_geometric(summary, out_dir / "geometric_accuracy_by_density.png")
    _plot_height(summary, out_dir / "height_accuracy_by_density.png")

    overall = {
        "reference_count": counts.get("reference_count"),
        "compared_count": counts.get("compared_count"),
        "matched_reference_count": counts.get("matched_reference_count"),
        "matched_compared_count": counts.get("matched_compared_count"),
        "matched_group_count": counts.get("matched_group_count"),
        "false_positive_count": counts.get("false_positive_count"),
        "false_negative_count": counts.get("false_negative_count"),
        "mean_group_iou": counts.get("mean_group_iou"),
        "mean_group_f1": counts.get("mean_group_f1"),
        "mean_group_precision": counts.get("mean_group_precision"),
        "mean_group_recall": counts.get("mean_group_recall"),
        "height_group_count": counts.get("height_group_count"),
        "height_mae_m": counts.get("height_mae_m"),
        "height_rmse_m": counts.get("height_rmse_m"),
        "height_mape_pct": counts.get("height_mape_pct"),
    }
    overall.update(pooled_metrics(
        int(overall["matched_reference_count"] or 0),
        int(overall["false_positive_count"] or 0),
        int(overall["false_negative_count"] or 0),
    ))

    metadata = {
        "generated_utc": _utc_now(),
        "input_mode": provenance.get("input_mode"),
        "gba_source": provenance.get("gba_source"),
        "tdbag_source": provenance.get("tdbag_source"),
        "remote_dataset_retrieval": provenance.get("remote_dataset_retrieval"),
        "reference_height_definition": reference_height_definition,
        "compared_height_definition": compared_height_definition,
        "working_crs": p.WORK_CRS,
        "grid_sizes_m": [float(s) for s in grid_sizes_m],
        "primary_grid_size_m": primary,
        "primary_grid_justification": (
            "The 250 m primary grid is chosen independently of the accuracy results. "
            "It matches the existing pipeline's 250 m 3DBAG retrieval cell size and "
            "approximates an urban block/neighbourhood scale. 100 m cells approach "
            "individual-building scale and produce many one-building cells; 500 m cells "
            "smooth local density variation."
        ),
        "density_definition": "number of 3DBAG reference buildings divided by cell area in km2",
        "density_centroid_definition": "3DBAG building centroid",
        "percentile_method": "numpy.percentile linear interpolation over non-empty cells",
        "p33_density_per_km2": primary_stats["p33_density_per_km2"],
        "p66_density_per_km2": primary_stats["p66_density_per_km2"],
        "class_rules": {
            "low": "density <= P33",
            "medium": "P33 < density <= P66",
            "high": "density > P66",
            "empty_cells": "unclassified, never low density",
        },
        "assignment": {
            "tp": "each matched 3DBAG reference building centroid",
            "fn": "each unmatched 3DBAG reference building centroid",
            "fp": "each unmatched GBA building centroid",
            "height_groups": "matched group assigned by the centroid of its 3DBAG reference union",
        },
        "matching": {
            "PAIR_IOU_MIN": p.PAIR_IOU_MIN,
            "CONTAINMENT_OVERLAP_MIN": p.CONTAINMENT_OVERLAP_MIN,
            "CENTROID_MAX_DIST_M": p.CENTROID_MAX_DIST_M,
            "CENTROID_MIN_IOU": p.CENTROID_MIN_IOU,
            "IOU_FN_THRESHOLD": p.IOU_FN_THRESHOLD,
            "MIN_BUILDING_AREA_M2": p.MIN_BUILDING_AREA_M2,
        },
        "sensitivity": sensitivity_rows,
        "overall_comparison": overall,
        "unclassified": {
            "tp": unclassified_tp,
            "fn": unclassified_fn,
            "fp_empty_cell": unclassified_fp_empty,
            "fp_outside_grid": unclassified_fp_outside,
            "matched_groups": unclassified_groups,
        },
        "ties": {
            "n_cells_at_p33": primary_stats["ties_at_p33"],
            "n_cells_at_p66": primary_stats["ties_at_p66"],
        },
        "supplementary_density_distribution_csv": str(
            out_dir / "density_distribution_by_grid.csv"
        ),
        "grid_geojson": str(grid_path),
    }
    (out_dir / "density_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    report = build_density_report(
        summary=summary,
        sensitivity=sensitivity,
        metadata=metadata,
        provenance=provenance,
        overall=overall,
    )
    (out_dir / "density_analysis_report.md").write_text(report, encoding="utf-8")
    p.log(f"Density analysis outputs written to {out_dir}")

    return {
        "summary": summary,
        "sensitivity": sensitivity,
        "metadata": metadata,
        "thresholds": {
            "p33_density_per_km2": primary_stats["p33_density_per_km2"],
            "p66_density_per_km2": primary_stats["p66_density_per_km2"],
            "primary_grid_size_m": primary,
        },
        "outputs": {
            "summary_csv": str(out_dir / "density_accuracy_summary.csv"),
            "map_png": str(out_dir / "density_map.png"),
            "geometric_png": str(out_dir / "geometric_accuracy_by_density.png"),
            "height_png": str(out_dir / "height_accuracy_by_density.png"),
            "report_md": str(out_dir / "density_analysis_report.md"),
            "sensitivity_csv": str(out_dir / "grid_size_sensitivity.csv"),
            "distribution_csv": str(out_dir / "density_distribution_by_grid.csv"),
            "grid_geojson": str(grid_path),
            "metadata_json": str(out_dir / "density_analysis_metadata.json"),
        },
    }


def build_density_report(
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    metadata: dict,
    provenance: dict,
    overall: dict,
) -> str:
    class_rows = summary[summary["density_class"].isin(DENSITY_CLASSES)].copy()
    geo_columns = [
        "density_class",
        "n_tp",
        "n_fp",
        "n_fn",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "n_grid_cells",
        "n_reference_buildings",
    ]
    height_columns = [
        "density_class",
        "n_matched_pairs",
        "n_valid_height_pairs",
        "n_excluded_height_pairs",
        "height_mae_m",
        "height_rmse_m",
        "height_bias_m",
    ]
    sens_columns = [
        "grid_size_m",
        "total_grid_cells",
        "nonempty_grid_cells",
        "n_reference_buildings",
        "min_density_per_km2",
        "max_density_per_km2",
        "mean_density_per_km2",
        "median_density_per_km2",
        "p33_density_per_km2",
        "p66_density_per_km2",
    ]
    unclassified = metadata["unclassified"]
    ties = metadata["ties"]
    gba = provenance.get("gba", {})
    tdbag = provenance.get("tdbag", {})
    class_cell_counts = {
        str(row["density_class"]): int(row["n_grid_cells"])
        for _, row in class_rows.iterrows()
    }
    lines = [
        "# Density-stratified accuracy analysis",
        "",
        f"Generated (UTC): {metadata['generated_utc']}",
        "",
        "## Inputs and execution mode",
        "",
        f"- Input mode: **{metadata['input_mode']}**",
        f"- GBA source: `{metadata['gba_source']}`",
        f"- 3DBAG source: `{metadata['tdbag_source']}`",
        f"- Remote dataset retrieval: **{metadata['remote_dataset_retrieval']}**",
        f"- Working/projected CRS: `{metadata['working_crs']}`",
        f"- Reference height definition: {metadata['reference_height_definition']}",
        f"- Compared height definition: {metadata['compared_height_definition']}",
        "",
    ]
    if gba:
        lines.extend(
            [
                "### GBA input provenance",
                "",
                f"- Path: `{gba.get('path')}`",
                f"- File size: {gba.get('size_bytes'):,} bytes",
                f"- File modified (filesystem metadata): {gba.get('modified_utc')}",
                f"- Features: {gba.get('feature_count'):,}",
                f"- Source CRS: `{gba.get('source_crs')}`",
                f"- ID field: `{gba.get('id_column')}`; height field: `{gba.get('height_column')}`",
                f"- Geometry repair: {gba.get('geometry', {}).get('geometries_repaired')} feature(s)",
                "",
            ]
        )
    if tdbag:
        lines.extend(
            [
                "### 3DBAG input provenance",
                "",
                f"- Path: `{tdbag.get('path')}`",
                f"- File size: {tdbag.get('size_bytes'):,} bytes",
                f"- File modified (filesystem metadata): {tdbag.get('modified_utc')}",
                f"- Features: {tdbag.get('feature_count'):,}",
                f"- Source CRS: `{tdbag.get('source_crs')}`",
                f"- ID field: `{tdbag.get('id_column')}`",
                f"- Height source: {tdbag.get('height_source')}",
                f"- Height definition used: {tdbag.get('height_definition')}",
                f"- Geometry repair: {tdbag.get('geometry', {}).get('geometries_repaired')} feature(s)",
                "",
            ]
        )
    lines.extend(
        [
            "The local file contents do not embed a verifiable dataset version or download "
            "date. The filesystem modification time is metadata about the file, not proof of "
            "the official release or download date. The locally supplied files are therefore "
            "not described as official WFS downloads.",
            "",
            "## Method",
            "",
            "Building density is defined as the number of 3DBAG reference buildings divided by "
            "the grid-cell area in square kilometres. Reference-building centroids are assigned "
            "to square grid cells. Grid construction and density calculations are performed in "
            f"`{metadata['working_crs']}`; no density is calculated in geographic coordinates.",
            "",
            f"The evaluated grid sizes are {', '.join(str(int(s)) for s in metadata['grid_sizes_m'])} m. "
            f"The primary grid size is **{int(metadata['primary_grid_size_m'])} m**. "
            + metadata["primary_grid_justification"],
            "",
            "For the primary grid, the 33rd and 66th percentiles are calculated over the density "
            "values of non-empty cells using linear interpolation. Classes are:",
            "",
            f"- Low: density <= P33 ({metadata['p33_density_per_km2']:.2f} buildings/km2)",
            f"- Medium: P33 < density <= P66 ({metadata['p66_density_per_km2']:.2f} buildings/km2)",
            f"- High: density > P66",
            "- Empty cells: unclassified, never low density",
            "",
            "Geometric TP/FP/FN assignment:",
            "",
            "- TP: each matched 3DBAG reference building, assigned by its own reference centroid.",
            "- FN: each unmatched 3DBAG reference building, assigned by its reference centroid.",
            "- FP: each unmatched GBA building, assigned by its GBA centroid.",
            "- A matched group is not counted twice; each reference building is counted once.",
            "",
            "Height metrics use the existing matched-group height values, sign convention "
            "(GBA - 3DBAG), and validity rules. Each matched group is assigned once to the "
            "density class of the centroid of its 3DBAG reference union. Only groups with valid "
            "heights in both datasets and a positive reference height enter MAE/RMSE/bias; "
            "missing or invalid heights are excluded and counted, never replaced with zero.",
            "",
            "## Overall comparison (existing methodology)",
            "",
            "The overall values below come from this run and use the existing comparison "
            "methodology. They can differ slightly from a previous Online Mode run because "
            "local files are revalidated, invalid geometries are repaired, and coordinates "
            "are reprojected from the stored GeoJSON. The original comparison outputs and "
            "metric definitions are preserved separately in `comparison_summary.csv`.",
            "",
            f"- Reference buildings: {overall.get('reference_count'):,}",
            f"- Compared GBA buildings: {overall.get('compared_count'):,}",
            f"- Matched reference buildings (TP): {overall.get('matched_reference_count'):,}",
            f"- False positives: {overall.get('false_positive_count'):,}",
            f"- False negatives: {overall.get('false_negative_count'):,}",
            f"- Matched groups: {overall.get('matched_group_count'):,}",
            f"- Pooled precision: {overall.get('precision')}",
            f"- Pooled recall: {overall.get('recall')}",
            f"- Pooled F1: {overall.get('f1')}",
            f"- Mean group IoU: {overall.get('mean_group_iou')}",
            f"- Height MAE / RMSE / MAPE: {overall.get('height_mae_m')} / "
            f"{overall.get('height_rmse_m')} / {overall.get('height_mape_pct')}",
            "",
            "## Geometric accuracy by density class",
            "",
            _md_table(class_rows, geo_columns),
            "",
            "Precision, recall, and F1 are pooled from class-level TP, FP, and FN counts, not "
            "averaged from per-cell values. A blank metric means an undefined zero denominator.",
            "",
            "## Height accuracy by density class",
            "",
            _md_table(class_rows, height_columns),
            "",
            "Matched pairs are the existing group-level matched units produced by the "
            "repository's matching algorithm. A multi-feature group is one matched pair "
            "unit for the height comparison and is assigned once. Missing or invalid "
            "heights are excluded and counted; they are never replaced.",
            "",
            "## Grid-size sensitivity",
            "",
            _md_table(sensitivity, sens_columns),
            "",
            "Full density distributions are in `density_distribution_by_grid.csv` and the "
            "complete distribution strings are in `grid_size_sensitivity.csv`. The primary "
            "grid size was fixed independently of the accuracy outcomes.",
            "",
            "## Empty cells, unclassified records, and ties",
            "",
            f"- TP outside classified cells: {unclassified['tp']}",
            f"- FN outside classified cells: {unclassified['fn']}",
            f"- FP in empty cells: {unclassified['fp_empty_cell']}",
            f"- FP outside the grid: {unclassified['fp_outside_grid']}",
            f"- Matched groups without a density class: {unclassified['matched_groups']}",
            f"- Non-empty cells equal to P33: {ties['n_cells_at_p33']}",
            f"- Non-empty cells equal to P66: {ties['n_cells_at_p66']}",
            f"- Observed class cell counts: low={class_cell_counts.get('low')}, "
            f"medium={class_cell_counts.get('medium')}, high={class_cell_counts.get('high')}",
            "",
            "Empty cells are not classified as low density. FP buildings in empty cells are "
            "reported in the unclassified row of `density_accuracy_summary.csv` and are not "
            "forced into a density class, so the sum of class-level FP counts can be smaller "
            "than the overall FP count. Tied density values at a percentile can produce "
            "unequal class sizes. The observed class sizes are reported above and in "
            "`density_accuracy_summary.csv`; they were not artificially forced to be equal.",
            "",
            "## Interpretation and limitations",
            "",
            "- The class differences are descriptive. No inferential statistical test was "
            "performed, so no claim of statistical significance is made.",
            "- Grid cells are spatially dependent; neighbouring cells share buildings, "
            "infrastructure, and error structure. The effective sample size is smaller than "
            "the number of cells, and cell-level observations are not independent.",
            "- The analysis is sensitive to the modifiable areal unit problem. Grid-size "
            "sensitivity is reported in `grid_size_sensitivity.csv`.",
            "- Multi-feature matched groups are assigned to one density class by the centroid of "
            "their reference union for height metrics; per-reference TP/FN counts are assigned "
            "by each reference centroid. This can make group-based and reference-based class "
            "counts differ slightly.",
            "- GBA and 3DBAG represent different acquisition times and production methods. "
            "Height agreement is an inter-dataset comparison, not absolute accuracy.",
            "- The local 3DBAG height may be rounded in the source file. The analysis preserves "
            "the existing height definitions and does not re-estimate missing heights.",
            "",
            "## Outputs",
            "",
            "- `density_accuracy_summary.csv`",
            "- `density_map.png`",
            "- `geometric_accuracy_by_density.png`",
            "- `height_accuracy_by_density.png`",
            "- `grid_size_sensitivity.csv`",
            "- `density_analysis_report.md`",
            "- `density_distribution_by_grid.csv`",
            f"- `{Path(metadata['grid_geojson']).name}`",
            "- `density_analysis_metadata.json`",
            "",
        ]
    )
    return "\n".join(lines)
