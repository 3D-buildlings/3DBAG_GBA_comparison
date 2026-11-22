#!/usr/bin/env python3
"""Create one map example for each GBA-vs-3DBAG matching category.

The examples are selected from the actual pipeline outputs.  The script writes
individual PNG/PDF maps, a six-panel paper figure, a selection table with the
full feature IDs, and a short README explaining where each category lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyogrio
from shapely import STRtree
from shapely.geometry import box

import pipeline_lib as p

COLOR_3DBAG = "#1565C0"
COLOR_GBA = "#E07B00"
COLOR_CONTEXT = "#E6E6E6"
COLOR_CONTEXT_EDGE = "#B0B0B0"
COLOR_3DBAG_EDGE = "#0D47A1"
COLOR_GBA_EDGE = "#B35C00"

EXAMPLES = [
    {
        "key": "1to1",
        "panel": "(a)",
        "title": "One-to-one",
        "category": "matched_1to1",
        "group_id": None,
        "ref_ids": ["NL.IMBAG.Pand.0363100012170217"],
        "cmp_ids": ["GBA_309048538"],
        "pad_m": 18.0,
    },
    {
        "key": "many_to_one",
        "panel": "(b)",
        "title": "Many-to-one",
        "category": "many_to_one_3dbag_to_compared",
        "group_id": "GBA_vs_3DBAG_G255705",
        "ref_ids": [
            "NL.IMBAG.Pand.0363100012207496",
            "NL.IMBAG.Pand.0363100012572337",
        ],
        "cmp_ids": ["GBA_126616333"],
        "pad_m": 18.0,
    },
    {
        "key": "one_to_many",
        "panel": "(c)",
        "title": "One-to-many",
        "category": "one_to_many_3dbag_to_compared",
        "group_id": "GBA_vs_3DBAG_G160337",
        "ref_ids": ["NL.IMBAG.Pand.0363100012118839"],
        "cmp_ids": ["GBA_308086822", "GBA_308072397"],
        "pad_m": 18.0,
    },
    {
        "key": "many_to_many",
        "panel": "(d)",
        "title": "Many-to-many",
        "category": "many_to_many",
        "group_id": "GBA_vs_3DBAG_G222669",
        "ref_ids": [
            "NL.IMBAG.Pand.0384100000003922",
            "NL.IMBAG.Pand.0384100000003923",
        ],
        "cmp_ids": ["GBA_309202106", "GBA_309203061"],
        "pad_m": 18.0,
    },
    {
        "key": "false_negative",
        "panel": "(e)",
        "title": "False negative",
        "category": "false_negative_missed_3dbag",
        "group_id": None,
        "ref_ids": ["NL.IMBAG.Pand.0344100000041309"],
        "cmp_ids": [],
        "pad_m": 55.0,
    },
    {
        "key": "false_positive",
        "panel": "(f)",
        "title": "False positive",
        "category": "false_positive_extra_detection",
        "group_id": None,
        "ref_ids": [],
        "cmp_ids": ["GBA_105542810"],
        "pad_m": 35.0,
    },
]


def _short(value: str, n: int = 10) -> str:
    text = str(value)
    return text[-n:] if len(text) > n else text


def _format_area(geom) -> str:
    return f"{geom.area:,.0f} m2"


def _scalebar(ax, bbox, length_m):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    x = x0 + width * 0.08
    y = y0 + height * 0.06
    ax.plot([x, x + length_m], [y, y], color="black", linewidth=3, solid_capstyle="butt", zorder=20)
    ax.plot([x, x], [y - height * 0.012, y + height * 0.012], color="black", linewidth=1, zorder=20)
    ax.plot([x + length_m, x + length_m], [y - height * 0.012, y + height * 0.012], color="black", linewidth=1, zorder=20)
    ax.text(x + length_m / 2, y + height * 0.025, f"{length_m:g} m", ha="center", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8), zorder=21)


def _draw_example(
    ax,
    example,
    ref_geoms,
    cmp_geoms,
    main,
    gba,
    main_tree,
    gba_tree,
    mode="overlay",
    show_ids=False,
):
    refs = [ref_geoms[x] for x in example["ref_ids"] if x in ref_geoms]
    cmps = [cmp_geoms[x] for x in example["cmp_ids"] if x in cmp_geoms]
    selected = refs + cmps
    if not selected:
        raise RuntimeError(f"No geometries found for example {example['key']}")
    x0, y0, x1, y1 = selected[0].bounds
    for geom in selected[1:]:
        bx0, by0, bx1, by1 = geom.bounds
        x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    pad = example["pad_m"]
    bbox = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    query_box = box(*bbox)

    ctx_ref_idx = sorted(main_tree.query(query_box).tolist())
    ctx_cmp_idx = sorted(gba_tree.query(query_box).tolist())
    ctx_ref = main.iloc[ctx_ref_idx]
    ctx_cmp = gba.iloc[ctx_cmp_idx]

    if len(ctx_ref):
        ctx_ref.plot(
            ax=ax,
            facecolor=COLOR_CONTEXT,
            edgecolor=COLOR_CONTEXT_EDGE,
            linewidth=0.35,
            alpha=0.8,
            zorder=1,
        )
    if len(ctx_cmp):
        ctx_cmp.plot(
            ax=ax,
            facecolor=COLOR_CONTEXT,
            edgecolor=COLOR_CONTEXT_EDGE,
            linewidth=0.35,
            alpha=0.8,
            zorder=2,
        )
    if mode in ("3dbag", "overlay") and refs:
        gpd.GeoSeries(refs, crs=p.WORK_CRS).plot(
            ax=ax,
            facecolor=COLOR_3DBAG,
            edgecolor=COLOR_3DBAG_EDGE,
            linewidth=1.0,
            alpha=0.65,
            zorder=4,
        )
    if mode in ("gba", "overlay") and cmps:
        gpd.GeoSeries(cmps, crs=p.WORK_CRS).plot(
            ax=ax,
            facecolor=COLOR_GBA,
            edgecolor=COLOR_GBA_EDGE,
            linewidth=1.0,
            alpha=0.65,
            zorder=5,
        )

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    width = bbox[2] - bbox[0]
    scale = 10 if width < 60 else 20 if width < 120 else 50
    _scalebar(ax, bbox, scale)
    if show_ids:
        lines = []
        if refs:
            lines.append("3DBAG: " + ", ".join(_short(x) for x in example["ref_ids"]))
        if cmps:
            lines.append("GBA: " + ", ".join(_short(x) for x in example["cmp_ids"]))
        if lines:
            ax.text(
                0.01,
                0.01,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5.8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.85),
                zorder=30,
            )
    return bbox


def main():
    parser = argparse.ArgumentParser(description="Create matching-category example maps")
    parser.add_argument("--out", required=True, help="pipeline output directory")
    parser.add_argument("--examples-dir", default=None, help="output directory for examples (default: <out>/figures/matching_examples)")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    examples_dir = Path(args.examples_dir).resolve() if args.examples_dir else out / "figures" / "matching_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    main = pyogrio.read_dataframe(out / "GBA_vs_3DBAG.geojson")
    fps = pyogrio.read_dataframe(out / "GBA_vs_3DBAG_false_positives.geojson")
    gba = pyogrio.read_dataframe(
        out / "raw" / "GBA_Amsterdam_Utrecht_raw.geojson",
        columns=["id", "height_m", "area_m2", "geometry"],
    )
    groups = pd.read_csv(out / "GBA_vs_3DBAG_structural_groups.csv", encoding="utf-8-sig")

    main = main.to_crs(p.WORK_CRS)
    fps = fps.to_crs(p.WORK_CRS)
    gba = gba.to_crs(p.WORK_CRS)
    ref_geoms = dict(zip(main["ref_id"].astype(str), main.geometry))
    cmp_geoms = dict(zip(gba["id"].astype(str), gba.geometry))
    main_tree = STRtree(main.geometry.values)
    gba_tree = STRtree(gba.geometry.values)

    rows = []
    for ex in EXAMPLES:
        if not all(x in ref_geoms for x in ex["ref_ids"]):
            missing = [x for x in ex["ref_ids"] if x not in ref_geoms]
            raise RuntimeError(f"Missing 3DBAG geometries for {ex['key']}: {missing}")
        if not all(x in cmp_geoms for x in ex["cmp_ids"]):
            missing = [x for x in ex["cmp_ids"] if x not in cmp_geoms]
            raise RuntimeError(f"Missing GBA geometries for {ex['key']}: {missing}")
        if ex["category"] == "matched_1to1":
            match = main.loc[main["ref_id"].astype(str) == ex["ref_ids"][0]]
            iou = float(match["polygon_iou"].iloc[0])
            ex["metric_text"] = f"IoU = {iou:.2f}"
        elif ex["category"] in ("many_to_one_3dbag_to_compared", "one_to_many_3dbag_to_compared", "many_to_many"):
            match = groups.loc[groups["group_id"] == ex["group_id"]]
            iou = float(match["group_iou"].iloc[0])
            ex["metric_text"] = f"group IoU = {iou:.2f}"
        elif ex["category"] == "false_negative_missed_3dbag":
            ex["metric_text"] = "no accepted GBA match"
        else:
            fp = fps.loc[fps["cmp_id_best"].astype(str) == ex["cmp_ids"][0]]
            ex["metric_text"] = f"GBA height = {float(fp['height_compared_m'].iloc[0]):.2f} m"
        ex["metric_text"] = ex["metric_text"]

    # Individual three-panel figures: GBA only, 3DBAG only, overlay
    for ex in EXAMPLES:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        modes = [
            ("gba", "GBA only", len(ex["cmp_ids"]), "GBA"),
            ("3dbag", "3DBAG only", len(ex["ref_ids"]), "3DBAG"),
            ("overlay", "Overlay", len(ex["ref_ids"]) + len(ex["cmp_ids"]), "both"),
        ]
        bbox = None
        for ax, (mode, label, count, source) in zip(axes, modes):
            bbox = _draw_example(
                ax,
                ex,
                ref_geoms,
                cmp_geoms,
                main,
                gba,
                main_tree,
                gba_tree,
                mode=mode,
                show_ids=False,
            )
            if mode == "gba":
                subtitle = f"{len(ex['cmp_ids'])} target polygon" + ("s" if len(ex["cmp_ids"]) != 1 else "")
            elif mode == "3dbag":
                subtitle = f"{len(ex['ref_ids'])} target polygon" + ("s" if len(ex["ref_ids"]) != 1 else "")
            else:
                subtitle = f"{len(ex['ref_ids'])} 3DBAG + {len(ex['cmp_ids'])} GBA"
            ax.set_title(f"{label}\n{subtitle}", fontsize=10, pad=5)
        fig.suptitle(
            f"{ex['panel']} {ex['title']}  |  {ex['metric_text']}",
            fontsize=12.5,
            y=0.985,
        )
        handles = [
            Patch(facecolor=COLOR_3DBAG, edgecolor=COLOR_3DBAG_EDGE, alpha=0.65, label="Selected 3DBAG reference"),
            Patch(facecolor=COLOR_GBA, edgecolor=COLOR_GBA_EDGE, alpha=0.65, label="Selected GBA comparison"),
            Patch(facecolor=COLOR_CONTEXT, edgecolor=COLOR_CONTEXT_EDGE, alpha=0.8, label="Context buildings"),
            Line2D([0], [0], color="black", linewidth=3, label="Scale bar"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.045),
        )
        fig.text(
            0.5,
            0.008,
            "Projected CRS EPSG:28992. Only the target polygons are coloured; gray polygons are nearby context buildings.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.20, wspace=0.04)
        png = examples_dir / f"{ex['key']}.png"
        pdf = examples_dir / f"{ex['key']}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        rows.append(
            {
                "panel": ex["panel"],
                "example": ex["key"],
                "category": ex["category"],
                "group_id": ex["group_id"],
                "ref_ids": ";".join(ex["ref_ids"]),
                "cmp_ids": ";".join(ex["cmp_ids"]),
                "metric": ex["metric_text"],
                "bbox_rd_minx": bbox[0],
                "bbox_rd_miny": bbox[1],
                "bbox_rd_maxx": bbox[2],
                "bbox_rd_maxy": bbox[3],
                "png": str(png),
                "pdf": str(pdf),
            }
        )

    # Six-panel paper figure
    fig, axes = plt.subplots(
        2, 3, figsize=(15.5, 10.8), gridspec_kw={"hspace": 0.20, "wspace": 0.04}
    )
    for ax, ex in zip(axes.ravel(), EXAMPLES):
        _draw_example(
            ax,
            ex,
            ref_geoms,
            cmp_geoms,
            main,
            gba,
            main_tree,
            gba_tree,
            mode="overlay",
            show_ids=False,
        )
        ax.set_title(
            f"{ex['panel']} {ex['title']}\n"
            f"{len(ex['ref_ids'])} x 3DBAG  |  {len(ex['cmp_ids'])} x GBA  |  {ex['metric_text']}",
            fontsize=10,
            pad=6,
        )
    handles = [
        Patch(facecolor=COLOR_3DBAG, edgecolor=COLOR_3DBAG_EDGE, alpha=0.65, label="Selected 3DBAG reference"),
        Patch(facecolor=COLOR_GBA, edgecolor=COLOR_GBA_EDGE, alpha=0.65, label="Selected GBA comparison"),
        Patch(facecolor=COLOR_CONTEXT, edgecolor=COLOR_CONTEXT_EDGE, alpha=0.8, label="Context buildings"),
        Line2D([0], [0], color="black", linewidth=3, label="Scale bar"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.040))
    fig.suptitle("Examples of GBA vs 3DBAG matching categories", fontsize=14, y=0.985)
    fig.text(
        0.5,
        0.008,
        "Projected CRS EPSG:28992. Selected features are coloured; gray polygons are nearby context buildings.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.925, bottom=0.085)
    panel_png = figures_dir / "matching_examples_panel.png"
    panel_pdf = figures_dir / "matching_examples_panel.pdf"
    fig.savefig(panel_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(panel_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selection = pd.DataFrame(rows)
    selection.to_csv(examples_dir / "matching_examples_selection.csv", index=False, encoding="utf-8-sig")

    table_columns = ["panel", "example", "category", "group_id", "ref_ids", "cmp_ids", "metric"]
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "|" + "|".join(["---"] * len(table_columns)) + "|",
    ]
    for _, row in selection[table_columns].iterrows():
        table_lines.append("| " + " | ".join(str(row[column]) for column in table_columns) + " |")

    readme = [
        "# Matching-category examples",
        "",
        f"Generated from `{out}`.",
        "",
        "## Where each category lives",
        "",
        "- One-to-one, many-to-one, one-to-many, many-to-many, and false negatives are in",
        "  `GBA_vs_3DBAG.geojson`; use the `match_category` field.",
        "- False positives are in `GBA_vs_3DBAG_false_positives.geojson`; all features have",
        "  `match_category = false_positive_extra_detection`.",
        "- Multi-feature group IDs and the complete `ref_ids` / `cmp_ids` lists are in",
        "  `GBA_vs_3DBAG_structural_groups.csv`.",
        "- `GBA_vs_3DBAG.geojson` geometry is the 3DBAG reference footprint.",
        "- `GBA_vs_3DBAG_false_positives.geojson` geometry is the GBA footprint.",
        "- GBA footprints for matched groups are in `raw/GBA_Amsterdam_Utrecht_raw.geojson`",
        "  (join on `id` = `cmp_ids` / `cmp_id_best`).",
        "",
        "## QGIS filter expressions",
        "",
        "```sql",
        "\"match_category\" = 'matched_1to1'",
        "\"match_category\" = 'many_to_one_3dbag_to_compared'",
        "\"match_category\" = 'one_to_many_3dbag_to_compared'",
        "\"match_category\" = 'many_to_many'",
        "\"match_category\" = 'false_negative_missed_3dbag'",
        "\"match_category\" = 'false_positive_extra_detection'",
        "```",
        "",
        "For a structural group, filter the main layer by the IDs in",
        "`ref_ids` and filter `raw/GBA_Amsterdam_Utrecht_raw.geojson` by the IDs in",
        "`cmp_ids` from `GBA_vs_3DBAG_structural_groups.csv`.",
        "",
        "## Selected examples",
        "",
        "\n".join(table_lines),
        "",
        "## Files",
        "",
        "- `matching_examples_panel.png` / `.pdf`: six-panel overview (in `figures/`).",
        "- Individual maps: " + ", ".join(f"`{ex['key']}.png` / `.pdf`" for ex in EXAMPLES) + ".",
        "- Each individual map has three panels: GBA only, 3DBAG only, and overlay.",
        "- Context buildings are gray; only the target polygons are colored.",
        "- `matching_examples_selection.csv`: full IDs, metrics, and bounds.",
    ]
    (examples_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote {examples_dir}")
    print(selection[["panel", "example", "category", "group_id", "ref_ids", "cmp_ids", "metric"]].to_string(index=False))


if __name__ == "__main__":
    main()
