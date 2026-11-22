from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import density_analysis as da
import pipeline_lib as p
from conftest import BASE_X, BASE_Y


def _points(xs, ys=None):
    if ys is None:
        ys = [BASE_Y] * len(xs)
    return gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=p.WORK_CRS)


def test_grid_is_built_in_projected_metric_crs():
    points = _points([BASE_X, BASE_X + 300, BASE_X + 400])
    grid = da.build_density_grid(points, 250.0)
    assert grid.cell_size_m == 250.0
    assert grid.total_cells > 0
    assert grid.counts.sum() == 3
    assert grid.x0 <= points.x.min()
    assert grid.y0 <= points.y.min()


def test_density_uses_reference_centroids():
    points = _points([BASE_X, BASE_X + 10, BASE_X + 300])
    grid = da.build_density_grid(points, 250.0)
    assert grid.counts.sum() == 3
    assert grid.counts.max() == 2
    expected = 2 / ((250.0 * 250.0) / 1_000_000.0)
    assert grid.density_per_km2.max() == pytest.approx(expected)


def test_percentile_thresholds_and_classes():
    grid = da.DensityGrid(
        cell_size_m=100.0,
        x0=0.0,
        y0=0.0,
        nx=4,
        ny=1,
        counts=np.array([1, 2, 3, 4], dtype=np.int64),
        density_per_km2=np.array([100.0, 200.0, 300.0, 400.0]),
    )
    stats = da.density_statistics(grid)
    assert stats["p33_density_per_km2"] == pytest.approx(199.0)
    assert stats["p66_density_per_km2"] == pytest.approx(298.0)
    classes = da.classify_cell_density(
        grid.density_per_km2, stats["p33_density_per_km2"], stats["p66_density_per_km2"]
    )
    assert list(classes) == ["low", "medium", "high", "high"]


def test_empty_cells_are_not_low_density():
    grid = da.DensityGrid(
        cell_size_m=100.0,
        x0=0.0,
        y0=0.0,
        nx=3,
        ny=1,
        counts=np.array([0, 1, 2], dtype=np.int64),
        density_per_km2=np.array([0.0, 100.0, 200.0]),
    )
    stats = da.density_statistics(grid)
    classes = da.classify_cell_density(
        grid.density_per_km2, stats["p33_density_per_km2"], stats["p66_density_per_km2"]
    )
    assert classes[0] == "unclassified_empty_cell"
    assert "low" not in classes[0]


def test_tied_density_values_are_reported_not_forced():
    grid = da.DensityGrid(
        cell_size_m=250.0,
        x0=0.0,
        y0=0.0,
        nx=4,
        ny=1,
        counts=np.array([1, 1, 1, 1], dtype=np.int64),
        density_per_km2=np.full(4, 16.0),
    )
    stats = da.density_statistics(grid)
    assert stats["p33_density_per_km2"] == pytest.approx(16.0)
    assert stats["p66_density_per_km2"] == pytest.approx(16.0)
    classes = da.classify_cell_density(
        grid.density_per_km2, stats["p33_density_per_km2"], stats["p66_density_per_km2"]
    )
    assert list(classes) == ["low", "low", "low", "low"]
    assert stats["ties_at_p33"] == 4
    assert stats["ties_at_p66"] == 4


def test_pooled_precision_recall_f1():
    metrics = da.pooled_metrics(tp=3, fp=1, fn=1)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.75)
    undefined = da.pooled_metrics(tp=0, fp=0, fn=0)
    assert undefined["precision"] is None
    assert undefined["recall"] is None
    assert undefined["f1"] is None


def test_height_metrics_use_only_valid_matched_pairs():
    frame = pd.DataFrame(
        {
            "height_3dbag_group_m": [10.0, 20.0, 30.0, 0.0, None],
            "height_compared_group_m": [11.0, 18.0, None, 5.0, 1.0],
        }
    )
    metrics = da.height_metrics(frame)
    assert metrics["n_groups"] == 5
    assert metrics["n_valid_height_groups"] == 2
    assert metrics["n_excluded_height_groups"] == 3
    assert metrics["height_mae_m"] == pytest.approx(1.5)
    assert metrics["height_rmse_m"] == pytest.approx(math.sqrt(2.5))
    assert metrics["height_bias_m"] == pytest.approx(-0.5)
    assert metrics["n_nonpositive_3dbag_height"] == 1


def test_density_grid_geodataframe_preserves_cell_positions():
    counts = np.zeros(12, dtype=np.int64)
    counts[0] = 1
    counts[1 * 4 + 2] = 30
    counts[2 * 4 + 3] = 5
    grid = da.DensityGrid(
        cell_size_m=250.0,
        x0=0.0,
        y0=0.0,
        nx=4,
        ny=3,
        counts=counts,
        density_per_km2=counts * 16.0,
    )
    stats = da.density_statistics(grid)
    gdf = da._build_primary_grid_geodataframe(grid, stats)
    cell = gdf[(gdf["ix"] == 2) & (gdf["iy"] == 1)].iloc[0]
    assert int(cell["n_ref_buildings"]) == 30
    assert cell["density_per_km2"] == pytest.approx(480.0)
    assert int(gdf[(gdf["ix"] == 0) & (gdf["iy"] == 0)]["n_ref_buildings"].iloc[0]) == 1
    assert int(gdf[(gdf["ix"] == 3) & (gdf["iy"] == 2)]["n_ref_buildings"].iloc[0]) == 5
    assert int(gdf["n_ref_buildings"].sum()) == 36


def test_density_outputs_and_tp_fp_fn_assignment(comparison_result, density_provenance, tmp_path):
    result = da.run_density_analysis(
        out_dir=tmp_path,
        main=comparison_result["main"],
        false_positives=comparison_result["false_positives"],
        groups=comparison_result["groups"],
        counts=comparison_result["counts"],
        provenance=density_provenance,
        reference_height_definition="test reference",
        compared_height_definition="test compared",
    )
    summary = result["summary"].set_index("density_class")
    assert summary.loc["low", "n_tp"] == 1
    assert summary.loc["low", "n_fp"] == 0
    assert summary.loc["low", "n_fn"] == 0
    assert summary.loc["low", "precision"] == pytest.approx(1.0)
    assert summary.loc["medium", "n_tp"] == 1
    assert summary.loc["medium", "n_fn"] == 1
    assert summary.loc["medium", "precision"] == pytest.approx(1.0)
    assert summary.loc["medium", "recall"] == pytest.approx(0.5)
    assert summary.loc["high", "n_tp"] == 3
    assert summary.loc["high", "n_fp"] == 1
    assert summary.loc["high", "precision"] == pytest.approx(0.75)
    assert summary.loc["high", "recall"] == pytest.approx(1.0)
    assert summary.loc["high", "f1"] == pytest.approx(2 * 0.75 * 1.0 / 1.75)
    assert summary.loc["high", "height_mae_m"] == pytest.approx(1.0)
    assert summary.loc["high", "height_rmse_m"] == pytest.approx(1.0)
    assert summary.loc["high", "height_bias_m"] == pytest.approx(1.0)
    assert summary.loc["medium", "n_valid_height_groups"] == 0
    assert summary.loc["medium", "n_excluded_height_groups"] == 1
    assert summary.loc["unclassified_empty_cell", "n_fp"] >= 1

    for name in (
        "density_accuracy_summary.csv",
        "density_map.png",
        "geometric_accuracy_by_density.png",
        "height_accuracy_by_density.png",
        "density_analysis_report.md",
        "grid_size_sensitivity.csv",
        "density_distribution_by_grid.csv",
        "density_grid_250m.geojson",
        "density_analysis_metadata.json",
    ):
        assert (tmp_path / name).exists()

    sensitivity = pd.read_csv(tmp_path / "grid_size_sensitivity.csv", encoding="utf-8-sig")
    assert set(sensitivity["grid_size_m"]) == {100.0, 250.0, 500.0}
    assert int(sensitivity.loc[sensitivity["is_primary_grid"], "grid_size_m"].iloc[0]) == 250

    grid = gpd.read_file(tmp_path / "density_grid_250m.geojson")
    assert len(grid) > 0
    assert set(grid["density_class"].dropna().unique()) <= {"low", "medium", "high"}
    assert grid["is_empty"].any()


def test_existing_overall_comparison_metrics_are_unchanged(comparison_result):
    counts = comparison_result["counts"]
    assert counts["reference_count"] == 6
    assert counts["compared_count"] == 7
    assert counts["matched_reference_count"] == 5
    assert counts["matched_compared_count"] == 5
    assert counts["matched_group_count"] == 5
    assert counts["false_negative_count"] == 1
    assert counts["false_positive_count"] == 2
    assert counts["height_group_count"] == 4
    assert counts["height_mae_m"] == pytest.approx(1.0)
    assert counts["height_rmse_m"] == pytest.approx(1.0)
    assert counts["groups_1to1"] == 5
