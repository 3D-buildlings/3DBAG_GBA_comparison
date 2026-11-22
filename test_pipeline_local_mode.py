from __future__ import annotations

import hashlib
import json
import sys

import pytest

import input_sources as ins
import run_pipeline as rp


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_mode_pipeline_makes_no_remote_requests(
    local_files, tmp_path, monkeypatch
):
    gba_path, bag_path, _, _ = local_files
    gba_hash = _sha256(gba_path)
    bag_hash = _sha256(bag_path)

    monkeypatch.setattr(ins, "REQUIRED_DRIVES", (tmp_path,))
    monkeypatch.setattr(
        rp, "download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote download called"))
    )
    monkeypatch.setattr(
        rp, "select_tiles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tile index used"))
    )
    monkeypatch.setattr(
        rp, "download_tiles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tile download called"))
    )
    monkeypatch.setattr(
        rp, "fetch_gba_tiled", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GBA WFS called"))
    )

    out = tmp_path / "out"
    work = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--out",
            str(out),
            "--work",
            str(work),
            "--gba-file",
            str(gba_path),
            "--bag-file",
            str(bag_path),
        ],
    )
    rp.main()

    assert (out / "GBA_vs_3DBAG.geojson").exists()
    assert (out / "GBA_vs_3DBAG_false_positives.geojson").exists()
    assert (out / "comparison_summary.csv").exists()
    assert (out / "comparison_category_breakdown.csv").exists()
    assert (out / "run_metadata.json").exists()
    assert (out / "density_accuracy_summary.csv").exists()
    assert (out / "density_map.png").exists()
    assert (out / "geometric_accuracy_by_density.png").exists()
    assert (out / "height_accuracy_by_density.png").exists()
    assert (out / "density_analysis_report.md").exists()
    assert (out / "grid_size_sensitivity.csv").exists()
    assert (out / "density_grid_250m.geojson").exists()
    assert (out / "density_analysis_outputs.zip").exists()
    assert (out / "comparison_outputs.zip").exists()
    assert (out / "raw" / "lod12_clean.zip").exists()

    assert not (work / "tile_index.fgb").exists()
    assert not (work / "3dbag_tiles").exists()
    assert not (work / "gba_tiles").exists()

    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["input_mode"] == "local"
    assert meta["remote_dataset_retrieval"] is False
    assert meta["tdbag_version"] is None
    assert meta["results"]["GBA_vs_3DBAG"]["reference_count"] == 3
    assert meta["results"]["GBA_vs_3DBAG"]["compared_count"] == 4
    assert meta["results"]["GBA_vs_3DBAG"]["matched_reference_count"] == 3
    assert meta["results"]["GBA_vs_3DBAG"]["false_positive_count"] == 1
    assert meta["results"]["GBA_vs_3DBAG"]["false_negative_count"] == 0
    assert meta["density_analysis"]["thresholds"]["primary_grid_size_m"] == 250.0
    assert meta["local_inputs"]["gba"]["geometry"]["geometries_repaired"] == 0
    assert meta["local_inputs"]["tdbag"]["geometry"]["geometries_repaired"] == 0

    assert _sha256(gba_path) == gba_hash
    assert _sha256(bag_path) == bag_hash

    summary = (out / "density_accuracy_summary.csv").read_text(encoding="utf-8-sig")
    assert "low" in summary
    assert "medium" in summary
    assert "high" in summary


def test_single_explicit_path_stops_pipeline_before_output(local_files, tmp_path, monkeypatch):
    gba_path, _, _, _ = local_files
    monkeypatch.setattr(ins, "REQUIRED_DRIVES", (tmp_path,))
    out = tmp_path / "out_single"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--out",
            str(out),
            "--gba-file",
            str(gba_path),
        ],
    )
    with pytest.raises(ins.InputSelectionError, match="Both --gba-file and --bag-file"):
        rp.main()
    assert not out.exists()


def test_invalid_local_file_stops_without_comparison_outputs(
    local_files, tmp_path, monkeypatch
):
    gba_path, bag_path, _, _ = local_files
    bag_path.write_text("this is not a geojson dataset", encoding="utf-8")
    monkeypatch.setattr(ins, "REQUIRED_DRIVES", (tmp_path,))
    out = tmp_path / "out_invalid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--out",
            str(out),
            "--gba-file",
            str(gba_path),
            "--bag-file",
            str(bag_path),
        ],
    )
    with pytest.raises(ins.InputSelectionError, match="not a readable vector dataset"):
        rp.main()
    assert not (out / "GBA_vs_3DBAG.geojson").exists()
    assert not (out / "density_accuracy_summary.csv").exists()


def test_figures_flag_calls_figure_generator(local_files, tmp_path, monkeypatch):
    gba_path, bag_path, _, _ = local_files
    monkeypatch.setattr(ins, "REQUIRED_DRIVES", (tmp_path,))
    monkeypatch.setattr(
        rp, "download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote download called"))
    )
    monkeypatch.setattr(
        rp, "select_tiles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tile index used"))
    )
    monkeypatch.setattr(
        rp, "download_tiles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tile download called"))
    )
    monkeypatch.setattr(
        rp, "fetch_gba_tiled", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GBA WFS called"))
    )
    import make_figures_folder

    called = {}
    monkeypatch.setattr(
        make_figures_folder,
        "generate_figures",
        lambda out, **kwargs: called.setdefault("out", out),
    )
    out = tmp_path / "out_figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--out",
            str(out),
            "--work",
            str(tmp_path / "work_figures"),
            "--gba-file",
            str(gba_path),
            "--bag-file",
            str(bag_path),
            "--figures",
        ],
    )
    rp.main()
    assert called["out"] == out.resolve()
    assert (out / "density_accuracy_summary.csv").exists()
