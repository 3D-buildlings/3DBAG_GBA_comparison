from __future__ import annotations

from pathlib import Path

import pytest

import input_sources as ins


def _touch(path: Path, content: str = "{}") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_both_default_files_present_selects_local(tmp_path):
    gba = _touch(tmp_path / "gba.geojson")
    bag = _touch(tmp_path / "bag.geojson")
    selection = ins.select_inputs(default_gba=gba, default_bag=bag)
    assert selection.mode == "local"
    assert selection.gba_path == gba
    assert selection.bag_path == bag
    assert selection.remote_dataset_retrieval is False


def test_both_default_files_absent_selects_online(tmp_path):
    selection = ins.select_inputs(
        default_gba=tmp_path / "missing_gba.geojson",
        default_bag=tmp_path / "missing_bag.geojson",
    )
    assert selection.mode == "online"
    assert selection.gba_path is None
    assert selection.bag_path is None
    assert selection.remote_dataset_retrieval is True


def test_only_gba_file_exists_stops(tmp_path):
    gba = _touch(tmp_path / "gba.geojson")
    with pytest.raises(ins.InputSelectionError, match="3DBAG file"):
        ins.select_inputs(
            default_gba=gba, default_bag=tmp_path / "missing_bag.geojson"
        )


def test_only_bag_file_exists_stops(tmp_path):
    bag = _touch(tmp_path / "bag.geojson")
    with pytest.raises(ins.InputSelectionError, match="GBA file"):
        ins.select_inputs(
            default_gba=tmp_path / "missing_gba.geojson", default_bag=bag
        )


def test_existing_directory_instead_of_file_stops(tmp_path):
    gba = tmp_path / "gba_dir"
    gba.mkdir()
    bag = _touch(tmp_path / "bag.geojson")
    with pytest.raises(ins.InputSelectionError, match="not a regular file"):
        ins.select_inputs(default_gba=gba, default_bag=bag)


def test_zero_byte_file_stops(tmp_path):
    gba = tmp_path / "gba.geojson"
    gba.write_bytes(b"")
    bag = _touch(tmp_path / "bag.geojson")
    with pytest.raises(ins.InputSelectionError, match="empty"):
        ins.select_inputs(default_gba=gba, default_bag=bag)


def test_one_explicit_path_stops(tmp_path):
    gba = _touch(tmp_path / "gba.geojson")
    with pytest.raises(ins.InputSelectionError, match="Both --gba-file and --bag-file"):
        ins.select_inputs(explicit_gba=gba)


def test_both_explicit_paths_select_local(tmp_path):
    gba = _touch(tmp_path / "custom_gba.geojson")
    bag = _touch(tmp_path / "custom_bag.geojson")
    selection = ins.select_inputs(explicit_gba=gba, explicit_bag=bag)
    assert selection.mode == "local"
    assert selection.explicit is True
    assert selection.gba_path == gba
    assert selection.bag_path == bag


def test_explicit_paths_take_precedence_over_defaults(tmp_path):
    default_gba = _touch(tmp_path / "default_gba.geojson")
    default_bag = _touch(tmp_path / "default_bag.geojson")
    custom_gba = _touch(tmp_path / "custom_gba.geojson")
    custom_bag = _touch(tmp_path / "custom_bag.geojson")
    selection = ins.select_inputs(
        explicit_gba=custom_gba,
        explicit_bag=custom_bag,
        default_gba=default_gba,
        default_bag=default_bag,
    )
    assert selection.gba_path == custom_gba
    assert selection.bag_path == custom_bag


def test_explicit_missing_file_stops_without_online_fallback(tmp_path):
    gba = _touch(tmp_path / "custom_gba.geojson")
    with pytest.raises(ins.InputSelectionError, match="does not exist"):
        ins.select_inputs(explicit_gba=gba, explicit_bag=tmp_path / "missing.geojson")


def test_explicit_unusable_file_stops_without_online_fallback(tmp_path):
    gba = tmp_path / "custom_gba_dir"
    gba.mkdir()
    bag = _touch(tmp_path / "custom_bag.geojson")
    with pytest.raises(ins.InputSelectionError, match="cannot be used"):
        ins.select_inputs(explicit_gba=gba, explicit_bag=bag)


def test_verify_environment_accepts_accessible_paths(tmp_path):
    report = ins.verify_environment(
        target_dir=tmp_path,
        dataset_dirs=[tmp_path],
        required_drives=[tmp_path],
    )
    assert report["checks"]
    assert not report["failures"]


def test_verify_environment_missing_drive_fails(tmp_path):
    with pytest.raises(ins.EnvironmentAccessError, match="Required drive"):
        ins.verify_environment(
            target_dir=tmp_path,
            dataset_dirs=[],
            required_drives=[tmp_path / "missing_drive"],
        )


def test_verify_environment_missing_target_fails(tmp_path):
    with pytest.raises(ins.EnvironmentAccessError, match="Target project directory"):
        ins.verify_environment(
            target_dir=tmp_path / "missing_target",
            dataset_dirs=[],
            required_drives=[tmp_path],
        )


def test_verify_environment_inaccessible_dataset_dir_fails(tmp_path):
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    (dataset_dir / "file.txt").write_text("x", encoding="utf-8")
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(ins.EnvironmentAccessError, match="dataset directory"):
        ins.verify_environment(
            target_dir=tmp_path,
            dataset_dirs=[not_a_dir],
            required_drives=[tmp_path],
        )


def test_verify_environment_missing_dataset_dir_is_not_fatal(tmp_path):
    report = ins.verify_environment(
        target_dir=tmp_path,
        dataset_dirs=[tmp_path / "absent_dataset_dir"],
        required_drives=[tmp_path],
    )
    assert report["checks"]


def test_ensure_no_input_overwrite_detects_collision(tmp_path):
    source = _touch(tmp_path / "source.geojson")
    with pytest.raises(ins.InputSelectionError, match="Refusing to overwrite"):
        ins.ensure_no_input_overwrite([source], [source])
    ins.ensure_no_input_overwrite([source], [tmp_path / "other.geojson"])
