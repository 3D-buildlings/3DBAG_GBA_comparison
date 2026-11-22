from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

import input_sources as ins
from run_pipeline import city_boxes_rd

from conftest import make_gba_gdf, make_reference_gdf


@pytest.fixture
def city_boxes():
    return city_boxes_rd()


def test_local_gba_validation_and_normalization(tmp_path, city_boxes):
    path = tmp_path / "gba.geojson"
    make_gba_gdf().to_file(path, driver="GeoJSON")
    gdf, info = ins.load_local_gba(path, city_boxes)
    assert len(gdf) == 7
    assert info["height_column"] == "height_m"
    assert info["geometry"]["geometries_repaired"] == 0
    assert info["height_stats"]["valid_for_height_comparison"] == 6
    assert info["height_stats"]["missing_values"] == 1


def test_local_gba_repairs_invalid_geometry(tmp_path, city_boxes):
    gdf = make_gba_gdf()
    gdf.loc[0, "geometry"] = Polygon(
        [
            (121000, 486000),
            (121010, 486010),
            (121010, 486000),
            (121000, 486010),
        ]
    )
    path = tmp_path / "gba_invalid.geojson"
    gdf.to_file(path, driver="GeoJSON")
    loaded, info = ins.load_local_gba(path, city_boxes)
    assert len(loaded) == len(gdf)
    assert info["geometry"]["invalid_geometries_found"] == 1
    assert info["geometry"]["geometries_repaired"] == 1
    assert loaded.geometry.is_valid.all()


def test_local_gba_missing_height_column_stops(tmp_path, city_boxes):
    gdf = make_gba_gdf().drop(columns=["height_m"])
    path = tmp_path / "gba_no_height.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ins.InputSelectionError, match="height column"):
        ins.load_local_gba(path, city_boxes)


def test_local_gba_all_non_numeric_heights_stops(tmp_path, city_boxes):
    gdf = make_gba_gdf()
    gdf["height_m"] = "not-a-number"
    path = tmp_path / "gba_bad_height.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ins.InputSelectionError, match="no finite numeric values"):
        ins.load_local_gba(path, city_boxes)


def test_local_gba_non_polygonal_geometry_stops(tmp_path, city_boxes):
    gdf = make_gba_gdf()
    gdf.loc[0, "geometry"] = Point(121000, 486000)
    path = tmp_path / "gba_point.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ins.InputSelectionError, match="cannot be repaired"):
        ins.load_local_gba(path, city_boxes)


def test_local_3dbag_uses_roof_ground_derivation(tmp_path, city_boxes):
    gdf = make_reference_gdf()
    path = tmp_path / "bag.geojson"
    gdf.to_file(path, driver="GeoJSON")
    loaded, info = ins.load_local_3dbag(path, city_boxes)
    assert len(loaded) == 6
    assert "derived" in info["height_source"]
    assert info["height_stats"]["valid_for_height_comparison"] == 6
    clean, ref, site_map = ins.build_3dbag_layers_from_local(
        loaded, tmp_path, city_boxes, info
    )
    assert len(clean) == 6
    assert len(ref) == 6
    assert len(site_map) == 6
    assert clean["height_m"].notna().all()
    assert clean["height_m"].iloc[0] == pytest.approx(10.0)


def test_local_3dbag_uses_height_attribute_when_derivation_unavailable(tmp_path, city_boxes):
    gdf = make_reference_gdf().drop(columns=["height_nap_m", "site_ground_nap_m"])
    path = tmp_path / "bag_height_only.geojson"
    gdf.to_file(path, driver="GeoJSON")
    loaded, info = ins.load_local_3dbag(path, city_boxes)
    assert info["height_source"] == "provided height column height_m"
    clean, ref, site_map = ins.build_3dbag_layers_from_local(
        loaded, tmp_path, city_boxes, info
    )
    assert clean["height_m"].iloc[0] == pytest.approx(10.0)


def test_local_3dbag_no_usable_height_stops(tmp_path, city_boxes):
    gdf = make_reference_gdf().drop(
        columns=["height_m", "height_nap_m", "site_ground_nap_m", "orig_height_nap_m"]
    )
    path = tmp_path / "bag_no_height.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ins.InputSelectionError, match="does not contain usable building heights"):
        ins.load_local_3dbag(path, city_boxes)


def test_local_3dbag_duplicate_ids_stop(tmp_path, city_boxes):
    gdf = make_reference_gdf()
    gdf.loc[1, "id"] = gdf.loc[0, "id"]
    path = tmp_path / "bag_duplicate_ids.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ins.InputSelectionError, match="duplicate identifiers"):
        ins.load_local_3dbag(path, city_boxes)
