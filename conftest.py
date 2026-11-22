from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_lib as p  # noqa: E402

BASE_X = 121000.0
BASE_Y = 486000.0


def _square(x, y, size=10.0):
    return box(x, y, x + size, y + size)


def make_reference_gdf() -> gpd.GeoDataFrame:
    ids = [f"NL.IMBAG.Pand.TEST{i}" for i in range(1, 7)]
    xs = [BASE_X, BASE_X + 300, BASE_X + 400, BASE_X + 600, BASE_X + 650, BASE_X + 700]
    heights = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    geoms = [_square(x, BASE_Y) for x in xs]
    return gpd.GeoDataFrame(
        {
            "id": ids,
            "height_m": heights,
            "height_nap_m": [h + 5.0 for h in heights],
            "area_m2": [100.0] * len(ids),
            "city": ["Amsterdam"] * len(ids),
            "orig_height_m": [h - 1.0 for h in heights],
            "orig_height_nap_m": [h + 5.0 for h in heights],
            "orig_area_m2": [100.0] * len(ids),
            "site_ground_nap_m": [5.0] * len(ids),
            "site_3dbag_id": ids,
            "site_match": ["matched"] * len(ids),
        },
        geometry=geoms,
        crs=p.WORK_CRS,
    )


def make_gba_gdf() -> gpd.GeoDataFrame:
    xs = [BASE_X, BASE_X + 400, BASE_X + 600, BASE_X + 650, BASE_X + 700]
    heights = [11.0, None, 41.0, 51.0, 61.0]
    geoms = [_square(x + 1.0, BASE_Y) for x in xs]
    geoms.append(_square(BASE_X + 720, BASE_Y))
    geoms.append(_square(BASE_X + 900, BASE_Y))
    heights = heights + [5.0, 6.0]
    ids = [f"GBA_TEST{i}" for i in range(1, 8)]
    return gpd.GeoDataFrame(
        {
            "id": ids,
            "height_m": heights,
            "height_nap_m": [None] * len(ids),
            "area_m2": [100.0] * len(ids),
            "city": ["Amsterdam"] * len(ids),
            "site_height_m": [None] * len(ids),
            "site_area_m2": [None] * len(ids),
            "site_height_nap_m": [None] * len(ids),
            "site_ground_nap_m": [None] * len(ids),
            "site_3dbag_id": [None] * len(ids),
            "site_match": ["no_3dbag_match"] * len(ids),
        },
        geometry=geoms,
        crs=p.WORK_CRS,
    )


def make_small_gba_gdf() -> gpd.GeoDataFrame:
    xs = [BASE_X, BASE_X + 300, BASE_X + 400, BASE_X + 500]
    heights = [11.0, 19.0, 31.0, 5.0]
    ids = [f"GBA_SMALL{i}" for i in range(1, 5)]
    return gpd.GeoDataFrame(
        {
            "id": ids,
            "height_m": heights,
            "height_nap_m": [None] * 4,
            "area_m2": [100.0] * 4,
            "city": ["Amsterdam"] * 4,
            "site_height_m": [None] * 4,
            "site_area_m2": [None] * 4,
            "site_height_nap_m": [None] * 4,
            "site_ground_nap_m": [None] * 4,
            "site_3dbag_id": [None] * 4,
            "site_match": ["no_3dbag_match"] * 4,
        },
        geometry=[_square(x + 1.0, BASE_Y) for x in xs],
        crs=p.WORK_CRS,
    )


@pytest.fixture
def local_files(tmp_path):
    ref = make_reference_gdf().iloc[:3].copy().reset_index(drop=True)
    cmp = make_small_gba_gdf()
    bag_path = tmp_path / "bag.geojson"
    gba_path = tmp_path / "gba.geojson"
    ref.to_file(bag_path, driver="GeoJSON")
    cmp.to_file(gba_path, driver="GeoJSON")
    return gba_path, bag_path, cmp, ref


@pytest.fixture
def comparison_result():
    ref = make_reference_gdf()
    cmp = make_gba_gdf()
    ref_internal = gpd.GeoDataFrame(
        {
            "ref_id": ref["id"].astype(str),
            "ref_bag_id": ref["id"].astype(str),
            "ref_source": "3DBAG",
            "height_3dbag_m": ref["height_m"],
            "height_3dbag_50p_m": ref["height_m"],
            "height_3dbag_max_m": ref["height_m"],
            "height_3dbag_definition": "test reference",
            "height_3dbag_max_definition": "test reference",
            "h_roof50_3dbag_m": ref["height_nap_m"],
            "h_roofmax_3dbag_m": ref["height_nap_m"],
            "h_ground_3dbag_m": ref["site_ground_nap_m"],
            "h_nok_3dbag_m": [None] * len(ref),
            "ref_area_m2": ref.geometry.area,
            "city": ref["city"],
        },
        geometry=ref.geometry,
        crs=p.WORK_CRS,
    )
    cmp_internal = gpd.GeoDataFrame(
        {
            "cmp_id": cmp["id"].astype(str),
            "cmp_source_id": cmp["id"].astype(str),
            "cmp_source": "GBA",
            "height_compared_m": cmp["height_m"],
            "height_compared_50p_m": cmp["height_m"],
            "height_compared_max_m": cmp["height_m"],
            "height_compared_method": "test compared",
            "height_compared_definition": "test compared",
            "h_roof50_cmp_m": [None] * len(cmp),
            "h_ground_cmp_m": [None] * len(cmp),
            "h_roof70_cmp_m": [None] * len(cmp),
            "h_roofmax_cmp_m": [None] * len(cmp),
            "year_built_cmp": [None] * len(cmp),
            "status_cmp": [None] * len(cmp),
            "n_parts_cmp": [1] * len(cmp),
            "gba_source": [None] * len(cmp),
            "gba_var": [None] * len(cmp),
            "cmp_area_m2": cmp.geometry.area,
            "city": cmp["city"],
        },
        geometry=cmp.geometry,
        crs=p.WORK_CRS,
    )
    res = p.compare_datasets(ref_internal, cmp_internal, "GBA_vs_3DBAG", height_mode="50p")
    return res


@pytest.fixture
def density_provenance():
    return {
        "input_mode": "local",
        "gba_source": "test gba",
        "tdbag_source": "test 3dbag",
        "remote_dataset_retrieval": False,
        "gba": {},
        "tdbag": {},
    }
