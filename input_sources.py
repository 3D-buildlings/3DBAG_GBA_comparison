#!/usr/bin/env python3
"""Local GeoJSON input selection, validation, and normalization.

The comparison pipeline can read both inputs from local GeoJSON files or from
the original online sources.  Local and online inputs are never mixed.  The
functions here perform the environment checks, decide the input mode, validate
the local datasets, repair invalid polygonal geometries where possible, and
normalize the local files to the internal structures used by the comparison
engine.
"""

from __future__ import annotations

import json
import math
import os
import stat as stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import Transformer

import pipeline_lib as p

DEFAULT_GBA_FILE = Path(r"D:\GBA_3DBAG_outputs\raw\GBA_Amsterdam_Utrecht_raw.geojson")
DEFAULT_BAG_FILE = Path(
    r"D:\GBA_3DBAG_outputs\raw\3DBAG_Amsterdam_Utrecht_lod12_clean.geojson"
)
DEFAULT_DATASET_DIR = DEFAULT_GBA_FILE.parent
def _env_required_drives():
    raw = os.environ.get("GBA_3DBAG_REQUIRED_DRIVES", "")
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return tuple(Path(value) for value in values)


REQUIRED_DRIVES = _env_required_drives()
CITY_LABEL = {"amsterdam": "Amsterdam", "utrecht": "Utrecht"}

GBA_ID_ALIASES = ("cmp_id", "id", "fid")
GBA_HEIGHT_ALIASES = ("orig_height_m", "height_compared_m", "height_m", "height")
BAG_ID_ALIASES = ("id", "ref_id", "pand_id", "bag_id")
BAG_HEIGHT_ALIASES = ("height_m", "height_3dbag_m", "height")
BAG_ROOF_ALIASES = ("height_nap_m", "h_roof50_3dbag_m", "zmax", "roof_nap_m")
BAG_GROUND_ALIASES = ("site_ground_nap_m", "h_ground_3dbag_m", "zmin", "ground_nap_m")
BAG_ORIG_HEIGHT_ALIASES = ("orig_height_m",)
BAG_ORIG_ROOF_ALIASES = ("orig_height_nap_m", "h_roof50_3dbag_m")
BAG_ORIG_AREA_ALIASES = ("orig_area_m2",)
BAG_SITE_AREA_ALIASES = ("area_m2", "b3_opp_grond", "site_area_m2")
BAG_SITE_MATCH_ALIASES = ("site_match",)

BAG_DERIVED_HEIGHT_DEFINITION = "3DBAG LoD1.2 site height (z_roof - z_base)"
BAG_ATTRIBUTE_HEIGHT_DEFINITION = "3DBAG building height from the local height attribute"


class EnvironmentAccessError(RuntimeError):
    """Raised when a required drive or directory is inaccessible."""


class InputSelectionError(RuntimeError):
    """Raised when the local/online input-selection rules cannot be satisfied."""


@dataclass
class FileStatus:
    path: Path
    label: str
    status: str
    detail: str


@dataclass
class InputSelection:
    mode: str
    gba_path: Path | None
    bag_path: Path | None
    reason: str
    explicit: bool
    remote_dataset_retrieval: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_directory(path: Path, label: str) -> dict:
    result = {
        "label": label,
        "path": str(path),
        "exists": False,
        "is_directory": False,
        "accessible": False,
        "detail": "",
    }
    try:
        exists = path.exists()
    except OSError as exc:
        result["detail"] = f"cannot stat path: {exc}"
        return result
    result["exists"] = bool(exists)
    if not exists:
        result["detail"] = "path does not exist"
        return result
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        result["detail"] = f"cannot determine whether path is a directory: {exc}"
        return result
    result["is_directory"] = bool(is_directory)
    if not is_directory:
        result["detail"] = "path exists but is not a directory"
        return result
    try:
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError as exc:
        result["detail"] = f"directory cannot be read: {exc}"
        return result
    result["accessible"] = True
    result["detail"] = "accessible"
    return result


def verify_environment(
    target_dir: Path,
    dataset_dirs=(),
    required_drives=None,
    fail_on_missing_dataset_dir: bool = False,
) -> dict:
    """Verify drives, target directory, and dataset directories before use."""
    if required_drives is None:
        required_drives = REQUIRED_DRIVES
    checks = []
    failures = []
    for drive in required_drives:
        check = _probe_directory(Path(drive), "required drive")
        checks.append(check)
        if check["accessible"]:
            p.log(f"Environment check: required drive {check['path']} is accessible.")
        else:
            message = f"Required drive {check['path']} is not accessible: {check['detail']}"
            failures.append(message)
            p.log(f"Environment check: FAILED - {message}")

    target_check = _probe_directory(Path(target_dir), "target directory")
    checks.append(target_check)
    if target_check["accessible"]:
        p.log(f"Environment check: target directory {target_check['path']} is accessible.")
    else:
        message = (
            f"Target project directory {target_check['path']} is not accessible: "
            f"{target_check['detail']}"
        )
        failures.append(message)
        p.log(f"Environment check: FAILED - {message}")

    for directory in dataset_dirs:
        check = _probe_directory(Path(directory), "configured dataset directory")
        checks.append(check)
        if check["accessible"]:
            p.log(f"Environment check: dataset directory {check['path']} is accessible.")
        elif not check["exists"]:
            message = (
                f"Configured dataset directory {check['path']} does not exist; "
                "configured files at this location are treated as absent."
            )
            if fail_on_missing_dataset_dir:
                failures.append(message)
                p.log(f"Environment check: FAILED - {message}")
            else:
                p.log(f"Environment check: {message}")
        else:
            message = (
                f"Configured dataset directory {check['path']} is not accessible: "
                f"{check['detail']}"
            )
            failures.append(message)
            p.log(f"Environment check: FAILED - {message}")

    if failures:
        raise EnvironmentAccessError(
            "Environment verification failed:\n" + "\n".join(f"- {m}" for m in failures)
        )

    return {
        "verified_utc": _utc_now(),
        "target_directory": str(Path(target_dir)),
        "checks": checks,
        "failures": [],
    }


def probe_file(path: Path, label: str) -> FileStatus:
    """Classify a configured local file as missing, present, or unusable."""
    path = Path(path)
    try:
        st = path.stat()
    except FileNotFoundError:
        return FileStatus(path, label, "missing", "file does not exist")
    except OSError as exc:
        return FileStatus(path, label, "error", f"cannot stat file: {exc}")

    if not stat_module.S_ISREG(st.st_mode):
        return FileStatus(path, label, "error", "path exists but is not a regular file")
    if st.st_size <= 0:
        return FileStatus(path, label, "error", "file is empty (0 bytes)")
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        return FileStatus(path, label, "error", f"file exists but cannot be read: {exc}")
    return FileStatus(path, label, "present", f"readable, {st.st_size:,} bytes")


def _file_statuses(gba_path: Path, bag_path: Path) -> tuple[FileStatus, FileStatus]:
    return probe_file(gba_path, "GBA"), probe_file(bag_path, "3DBAG")


def select_inputs(
    explicit_gba: str | Path | None = None,
    explicit_bag: str | Path | None = None,
    default_gba: Path = DEFAULT_GBA_FILE,
    default_bag: Path = DEFAULT_BAG_FILE,
) -> InputSelection:
    """Apply the local/online input-selection rules exactly."""
    has_gba = explicit_gba is not None
    has_bag = explicit_bag is not None
    if has_gba != has_bag:
        missing_flag = "--bag-file" if has_gba else "--gba-file"
        raise InputSelectionError(
            "Both --gba-file and --bag-file must be supplied together. "
            f"Only one explicit local path was supplied; {missing_flag} is missing. "
            "Local and online sources are never mixed."
        )

    if has_gba and has_bag:
        gba_path = Path(explicit_gba).expanduser()
        bag_path = Path(explicit_bag).expanduser()
        gba_status, bag_status = _file_statuses(gba_path, bag_path)
        for status in (gba_status, bag_status):
            if status.status == "error":
                raise InputSelectionError(
                    f"The explicitly supplied {status.label} file cannot be used: "
                    f"{status.path}: {status.detail}. Online fallback is not allowed."
                )
            if status.status == "missing":
                raise InputSelectionError(
                    f"The explicitly supplied {status.label} file does not exist: "
                    f"{status.path}. Online fallback is not allowed."
                )
        selection = InputSelection(
            mode="local",
            gba_path=gba_path,
            bag_path=bag_path,
            reason="Both explicit local dataset paths were supplied and are readable.",
            explicit=True,
            remote_dataset_retrieval=False,
        )
        p.log("Input selection: Local Input Mode (explicit --gba-file and --bag-file).")
        p.log(f"  GBA source: {gba_path}")
        p.log(f"  3DBAG source: {bag_path}")
        p.log("  Remote dataset retrieval: disabled.")
        return selection

    gba_status, bag_status = _file_statuses(Path(default_gba), Path(default_bag))
    for status in (gba_status, bag_status):
        if status.status == "error":
            raise InputSelectionError(
                f"The configured local {status.label} file exists but cannot be used: "
                f"{status.path}: {status.detail}. The run stops; online fallback is not allowed."
            )

    gba_state = gba_status.status
    bag_state = bag_status.status
    if gba_state == "present" and bag_state == "present":
        selection = InputSelection(
            mode="local",
            gba_path=Path(default_gba),
            bag_path=Path(default_bag),
            reason="Both configured default local files are present and readable.",
            explicit=False,
            remote_dataset_retrieval=False,
        )
        p.log("Input selection: Local Input Mode (both default local files are present).")
        p.log(f"  GBA source: {selection.gba_path}")
        p.log(f"  3DBAG source: {selection.bag_path}")
        p.log("  Remote dataset retrieval: disabled.")
        return selection

    if gba_state == "missing" and bag_state == "missing":
        selection = InputSelection(
            mode="online",
            gba_path=None,
            bag_path=None,
            reason="Neither configured default local file is present.",
            explicit=False,
            remote_dataset_retrieval=True,
        )
        p.log("Input selection: Online Mode (both default local files are absent).")
        p.log(f"  GBA source: GBA viewer WFS ({p.GBA_WFS})")
        p.log(f"  3DBAG source: pinned 3DBAG CityJSON tiles")
        p.log("  Remote dataset retrieval: enabled.")
        return selection

    if gba_state == "present" and bag_state == "missing":
        raise InputSelectionError(
            f"The GBA file exists at {gba_status.path}, but the configured 3DBAG file "
            f"is missing at {bag_status.path}. The run stops; local and online sources "
            "are never mixed. Supply both local files or remove the existing local file "
            "to use Online Mode."
        )
    if gba_state == "missing" and bag_state == "present":
        raise InputSelectionError(
            f"The 3DBAG file exists at {bag_status.path}, but the configured GBA file "
            f"is missing at {gba_status.path}. The run stops; local and online sources "
            "are never mixed. Supply both local files or remove the existing local file "
            "to use Online Mode."
        )
    raise InputSelectionError(
        f"Unhandled input-selection state: GBA={gba_state}, 3DBAG={bag_state}."
    )


def ensure_no_input_overwrite(input_paths, output_paths) -> None:
    inputs = {Path(path).resolve() for path in input_paths if path is not None}
    collisions = []
    for path in output_paths:
        if Path(path).resolve() in inputs:
            collisions.append(str(Path(path).resolve()))
    if collisions:
        raise InputSelectionError(
            "Refusing to overwrite a local input file. Choose a different --out "
            f"directory. Colliding output path(s): {', '.join(collisions)}"
        )


def _find_column(gdf: gpd.GeoDataFrame, aliases) -> str | None:
    for name in aliases:
        if name in gdf.columns:
            return name
    return None


def _read_vector(path: Path, label: str) -> gpd.GeoDataFrame:
    try:
        gdf = pyogrio.read_dataframe(path)
    except Exception as exc:
        raise InputSelectionError(
            f"The local {label} file is not a readable vector dataset: {path}: {exc}"
        ) from exc
    if gdf.empty:
        raise InputSelectionError(f"The local {label} file contains no features: {path}")
    gdf = gdf.reset_index(drop=True)
    return gdf


def _validate_crs(gdf: gpd.GeoDataFrame, label: str, path: Path) -> str:
    if gdf.crs is None:
        raise InputSelectionError(
            f"The local {label} file has no CRS definition: {path}. "
            "A CRS is required for metric comparison."
        )
    try:
        Transformer.from_crs(gdf.crs, p.WORK_CRS, always_xy=True)
    except Exception as exc:
        raise InputSelectionError(
            f"The local {label} file has an unsupported CRS ({gdf.crs}): {path}: {exc}"
        ) from exc
    return str(gdf.crs)


def _repair_geometries(gdf: gpd.GeoDataFrame, label: str, path: Path):
    geom = gdf.geometry
    missing = int(geom.isna().sum())
    if missing:
        raise InputSelectionError(
            f"The local {label} file contains {missing} missing geometries: {path}. "
            "Missing geometries are not dropped silently."
        )
    empty = int(geom.is_empty.sum())
    if empty:
        raise InputSelectionError(
            f"The local {label} file contains {empty} empty geometries: {path}. "
            "Empty geometries are not dropped silently."
        )

    type_counts = {str(k): int(v) for k, v in geom.geom_type.value_counts(dropna=False).items()}
    allowed = np.isin(geom.geom_type.values, ["Polygon", "MultiPolygon"])
    invalid = ~shapely.is_valid(geom.values)
    needs_repair = (~allowed) | invalid
    repair_positions = np.where(needs_repair)[0]
    non_polygonal = int((~allowed).sum())
    invalid_count = int(invalid.sum())

    if len(repair_positions):
        repaired = geom.values.copy()
        for position in repair_positions:
            new_geom = p.clean_polygonal(repaired[position])
            if new_geom is None:
                raise InputSelectionError(
                    f"The local {label} file contains a geometry that cannot be repaired "
                    f"to a polygonal footprint (feature index {int(position)}, "
                    f"type {geom.iloc[position].geom_type}): {path}."
                )
            repaired[position] = new_geom
        gdf = gdf.set_geometry(gpd.GeoSeries(repaired, index=gdf.index, crs=gdf.crs))
        p.log(
            f"Local {label} geometry repair: {len(repair_positions)} feature(s) repaired "
            f"({invalid_count} invalid, {non_polygonal} non-polygonal)."
        )

    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise InputSelectionError(
            f"The local {label} file still contains missing or empty geometries after repair: {path}."
        )
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise InputSelectionError(
            f"The local {label} file contains non-polygonal geometries after repair: {path}."
        )
    if not bool(shapely.is_valid(gdf.geometry.values).all()):
        raise InputSelectionError(
            f"The local {label} file still contains invalid geometries after repair: {path}."
        )

    return gdf, {
        "geometry_types": type_counts,
        "invalid_geometries_found": invalid_count,
        "non_polygonal_geometries_found": non_polygonal,
        "geometries_repaired": int(len(repair_positions)),
        "missing_geometries": missing,
        "empty_geometries": empty,
    }


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite(series: pd.Series) -> pd.Series:
    values = _numeric_series(series)
    return values.notna() & np.isfinite(values)


def _height_value_stats(series: pd.Series, label: str, require_positive: bool = False) -> dict:
    raw = series
    numeric = _numeric_series(raw)
    raw_non_null = raw.notna()
    numeric_non_null = numeric.notna()
    non_numeric = int((raw_non_null & ~numeric_non_null).sum())
    non_finite = int((numeric_non_null & ~np.isfinite(numeric)).sum())
    finite_mask = numeric_non_null & np.isfinite(numeric)
    positive_mask = finite_mask & (numeric > 0)
    stats = {
        "field": label,
        "count": int(len(raw)),
        "missing_values": int((~raw_non_null).sum()),
        "non_numeric_values": non_numeric,
        "non_finite_values": non_finite,
        "finite_values": int(finite_mask.sum()),
        "non_positive_values": int((finite_mask & ~positive_mask).sum()),
    }
    if require_positive:
        stats["valid_for_height_comparison"] = int(positive_mask.sum())
    else:
        stats["valid_for_height_comparison"] = int(finite_mask.sum())
    return stats


def load_local_gba(path: str | Path, city_boxes) -> tuple[gpd.GeoDataFrame, dict]:
    """Read, validate, and repair a local GBA GeoJSON file."""
    path = Path(path)
    p.log(f"Reading local GBA file: {path}")
    gdf = _read_vector(path, "GBA")
    source_crs = _validate_crs(gdf, "GBA", path)
    gdf, geometry_info = _repair_geometries(gdf, "GBA", path)

    id_col = _find_column(gdf, GBA_ID_ALIASES)
    generated_ids = 0
    if id_col is None:
        p.log("Local GBA file has no id/cmp_id/fid column; generating GBA_<row> identifiers.")
        gdf["cmp_id"] = [f"GBA_{i}" for i in range(len(gdf))]
        id_col = "cmp_id"
        generated_ids = len(gdf)
    else:
        ids = gdf[id_col].astype(object)
        missing_ids = ids.isna()
        if bool(missing_ids.any()):
            generated_ids = int(missing_ids.sum())
            p.log(
                f"Local GBA file has {generated_ids} missing identifiers in {id_col}; "
                "filling only those rows with GBA_<row> identifiers."
            )
            ids = ids.where(~missing_ids, pd.Series([f"GBA_{i}" for i in range(len(gdf))], index=gdf.index))
            gdf[id_col] = ids
    if gdf[id_col].astype(str).duplicated().any():
        duplicates = int(gdf[id_col].astype(str).duplicated().sum())
        raise InputSelectionError(
            f"The local GBA file has {duplicates} duplicate identifiers in {id_col}: {path}. "
            "Identifiers must be unique for matching."
        )

    height_col = _find_column(gdf, GBA_HEIGHT_ALIASES)
    if height_col is None:
        raise InputSelectionError(
            f"The local GBA file does not contain a usable height column. "
            f"Expected one of {list(GBA_HEIGHT_ALIASES)}: {path}."
        )
    height_numeric = _numeric_series(gdf[height_col])
    height_stats = _height_value_stats(gdf[height_col], height_col)
    if not bool((height_numeric.notna() & np.isfinite(height_numeric)).any()):
        raise InputSelectionError(
            f"The local GBA height column {height_col!r} contains no finite numeric values: {path}."
        )
    if height_stats["non_numeric_values"] or height_stats["non_finite_values"]:
        p.log(
            f"Local GBA height validation: {height_stats['non_numeric_values']} non-numeric "
            f"and {height_stats['non_finite_values']} non-finite values will remain null "
            "and are not replaced."
        )
    if height_stats["non_positive_values"]:
        p.log(
            f"Local GBA height validation: {height_stats['non_positive_values']} non-positive "
            "height value(s) are retained as recorded; height metrics use the existing rules."
        )

    info = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "source_crs": source_crs,
        "working_crs": p.WORK_CRS,
        "feature_count": int(len(gdf)),
        "id_column": id_col,
        "generated_ids": generated_ids,
        "height_column": height_col,
        "height_stats": height_stats,
        "geometry": geometry_info,
        "columns": [str(c) for c in gdf.columns],
    }
    p.log(
        f"Local GBA validated: {len(gdf)} features, CRS {source_crs}, "
        f"id={id_col}, height={height_col}, repaired geometries={geometry_info['geometries_repaired']}."
    )
    return gdf, info


def _numeric_or_none(series: pd.Series) -> pd.Series:
    values = _numeric_series(series)
    values = values.where(np.isfinite(values), np.nan)
    return values


def load_local_3dbag(path: str | Path, city_boxes) -> tuple[gpd.GeoDataFrame, dict]:
    """Read, validate, and repair a local 3DBAG GeoJSON file."""
    path = Path(path)
    p.log(f"Reading local 3DBAG file: {path}")
    gdf = _read_vector(path, "3DBAG")
    source_crs = _validate_crs(gdf, "3DBAG", path)
    gdf, geometry_info = _repair_geometries(gdf, "3DBAG", path)

    id_col = _find_column(gdf, BAG_ID_ALIASES)
    generated_ids = 0
    if id_col is None:
        p.log("Local 3DBAG file has no id/ref_id/pand_id/bag_id column; generating 3DBAG_<row> identifiers.")
        gdf["id"] = [f"3DBAG_{i}" for i in range(len(gdf))]
        id_col = "id"
        generated_ids = len(gdf)
    else:
        ids = gdf[id_col].astype(object)
        missing_ids = ids.isna()
        if bool(missing_ids.any()):
            generated_ids = int(missing_ids.sum())
            p.log(
                f"Local 3DBAG file has {generated_ids} missing identifiers in {id_col}; "
                "filling only those rows with 3DBAG_<row> identifiers."
            )
            ids = ids.where(~missing_ids, pd.Series([f"3DBAG_{i}" for i in range(len(gdf))], index=gdf.index))
            gdf[id_col] = ids
    if gdf[id_col].astype(str).duplicated().any():
        duplicates = int(gdf[id_col].astype(str).duplicated().sum())
        raise InputSelectionError(
            f"The local 3DBAG file has {duplicates} duplicate identifiers in {id_col}: {path}. "
            "Identifiers must be unique for matching."
        )

    height_col = _find_column(gdf, BAG_HEIGHT_ALIASES)
    roof_col = _find_column(gdf, BAG_ROOF_ALIASES)
    ground_col = _find_column(gdf, BAG_GROUND_ALIASES)
    orig_height_col = _find_column(gdf, BAG_ORIG_HEIGHT_ALIASES)
    orig_roof_col = _find_column(gdf, BAG_ORIG_ROOF_ALIASES)
    site_area_col = _find_column(gdf, BAG_SITE_AREA_ALIASES)
    site_match_col = _find_column(gdf, BAG_SITE_MATCH_ALIASES)

    roof = _numeric_or_none(gdf[roof_col]) if roof_col else None
    ground = _numeric_or_none(gdf[ground_col]) if ground_col else None
    provided_height = _numeric_or_none(gdf[height_col]) if height_col else None

    derived_height = None
    derived_rows = 0
    if roof is not None and ground is not None:
        both = roof.notna() & ground.notna()
        derived_rows = int(both.sum())
        if derived_rows:
            derived_height = (roof - ground).where(both, np.nan)

    consistency = None
    if derived_height is not None and provided_height is not None:
        both = derived_height.notna() & provided_height.notna()
        if bool(both.any()):
            diff = (provided_height[both] - derived_height[both]).abs()
            consistency = {
                "compared_rows": int(both.sum()),
                "max_abs_difference_m": float(diff.max()),
                "mean_abs_difference_m": float(diff.mean()),
                "rows_above_0_06_m": int((diff > 0.06).sum()),
            }
            p.log(
                "Local 3DBAG height consistency: provided height field vs roof-ground "
                f"derivation, n={consistency['compared_rows']}, max |diff|="
                f"{consistency['max_abs_difference_m']:.3f} m, rows>0.06 m="
                f"{consistency['rows_above_0_06_m']}."
            )

    if derived_height is not None and bool(derived_height.notna().any()):
        height_values = derived_height
        height_definition = BAG_DERIVED_HEIGHT_DEFINITION
        height_source = f"derived as {roof_col} - {ground_col}"
        height_field = f"{roof_col} - {ground_col}"
    elif provided_height is not None and bool(provided_height.notna().any()):
        height_values = provided_height
        height_definition = BAG_ATTRIBUTE_HEIGHT_DEFINITION
        height_source = f"provided height column {height_col}"
        height_field = str(height_col)
    else:
        raise InputSelectionError(
            f"The local 3DBAG file does not contain usable building heights. Expected "
            f"one of {list(BAG_HEIGHT_ALIASES)} or a roof/ground pair from "
            f"{list(BAG_ROOF_ALIASES)} and {list(BAG_GROUND_ALIASES)}: {path}."
        )

    height_stats = _height_value_stats(height_values, height_field)
    if not bool((height_values.notna() & np.isfinite(height_values)).any()):
        raise InputSelectionError(
            f"The local 3DBAG height values contain no finite numeric values: {path}."
        )
    if height_stats["non_positive_values"]:
        p.log(
            f"Local 3DBAG height validation: {height_stats['non_positive_values']} non-positive "
            "height value(s) are retained as recorded; height metrics use the existing rules."
        )
    if provided_height is not None and height_col != height_field:
        provided_stats = _height_value_stats(provided_height, str(height_col))
        p.log(
            f"Local 3DBAG provided height field {height_col!r}: "
            f"{provided_stats['valid_for_height_comparison']} finite value(s), "
            f"{provided_stats['missing_values']} missing, "
            f"{provided_stats['non_numeric_values']} non-numeric."
        )

    if site_area_col is not None:
        site_area = _numeric_or_none(gdf[site_area_col])
        area_source = f"provided column {site_area_col}"
        area_stats = _height_value_stats(gdf[site_area_col], site_area_col)
    else:
        site_area = None
        area_source = "geometry area (no area attribute present)"
        area_stats = {"field": None, "count": int(len(gdf)), "missing_values": int(len(gdf))}
        p.log(
            "Local 3DBAG file has no area_m2/b3_opp_grond/site_area_m2 column; "
            "site areas will be derived from reprojected geometry where needed."
        )

    if orig_roof_col is not None:
        orig_roof = _numeric_or_none(gdf[orig_roof_col])
    else:
        orig_roof = None
    orig_height = None
    if orig_roof is not None and ground is not None:
        both = orig_roof.notna() & ground.notna()
        if bool(both.any()):
            orig_height = (orig_roof - ground).where(both, np.nan)
    if orig_height is None and orig_height_col is not None:
        orig_height = _numeric_or_none(gdf[orig_height_col])
    if orig_height is None:
        orig_height = pd.Series([np.nan] * len(gdf), index=gdf.index)

    info = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "source_crs": source_crs,
        "working_crs": p.WORK_CRS,
        "feature_count": int(len(gdf)),
        "id_column": id_col,
        "generated_ids": generated_ids,
        "height_field": height_field,
        "height_source": height_source,
        "height_definition": height_definition,
        "height_stats": height_stats,
        "derived_height_rows": derived_rows,
        "roof_column": roof_col,
        "ground_column": ground_col,
        "provided_height_column": height_col,
        "height_consistency": consistency,
        "site_area_column": site_area_col,
        "site_area_source": area_source,
        "site_area_stats": area_stats,
        "orig_height_column": orig_height_col,
        "orig_roof_column": orig_roof_col,
        "site_match_column": site_match_col,
        "geometry": geometry_info,
        "columns": [str(c) for c in gdf.columns],
    }
    p.log(
        f"Local 3DBAG validated: {len(gdf)} features, CRS {source_crs}, "
        f"id={id_col}, height={height_field}, repaired geometries={geometry_info['geometries_repaired']}."
    )
    return gdf, info


def _round_half_up(value: float, ndigits: int) -> float:
    factor = 10**ndigits
    return math.floor(value * factor + 0.5) / factor


def _round_or_none(value, ndigits: int, half: bool = False):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    number = float(value)
    return _round_half_up(number, ndigits) if half else round(number, ndigits)


def build_3dbag_layers_from_local(
    gdf: gpd.GeoDataFrame,
    out_raw: Path,
    city_boxes,
    height_info: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Build the same clean/reference structures used by the online pipeline."""
    g = gdf.to_crs(p.WORK_CRS).copy()
    geom_area = g.geometry.area.astype(float)

    if height_info["roof_column"] is not None:
        roof = _numeric_or_none(g[height_info["roof_column"]])
    else:
        roof = None
    if height_info["ground_column"] is not None:
        ground = _numeric_or_none(g[height_info["ground_column"]])
    else:
        ground = None
    if height_info["provided_height_column"] is not None:
        provided_height = _numeric_or_none(g[height_info["provided_height_column"]])
    else:
        provided_height = None

    if roof is not None and ground is not None:
        both = roof.notna() & ground.notna()
        site_h = (roof - ground).where(both, np.nan)
    elif provided_height is not None:
        site_h = provided_height.copy()
    else:
        raise InputSelectionError("No usable 3DBAG height values were available for normalization.")

    if height_info["site_area_column"] is not None:
        site_area = _numeric_or_none(g[height_info["site_area_column"]])
    else:
        site_area = geom_area.copy()

    orig_roof = None
    if height_info["orig_roof_column"] is not None:
        orig_roof = _numeric_or_none(g[height_info["orig_roof_column"]])
    if orig_roof is not None and ground is not None:
        both = orig_roof.notna() & ground.notna()
        orig_h = (orig_roof - ground).where(both, np.nan)
    elif height_info.get("orig_height_column") and height_info["orig_height_column"] in g.columns:
        orig_h = _numeric_or_none(g[height_info["orig_height_column"]])
    else:
        orig_h = pd.Series([np.nan] * len(g), index=g.index)

    matched = site_h.notna() & np.isfinite(site_h) & site_area.notna() & np.isfinite(site_area)
    no_lod12 = ~(site_h.notna() & np.isfinite(site_h))
    site_match = np.where(matched, "matched", np.where(no_lod12, "no_lod12", "no_site_area"))

    height_m_source = site_h.where(matched, orig_h)
    if roof is not None:
        if orig_roof is not None:
            height_nap_source = roof.where(matched, orig_roof)
        else:
            height_nap_source = roof.where(matched, np.nan)
    else:
        height_nap_source = pd.Series([np.nan] * len(g), index=g.index)
    area_m2_source = site_area.where(matched, geom_area)

    height_m = [
        _round_or_none(v, 1 if m else 3, bool(m))
        for v, m in zip(height_m_source, matched)
    ]
    height_nap = [_round_or_none(v, 3) for v in height_nap_source]
    area_m2 = [
        _round_or_none(v, 2 if m else 3)
        for v, m in zip(area_m2_source, matched)
    ]
    orig_height_m = [_round_or_none(v, 3) for v in orig_h]
    orig_height_nap_m = (
        [_round_or_none(v, 3) for v in roof] if roof is not None else [None] * len(g)
    )
    orig_area_m2 = [_round_or_none(v, 3) for v in geom_area]
    site_ground_nap_m = (
        [_round_or_none(v, 3) for v in ground] if ground is not None else [None] * len(g)
    )

    if "city" in g.columns:
        fallback = [CITY_LABEL.get(_city_for(geom, city_boxes), None) for geom in g.geometry]
        city = g["city"].where(g["city"].notna(), pd.Series(fallback, index=g.index))
    else:
        city = pd.Series(
            [CITY_LABEL.get(_city_for(geom, city_boxes), None) for geom in g.geometry],
            index=g.index,
        )

    clean = gpd.GeoDataFrame(
        {
            "id": g[height_info["id_column"]].astype(str),
            "height_m": height_m,
            "height_nap_m": height_nap,
            "area_m2": area_m2,
            "city": city,
            "orig_height_m": orig_height_m,
            "orig_height_nap_m": orig_height_nap_m,
            "orig_area_m2": orig_area_m2,
            "site_ground_nap_m": site_ground_nap_m,
            "site_3dbag_id": np.where(matched, g[height_info["id_column"]].astype(str), None),
            "site_match": site_match,
        },
        geometry=g.geometry,
        crs=p.WORK_CRS,
    )

    clean_path = out_raw / "3DBAG_Amsterdam_Utrecht_lod12_clean.geojson"
    pyogrio.write_dataframe(
        clean.to_crs(p.WGS84),
        clean_path,
        driver="GeoJSON",
        layer_options={"COORDINATE_PRECISION": 6},
    )
    p.log(f"wrote {clean_path.name}: {len(clean)} features (local 3DBAG input)")

    definition = height_info["height_definition"]
    ref = gpd.GeoDataFrame(
        {
            "ref_id": g[height_info["id_column"]].astype(str),
            "ref_bag_id": g[height_info["id_column"]].astype(str),
            "ref_source": "3DBAG",
            "height_3dbag_m": clean["height_m"],
            "height_3dbag_50p_m": clean["height_m"],
            "height_3dbag_max_m": clean["height_m"],
            "height_3dbag_definition": definition,
            "height_3dbag_max_definition": definition,
            "h_roof50_3dbag_m": clean["height_nap_m"],
            "h_roofmax_3dbag_m": clean["height_nap_m"],
            "h_ground_3dbag_m": clean["site_ground_nap_m"],
            "h_nok_3dbag_m": None,
            "ref_area_m2": geom_area,
            "city": clean["city"],
        },
        geometry=clean.geometry,
        crs=p.WORK_CRS,
    )

    site_map = {
        str(row.id): {
            "height": row.height_m,
            "height_nap": row.height_nap_m,
            "area": row.area_m2,
            "ground": row.site_ground_nap_m,
        }
        for row in clean.itertuples(index=False)
        if row.site_match == "matched"
    }
    return clean, ref, site_map


def _city_for(geom, city_boxes):
    centroid = geom.centroid
    for name, building_box in city_boxes.items():
        if building_box.contains(centroid):
            return name
    best = None
    best_area = 0.0
    for name, building_box in city_boxes.items():
        if geom.intersects(building_box):
            try:
                area = float(geom.intersection(building_box).area)
            except Exception:
                area = 0.0
            if area > best_area:
                best = name
                best_area = area
    return best


def environment_and_input_summary(selection: InputSelection, environment: dict) -> dict:
    return {
        "input_mode": selection.mode,
        "input_selection_reason": selection.reason,
        "explicit_paths": selection.explicit,
        "gba_source": str(selection.gba_path) if selection.gba_path else f"GBA viewer WFS ({p.GBA_WFS})",
        "tdbag_source": str(selection.bag_path) if selection.bag_path else "pinned 3DBAG CityJSON tiles",
        "remote_dataset_retrieval": selection.remote_dataset_retrieval,
        "environment_verification": environment,
    }


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
