# GBA vs 3DBAG building-height pipeline (Amsterdam + Utrecht)

This repository contains an end-to-end, reproducible pipeline that builds per-building
height/area layers for **3DBAG** and the **Global Building Atlas (GBA)**, then compares them.

Repository outputs are generated from scratch and can be written directly to Google Drive
(e.g. from Colab).

## What it produces

For both study cities (Amsterdam and Utrecht):

- `raw/3DBAG_Amsterdam_Utrecht_lod12_clean.geojson`
  - one polygon per 3DBAG building
  - `height_m`: 3DBAG LoD1.2 site height = `z_roof - z_base` (roof is the flat top of the LoD1.2 model)
  - `area_m2`: `b3_opp_grond` (area shown by the 3DBAG viewer)
  - `height_nap_m`, `site_ground_nap_m`, `orig_*`, `city`
- `raw/GBA_Amsterdam_Utrecht_lod12_clean.geojson`
  - the GBA footprint with its matched 3DBAG site height/area in `height_m` / `area_m2`
  - the GBA own height/area in `orig_height_m` / `orig_area_m2`
- `raw/GBA_Amsterdam_Utrecht_raw.geojson`
  - the GBA footprint with the GBA own height/area as `height_m` / `area_m2`
  - the matched 3DBAG site values in `site_height_m`, `site_area_m2`, `site_3dbag_id`
- `GBA_vs_3DBAG.geojson` (one feature per 3DBAG footprint, incl. false negatives)
- `GBA_vs_3DBAG_false_positives.geojson` (GBA footprints with no accepted 3DBAG match)
- `GBA_vs_3DBAG_structural_groups.csv`
- `comparison_summary.csv`, `comparison_category_breakdown.csv`, `run_metadata.json`
- `comparison_outputs.zip`, `raw/lod12_clean.zip`, `density_analysis_outputs.zip`
- `density_accuracy_summary.csv`, `grid_size_sensitivity.csv`,
  `density_distribution_by_grid.csv`, `density_analysis_report.md`,
  `density_analysis_metadata.json`
- `density_map.png`, `geometric_accuracy_by_density.png`,
  `height_accuracy_by_density.png`, `density_grid_250m.geojson`
- `figures/` with the complete paper figure set:
  - `figure2_overall_comparison.png` / `.pdf`
  - `figure3_iou_by_h3_density.png` / `.pdf`
  - `figure3_iou_by_h3_density_gaussian.png` / `.pdf`
  - `figure4_height_error_by_h3_density.png` / `.pdf`
  - `figure4_height_error_by_h3_density_gaussian.png` / `.pdf`
  - `figure5_h3_spatial_performance.png` / `.pdf`
  - `study_areas_netherlands.png` / `.pdf`
  - copies of the density-stratified figures
  - `README.md` with the numbers used in the figures
- `h3/h3_density_summary.csv`, `h3/h3_cell_summary.csv`,
  `h3/h3_group_metrics.geojson`, H3 figures and `h3/h3_metadata.json`

## Data sources and input modes

The pipeline supports two execution modes and never mixes them:

- **Local Input Mode** reads both datasets from local GeoJSON/vector files.
- **Online Mode** retrieves both datasets with the original online mechanisms:
  3DBAG CityJSON tiles pinned to release **v2025.09.03**
  (`https://data.3dbag.nl/v20250903/`) and the live GBA viewer WFS
  (`https://tubvsig-so2sat-vm1.srv.mwn.de`).

Default local paths:

```
D:\GBA_3DBAG_outputs\raw\GBA_Amsterdam_Utrecht_raw.geojson
D:\GBA_3DBAG_outputs\raw\3DBAG_Amsterdam_Utrecht_lod12_clean.geojson
```

Automatic input selection:

| GBA file | 3DBAG file | Behavior |
|---|---|---|
| present and valid | present and valid | Local Input Mode for both |
| missing | missing | Online Mode for both |
| present | missing | stop with an error naming the missing 3DBAG file |
| missing | present | stop with an error naming the missing GBA file |
| present but unusable/invalid | any state | stop with an error |
| any state | present but unusable/invalid | stop with an error |

An explicit `--gba-file` or `--bag-file` takes precedence over the default paths.
Both explicit paths must be supplied together. If an explicit file is missing,
unreadable, or incompatible, the run stops; it never falls back to Online Mode.

Before any dataset operation the pipeline verifies the project directory and the
configured dataset directories. A failure there stops the run. If both configured
dataset files are absent and their directories are accessible, Online Mode is selected.
Required drive letters are not hardcoded, so the repository can run on machines without
`D:` or `E:`; to enforce strict drive checks, set the environment variable
`GBA_3DBAG_REQUIRED_DRIVES` (for example `D:\;E:\`). The default local paths use `D:`;
on other machines, either pass `--gba-file` and `--bag-file`, or let Online Mode run.

Online Mode downloads the GBA LoD1 data from the official viewer WFS and the 3DBAG
CityJSON tiles for the Amsterdam + Utrecht study areas only. It does not download the
entire Netherlands. The GBA authors request that their WFS not be used for bulk or
automated extraction; for a public, reproducible run, supply a local GBA file obtained
from the official GBA.LoD1 release and keep it with your project or archive.

## Setup

```bash
python -m pip install -r requirements.txt
```

The pipeline was tested with `geopandas`, `pyogrio`, `shapely >= 2`, `pyproj`, `pandas`,
`numpy`, and `requests`. The tests additionally require `pytest`.

## Run

Online mode (both default local files absent):

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work
```

Local Input Mode using the exact default files:

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work
```

Local Input Mode with explicit custom files:

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work ^
  --gba-file "D:\GBA_3DBAG_outputs\raw\GBA_Amsterdam_Utrecht_raw.geojson" ^
  --bag-file "D:\GBA_3DBAG_outputs\raw\3DBAG_Amsterdam_Utrecht_lod12_clean.geojson"
```

`--tdbag-file` is accepted as an alias for `--bag-file`.

Useful options:

| option | meaning |
|---|---|
| `--out` | output directory (use your Google Drive folder in Colab) |
| `--work` | cache/work directory (default: `<out>/work`) |
| `--gba-file` | local GBA vector file; requires `--bag-file` |
| `--bag-file` | local 3DBAG vector file; requires `--gba-file` |
| `--gba-tile-step` | GBA WFS request tile size in degrees (default `0.03`; online only) |
| `--threads` | parallel downloads for 3DBAG tiles (online only) |
| `--figures` | generate the complete `figures/` folder after the pipeline |

The selected mode, exact sources, and whether remote dataset retrieval was used are
printed and recorded in `run_metadata.json`. Local files are validated before use:
CRS, polygonal geometry, empty/missing/invalid geometries, identifiers, and height
fields are checked. Invalid polygonal geometries are repaired with `make_valid`;
the number of repaired features is reported. Unrepairable geometries stop the run.
Original input files are never overwritten; the pipeline refuses an `--out` location
that would collide with an input file.

For the standard local files, the GBA height is read from `height_m` (the GBA own
height) and the 3DBAG height is derived from `height_nap_m - site_ground_nap_m`
(the existing LoD1.2 site-height definition). Other common field names are mapped
when present. Missing or invalid heights are never replaced with zero or estimates;
they are excluded from height metrics and counted.

### Reproduce the complete output set

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work --figures
```

`--figures` runs the comparison and density analysis, then H3 analysis, the paper
figures, the study-area map, the Gaussian-fit figures, the density figures, and the
matching-category example maps. The resulting directory has the same structure as
the reference `outputs_local` run:

- `comparison_summary.csv`, `comparison_category_breakdown.csv`,
  `GBA_vs_3DBAG*.geojson`, `GBA_vs_3DBAG_structural_groups.csv`, `run_metadata.json`
- `raw/` clean/raw layers and zip files
- `density_*`, `grid_size_sensitivity.csv`, `geometric_accuracy_by_density.png`,
  `height_accuracy_by_density.png`, `density_grid_250m.geojson`
- `h3/` tables, group GeoJSON, figures, and metadata
- `figures/` paper figures, Gaussian fits, study-area map, density figures, and
  `matching_examples/` three-panel category examples

If `--figures` is omitted, the comparison and density outputs are still generated;
H3 and figures can be produced afterwards with:

```bash
python make_figures_folder.py --out /path/to/output
```

The output **structure** is fixed, but values depend on the input data. A custom
GeoJSON must contain polygonal building footprints and compatible height fields; the
loader maps common aliases and stops with a clear error for incompatible data. The
pipeline's configured study area and summary area remain Amsterdam + Utrecht; for a
different region, provide local files and adjust the study-area settings/figures as
needed.

### Google Colab

Open `GBA_3DBAG_Colab.ipynb` in Colab, set the repository path, mount Drive,
and run the final cell. Outputs are written to:

```
/content/drive/MyDrive/GBA_3DBAG_outputs
```

Typical online runtime is about 40-90 minutes (3DBAG tile download + GBA WFS fetch +
matching), depending on network speed. Colab high-RAM is recommended for the matching
step. Local Input Mode skips all remote dataset retrieval.

### Density-stratified accuracy analysis

The density analysis runs automatically at the end of `run_pipeline.py`, in addition
to (not instead of) the existing overall comparison. Building density is:

```
number of 3DBAG reference buildings / grid-cell area in km2
```

Reference-building centroids are assigned to square grid cells in the metric working
CRS (`EPSG:28992`). Grid sizes 100 m, 250 m, and 500 m are evaluated for sensitivity.
The primary grid is **250 m**, chosen before inspecting accuracy outcomes because it
matches the pipeline's existing 250 m 3DBAG retrieval cell size and approximates an
urban block/neighbourhood scale.

For the primary grid, the 33rd and 66th percentiles of non-empty-cell density define:

- Low: density <= P33
- Medium: P33 < density <= P66
- High: density > P66
- Empty cells: unclassified, never low density

Geometric classes use each matched 3DBAG reference building as TP, each unmatched
3DBAG building as FN, and each unmatched GBA building as FP. Class precision, recall,
and F1 are pooled from class-level TP/FP/FN counts. Height metrics reuse the existing
matched-group heights, sign convention (`GBA - 3DBAG`), and validity rules; each group
is assigned once using the centroid of its 3DBAG reference union. Missing/invalid
heights are excluded and counted. Ties at a percentile can produce unequal class
sizes; this is reported and not forced.

The class comparisons are descriptive only. No significance test is performed, and
the analysis explicitly notes spatial dependence between neighbouring cells and the
modifiable-areal-unit sensitivity. See `density_analysis_report.md` for the generated
methodology, thresholds, sample counts, metrics, and limitations.

### H3 density analysis (after the pipeline)

```bash
python h3_analysis.py --out /path/to/output --resolution 8
```

This optional, separate analysis reads the comparison outputs and writes `h3/` tables,
group points, figures, and metadata. Density is the number of 3DBAG reference buildings
per occupied H3 cell; cells at or above the median are `Dense`, the rest `Sparse`.
Match groups are assigned to the H3 cell of the reference centroid (1:1) or the
centroid of the union of their reference buildings (multi-feature groups). It is not
replaced by the grid-based density analysis above.

### Complete figures folder

Generate the same figure set as the paper/reference run from an existing output
directory:

```bash
python make_figures_folder.py --out /path/to/output
```

Or generate it automatically at the end of a pipeline run:

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work --figures
```

The `figures/` folder contains:

- `figure2_overall_comparison.png` / `.pdf` — overall building counts, group IoU/F1/
  precision/recall, and height MAE/RMSE with the current numeric values printed on the
  figure.
- `figure3_iou_by_h3_density.png` / `.pdf` — group IoU by H3 Sparse/Dense cells, with
  sample sizes, means, and medians.
- `figure3_iou_by_h3_density_gaussian.png` / `.pdf` — IoU boxplot plus empirical
  distribution and Gaussian fit; IoU is bounded and strongly skewed, so the normal curve
  is a reference overlay only.
- `figure4_height_error_by_h3_density.png` / `.pdf` — signed height difference by H3
  density, with sample sizes, means, and medians.
- `figure4_height_error_by_h3_density_gaussian.png` / `.pdf` — height-error boxplot plus
  empirical distribution and Gaussian fit; the empirical tails are heavier than normal.
- `figure5_h3_spatial_performance.png` / `.pdf` — spatial H3 maps of mean IoU, signed
  height error, and absolute height error.
- `study_areas_netherlands.png` / `.pdf` — study-area context map and insets.
- `density_map.png`, `geometric_accuracy_by_density.png`,
  `height_accuracy_by_density.png` ? the grid-based low/medium/high analysis figures.
- `matching_examples_panel.png` / `.pdf` and `matching_examples/` ? three-panel GBA-only,
  3DBAG-only, and overlay examples for each matching category.
- `README.md` ? the exact overall, H3, density-class, and grid-sensitivity numbers used
  in the figures, plus provenance and provenance caveats.

`make_figures_folder.py` reuses existing H3 outputs when present and runs
`h3_analysis.py` if they are missing. The paper figures and study-area map download
Sentinel-2 cloudless basemap tiles from EOX/Copernicus; if the network is unavailable,
the scripts use grey fallback tiles. Figure generation is separate from dataset
retrieval and does not change the comparison results.

### Tests

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests cover environment checks, automatic and explicit input selection, local-file
validation and geometry repair, no-remote-request local execution, grid construction,
percentile thresholds and tied values, empty-cell handling, TP/FP/FN assignment, pooled
precision/recall/F1, height-metric validity rules, output generation, and a regression
check of the existing comparison counts and height metrics.


## Definitions

- **3DBAG site height**: for each building, all vertices of its LoD1.2 geometry are
  projected on the local up direction; the reported height is `max(z) - min(z)`.
  This is exactly the height the 3DBAG viewer shows when clicking the flat roof in the
  LoD1.2 view (`height.toFixed(1)`).
- **3DBAG area**: `b3_opp_grond` from the 3DBAG attributes.
- **GBA own height**: the `height` attribute from the GBA LoD1 footprint.
- **GBA clean file**: the GBA footprint with the matched 3DBAG site height/area for
  inspection; the GBA own values are kept in the `orig_*` fields.
- **Comparison**: 3DBAG reference = LoD1.2 site height; GBA compared = GBA own height.
  Height metrics are group-level (`MAE`, `RMSE`, `MAPE`) using the original matching
  algorithm of the study.
## Figures

python make_paper_figures.py --out /path/to/output

python study_area_map.py --out /path/to/output

python study_area_map.py --out /path/to/output --area "Name:min_lon,min_lat,max_lon,max_lat"

# any dataset with different column names (interactive or JSON mapping):

python make_figures_custom.py --out /path/to/output

python make_figures_custom.py --init-map columns.json

python make_figures_custom.py --out /path/to/output --map columns.json
## Method summary

For each 3DBAG building, all vertices of the LoD1.2 geometry are projected on the local
up direction; the site height is `max(z) - min(z)`, i.e. the flat-roof height shown by
the 3DBAG LoD1.2 viewer. The site area is `b3_opp_grond`. GBA provides one LoD1 footprint
and one height attribute per building. The two datasets are matched with the overlap-graph
algorithm in `pipeline_lib.py` (1:1, many-to-one, one-to-many, many-to-many;
false negatives and false positives are reported separately). Height agreement is measured
at group level with MAE, RMSE, and MAPE.

## Reference run

Command used for the reference output:

```bash
python run_pipeline.py --out D:\GBA_3DBAG_outputs --work D:\GBA_3DBAG_work
```

Data: 3DBAG `v2025.09.03` (pinned) + GBA WFS fetched 2026-11-16.

| item | value |
|---|---|
| 3DBAG buildings | 473,690 |
| GBA buildings | 512,327 |
| matched 3DBAG buildings | 471,840 |
| false negatives | 1,850 |
| false positives | 39,639 |
| matched overlap groups | 468,999 |
| height MAE | 2.4663 m |
| height RMSE | 3.4057 m |
| height MAPE | 43.15% |

Small variations in these counts are expected if the live GBA WFS changes. For exact
reproduction, supply fixed local inputs with both `--gba-file` and `--bag-file`.

### Reference H3 results (Amsterdam + Utrecht, resolution 8)

- Occupied H3 cells: 877 (874 contain matched groups)
- Density threshold: median = 127 reference buildings per occupied cell
  (`Dense`: >= 127; `Sparse`: < 127)
- Groups: 468,999 (466,184 one-to-one + 2,815 multi-feature groups)

| Density | Groups | Mean IoU | Median IoU | Mean signed height error (m) | Median signed height error (m) | Mean absolute height error (m) | RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 15,670 | 0.9619 | 0.9948 | -1.589 | -1.348 | 3.070 | 5.302 |
| Dense | 453,329 | 0.9727 | 0.9912 | -0.999 | -0.813 | 2.445 | 3.321 |

The signed error is `GBA - 3DBAG`. Dense cells show a higher mean IoU and lower absolute
height error, while sparse cells have a higher median IoU but a heavier lower tail.

## Notes

- 3DBAG tile index and CityJSON files are pinned to `v2025.09.03` in Online Mode so the
  3DBAG side is reproducible.
- Building matching uses spatial overlap groups (1:1, many-to-one, one-to-many, many-to-many).
- False negatives are 3DBAG buildings without an accepted GBA match; false positives are
  GBA footprints without an accepted 3DBAG match.
- The bundled `pipeline_lib.py` is the comparison engine used in the original study.
  Only its GBA and comparison functions are exercised by `run_pipeline.py`.
- Local files are validated and normalized before comparison. Invalid polygonal
  geometries are repaired and counted; unrepairable or empty geometries stop the run.
  Local file provenance (dataset version and download date) is not verified from the
  file contents, and filesystem timestamps are not treated as download dates.

## Repository layout

```text
run_pipeline.py                  end-to-end pipeline and mode selection
pipeline_lib.py                  matching and comparison engine
input_sources.py                 local input selection, validation, normalization
density_analysis.py              grid-based low/medium/high density analysis
h3_analysis.py                   H3 resolution-8 density analysis (separate)
make_figures_folder.py           complete figures/ folder from pipeline outputs
make_matching_examples.py        three-panel matching-category example maps
make_paper_figures.py            paper figures from pipeline outputs
GBA_3DBAG_Colab.ipynb            Colab runner
make_figures_custom.py           same figures from any dataset via column mapping
study_area_map.py                study-area map (default: Amsterdam + Utrecht)
tests/                           pytest suite for input modes and density analysis
requirements.txt
README.md
LICENSE
```

## Numbers and provenance

docs/paper_numbers_provenance.md maps every reported number in the manuscript to
the output file and field (or the code constant) that produced it.

## Code availability

https://github.com/farnia04/GBA_3DBAG-comparison

## Citation

If you use this code in a publication, please cite the repository above and the
3DBAG and GBA data sources listed under Attribution.

## Attribution

- 3DBAG by the 3D Geoinformation research group (TU Delft) and 3DGI, licensed under
  CC BY 4.0 (https://3dbag.nl).
- Global Building Atlas (GBA) data from its official web service; check the GBA terms of use.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Data licenses remain with the respective data providers.
