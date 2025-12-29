# Weather Warning Pipeline

End-to-end system for building gridded features from ERA5, joining storm tracks,
engineering geometric / spherical features, training models, generating alerts,
and producing per-run reports.

This repo is organized around thin "manager" scripts plus focused workers.
Everything is orchestrated from a single YAML config via `run_pipeline.py`.

## Recent updates

- Reporting now writes into a dated run folder (`YYYYMMDD_run###`) and includes
  `reporting_v2` outputs (MD/JSON + storm pages).
- Object matching uses scored selection (top-K + spatial de-dup) and reports
  both motion-based and flow-based directions.
- Pipeline validation is fail-fast (inputs/columns/ordering) with an autofix
  mode to enable missing dependencies.
- YAML normalization is stricter: key aliases, whitespace stripping, and
  canonical config output are built in.
- `skip_if_exists` is supported in more tools (sweep best-constrained + eval).
- Alerts now preserve `row_id` through denoise so specialist training can
  use alert IDs.

## Repository layout (current)

Key files and directories referenced by the pipeline:

```text
run_pipeline.py                   # Top-level orchestrator driven by YAML
pipeline_contracts.py             # Step contracts, dependencies, column checks

fetch_subprocess/
  fetch_manager.py                # Front door for fetch modes
  ...

features_subprocess/
  features_manager.py             # Front door for feature steps
  build_features_grid.py          # Flatten ERA5 single levels -> grid features
  features_bulk_shear.py          # Bulk shear from pressure-level winds
  features_patch.py               # Rolling stats, shear proxy, S3, derivatives
  compute_gka_features.py         # GKA / geometric-kernel features
  integrate_era5_thermo.py         # Thermo integration
  compute_spherical_feedback.py   # Spherical feedback / SFI features

data_subprocess/
  data_stage_manager.py           # Data staging (ids, transitions, panels)
  train_calibrate_eval.py          # Base model training + calibration
  train_alert_specialist.py        # Alert specialist training
  predict_and_alert.py             # Base + specialist blending

sweep_subprocess/
  sweep_manager.py                 # Sweep orchestration
  find_best_f1_thresholds_constrained.py

alerts_logic_subprocess/
  alerts_logic_manager.py          # Alert pipeline
  apply_thresholds.py              # Base alerts
  throttle_by_percentile.py        # Throttle
  denoise_alerts.py                # Denoise (keeps row_id when present)

eval_subprocess/
  eval_manager.py                  # Eval front door
  eval_viability_leads.py
  hourly_metrics.py
  hourly_rollup.py

seeds_subprocess/
  seeds_tracks.py                  # Seeds / proto-outcomes / starts-vs-tracks

reports_subprocess/
  reports_and_maps_manager.py      # Bundle reports into per-run folder
  report_pack.py                   # Tables for reporting
  reporting_v2.py                  # MD/JSON report + storm pages
  match_objects_to_tracks.py       # Scored matching + directionality
  plot_object_matches.py           # Maps: all objects, candidates, matches

utils/
  run_naming.py                    # YYYYMMDD_run### folder naming
  config_normalize.py              # YAML normalization + aliasing
  quick_gka_summary.py             # Lightweight stats for large GKA tables
```

## Pipeline orchestration

`run_pipeline.py` is the only entry point you need. It validates the config,
prints the plan, and dispatches each section to the appropriate manager.

Default section order:

```text
fetch -> features -> data_stage -> training -> sweep -> score -> alerts_logic
-> eval -> seeds -> report -> misc
```

### Config normalization and validation

- Configs are normalized on load (aliases, dashes to underscores, whitespace
  stripping). A canonical config is written next to the input YAML.
- `validate_pipeline()` checks required inputs/columns and dependency order.
  Missing prerequisites fail fast with a clear error.
- `--autofix-config` can auto-enable missing upstream steps; add overwrite
  flags only when `--autofix-add-overwrite` is supplied.

### Run folder naming

Reports are organized into a per-run directory created under `run-root`
(default `results/reports`) with the format:

```
YYYYMMDD_run###
```

Example: `results/reports/20251224_run003/`

## Reporting and maps

The report bundle step (recommended) produces:

- `results/reports/<YYYYMMDD_run###>/<run_name>_report.txt`
- `results/reports/<YYYYMMDD_run###>/<run_name>_report.md`
- `results/reports/<YYYYMMDD_run###>/<run_name>_report.json`
- `results/reports/<YYYYMMDD_run###>/<run_name>_tables/` (CSV/Parquet tables)
- `results/reports/<YYYYMMDD_run###>/storm_pages/` (storm-by-storm pages)

Key behaviors:

- `report_pack.py` writes tables into `<run_name>_tables/` under the run folder.
- `reporting_v2.py` builds the MD/JSON report and storm pages.
- Map outputs include all predicted objects, candidates, and matched objects.

## Object matching and directionality

Matching is now scored and de-duplicated per hour:

- Score uses probability, distance penalty, compactness, and persistence.
- Top-K candidates are kept per hour, then non-maximum suppression enforces
  a minimum separation in km.
- Direction is reported two ways:
  - Motion direction from centroid displacement (preferred).
  - Flow direction from mean u10/v10 within the object mask.

Maps show all objects, candidates, and matches with arrows labeled by
speed (km/h) and bearing.

## Training and alerts pipeline

Training chain (typical order):

1. `data_stage` steps: add-ids -> state-transitions -> (optional panels)
2. `train-base` (base model)
3. `train-alert-specialist` (uses alerts + storm labels)
4. `predict-alerts` (blend base + specialist)
5. `track-objects` (object extraction)

Alerts logic (viability pipeline) produces:

- `prob_viable`
- `alert_base`, `alert_final`
- `row_id` is preserved through denoise for specialist training

## Evaluation

Eval tools are run via `eval_subprocess/eval_manager.py` and include:

- `viability-leads` (lead-time skill vs `t_to_storm_min_h`)
- `hourly-metrics` (alert coverage/cluster stats)
- `hourly-rollup` (per-hour aggregates)

`skip_if_exists` is supported in the above eval tools.

## Example: minimal pipeline YAML (current shape)

```yaml
run_name: coral_sea_demo
workdir: .
reports_dir: results/reports
table_format: parquet
force_keep_quantile: 1.0

defaults:
  start: &start "2025-02-01"
  end:   &end   "2025-05-01"
  hours: &hours "0..23"
  normalize_lon: &norm "-180..180"
  area: &area "-5,125,-35,175"

fetch:
  enabled: false
  steps:
    - mode: ibtracs
      start: *start
      end: *end
      area: *area
      normalize-lon: *norm
      out: data/tracks/tracks_subset.csv

features:
  enabled: true
  steps:
    - mode: build
      nc-glob: "data_era5/extracted/**/era5_single_*_*.nc"
      out: data/grid_FMA_base.parquet
      normalize-lon: *norm
      area: *area
      export-uv: true
      with-vortdiv: true
      emit-grid-index: true

    - mode: bulk-shear
      pl-glob: "data_era5/extracted/**/era5_pl_*_uv.nc"
      out: data/grid_FMA_shear.parquet
      low-pair: [1000, 925]
      deep-pair: [1000, 500]

    - mode: join-features
      left: data/grid_FMA_base.parquet
      right: data/grid_FMA_shear.parquet
      on: [time, lat, lon]
      out: data/grid_FMA_joined.parquet

    - mode: patch
      in: data/grid_FMA_joined.parquet
      out: data/grid_FMA_patched.parquet

    - mode: gka
      infile: data/grid_FMA_patched.parquet
      outfile: data/grid_FMA_gka.parquet

    - mode: integrate-thermo
      features: data/grid_FMA_gka.parquet
      thermo-glob: "data_era5/extracted/**/era5_single_*_*.nc"
      out: data/grid_FMA_gka_realthermo.parquet

    - mode: spherical-feedback
      labelled: data/grid_FMA_gka_realthermo.parquet
      out: data/grid_FMA_sph.parquet

training:
  enabled: true
  steps:
    - mode: train-base
      train: data/grid_labelled_FMA_gka_realthermo_sph_ms_id_state.parquet
      label: storm
      train-end: "2025-03-31"
      val-end: "2025-04-30"
      model-out: models/base_model.pkl
      calibration-out: models/base_calibrator.pkl

    - mode: train-alert-specialist
      train: data/grid_labelled_FMA_gka_realthermo_sph_ms_id_state.parquet
      alerts: results/alerts/alerts_coral_sea_demo_final.parquet
      label: storm
      model-out: models/alert_specialist.pkl

    - mode: predict-alerts
      features: data/grid_labelled_FMA_gka_realthermo_sph_ms_id_state.parquet
      base-model: models/base_model.pkl
      base-calibrator: models/base_calibrator.pkl
      specialist-model: models/alert_specialist.pkl
      outfile: results/predictions_base_specialist.parquet

    - mode: track-objects
      infile: results/predictions_base_specialist.parquet
      mask-col: P_final
      threshold: 0.6
      objects-out: results/objects/objects.parquet
      join-out: results/objects/cell_objects.parquet

alerts_logic:
  enabled: true
  steps:
    - mode: viability-pipeline
      run-name: coral_sea_demo
      labelled: data/grid_train_gse_panel_targets.parquet
      model: models/viability_model.pkl
      thr: 0.37
      keep-quantile: 1
      base-out: results/alerts/alerts_coral_sea_demo_base.parquet
      thr-out: results/alerts/alerts_coral_sea_demo_thr.parquet
      out: results/alerts/alerts_coral_sea_demo_final.parquet

sweep:
  enabled: true
  steps:
    - mode: best-constrained
      labelled: data/grid_train_gse_panel_targets.parquet
      model: models/viability_model.pkl
      metrics-json: models/viability_model_metrics.json
      out: results/sweeps/viability_best_thresholds.csv
      skip_if_exists: true

eval:
  enabled: true
  steps:
    - mode: viability-leads
      run-name: coral_sea_demo
      out: results/metrics/coral_sea_demo_viability_leads.csv
      skip_if_exists: true

    - mode: hourly-metrics
      run-name: coral_sea_demo
      out: results/metrics/coral_sea_demo_hourly_metrics.csv
      skip_if_exists: true

    - mode: hourly-rollup
      run-name: coral_sea_demo
      out: results/metrics/coral_sea_demo_hourly_rollup.parquet
      skip_if_exists: true

report:
  enabled: true
  steps:
    - mode: bundle
      run-name: coral_sea_demo
      run-root: results/reports
      objects-in: results/alerts/alerts_coral_sea_demo_final.parquet
      object-matches-out: results/matches/storm_object_matches.parquet
      ibtracs: data/tracks/tracks_subset.csv
      ibtracs-normalize-lon: *norm
```

## Running the pipeline

Run all sections:

```bash
python run_pipeline.py --config config/pipeline.yaml
```

Run only a subset:

```bash
python run_pipeline.py --config config/pipeline.yaml --sections=features,data_stage,training
```

Generate an autofix config:

```bash
python run_pipeline.py --config config/pipeline.yaml --autofix-config
```

## Performance notes

- Prefer Parquet for large tables.
- Many workers accept `--chunk-rows` and `--parquet-rows` to stay memory-safe.
- AOI cropping and shorter time spans are the fastest way to reduce load.

## Troubleshooting

- Single-class metrics in training or eval usually mean the validation window
  has no positives. Adjust `train-end`/`val-end` or use a random split.
- `alerts_used=0` in specialist training usually means the alerts file lacks
  `row_id`. Re-run the alerts pipeline so denoise outputs preserve `row_id`.
- If AOI crop removes all rows, confirm `normalize-lon` and AOI bounds use the
  same longitude frame.
