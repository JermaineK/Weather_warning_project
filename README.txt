The Weather Warning Project - Jermaine Kotlar

---

# 🌪️ Grid Forecast Pipeline

A full end-to-end system for **spatiotemporal forecasting** and **alert generation** using
gridded data. The pipeline handles model training, calibration, lead-time evaluation,
threshold optimization, spatial–temporal denoising, throttling, diagnostics, and
automated reporting.

---

## 🧩 Pipeline Overview

| Stage | Script | Description |
|-------|---------|-------------|
| **1. Integrate ERA5** | `integrate_era5_thermo.py` | Optionally merges ERA5 reanalysis (thermo or other features) into the labelled grid. |
| **2. Train model** | `train_grid_logit_from_csv.py` | Fits a logistic regression or other ML model on the labelled data. |
| **3. Calibrate model** | `calibrate_grid.py` | Performs isotonic (or other) calibration on model probabilities. |
| **4. Evaluate lead-time metrics** | `eval_leadtime_grid_progress.py` | Measures model performance at various forecast leads (AUC, Brier, etc.). |
| **5. Find thresholds** | `find_best_f1_thresholds_constrained.py` | Chooses thresholds under coverage/precision constraints (e.g. Fβ). |
| **6. Apply thresholds → Denoise → Throttle** | `apply_thresholds.py`, `denoise_alerts.py`, `throttle_by_percentile.py` | Generates binary alert maps, smooths them in space/time, and limits alert density. |
| **7. Evaluate hits** | `eval_alert_hits.py` | Compares predicted alerts to future ground truth within a forecast lead window. |
| **8. Summarize & visualize** | `run_pipeline.py` | Creates Markdown, text, and optional PDF reports with diagnostics and plots. |

---

## 🧠 Key Features

- **Sparse or full grids supported** – missing cells auto-filled as zeros.  
- **Robust time parsing** – ISO strings, UNIX epochs, or custom formats.  
- **Spatial + temporal denoising** – morphology, neighbor filters, persistence.  
- **Throttle control** – keeps only the top quantile of alert confidence.  
- **Diagnostics** – detects incomplete grids, duplicate keys, and large alert drops.  
- **Auto-reporting** – Markdown summary, text log, and optional PDF charts.

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt

Minimal dependencies (for quick data exploration or running lightweight scripts):

- numpy
- pandas
- scipy
- matplotlib
- pyyaml
- scikit-learn

Full pipeline dependencies (mirrors `requirements.txt`):

- numpy
- pandas
- xarray
- pyarrow
- lightgbm
- scikit-learn
- matplotlib
- cartopy
- scipy
- joblib
- pyyaml


---

🚀 Running the Pipeline

Basic run

python run_pipeline.py --config config/pipeline.yaml

Re-run all stages (overwrite existing results)

python run_pipeline.py --config config/pipeline.yaml --overwrite


---

🧾 Example Configuration (config/pipeline.yaml)

run_name: gka_fma_realthermo_public

labelled_csv: data/grid_labelled_FMA_gka_realthermo.csv.gz
target: pregen

# Optional ERA5 integration
era5:
  enabled: false
  nc_glob: data/era5/*.nc
  out_csv: data/grid_labelled_era5.csv.gz
  nearest: true
  nearest_maxdeg: 0.4

# Model outputs
model_out: models/grid_logit_pregen.pkl
model_out_cal: models/grid_logit_pregen_cal.pkl

test_size: 0.1
impute: median
clip_quantile: 0.999
class_weight: balanced

# Threshold constraints
constraints:
  out_csv: results/best_fbeta_thresholds.csv
  min_precision: 0.15
  max_coverage: 0.10
  fbeta: 0.5
  verbose: true

# Alert generation
alerts:
  out_dir: results/alerts
  flag_col: alert_final
  denoise:
    persist_hours: 1
    min_neighbors: 3
    connectivity: 4
    min_area: 0
    sparse_output: false
    overwrite: true
  throttle:
    per_lead_keep_quantile:
      24: 0.95
      48: 0.95
      72: 0.95
      120: 0.95
      240: 0.95

# Reports
summary_md: results/summary_report.md
diagnostics_csv: results/diagnostics.csv
report_pdf: results/summary_report.pdf

# Overwrite controls
overwrite:
  train: false
  calibrate: false
  thresholds: true
  alerts: true
  reports: true


---

🧹 Denoising Options

denoise_alerts.py can be run standalone or via the pipeline.

Command-line example

python denoise_alerts.py \
  --alerts results/alerts/alerts_run_lead24_thr0.045.csv.gz \
  --persist-hours 2 \
  --min-neighbors 3 \
  --connectivity 8 \
  --min-area 4 \
  --sparse-output \
  --out results/alerts/alerts_run_lead24_denoised.csv.gz \
  --overwrite

Key Parameters

Option	Description

--persist-hours	Keep alerts that persist within N hours (rolling max).
--min-neighbors	Remove isolated pixels with fewer than N active neighbors.
--connectivity	Use 4 or 8 neighbor connectivity for morphology.
--min-area	Remove connected regions smaller than this threshold.
--sparse-output	Write only rows where the alert flag is 1 (saves space).
--overwrite	Overwrite existing output file if present.



---

📊 Reports and Diagnostics

After each run, the following files are generated automatically:

File	Description

results/summary_report.md	Markdown summary with tables and pipeline diagnostics.
results/summary_report.txt	Plain-text version for logs or CLI view.
results/summary_report.pdf	Optional PDF with quick plots (coverage & time series).
results/diagnostics.csv	Per-lead diagnostics including completeness, retention, and warnings.


Example Coverage Summary

Coverage (post-throttle):
  Lead 24h  thr=0.0435  hours=90  active=18,840/444,690  cov=0.042
  Lead 72h  thr=0.0375  hours=90  active=22,512/444,690  cov=0.051
  Lead 240h thr=0.9884  hours=90  active=867/444,690    cov=0.002


---

🧪 Diagnostics Explained

Each lead has a diagnostic row summarizing data integrity and performance:

Field	Meaning

base_rows	Number of rows in raw alert grid.
base_bad_time	Count of invalid timestamps dropped.
base_cov	Fraction of active alerts before cleanup.
den_cov / thr_cov	Coverage after denoise / throttle.
keep_denoise	Fraction retained after denoising.
keep_throttle	Fraction retained after throttling.
flags	Warning flags like incomplete_grid, large_drop_after_denoise, etc.



---

🔍 Example Diagnostic Flags

Flag	Meaning

base_incomplete_grid	Input grid missing expected (time×lat×lon) cells.
large_drop_after_denoise	Too many alerts removed in smoothing step.
large_drop_after_throttle	Coverage reduction exceeded 50%.
dup_keys	Duplicate (time, lat, lon) entries detected.
bad_time	Invalid or unparsable timestamps found.



---

🧠 Tips for Better Forecast Skill

Increase min_neighbors (3–5) to suppress false positives.

Use higher persist_hours (2–4) for more coherent, storm-like patterns.

8-connectivity + min_area > 3 smooths regional noise.

Re-run calibration periodically to maintain consistent coverage.

Inspect diagnostics.csv after each run for early anomalies.



---

📈 Outputs and Visualization

The optional PDF report (results/summary_report.pdf) includes:

1. Coverage by lead:
Bar chart showing alert coverage fraction for each forecast lead.


2. Active cells per hour:
Line plot of alert frequency across time for the longest lead.




---

🔧 Overwrite Logic

You can re-run specific stages selectively:

overwrite:
  train: false
  calibrate: false
  thresholds: true
  alerts: true
  reports: true

Or globally:

python run_pipeline.py --config config/pipeline.yaml --overwrite


---

🗂️ Typical Workflow

1. Train and calibrate model once.


2. Tune thresholds (precision vs. coverage).


3. Generate multi-lead forecasts.


4. Adjust denoise parameters for better coherence.


5. Evaluate and visualize outputs.


6. Publish results or share alert maps.




---

🧱 Recommended Folder Structure

Weather warning project/
├─ config/
│  └─ pipeline.yaml
├─ data/
│  └─ grid_labelled_FMA_gka_realthermo.csv.gz
├─ models/
│  ├─ grid_logit_pregen.pkl
│  └─ grid_logit_pregen_cal.pkl
├─ results/
│  ├─ alerts/
│  ├─ diagnostics.csv
│  ├─ summary_report.md
│  ├─ summary_report.txt
│  └─ summary_report.pdf
├─ scripts/
│  ├─ apply_thresholds.py
│  ├─ calibrate_grid.py
│  ├─ denoise_alerts.py
│  ├─ eval_alert_hits.py
│  ├─ eval_leadtime_grid_progress.py
│  ├─ find_best_f1_thresholds_constrained.py
│  ├─ throttle_by_percentile.py
│  └─ train_grid_logit_from_csv.py
└─ run_pipeline.py


---

🧪 Example: Full Run

python run_pipeline.py --config config/pipeline.yaml --overwrite

Example output snippet:

== Denoise alerts (lead=240) ==
Grid inferred (from sparse): H=61 W=81 T=90
Wrote results/alerts/alerts_run_lead240_denoised.csv.gz | active cells: 867/444,690
[DENOISE] Summary: T=90 H=61 W=81 persist=1h min_neighbors=3 connectivity=4 min_area=0 coverage=0.002


---

Developed for open, reproducible, and scalable spatiotemporal forecasting systems.

---


