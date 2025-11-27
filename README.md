\# Weather Warning Pipeline



End-to-end system for building gridded features from ERA5, joining storm tracks, engineering geometric / spherical features, sweeping ML models, and emitting alert-style products.



The project is structured as a set of thin “manager” scripts plus small, focused workers. Everything is orchestrated from a single YAML config and `run\_pipeline.py`.



---



\## High-level overview



Data flow, in rough order:



1\. \*\*Fetch\*\*

&nbsp;  - Download / extract ERA5 single-level and pressure-level fields.

&nbsp;  - Pull historical storm tracks (e.g. IBTrACS subsets).



2\. \*\*Feature engineering (gridded)\*\*

&nbsp;  - Flatten ERA5 single levels into `(time, lat, lon, vars)` features.

&nbsp;  - Compute bulk shear from pressure-level winds.

&nbsp;  - Patch in rolling / derivative / shear diagnostics.

&nbsp;  - Add geometric-kernel analysis (GKA) features.

&nbsp;  - Integrate “real” thermo back into the flattened table.

&nbsp;  - Compute spherical feedback / SFI-style features.



3\. \*\*Labels \& data staging\*\*

&nbsp;  - Join storm tracks to the grid.

&nbsp;  - Prepare model-ready train/val/test tables.



4\. \*\*Sweeps, scoring, alerts\*\*

&nbsp;  - Hyperparameter sweeps and model selection.

&nbsp;  - Gridded scores.

&nbsp;  - Alert logic and seeds/tracks analyses.

&nbsp;  - Final reporting.



You can run the whole thing with a single command and control each stage from YAML.



---



\## Repository layout



Key files and directories (only listing the pieces referenced by the current pipeline):



```text

run\_pipeline.py                # Top-level orchestrator driven by YAML



fetch\_subprocess/

&nbsp; fetch\_manager.py             # Front-door for all fetch modes

&nbsp; era5\_merge\_singlelevels.py   # Merge ERA5 single-level monthly chunks

&nbsp; ...                          # Other fetch\_\* helpers



features\_subprocess/

&nbsp; features\_manager.py          # Front-door for feature steps



&nbsp; build\_features\_grid.py       # Flatten ERA5 single levels → grid features

&nbsp; features\_bulk\_shear.py       # Bulk shear from ERA5 pressure-level U/V

&nbsp; features\_join\_features.py    # Join multiple feature tables

&nbsp; join\_labels\_grid.py          # Join grid features with storm tracks

&nbsp; features\_patch.py            # Rolling stats, shear proxy, S3, derivatives, gradients

&nbsp; compute\_gka\_features.py      # GKA / geometric-kernel features on flat table

&nbsp; integrate\_era5\_thermo.py     # Integrate ERA5 thermo back into large features

&nbsp; compute\_spherical\_feedback.py# Spherical feedback / SFI features



data\_subprocess/

&nbsp; stage\_data.py                # Stage features + labels into model-ready tables

&nbsp; build\_features\_and\_labels.py # Additional assembly, if used

&nbsp; train\_calibrate\_eval.py      # Train models, calibrate, evaluate

&nbsp; thresholds\_and\_alerts.py     # Threshold selection \& alert tables

&nbsp; diagnostics\_packager.py      # Bundle diagnostics / artifacts



sweep\_subprocess/

&nbsp; sweep\_manager.py             # Model sweeps / chains



score\_subprocess/

&nbsp; grid\_score.py                # Unified scoring script for grid outputs



alerts\_logic\_subprocess/

&nbsp; alerts\_logic\_manager.py      # Alert logic pipeline



eval\_subprocess/

&nbsp; eval\_manager.py              # Evaluation rollups



runtime\_subprocess/

&nbsp; runtime\_manager.py           # Runtime / environment checks



utils/

&nbsp; quick\_gka\_summary.py         # Lightweight stats/summary for large GKA parquet files



seeds\_tracks.py                # Seeds / proto-outcomes / starts-vs-tracks utilities

reporting\_and\_results.py       # Final stitching \& reporting



data/                          # Local outputs (parquet / csv / gz)

data\_era5/extracted/           # ERA5 extracted NetCDFs

&nbsp; .../era5\_single\_YYYYMM\_oper.nc

&nbsp; .../era5\_pl\_\*\_uv.nc

data/tracks/

&nbsp; tracks\_subset.csv            # Track subset used for labelling

````



---



\## Installation



Use Python 3.10+ and a virtual environment.



```bash

python -m venv .venv

source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt

```



The exact `requirements.txt` will vary, but the project uses at least:



\* `numpy`

\* `pandas`

\* `xarray`

\* `pyarrow`

\* `netcdf4` and/or `h5netcdf` and/or `scipy` (for NetCDF)

\* `pyyaml`

\* Standard scientific Python stack (scikit-learn, etc.) for downstream stages.



---



\## Data requirements



At minimum:



\* \*\*ERA5 single-level NetCDFs\*\* for:



&nbsp; \* `u10`, `v10` (10-m winds)

&nbsp; \* `msl` (mean sea–level pressure)

&nbsp; \* `t2m` (2-m temperature)

\* \*\*ERA5 pressure-level NetCDFs\*\* for:



&nbsp; \* U/V winds on pressure levels including at least 1000, 925, 500 hPa (for bulk shear).

\* \*\*Storm tracks\*\* (e.g. IBTrACS subset) with:



&nbsp; \* `time`, `lat`, `lon`, intensity / ID fields used by your labelling logic.



All ERA5 files are expected under `data\_era5/extracted/` with patterns like:



\* `data\_era5/extracted/\*\*/era5\_single\_YYYYMM\_oper.nc`

\* `data\_era5/extracted/\*\*/era5\_pl\_\*\_uv.nc`



Storm tracks are typically under `data/tracks/tracks\_subset.csv`.



---



\## Orchestration: `run\_pipeline.py` + YAML



`run\_pipeline.py` is the only script you \*have\* to call directly.

It reads a YAML config and calls the section managers in order.



\### Section order



In code, the default order is:



```python

SECTION\_ORDER = \[

&nbsp;   "fetch",

&nbsp;   "features",

&nbsp;   "data\_stage",

&nbsp;   "sweep",

&nbsp;   "score",

&nbsp;   "alerts\_logic",

&nbsp;   "eval",

&nbsp;   "runtime",

&nbsp;   "seeds",

&nbsp;   "report",

]

```



Each section has:



\* `enabled: true/false`

\* A `steps:` (or `jobs:`) list with:



&nbsp; \* `mode` (or `recipe`) that selects the manager sub-mode.

&nbsp; \* Additional key/value pairs forwarded as CLI flags.



Extra keys are flattened to `--k v`, with nested keys becoming dotted flags:

`{"pick": {"metric": "F1"}} → --pick.metric F1`.



\### Example: minimal pipeline YAML (current structure)



This example matches the current scripts and file names that are in use:



```yaml

run\_name: fma\_run

workdir: .



defaults:

&nbsp; normalize\_lon: "-180..180"



fetch:

&nbsp; enabled: true

&nbsp; steps:

&nbsp;   - mode: era5-both

&nbsp;     years: \[2025]

&nbsp;     months: \[2, 3, 4, 5]

&nbsp;     area: "-5,125,-35,175"

&nbsp;     out\_dir: "data\_era5/extracted"

&nbsp;   - mode: ibtracs

&nbsp;     basin: "SHEM"

&nbsp;     out\_csv: "data/tracks/tracks\_subset.csv"



features:

&nbsp; enabled: true

&nbsp; steps:

&nbsp;   - mode: build

&nbsp;     nc\_glob: "data\_era5/extracted/\*\*/era5\_single\_\*\_\*.nc"

&nbsp;     out: "data/features\_eoi.parquet"

&nbsp;     normalize\_lon: "-180..180"

&nbsp;     area: "-5,125,-35,175"

&nbsp;     export\_uv: true

&nbsp;     with\_vortdiv: true

&nbsp;     require\_vars: \["u10", "v10", "msl", "t2m"]

&nbsp;     dedup: "time\_lat\_lon"

&nbsp;     emit\_grid\_index: true



&nbsp;   - mode: bulk-shear

&nbsp;     pl\_glob: "data\_era5/extracted/\*\*/era5\_pl\_\*\_uv.nc"

&nbsp;     out: "data/features\_bulk\_shear.parquet"

&nbsp;     low\_pair: \[1000, 925]

&nbsp;     deep\_pair: \[1000, 500]

&nbsp;     s3\_window: 3



&nbsp;   - mode: join-features

&nbsp;     left: "data/features\_eoi.parquet"

&nbsp;     right: "data/features\_bulk\_shear.parquet"

&nbsp;     on: \["time", "lat", "lon"]

&nbsp;     out: "data/features\_merged.parquet"



&nbsp;   - mode: patch

&nbsp;     in: "data/features\_merged.parquet"

&nbsp;     out: "data/features\_merged\_patched.parquet"

&nbsp;     prefer\_shear: "shear\_06km"

&nbsp;     s3\_window: 3



&nbsp;   - mode: gka

&nbsp;     infile: "data/features\_merged\_patched.parquet"

&nbsp;     outfile: "data/grid\_labelled\_FMA\_gka.parquet"

&nbsp;     overwrite: true



&nbsp;   - mode: integrate-thermo

&nbsp;     features: "data/grid\_labelled\_FMA\_gka.parquet"

&nbsp;     thermo\_glob: "data\_era5/extracted/\*\*/era5\_single\_\*\_\*.nc"

&nbsp;     normalize\_lon: "-180..180"

&nbsp;     area: "-5,125,-35,175"

&nbsp;     out: "data/grid\_labelled\_FMA\_gka\_realthermo.parquet"



&nbsp;   - mode: spherical-feedback

&nbsp;     labelled: "data/grid\_labelled\_FMA\_gka\_realthermo.parquet"

&nbsp;     normalize\_lon: "-180..180"

&nbsp;     area: "-5,125,-35,175"

&nbsp;     neighbor\_step: 0.0

&nbsp;     radius\_cells: 1

&nbsp;     out: "data/spherical\_feedback.csv.gz"



labels\_stage:

&nbsp; # This is used by join\_labels\_grid via run\_pipeline.run\_join\_labels, if enabled

&nbsp; features\_csv: "data/features\_eoi.parquet"

&nbsp; labels\_csv:   "data/tracks/tracks\_subset.csv"

&nbsp; out\_csv:      "data/grid\_labelled\_base.parquet"

&nbsp; storm\_radius\_deg: 2.0

&nbsp; storm\_time\_h: 6.0

&nbsp; near\_radius\_deg: 8.0

&nbsp; near\_time\_h: 24.0

&nbsp; pregen\_radius\_deg: 8.0

&nbsp; pregen\_hours: "1..240"

&nbsp; pregen\_step: 1

&nbsp; normalize\_lon: "-180..180"



pipeline:

&nbsp; run\_labels\_stage: false



data\_stage:

&nbsp; enabled: false

&nbsp; steps: \[]      # stage\_data / build\_features\_and\_labels etc.



sweep:

&nbsp; enabled: false

&nbsp; steps: \[]      # sweep\_manager modes



score:

&nbsp; enabled: false

&nbsp; jobs: \[]       # grid\_score jobs



alerts\_logic:

&nbsp; enabled: false

&nbsp; steps: \[]      # alerts\_logic\_manager modes



eval:

&nbsp; enabled: false

&nbsp; steps: \[]      # eval\_manager modes



runtime:

&nbsp; enabled: false

&nbsp; steps: \[]      # runtime\_manager checks



seeds:

&nbsp; enabled: false



report:

&nbsp; enabled: false

```



You can turn sections on/off by toggling `enabled` fields, and you can also restrict which sections run with `--sections` on the CLI.



---



\## Running the pipeline



Example: run the full default section order:



```bash

python run\_pipeline.py --config configs/run\_fma.yaml

```



Run only `fetch` and `features`:



```bash

python run\_pipeline.py --config configs/run\_fma.yaml --sections=fetch,features

```



Each section logs the exact command lines it invokes, so you can copy/paste to debug or run steps manually.



---



\## Key managers and their modes



\### `fetch\_subprocess/fetch\_manager.py`



Front-door for all fetch logic. You normally won’t call it directly when using `run\_pipeline.py`, but you can.



Modes include (exact set may vary with your current version):



\* `era5` / `era5-pl` / `era5-both` / `era5-shear`

\* `ibtracs`

\* `intensity`



All extra YAML key/vals in the `fetch.steps` entries are passed as CLI flags.



---



\### `features\_subprocess/features\_manager.py`



Thin delegator; it only chooses the script and passes through all other arguments.



Current routing:



\* `build`              → `build\_features\_grid.py`

\* `patch`              → `features\_patch.py`

\* `join`               → `join\_labels\_grid.py`

\* `gka`                → `compute\_gka\_features.py`

\* `integrate-thermo`   → `integrate\_era5\_thermo.py`

\* `spherical-feedback` → `compute\_spherical\_feedback.py`

\* `spherical`          → `compute\_spherical\_feedback.py` (alias)

\* `bulk-shear`         → `features\_bulk\_shear.py`

\* `join-features`      → `features\_join\_features.py`



Examples:



```bash

\# Build base gridded ERA5 features

python features\_subprocess/features\_manager.py build \\

&nbsp; --nc-glob "data\_era5/extracted/\*\*/era5\_single\_\*\_\*.nc" \\

&nbsp; --out data/features\_eoi.parquet \\

&nbsp; --normalize-lon "-180..180" \\

&nbsp; --area "-5,125,-35,175" \\

&nbsp; --export-uv \\

&nbsp; --with-vortdiv \\

&nbsp; --require-vars u10,v10,msl,t2m \\

&nbsp; --dedup time\_lat\_lon \\

&nbsp; --emit-grid-index



\# Bulk shear from pressure levels

python features\_subprocess/features\_manager.py bulk-shear \\

&nbsp; --pl-glob "data\_era5/extracted/\*\*/era5\_pl\_\*\_uv.nc" \\

&nbsp; --out data/features\_bulk\_shear.parquet \\

&nbsp; --low-pair 1000,925 \\

&nbsp; --deep-pair 1000,500 \\

&nbsp; --s3-window 3



\# Join base features + shear

python features\_subprocess/features\_manager.py join-features \\

&nbsp; --left data/features\_eoi.parquet \\

&nbsp; --right data/features\_bulk\_shear.parquet \\

&nbsp; --on time,lat,lon \\

&nbsp; --out data/features\_merged.parquet



\# Patch in rolling, derivatives, S3, etc.

python features\_subprocess/features\_manager.py patch \\

&nbsp; --in data/features\_merged.parquet \\

&nbsp; --out data/features\_merged\_patched.parquet \\

&nbsp; --prefer-shear shear\_06km \\

&nbsp; --s3-window 3



\# Add GKA features

python features\_subprocess/features\_manager.py gka \\

&nbsp; --infile data/features\_merged\_patched.parquet \\

&nbsp; --outfile data/grid\_labelled\_FMA\_gka.parquet \\

&nbsp; --overwrite



\# Integrate thermo

python features\_subprocess/features\_manager.py integrate-thermo \\

&nbsp; --features data/grid\_labelled\_FMA\_gka.parquet \\

&nbsp; --thermo-glob "data\_era5/extracted/\*\*/era5\_single\_\*\_\*.nc" \\

&nbsp; --out data/grid\_labelled\_FMA\_gka\_realthermo.parquet \\

&nbsp; --normalize-lon "-180..180" \\

&nbsp; --area "-5,125,-35,175"



\# Spherical feedback / SFI

python features\_subprocess/features\_manager.py spherical-feedback \\

&nbsp; --labelled data/grid\_labelled\_FMA\_gka\_realthermo.parquet \\

&nbsp; --out data/spherical\_feedback.csv.gz \\

&nbsp; --neighbor-step 0.0 \\

&nbsp; --radius-cells 1 \\

&nbsp; --normalize-lon "-180..180" \\

&nbsp; --area "-5,125,-35,175"

```



---



\### `data\_manager.py`



Thin wrapper for the “data” stage scripts living in `data\_subprocess/`.



Subcommands:



\* `stage-data`            → `stage\_data.py`

\* `build-features-labels` → `build\_features\_and\_labels.py`

\* `train-calibrate-eval`  → `train\_calibrate\_eval.py`

\* `thresholds-and-alerts` → `thresholds\_and\_alerts.py`

\* `diagnostics-packager`  → `diagnostics\_packager.py`



Usage:



```bash

python data\_manager.py stage-data --config configs/my\_data\_stage.yaml

python data\_manager.py train-calibrate-eval --config configs/my\_training.yaml

```



All arguments after the subcommand are passed straight to the underlying script.



---



\## Utilities



\### Quick GKA summary



`utils/quick\_gka\_summary.py` is built for large GKA parquet tables (~70M rows, ~10 GB).



Example:



```bash

python utils/quick\_gka\_summary.py data/grid\_labelled\_FMA\_gka.parquet

```



Outputs:



\* File stats: rows, row groups, size, basic column ranges.

\* Quantiles (e.g. 5/50/95%) for the `gka\_` feature family.

\* Basic null counts.



This is designed to work incrementally via `pyarrow.parquet.ParquetFile` and to avoid reading the whole table into memory.



---



\## Memory \& performance notes



Current data volumes discussed in this pipeline:



\* Grid for FMA (Feb–May) at 0.25° resolution:



&nbsp; \* ≈ \*\*70,044,480 rows\*\*

&nbsp; \* GKA table around \*\*10.2 GB\*\* as Parquet (`grid\_labelled\_FMA\_gka.parquet`).



Some guidance:



\* Prefer \*\*Parquet\*\* over CSV for large intermediate tables.

\* Scripts like `features\_patch.py`, `compute\_gka\_features.py`, and `compute\_spherical\_feedback.py` are written to:



&nbsp; \* Downcast floats to `float32` where safe.

&nbsp; \* Use per-time or chunked processing for expensive neighbor operations.

\* For anything that still runs out of memory:



&nbsp; \* Reduce temporal span (fewer months) for explorations.

&nbsp; \* Restrict spatial area via `--area` before going back to larger regions.

&nbsp; \* Consider turning off non-essential diagnostics in YAML for initial smoke tests.



---



\## Troubleshooting (current known patterns)



\* \*\*KeyError(\[... not in index]) in thermo integration\*\*

&nbsp; Addressed by the current `integrate\_era5\_thermo.py`, which uses a stable `\_\_idx` column rather than global integer index lists.



\* \*\*MemoryError in GKA parity / SFI neighbor logic\*\*

&nbsp; Current versions:



&nbsp; \* Use chunking (CSV streaming) or per-hour grouping.

&nbsp; \* Avoid storing huge intermediate matrices.

&nbsp; \* Cast to `float32` where possible.



\* \*\*“No rows after normalization/AOI filter” in spherical feedback\*\*

&nbsp; Typically means:



&nbsp; \* Input file has been filtered to a different lon convention than the one you’re passing.

&nbsp; \* `--normalize-lon` + `--area` combination excludes everything.

&nbsp;   Fix by ensuring:

&nbsp; \* All previous stages use the same lon mode (e.g. `-180..180`).

&nbsp; \* AOI bounds match that lon mode.



---



\## Extending the pipeline



The current structure is modular:



\* Add new feature steps as a script under `features\_subprocess/` and wire it into `features\_manager.ROUTING`.

\* Add new “data stage” steps under `data\_subprocess/` and expose them via `data\_manager.py`.

\* New metrics or evaluation logic can be plugged into:



&nbsp; \* `score\_subprocess/grid\_score.py`

&nbsp; \* `eval\_subprocess/eval\_manager.py`

\* New alert logic recipes can be added via `alerts\_logic\_subprocess/alerts\_logic\_manager.py`.



Each new piece can then be activated from YAML with a new `mode` in the appropriate section, without changing `run\_pipeline.py`.



---



```

```



