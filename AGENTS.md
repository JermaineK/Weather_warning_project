# AGENTS.md — How to Touch This Project Without Breaking It

This file is for *all* agents working on this repo — humans, LLMs, and other helpers.

The priorities are:

1. **Math stays correct.**  
   Spiral / kernel / shear / feedback maths must never be casually changed.
2. **Data stays usable.**  
   Every pipeline run should produce consistent, analyzable tables with stable schemas.
3. **Modular, spiral-like design.**  
   Code should be built from small, composable steps that can be tested and swapped without breaking the whole structure.

If you are an automated coding agent: read this as a *contract*.


---

## 1. Project Intent (Short Version)

This repo builds a **modular severe-weather feature pipeline** with:

- ERA5 → flattened grid features (winds, pressure, shear, etc.)
- Derived fields: vorticity/divergence, S, S3, GKA metrics, spherical feedback indices (SFI)
- Label joining and pregen windows
- Downstream training / calibration / alerts (separate stage)

The physics motivation is a **geometric / spiral kernel framework**, so features like:

- `zeta`, `div`, `S`, `S3`
- `gka_*` fields
- `sph_*` and `SFI`

are **not optional noise** — they’re the skeleton.


---

## 2. Never-Break Invariants

These rules override “clever ideas.”

### 2.1 Coordinate & Timebase

- `time`:  
  - Must be **timezone-naive UTC** (`datetime64[ns]` without tz).  
  - No local time, no offsets.
- `lat`, `lon`:
  - `lat` in degrees, -90 → 90
  - `lon` must be **normalized consistently per run**:
    - Preferred: `[-180, 180]`
  - Any change to lon-mode must be explicit and documented.

### 2.2 Column Contract

Unless a script is explicitly documented otherwise, the following **must exist** in the main merged feature table before modelling:

- Identity:
  - `time`, `lat`, `lon`
- Surface fields:
  - `msl`, `t2m`
- Winds:
  - `u10`, `v10`, `wspd`
- Vorticity / divergence / strength:
  - `zeta`, `div`, `S`
- Shear:
  - `shear10_def`, `S3`, `S3_src`
- GKA:
  - `gka_kappa`, `gka_tau`, `gka_parity_eta`, `gka_A_overlap`
  - `gka_F`, `gka_msl_nd`, `gka_knee_ratio`
  - `gka_chirality`, `gka_Q`, `gka_dir_var`, `gka_vortdiv_ratio`
- Spherical feedback:
  - `sph_center`, `sph_radial_signed`, `sph_radial_abs`, `SFI`

If you remove or rename any of these:

- You **must** update:
  - The contract/tests (e.g. `CONTRACT.yml`, `tests/`)
  - Any downstream scripts that rely on them
- And leave a clear note in the relevant doc (e.g. ADR or THEORY_ALIGNMENT).

### 2.3 Units & Ranges

Agents must preserve:

- `msl` in **hPa** (hectopascals).  
  - If input is Pascals, it must be divided by 100 once (and only once).
- `zeta`, `div`, `S`:
  - Magnitudes are small, typically \|value\| ≲ `1e-3`.
  - Abnormally huge values usually mean a bug in grid spacing or units.
- Shear magnitudes:
  - `shear10_def`, `shear_low`, `shear_deep` should be finite and non-negative.
  - NaN storms of shear are not acceptable unless explicitly masked.


---

## 3. Modular / Spiral Design Guidelines

Think like this:

- **Outer spiral**: high-level orchestrators / managers:
  - `features_manager.py`
  - `data_manager.py`
- **Mid spiral**: single-purpose feature scripts:
  - `build_features_grid.py`
  - `features_bulk_shear.py`
  - `features_patch.py`
  - `compute_gka_features.py`
  - `compute_spherical_feedback.py`
- **Inner spiral**: math kernels:
  - Functions like `compute_zeta_div`, `_deformation_shear_from_uv`, `_compute_chunk_features`, `per_time_neighbors_block`.

Rules:

- High-level scripts should **delegate**, not re-implement math.
- Inner kernels should be **small, pure, and testable**, with clear docstrings.
- When changing behaviour:
  - Prefer adding **new derived fields** instead of mutating old ones.
  - Keep old features available while new ones are trialled, where possible.


---

## 4. What Agents MAY Change

It is acceptable (and encouraged) for an agent to:

- Improve **performance / memory usage**:
  - Chunked reads/writes
  - dtype downcasting (`float64` → `float32`, where safe)
  - hourly / spatial tiling
- Add **new** features that:
  - Don’t silently replace existing physics
  - Are described in a comment or short doc section
- Add or refine **logging, errors, and diagnostics**:
  - More explicit error messages (`KeyError: missing u10/v10`)
  - Summary stats at the end of a run
- Strengthen **robustness**:
  - Alias handling for column names
  - NaN guards and sensible fallbacks (e.g. `shear_proxy` if no bulk shear)

Whenever possible:

- Prefer **non-breaking** extensions:
  - E.g., add `gka2_*` rather than subtly changing `gka_*` maths in place.


---

## 5. What Agents MUST NOT Do (Without Explicit Approval)

Please do **not**:

1. **Change the mathematics** of:
   - `compute_zeta_div`
   - `S` and `S3` definitions
   - GKA feature formulas
   - Spherical feedback kernel (`sph_*`, `SFI`)
   
   …unless:
   - The change is documented in an ADR / theory note, and
   - Tests/golden stats are updated accordingly.

2. **Silently drop or rename** core columns listed in §2.2.

3. **Change lon normalization** (`0..360` vs `-180..180`) without:
   - Applying it consistently across stages, and
   - Calling it out in docs / commit messages.

4. Inject hidden data dependencies:
   - No hard-coded local paths.
   - No silent schema assumptions not reflected in tests / docs.

5. Reformat large files in ways that obscure functional diffs:
   - Massive reformat + functional change in the same commit = hard to audit.


---

## 6. Expectations for Automated Coding Agents

If you are an LLM-based agent (like a code assistant), you should:

1. **State your intent in comments/commits**  
   - E.g. `# Agent: reduce memory usage in per_time_neighbors_block without changing outputs`.

2. **Prefer local, minimal patches** over rewrites:
   - Touch only the functions needed to solve the problem.
   - Don’t restructure whole modules unless asked.

3. **Respect tests and contracts**:
   - If there is a failure, fix the *root cause* instead of masking it.
   - Never “fix” a test by deleting it or loosening it without explanation.

4. **Preserve numerical behaviour**:
   - Be very cautious around:
     - order of operations
     - type promotion
     - use of `float32` vs `float64`
   - If you change numeric code, add a short comment explaining stability considerations.

5. **Keep the spiral**:
   - New logic belongs in a dedicated function where possible.
   - Avoid monolithic “god functions” that mix IO, control flow, and math.


---

## 7. Quick Checklist Before/After Changes

**Before editing:**

- Identify whether you’re touching:
  - IO / plumbing  
  - Performance  
  - Math / physics
- If touching math: treat it as a **breaking change** and document accordingly.

**After editing (ideal behaviour for humans *and* agents):**

- Run the smallest available pipeline slice (mini AOI, short time).
- Check:
  - Output schema still contains core columns.
  - Basic stats look sane (no wild explosions).
- For math changes:
  - Compare summary stats or golden quantiles vs previous version.

If in doubt: **prefer correctness over cleverness**. It's better to be slightly slow and right than blazing fast and wrong.


---


  Machine-read schema + unit expectations.
- `tests/`  
  What must never silently break.
- `features_subprocess/*.py`  
  Pipeline modules: each one should have a clear single responsibility.

If you are unsure whether a change violates the intent: err on the side of **not changing the mathematics** and focus on clarity, robustness, and performance only.

---

# Agent Briefing: Geometric Kernel Algorithm (GKA) & Weather Pipeline

This repository implements a **Geometric Kernel Algorithm (GKA)** for weather-like spatiotemporal fields. The goal is to treat GKA as a **spiral spectral probe**, not just a random feature bundle.

You (the coding agent) only see this repo, not prior conversations. This document tells you what the math is supposed to mean and how to extend the code **without breaking that meaning**.

---

## 1. Conceptual model you must preserve

We treat the atmosphere as a field \( F(x, t) \) on a 2D spatial domain. There is a **spiral operator** (\mathcal{S}) that acts (conceptually) by scaling + rotating:

* \((\mathcal{S} f)(z) = f(\lambda e^{i\theta_0} z)\)

The **Geometric Kernel Algorithm (GKA)** is a domain-agnostic procedure that estimates how strongly local patches of the data align with **spiral eigenmodes** of this operator.

In practice that means:

1. Identify a **local patch** where (\mathcal{S}) is approximately well-defined.
2. Extract statistics that estimate:

   * amplitude of the leading spiral-like mode,
   * stability / coherence of that mode,
   * how shear is destroying or preserving it.
3. Track how these mode coefficients evolve in time (build vs relax) and under parity flip (handedness).
4. Measure **scale dependence** via knee-like ratios (small-scale vs smoothed behaviour).

You must treat the GKA feature set as a coherent implementation of this idea.

---

## 2. Existing GKA-related code and what it already does

You will see at least the following scripts (names may vary slightly):

* `features_subprocess/compute_gka_features.py`
  Produces columns like:

  * `gka_kappa ~ zeta`
  * `gka_tau   ~ -div`
  * `gka_parity_eta`
  * `gka_A_overlap`
  * `gka_F`
  * `gka_msl_nd`
  * `gka_knee_ratio`
  * `gka_chirality`
  * `gka_Q`
  * `gka_dir_var`
  * `gka_vortdiv_ratio`

* `features_subprocess/features_patch.py`
  Produces:

  * temporal rolls and tendencies (`*_mean3h`, `*_std3h`, `dS_dt`, etc.)
  * shear-related quantities (`shear10_def`, `shear_proxy`, `S3`)

* `features_subprocess/compute_spherical_feedback.py`
  Produces per-time local patch features:

  * `sph_center`  (pressure center-ness)
  * `sph_radial_signed`, `sph_radial_abs` (radial wind alignment)
  * `sph_lightning` (optional)
  * `SFI` (spherical feedback index)

Interpretation (do not break this):

* `zeta`, `div`, `S`, `S3`, `shear*`, `gka_*`, and `SFI` are all **different projections of how spiral-like and stable the flow is**, at various scales.

---

## 3. General constraints

When modifying / extending:

1. **Do not break math semantics.**

   * `gka_*` features should remain interpretable as spiral-mode amplitude, coherence, or parity / shear descriptors.
   * Don’t repurpose GKA columns for unrelated calculations.

2. **Keep the pipeline modular and streaming-friendly.**

   * Scripts are designed to run on large datasets (10+ GB) on modest hardware.
   * Prefer chunked IO, `float32` where safe, and avoid unnecessary `.copy()` on big DataFrames.
   * Any new heavy transform should either:

     * operate *per chunk*, or
     * operate *per time slice* with minimal memory overhead.

3. **Don’t remove or rename existing CLI flags / YAML toggles.**

   * They are used by higher-level managers (`features_manager.py`, `data_manager.py`, YAML configs).
   * You may add new flags, but keep default behaviour backward compatible.

4. **Always check column presence explicitly.**

   * Many scripts run on different subsets of ERA5 / label grids.
   * Use alias binding and “if column exists, compute; otherwise use 0 / NaN with clear comments”.

---

## 4. High-priority tasks (GKA as spiral spectral probe)

### Task A: Implement composite GKA alignment indices

Location: `features_subprocess/compute_gka_features.py`

Add 1–2 **composite indices** that explicitly encode “how spiral-eigenmode-like is this patch”. Example:

* `gka_SAI` — **Spiral Alignment Index**

  * High when the local flow is a coherent spiral mode:

    * large |`gka_kappa`| (vorticity magnitude),
    * high `gka_F` (low destructive shear / high survivorship),
    * low `gka_dir_var` (consistent direction),
    * high `gka_A_overlap` (vorticity-dominant over divergence),
    * moderate `gka_knee_ratio` (non-trivial small-scale structure, but not pure noise).

* `gka_SII` — **Spiral Instability Index**

  * High when a spiral mode exists and is being torn apart:

    * large |`gka_kappa`| and |`gka_tau`|,
    * high `S3` or other shear-related magnitude,
    * large |`gka_vortdiv_ratio`|,
    * large |`dS_dt`| or similar time-derivative indicators (from patched features, if present).

Implementation guidance:

* Use **robust scaling** where possible (e.g. `robust01` style, quantile-based, or median/MAD).
* Provide clear comments explaining which terms correspond to amplitude, coherence, parity, or scale.
* Fail-safe: if needed inputs are missing, set `gka_SAI` / `gka_SII` to 0.0 and document in logs.

### Task B: Patch-level aggregation (optional but desirable)

Either:

* Add a new script, e.g. `features_subprocess/compute_gka_patch_features.py`, or
* Add an optional, **opt-in** block to `features_patch.py`.

Goal: compute **patch-level statistics** over small spatial neighbourhoods (3x3 or 5x5 cells) and maybe a short temporal window, using existing GKA and SFI features.

Examples (per cell):

* Local mean / max of:

  * `gka_kappa`, `gka_F`, `S3`, `SFI`, `gka_SAI`, `gka_SII`.
* Local variance of:

  * `gka_parity_eta` (parity coherence),
  * `gka_vortdiv_ratio`.

Constraints:

* Use grid identity where available (`ilat`, `ilon`).
* Neighborhood size should be small and configurable (e.g. `--patch-radius-cells`).
* Implementation should be “rolling patch” style, not full dense matrices if memory is tight.
* If options are added, make them **off by default** so current pipelines are unchanged.

### Task C: Lead-time spiral features for forecasting

Goal: provide **forecaster-friendly lead features** based on spiral alignment.

Where to implement (flexible):

* Either in a dedicated script, e.g. `features_subprocess/compute_gka_lead_features.py`, or
* As an optional flag in an existing patch stage.

Examples:

* For each point `(time, lat, lon)` compute:

  * Lagged versions of composite indices:
    `gka_SAI_lag3h`, `gka_SAI_lag6h`, `gka_SII_lag3h`, etc. (when prior data exists).
  * “Future-any” style indicators similar to `future_any_by_point` used for pregen:

    * e.g. max GKA over next window vs storm labels (for analysis).

Constraints:

* Time alignment must be clear: document whether `lag3h` refers to t−3h or a rolling window.
* Use `groupby(["lat","lon"])` with memory-safe patterns for big tables.

---

## 5. Testing and validation expectations

When you change or extend the code:

1. **Smoke tests on small subsets**

   * Create a minimal synthetic or downsampled ERA5-like table (few thousand rows).
   * Run:

     * `build_features_grid.py`
     * `features_patch.py`
     * `compute_gka_features.py`
     * `compute_spherical_feedback.py`
   * Confirm:

     * No crashes.
     * New columns exist and have reasonable ranges (no wild NaNs, no all-zeros unless expected).

2. **Interpretability checks (non-strict but important)**

   * For `gka_SAI`: values in [0, 1]-ish (or clearly bounded) and higher in obviously rotational / low-shear scenes.
   * For `gka_SII`: high where shear and vorticity are both strong.

3. **Memory usage**

   * Avoid operations that create huge intermediate arrays unnecessarily (`.copy()` of entire frames, `groupby.apply` on 70M-row frames without care, etc.).
   * Prefer:

     * per-time-slice loops (like `compute_spherical_feedback.py`),
     * `float32` where precision is non-critical,
     * `chunked` reading/writing for CSV/Parquet when possible.

---

## 6. What *not* to do

* Do not discard or overwrite GKA columns unless the new version is a strict improvement and backward compatible.
* Do not “simplify away” geometric / spiral aspects for convenience (e.g. replacing GKA with raw thresholds on wspd/msl).
* Do not introduce hard-coded values that depend on a specific region or dataset unless guarded by flags and documented.

---

If you follow this document, the code changes you make will remain faithful to the underlying geometric / spiral physics while still being practical for large-scale weather forecasting experiments.
