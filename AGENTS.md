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

If in doubt: **prefer correctness over cleverness**. It’s better to be slightly slow and right than blazing fast and wrong.


---


  Machine-read schema + unit expectations.
- `tests/`  
  What must never silently break.
- `features_subprocess/*.py`  
  Pipeline modules: each one should have a clear single responsibility.

If you are unsure whether a change violates the intent: err on the side of **not changing the mathematics** and focus on clarity, robustness, and performance only.
