# Preregistration V3.2 — Composite Phi Threshold Test

**Date:** 2026-04-30

---

## Feature Definition

**gka_phi** (Φ) is a composite signal combining relaxation resistance, spiral agreement, and
local vorticity variance:

```
Phi = (1 / (|relax| + 1e-6)) * agree / (V_zeta + 1e-6)
```

where:
- `relax` — column bound via alias `["relax", "relaxation", "relax_rate"]`
- `agree` — agreement/overlap column (alias `["agree", "agreement", "overlap"]`)
- `V_zeta` — rolling variance of zeta (alias `["zeta", "zeta_mean"]`) over a 7-step centered
  window per `(lat, lon)` group, `min_periods=3`
- `ε = 1e-6` guards against division by zero in both denominators

The raw Phi is robust-scaled (median/MAD) to produce `gka_phi`.

If the `relax` column is absent, `gka_phi` is set to NaN and a warning is emitted to stderr;
the column is **never zero-filled** so downstream NaN checks can detect missing input.

### Code snippet (from `features_subprocess/compute_gka_features.py`)

```python
v_zeta = _roll_group_var(
    _safe_num(zeta_for_var),
    out["lat"], out["lon"],
    out["ilat"] if "ilat" in out.columns else None,
    out["ilon"] if "ilon" in out.columns else None,
    window=7,
)
phi_raw = (1.0 / (np.abs(r_arr) + 1e-6)) * a_arr / (v_zeta + 1e-6)
phi_med = np.nanmedian(phi_raw)
phi_mad = np.nanmean(np.abs(phi_raw - phi_med)) + 1e-6
out["gka_phi"] = (phi_raw - phi_med) / phi_mad
```

---

## Hypothesis

`gka_phi` threshold separates intensifying from dissipating pre-cyclone cells with sharper
discrimination than the Genesis Potential Index (GPI).  Specifically, a logistic model with
a structural break at threshold φ_c will achieve both:

1. A statistically significant break in the slope of P(intensify | φ) at φ_c (bootstrap 95%
   CI for slope difference excludes zero), and
2. AUC gain over a GPI baseline ≥ 0.02 on the held-out test period.

---

## Data Split

| Split | Period            | Role      |
|-------|-------------------|-----------|
| Train | Feb–Mar 2025      | Fit M_smooth + M_break, grid-search φ_c |
| Test  | Apr 2025          | Evaluate AUC, Brier, bootstrap CI       |

---

## Falsification Criterion

The hypothesis is **not supported** if either:

- The bootstrap 95% CI for the break slope difference overlaps the smooth (no-break) slope, **or**
- The AUC gain of M_break over the GPI baseline is < 0.02 on the test set.

Both criteria must be met for the result to be labelled **SUPPORTED**.

---

## Source file hash

SHA-256 of `features_subprocess/compute_gka_features.py` at Task-1 commit
(`01a378a`):

```
b79bd4be67c7df02d398ce25cf938b8f69090f01856ad4630145370397b6e854
```
