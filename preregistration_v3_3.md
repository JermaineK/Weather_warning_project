# Preregistration V3.3 — Geometric-Pumping Loop-Area Test

**Date:** 2026-07-23

This preregistration is **independent of and does not modify** the sealed
`preregistration_v3_2.md` (composite Phi threshold). V3.2 remains the fixed,
hash-sealed criterion for the `gka_phi` grid feature. V3.3 registers a separate
hypothesis at a different data granularity (per storm **track**), tested with a
different, leakage-hardened protocol.

---

## Theory

The geometric-pumping reading treats pre-cyclone intensification as a response
to the **signed area enclosed by the storm's path in position space**. The
handedness (sign) of that loop encodes whether the circulation is being pumped
toward or away from genesis, and — because pumping is a geometric (adiabatic)
effect — the response should:

1. **flip sign between hemispheres** (Coriolis handedness reverses),
2. **scale with the enclosed |area|**, and
3. be **invariant to traversal speed** (how fast the loop is walked).

These are three *independent structured invariances*. Noise fitted to one
basin/season cannot reproduce all three, so they are free confirmatory tests
that do not consume held-out data.

---

## Feature Definition

For each storm track (grouped by `storm_id`), the ordered `(lon, lat)` path is
treated as a closed loop and its **signed shoelace area** is computed:

```python
def signed_loop_area(u1, u2):
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float)
    if u1.size < 3:
        return 0.0
    return 0.5 * float(np.sum(u1 * np.roll(u2, -1) - np.roll(u1, -1) * u2))
```

The fixed (no-free-knobs) feature set per track is
`[area_signed, area_abs, area_norm]`, where `area_norm = area_signed / path_len`
(the speed-normalised, traversal-invariant form). One sample == one track.

---

## Hypothesis

The signed loop area of a pre-cyclone track path — orientation-corrected by
hemisphere — carries genuine, non-leaked skill for cyclogenesis: a logistic
model over the fixed loop-feature set scores **above a within-block permutation
null** built with the identical block-CV pipeline, **and** the three physics
invariance guards hold.

---

## Protocol

| Element        | Setting                                                        |
|----------------|----------------------------------------------------------------|
| Sample unit    | One storm track (`storm_id`)                                   |
| Cross-val      | Leave-one-block-out, blocks = **basin × season**              |
| Null           | Labels permuted **within block**, same pipeline, n_perm ≥ 200 |
| Metrics        | PR-AUC (average precision), Brier skill score vs climatology  |
| Model          | Ridge logistic (IRLS), no sklearn, no tuned hyperparameters   |

Block-CV by both basin and season prevents spatial and temporal
autocorrelation from leaking between train and test folds.

---

## Falsification Criterion

The hypothesis is **NOT supported** if **any** of:

- Observed PR-AUC lies **inside** the within-block permutation null
  (p ≥ 0.05), **or**
- The **hemisphere sign-flip** guard fails (signed-area↔genesis rank
  correlation does not reverse sign between N and S), **or**
- The **area-scaling** guard fails (genesis rate is not monotonic across
  orientation-corrected |area| quintiles).

Speed independence (guard c) is **confirmatory** and reported, but does not
gate the verdict.

The harness ships with a mandatory **self-test** (`--self-test`) that plants a
known signal and also runs pure noise; the validator is only trustworthy if the
noise case lands *inside null with guards failing* and the planted-signal case
lands *above null with guards passing*.

---

## Source file hash

SHA-256 of `storm_geometric_validation.py` at its adding commit (`9bf121b`):

```
59978d021381de3e44abd11459c095352686ecf2d6e89e1482825c5a41ee73f7
```
