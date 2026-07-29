# Preregistration V4.0 — Sequential Sector Genesis Detector

**Date:** 2026-07-29

Registered **before** any detector code is written. Supersedes nothing: V3.2 and
V3.3 remain sealed and falsified. This registers a change of *unit and structure*,
not another feature.

---

## 1. Why a pivot, and what the V3.x work established

V3.x built a **per-cell, instantaneous, additive** classifier. It works, modestly:
held-out AUC **0.787** (95% CI [0.753, 0.821]) over 24 storms / 5 seasons,
season-blocked, strict genesis labels; skill decaying 0.930 (6 h) → 0.692 (48 h).

It also hit a wall that three independent tests agree on:

| finding | evidence |
|---|---|
| Feature saturation | `shear_low` 0.810 standalone → **+0.000** incremental; `cape` 0.706 standalone → **−0.001** incremental; 18-feature model underperforms the curated 4 |
| Low sharpness | Calibration: reliability 0.00057 (excellent) but resolution 0.00474, BSS **+0.073** |
| Over-confidence at the top | Forecast 0.93 → observed 0.47; precision caps ~54% |

**The additive-scalar approach is exhausted.** Adding more environmental scalars
to a logistic model does not help.

Two results specifically motivate a *different structure*:

* **CAPE is a gate, not a feature.** It adds nothing additively, yet in the
  event-centred composite it is the **earliest** field to separate from control
  (T−57 h; 483 vs 358 J/kg). A gate changes the population scanned; a feature
  changes the score within a fixed population. These are different objects and
  the V3.x null does not apply to the former.
* **The sequence is ordered.** Composite separation times: CAPE T−57 h →
  `gka_SII` T−39 h → discriminator score T−45 h → `shear_low` T−15 h. The
  ingredients arrive in a consistent order, which a snapshot classifier discards.

### What was tested and rejected on the way here

The originating intuition included an "activity then lull" signature. It is
**not supported**. Event-centred, longitudinal, season-blocked compositing over
24 genesis events shows a **monotonic build** (0.0420 → 0.0779) with a flat
control (0.0399–0.0421) and **no interior local minimum** deeper than its
bootstrap CI. The earlier build→dip→surge was an artifact of (a) the generous
`near_storm` label and (b) cross-sectional rather than longitudinal averaging.

**V4.0 therefore registers a monotonic-build detector, with no lull stage.**
This is a deliberate simplification that removes two free parameters (lull depth,
lull duration) that would otherwise be the main overfitting surface.

---

## 2. Hypothesis

> Genesis is preceded by an **ordered, sustained sequence** of environmental and
> structural changes within a spatial sector. A detector that requires this
> sequence — CAPE reservoir, then building instability and organisation, then
> low-level shear confirmation — will identify genesis sectors with **better
> precision at equal or greater lead time** than max-pooling the V3.3 per-cell
> score over the same sector.

The comparison is deliberately against the strongest simple alternative. If the
sequential structure adds nothing over pooling the existing score, it is not
worth its complexity, and V4.0 is not supported.

---

## 3. Definitions (fixed in advance)

**Sector.** Fixed, non-overlapping 5° × 5° lat/lon boxes on the analysis domain,
evaluated hourly. Fixed tiling is chosen over object detection specifically to
avoid free parameters in sector formation.

**Sector state at hour t** — aggregates over spiral cells (`pregen == 1`) in the sector:
- `C(t)` = median `cape`
- `I(t)` = median `gka_SII`
- `G(t)` = mean V3.3 discriminator score (4 curated features, season-blocked model)
- `H(t)` = median `shear_low`

**Stages** (each must hold for at least `D` consecutive hours):
1. **S1 Reservoir** — `C(t) ≥ c*`
2. **S2 Build** — `I(t)` and `G(t)` both non-decreasing over a trailing `W` hours,
   with `G(t) − G(t−W) ≥ g*`
3. **S3 Confirm** — `H(t) ≥ h*`

A **detection** fires at the first hour S1→S2→S3 have all been satisfied **in order**.

**Episode.** A maximal run of hours in which a sector satisfies S1. Episodes are
the unit of analysis; each is labelled positive if a genesis event (first track
time with vmax ≥ 34 kt) occurs inside that sector within 72 h of the episode's
detection hour, else negative.

**Free parameters:** exactly six — `c*, g*, h*, D, W`, and the S2 trailing window
step. No others may be introduced. Sector size (5°) and the genesis definition
(34 kt, 72 h) are **fixed, not tuned**.

---

## 4. Protocol

| | |
|---|---|
| **Fit** | 2021 + 2022 only (11 storms). All six parameters selected here by maximising sector-episode F1. |
| **Freeze** | Parameters written to `results/metrics/v4_params.json` and hashed **before** any test-season evaluation. |
| **Test** | 2023 + 2024 + 2025 (19 storms), evaluated **once** with frozen parameters. |
| **Baseline** | Same sectors, same episodes: max-pool the V3.3 per-cell score over the sector, threshold swept on 2021–2022 only. |
| **Statistics** | Bootstrap over **genesis events**, not episodes or cells. |

The test seasons are evaluated **once**. Any re-tuning after seeing test results
invalidates this preregistration and requires a V4.1.

---

## 5. Falsification criteria

V4.0 is **NOT SUPPORTED** if any of the following holds on the test seasons:

1. **No precision gain.** Sector-episode precision at matched recall does not
   exceed the max-pool baseline by **≥ 0.05**, with the paired bootstrap 95% CI
   for the difference excluding zero.
2. **No lead advantage.** Median detection lead time is **< 24 h** before genesis,
   or is not greater than the baseline's median lead.
3. **Order does not matter.** A permuted-order control — requiring the same three
   stages but in a randomised order — performs within the 95% CI of the ordered
   detector. If order is irrelevant, the "sequence" claim is empty.
4. **No skill over the gate alone.** The full detector does not beat S1 (CAPE
   gate) plus a simple threshold on `G(t)`, i.e. the staging adds nothing beyond
   gate-and-score.

Criterion 3 is the one that specifically tests the *sequential* claim, as opposed
to merely conjoining conditions. Criterion 4 guards against the detector's
apparent skill coming entirely from the CAPE gate.

---

## 6. Known power limitation (stated in advance)

The test set contains **19 storms, of which ~15 produce genesis events** meeting
the 34 kt criterion. This is a small positive sample and the study is
**underpowered to detect small effects**. The 0.05 precision threshold in
criterion 1 is set deliberately above what this sample can resolve by chance;
effects smaller than that will be reported as "not supported", not as trends.

We commit in advance to reporting the result **whatever it is**, including a null,
and to reporting the number of contributing events at every operating point.

---

## 7. Anti-overfitting commitments

- Parameters fitted on 2021–2022 only, frozen and hashed before test evaluation.
- Exactly six free parameters; no post-hoc additions.
- Sector size and genesis definition fixed, not tuned.
- Test seasons evaluated once.
- Permuted-order and gate-only controls run alongside the main result.
- Bootstrap resampling at the **event** level throughout.

---

## Source file hash

The V4.0 hypothesis rests on results produced by the scripts below. Their
SHA-256 values are recorded so the evidence base is reproducible and cannot
be silently revised after the fact.

| file | SHA-256 |
|---|---|
| `eval_lull_composite.py` | `2c4b02a6c347f3f592fb46d6c85a974c49228b4660136cff028670556e794725` |
| `eval_multiseason_battery.py` | `6192415806ee8aca7720c54e4d00ae6e0f5be7c440d97788d258669cbcee6675` |
| `eval_calibration.py` | `58fd0bcb05268f4ed269b6390e21dc2f9620661349638773f2bdbf5f875c5af3` |
| `geomval_seasons.py` | `831d1c7217267518da2ed56fc1ae5021f9c71de08018204ab19ef14f2f711280` |
| `eval_spiral_genesis.py` | `cc9ee70ec340b29a2e490f7f614d376c117d16372583aeb6959dbd47b9efed0e` |

Detector implementation and fitted parameters will be hashed separately in
`results/metrics/v4_params.json` at the freeze step, before any test-season
evaluation.
