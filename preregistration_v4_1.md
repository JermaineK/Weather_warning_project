# Preregistration V4.1 — Sequential Sector Genesis Detector (reduced parameterisation)

**Date:** 2026-07-29

Supersedes the *protocol* of V4.0. The **hypothesis is unchanged**; V4.0's
parameter-fitting procedure failed and its verdict is withdrawn as uninformative
rather than treated as a falsification. V3.2 and V3.3 remain sealed and falsified.

---

## 1. Why V4.0 has to be redone rather than believed

V4.0 returned **NOT SUPPORTED** on all four criteria: the sequential detector
fired **zero times** on the test seasons. That verdict is **not credible as
evidence about the hypothesis**, for a reason visible in the fit output:

| V4.0 fit diagnostics (2021–2022) | value |
|---|---|
| genesis events in fit seasons | 9 |
| events in sectors with a qualifying S1 episode | **3** |
| episodes fired at the selected parameters | **4** |
| selected F1 | 0.571 |

Six free parameters were selected by maximising F1 computed on **four
detections**. With three positive events, F1 is maximised by a detector that
almost never fires — so the grid search chose the most restrictive corner
available (`c*` at the 70th percentile, `g*` at the grid maximum, `D = 6`).
Those thresholds then fired never on held-out data.

**V4.0 tested the fitting procedure, not the hypothesis.** All 9 fit-season and
all 17 test-season genesis events lie in sectors with panel coverage, so this is
not a data-coverage failure — it is parameter selection collapsing on a tiny
positive sample.

### What V4.0 did establish

The registered **baseline** ran cleanly and is retained as the number to beat:

| detector | precision | recall | median lead | events |
|---|---|---|---|---|
| baseline max-pool of V3.3 score | 0.056 | 1.000 | 15.5 h | 6/6 |
| gate-only (CAPE + score threshold) | 0.034 | 0.625 | 11.0 h | 5/8 |

---

## 2. Hypothesis (unchanged from V4.0)

> Genesis is preceded by an **ordered, sustained sequence** within a spatial
> sector — CAPE reservoir, then building instability and organisation, then
> low-level shear confirmation. A detector requiring this sequence will identify
> genesis sectors with **better precision at equal or greater lead time** than
> max-pooling the V3.3 per-cell score over the same sector.

---

## 3. Changes from V4.0

### 3.1 Three free parameters, not six

`D` and `W` are **fixed in advance from the V3.x event-centred composite**
(`eval_lull_composite.py`, hashed below) — evidence that predates this
registration and is independent of the test seasons:

| fixed | value | justification |
|---|---|---|
| `W` (build window) | **24 h** | The composite rises materially over ~24 h intervals; separation from control emerges T−45 h and grows monotonically to genesis. |
| `D` (persistence) | **6 h** | Long enough to reject hourly noise, short relative to the 72 h genesis window. |
| S2 window step | **1 h** | Native data resolution; not a tuned quantity. |

**Remaining free parameters: exactly three — `c*`, `g*`, `h*`.**

### 3.2 A fitting objective that does not collapse on rare events

F1 is pathological at this sample size. V4.1 instead selects parameters by:

> **Maximise events caught, subject to a false-alarm ceiling of 1 fired episode
> per 20 sector-episodes (5%).**

This is the operational framing (maximise detection at a tolerable alarm rate)
and, unlike F1, it does not reward a detector that never fires — a detector that
fires zero times catches zero events and scores worst, not best.

### 3.3 Leave-one-event-out fitting within the fit seasons

Parameters are selected by **leave-one-event-out** across the fit seasons'
genesis events, scoring each candidate on the held-out event. This prevents the
selection being driven by a single event, which is the specific failure mode that
produced V4.0's degenerate corner solution.

### 3.4 Split unchanged

Fit on **2021 + 2022**; test **once** on **2023 + 2024 + 2025**. Deliberately not
rebalanced: protecting the test set matters more than easing the fit, and the
fit-side problem is addressed by 3.1–3.3 rather than by borrowing test data.

---

## 4. Falsification criteria (unchanged from V4.0)

V4.1 is **NOT SUPPORTED** if, on the test seasons:

1. **No precision gain** — precision at matched recall does not exceed the
   max-pool baseline by ≥ **0.05**, paired bootstrap 95% CI excluding zero.
2. **No lead advantage** — median detection lead < **24 h**, or not greater than
   the baseline's median lead.
3. **Order does not matter** — a permuted-order control performs within the
   ordered detector's 95% CI.
4. **No skill over the gate alone** — the full detector does not beat CAPE-gate
   plus a plain threshold on `G(t)`.

---

## 5. Inconclusive criterion (NEW — declared before running)

V4.0's zero-fire result was explained away **after** it was seen. That is exactly
the move preregistration exists to prevent, so the distinction is now fixed in
advance.

The result is reported as **INCONCLUSIVE (insufficient power)**, and explicitly
**not** as a falsification, if on the test seasons:

- the detector fires **fewer than 5 episodes**, **or**
- **fewer than 2 genesis events** are available to be caught at the frozen
  parameters.

An inconclusive result is a failure of this study design, not evidence against
the hypothesis, and must be reported as such. A **NOT SUPPORTED** verdict
requires the detector to have fired enough to be measurable **and** to have lost
on the criteria in §4.

Symmetrically: a fit that produces fewer than **5 fired episodes on the fit
seasons** is declared a failed fit; the test is not run, and the design is
revised under a V4.2 rather than burning the test set.

---

## 6. Power statement

Total: **26 genesis events** — 9 fit, 17 test. Three free parameters against 9
fit-season events remains thin, and the study can only detect large effects. The
§4 thresholds are deliberately set above what this sample resolves by chance.

We commit in advance to reporting the outcome **whatever it is** — supported,
not supported, or inconclusive — with the number of contributing events at every
operating point.

---

## 7. Anti-overfitting commitments

- Three free parameters; `D`, `W`, sector size, and the genesis definition fixed.
- `D`/`W` justified from pre-existing composite evidence, not tuned on any data.
- Leave-one-event-out selection within fit seasons.
- Parameters frozen and hashed before any test evaluation.
- Test seasons evaluated **once**.
- Permuted-order and gate-only controls reported alongside the main result.
- Bootstrap resampling at the **event** level.
- Failed-fit and inconclusive conditions declared above, before running.

---

## Source file hash

The V4.1 design rests on results and code below. `eval_lull_composite.py` is
hashed specifically because the fixed values of `D` and `W` are justified from
its output, so that justification cannot be silently revised.

| file | SHA-256 |
|---|---|
| `eval_lull_composite.py` | `2c4b02a6c347f3f592fb46d6c85a974c49228b4660136cff028670556e794725` |
| `v4_sector_detector.py` | `965a1df1f59dd449211ce97a089a4d52a3495c7078f91a87704437177aea9130` |
| `geomval_seasons.py` | `831d1c7217267518da2ed56fc1ae5021f9c71de08018204ab19ef14f2f711280` |
| `eval_multiseason_battery.py` | `6192415806ee8aca7720c54e4d00ae6e0f5be7c440d97788d258669cbcee6675` |
| `eval_spiral_genesis.py` | `cc9ee70ec340b29a2e490f7f614d376c117d16372583aeb6959dbd47b9efed0e` |

Fitted parameters will be hashed in `results/metrics/v4_1_params.json` at the
freeze step, before any test-season evaluation.
