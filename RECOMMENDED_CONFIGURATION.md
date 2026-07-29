# Recommended Configuration — CAPE-gated genesis screening

**Date:** 2026-07-30
**Status:** recommended for screening use; **selection is post-hoc** (see §5)

---

## 1. Recommendation

Use a **CAPE gate followed by a threshold on the V3.3 discriminator score**,
evaluated over 5° sectors:

```
sector = 5° x 5° box, hourly
  C(t) = median cape        over spiral cells (pregen == 1) in the sector
  G(t) = mean V3.3 score    over the same cells
FIRE when   C(t) >= 393.7 J/kg   AND   G(t) >= 0.0571
```

where the V3.3 score is the 4-feature ridge logistic
(`gka_shear_quench`, `gka_msl_nd`, `gka_SII`, `gka_knee_ratio`), and both
thresholds were set on the **fit seasons (2021–2022) only**.

Do **not** use: the sequential staging (V4.0/V4.1, falsified), the trailing
accumulation trigger (falsified), the shape filter (not supported), or `gka_phi`
(falsified).

---

## 2. Measured performance

Test seasons **2023–2025**, evaluated once, 17 genesis events (13 catchable at
these thresholds), 46,801 sector-hours:

| configuration | fired | precision | recall | median lead | events caught |
|---|---|---|---|---|---|
| **CAPE-gate + score (recommended)** | 248 | **0.085** | **0.923** | 14.0 h | **12 / 13** |
| max-pool score only (baseline) | 137 | 0.022 | 1.000 | **36.0 h** | 3 / 3 |
| sequential staging (V4.1) | 43 | 0.070 | 0.231 | 37.0 h | 3 / 13 |

**The gain is precision at fixed recall:** ~3.9× the baseline's precision
(0.085 vs 0.022) while still catching 12 of 13 events.

---

## 3. This is a trade, not a dominance

The three configurations are **not at a common operating point**, and the honest
comparison is:

- **Want lead time** → max-pool, ~36 h median lead, but precision 0.022
  (≈1 in 45 alarms is real).
- **Want precision** → CAPE-gate + score, precision 0.085 (≈1 in 12), but median
  lead drops to ~14 h.

The recommended configuration fires **later** because its score threshold (90th
percentile of fit-season `G`) is high, so it triggers closer to formation. The
lead difference is a consequence of threshold placement, not of the CAPE gate
itself. **A like-for-like comparison at matched precision or matched lead has not
been run** and would be needed to state the trade precisely.

---

## 4. Why the CAPE gate works when CAPE-as-a-feature did not

This is the substantive finding.

| use of CAPE | result |
|---|---|
| additive 5th model feature | paired ΔAUC **−0.0005**, CI [−0.0054, +0.0042], 24 storms — **no value** |
| standalone discriminator | AUC **0.706**; pre-genesis 488 vs 178 J/kg |
| **gate on the scanned population** | **precision 0.022 → 0.085** at equal recall |

A gate changes *which cells are scanned*; a feature changes *the score within a
fixed population*. They are different operations, and the null for one does not
transfer to the other. The event-centred composite independently supports the
gate reading: CAPE is the **earliest** field to separate from control
(T−57 h; 483 vs 358 J/kg), before instability (T−39 h) or low-level shear
(T−15 h).

---

## 5. Provenance and the post-hoc caveat (read before quoting numbers)

- Thresholds were fitted on **2021–2022 only** and applied unchanged to
  2023–2025, so there is **no threshold leakage** into the reported numbers.
- **However**, gate-only entered the test as a *registered control* under
  `preregistration_v4_1.md`, not as the primary hypothesis. It is recommended
  here **because it won on the test seasons** — that is selection on the test
  set, over four configurations.
- Consequently **0.085 precision should be treated as an optimistic estimate.**
  Selecting the best of four configurations on a 13-event test set carries real
  selection bias.
- Confirming it properly requires a fresh preregistration with gate-only as the
  **primary** hypothesis, evaluated on data not used here — a new season, a new
  basin, or a held-back split.

---

## 6. Scope limits

- **Reanalysis, not forecast.** Features come from ERA5 analysis at time *t*. An
  operational system would use forecast fields with their own error growth.
- **Storm-windowed negatives.** The season panels cover ±6° / −120 h..+24 h around
  real tracks, so "false alarms" are *storm-adjacent* sectors, not a global
  background. Domain-wide false-alarm rates will differ.
- **Screening, not warning.** At ~1 in 12 alarms real, this narrows attention; it
  does not support a public warning. Calibration analysis found the model well
  calibrated but low-resolution (BSS +0.073), with precision capping ~54% even at
  extreme thresholds.
- **13 catchable events.** Every number here rests on a small positive sample.

---

## 7. Reproduction

```bash
# 1. build the hourly 5deg sector series (score model fitted on 2021-2022)
python v4_sector_detector.py --mode sectors

# 2. fit + freeze thresholds on 2021-2022, then evaluate once on 2023-2025
python v4_1_detector.py --mode fit
python v4_1_detector.py --mode test
```

The `gate_only` row of `results/metrics/v4_1_test_results.json` is the
recommended configuration. Frozen parameters and their SHA-256 are in
`results/metrics/v4_1_params.json`.

---

## 8. Related records

| document | subject | verdict |
|---|---|---|
| `preregistration_v3_2.md` | composite Phi threshold | **falsified** |
| `preregistration_v3_3.md` | geometric loop-area | **not supported** |
| `preregistration_v4_0.md` | sequential sector detector | withdrawn (failed fit) |
| `preregistration_v4_1.md` | sequential sector detector, reduced params | **not supported** |
