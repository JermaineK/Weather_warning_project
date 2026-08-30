# Recommended Configuration — CAPE-gated genesis screening

**Date:** 2026-07-30 (revised 2026-07-30 with matched-rate attribution)
**Status:** recommended for screening use; **selection is post-hoc** (§5) and the
gate's contribution is **much smaller than the sector comparison implied** (§2b)

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

## 2b. CORRECTION — how much of that gain is actually the CAPE gate?

The §2 table compares two configurations **as configured**, and they fired at
different rates (248 vs 137 episodes). That confounds the gate's effect with
threshold placement, so it is **not** a clean measurement of what the gate
contributes.

A matched comparison was run at cell level (`eval_cape_gate_cells.py`): strict
genesis labels, leave-one-season-out scoring, 24 storms, paired per storm, with
**every configuration evaluated at the same alert rate** so precision differences
are attributable to the gate alone.

| CAPE gate | Δ precision (matched rate) | 95% CI | verdict |
|---|---|---|---|
| **≥ 12 J/kg** (30th pct of fit-season cape) | **+0.008 … +0.017** | excludes 0 at **all 5** alert rates | **helps** |
| ≥ 92 J/kg (50th pct) | +0.005 … +0.019 | spans 0 | not significant |
| ≥ 370 J/kg (70th pct) | **−0.030 … −0.063** | excludes 0 | **hurts** |

Lead time is essentially unchanged (+0.2 to +0.7 h).

**Two conclusions:**

1. **The gate's real contribution is ~+0.014 precision, not the ~+0.063 the
   sector table implies.** Most of the sector-level difference came from the
   recommended configuration's higher score threshold — i.e. from firing later and
   less often — not from CAPE. The §2 numbers remain correct as a description of
   those two configurations; they are wrong if read as "CAPE gives 4x precision".
2. **Only a permissive gate helps.** An aggressive CAPE threshold discards
   genuinely pre-genesis cells and significantly *degrades* precision. If the gate
   is used, it should mean "some convective energy present", not "high CAPE".

The configuration in §1 still stands — it is a reasonable operating point and the
gate does add a small robust increment. What changes is the **attribution**: the
score threshold sets the operating point, and CAPE contributes a modest extra.

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
itself — confirmed in §2b, where holding the alert rate fixed leaves the gate's
lead effect at +0.2 to +0.7 h.

A **matched-alert-rate** comparison has now been run (§2b). Comparisons at
**matched precision** or **matched lead** have not, and would be needed to state
the precision/lead trade with full precision.

---

## 4. Why the CAPE gate works when CAPE-as-a-feature did not

This is the substantive finding.

| use of CAPE | result |
|---|---|
| additive 5th model feature | paired ΔAUC **−0.0005**, CI [−0.0054, +0.0042], 24 storms — **no value** |
| standalone discriminator | AUC **0.706**; pre-genesis 488 vs 178 J/kg |
| **gate on the scanned population** | **+0.014 precision at matched alert rate** (CI excludes 0, 24 storms, permissive threshold only) — the sector-level 0.022 → 0.085 mostly reflects threshold placement, see §2b |

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

```bash
# 3. matched-alert-rate attribution of the gate at cell level
python eval_cape_gate_cells.py     --panels "data/genesis_*_slim_cape.parquet"     --tracks "data/tracks/tracks_2021.parquet,...,data/tracks/tracks_geomval.parquet"
```

The `gate_only` row of `results/metrics/v4_1_test_results.json` is the
recommended configuration; `results/metrics/cape_gate_cells.csv` carries the
matched-rate attribution. Frozen parameters and their SHA-256 are in
`results/metrics/v4_1_params.json`.

---

## 8. Related records

| document | subject | verdict |
|---|---|---|
| `preregistration_v3_2.md` | composite Phi threshold | **falsified** |
| `preregistration_v3_3.md` | geometric loop-area | **not supported** |
| `preregistration_v4_0.md` | sequential sector detector | withdrawn (failed fit) |
| `preregistration_v4_1.md` | sequential sector detector, reduced params | **not supported** |

---

## 9. Pipeline availability

The gate is wired into the cell-level trigger as **opt-in**:

```bash
python alerts_logic_manager.py genesis-trigger     --labelled <grid> --out <alerts>     --cape-nc-glob "data_era5/extracted/2025/**/era5_2025*_cape.nc"     --cape-min-quantile 0.30
```

CAPE is read from the ERA5 netCDFs per `ilat` stripe, so the full-domain field is
never held in memory; the join is guarded (a row-count change or missing CAPE
files raises rather than silently dropping the gate).

**Default is off**, because the gate needs a CAPE source the base pipeline does
not build — a default-on setting would fail or error depending on whether CAPE had
been fetched. The validated permissive threshold and the warning about aggressive
thresholds are in the `--help` text.
