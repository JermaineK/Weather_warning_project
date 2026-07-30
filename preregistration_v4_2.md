# Preregistration V4.2 — Out-of-sample confirmation of the CAPE-gated configuration

**Date:** 2026-07-30

**Written before the confirmation data existed in analysable form.** At the time
of sealing, ERA5 for 2020 had just been processed and 2016–2019 were still
downloading; **no 2016–2020 slim panel had been evaluated against any
configuration.** This registration therefore cannot have been shaped by the
result it is designed to test.

---

## 1. What is being confirmed, and why it needs confirming

`RECOMMENDED_CONFIGURATION.md` recommends a CAPE-gated screening configuration on
the basis of a **post-hoc selection**: gate-only entered the V4.1 test as a
registered *control*, not the primary hypothesis, and was recommended because it
won on the test seasons — chosen from four configurations on **13 catchable
events**. Its precision (0.085) is therefore an **optimistic estimate**.

This registration converts that into a genuine out-of-sample test on **five
independent seasons that no part of the project has touched**.

| | fitted / already used | confirmation set |
|---|---|---|
| seasons | 2021–2025 | **2016–2020** |
| genesis events (vmax ≥ 34 kt, in domain) | 26 | **27** |

---

## 2. Frozen artefacts — nothing is refitted

The configuration is applied **exactly as it stands**. Specifically frozen:

| item | value | source |
|---|---|---|
| score model | 4-feature ridge logistic, `gka_shear_quench`, `gka_msl_nd`, `gka_SII`, `gka_knee_ratio` | fitted on **2021–2022 only** |
| sector CAPE threshold `c*` | **393.68 J/kg** | `results/metrics/v4_1_params.json` |
| sector score threshold | **0.05711** (90th pct of fit-season `G`) | fit seasons only |
| sector size | 5° × 5° | fixed |
| genesis definition | first track time at vmax ≥ 34 kt, 72 h window | fixed |

**No parameter, threshold, or model coefficient may be re-estimated on
2016–2020.** The score model's standardisation (`mu`, `sd`) is also carried over
unchanged. Doing otherwise would reproduce the flaw this test exists to remove.

---

## 3. Primary hypothesis

> The recommended configuration's measured precision on 2023–2025 (**0.085** at
> 92% recall) is not an artifact of post-hoc selection: applied unchanged to five
> unseen seasons, it will achieve precision meaningfully above the max-pool
> baseline evaluated on the same seasons.

---

## 4. Falsification criteria

Evaluated **once** on 2016–2020.

**CONFIRMED** requires all of:

1. **Precision above baseline** — sector-episode precision exceeds the max-pool
   baseline (same sectors, same episodes, baseline threshold also frozen from
   2021–2022) by **≥ 0.03**, paired bootstrap 95% CI over genesis events excluding
   zero.
2. **Recall retained** — recall ≥ **0.60**. The recommended configuration achieved
   0.92; a large recall collapse would mean the thresholds do not transfer even if
   precision holds.
3. **Precision not collapsed** — measured precision ≥ **0.04**, i.e. at least
   roughly half the 0.085 originally reported. This is the specific test of whether
   0.085 was inflated by selection.

**NOT CONFIRMED** if any of the above fails. In that case the recommendation is
downgraded: the configuration is reported as fitting 2021–2025 but not
generalising, and `RECOMMENDED_CONFIGURATION.md` must be revised accordingly.

We state in advance: **a NOT CONFIRMED outcome is the expected consequence of
post-hoc selection bias** and will be reported as such, not explained away.

---

## 5. Inconclusive criterion (carried forward from V4.1 §5)

Reported as **INCONCLUSIVE (insufficient power)** — explicitly **not** a
falsification — if on 2016–2020:

- the configuration fires **fewer than 5 episodes**, **or**
- **fewer than 5 genesis events** are catchable at the frozen thresholds.

Note the event floor is raised from 2 (V4.1) to 5, because this set is expected to
supply ~27 events; a yield below 5 would indicate a data or coverage failure
rather than a genuine null.

---

## 6. Secondary (reported, not gating)

These are recorded for information and **cannot** change the §4 verdict:

- Cell-level CAPE gate at matched alert rates on 2016–2020, to check whether the
  **+0.014** matched-rate gain replicates.
- Discriminator AUC on 2016–2020 with the frozen model, versus the 0.787
  established on 2021–2025.
- Median detection lead.

---

## 7. Known limits, unchanged by more data

More seasons fix small-N and selection bias. They do **not** fix:

- **Reanalysis, not forecast** — features come from ERA5 analysis at time *t*.
- **Storm-windowed negatives** — false alarms are storm-adjacent sectors, not a
  global background.
- **Screening, not warning** — the operating point remains low-precision.

---

## 8. Commitments

- Configuration applied unchanged; nothing refitted on 2016–2020.
- Evaluated **once**; any subsequent variation is a new registration.
- Outcome reported whatever it is — confirmed, not confirmed, or inconclusive.
- Event counts reported at every operating point.
- Baseline threshold also frozen from 2021–2022, so the comparison is fair.

---

## Source file hash

The configuration under test and the code that will evaluate it are hashed so
neither can be silently revised between this registration and the result.

| file | SHA-256 |
|---|---|
| `v4_sector_detector.py` | `965a1df1f59dd449211ce97a089a4d52a3495c7078f91a87704437177aea9130` |
| `v4_1_detector.py` | `77322bc866bf0f10f1dc487d89dc9910b08a9c2c33783dff0669267a7ef5d2ea` |
| `eval_cape_gate_cells.py` | `283a5a9ee59aebcb2894d634311db2980b551e8a0043dd4e129dccdf78024dbc` |
| `geomval_seasons.py` | `831d1c7217267518da2ed56fc1ae5021f9c71de08018204ab19ef14f2f711280` |
| `eval_spiral_genesis.py` | `cc9ee70ec340b29a2e490f7f614d376c117d16372583aeb6959dbd47b9efed0e` |
| `RECOMMENDED_CONFIGURATION.md` | `a552439b9e0bd5d2da28f9a10c344ed67c1760e630089e0ee88c41ef45b79e62` |

Frozen V4.1 parameters carried over unchanged (`results/metrics/v4_1_params.json`, self_sha256 `d2b689e4dc310b34f1cb3d7854dd7a13c771c7008940bc408b4f413c7a12cda5`):

```
{
  "c_star": 393.68124999999986,
  "g_star": 0.005,
  "h_star": 1.136738336086273
}
```

