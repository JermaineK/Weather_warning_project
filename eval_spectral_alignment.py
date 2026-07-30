#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spectral_alignment.py — V5.0: is there PERIODICITY in the pre-genesis signal,
and does its timescale scale with system size?

HYPOTHESIS (beat / commensurability)
    Two coupled rotating structures with different periods produce feedback when
    their leading edges align. Alignment recurs on a beat timescale, and a larger
    "influenced" spiral has a longer interval between alignments. The empirical
    prediction is therefore:
      (a) periodicity in the pre-genesis signal, over and above the monotonic build
      (b) a per-event characteristic period that INCREASES with system size

WHY THIS IS NOT ALREADY ANSWERED
    eval_lull_composite.py searched for a SINGLE interior minimum deeper than its
    bootstrap CI, and found none (monotonic build, 0.0420 -> 0.0779). A beat
    pattern is several shallow oscillations, which that test rejects by
    construction. "No lull" therefore does NOT imply "no periodicity" — they are
    different hypotheses and only the first was tested.

PRE-STATED CRITERIA (fixed before the first run; see git history)

  Resolvable band. A 72 h window at 1 h resolution resolves periods ~6-36 h.
  Periods > 36 h are indistinguishable from the trend and are NOT tested. If the
  hypothesis predicts alignment gaps > 36 h at storm scale, THIS DATA CANNOT TEST
  IT and the result must be reported as out-of-band, not as a null.

  T1 EXCESS POWER — pre-genesis periodogram power exceeds the matched control's at
     some period in band, bootstrap-over-events 95% CI excluding zero.
  T2 CLUSTERING — per-event peak periods cluster more tightly than uniform over
     the band (bootstrap CI on circular-style dispersion excludes the uniform
     expectation).
  T3 SIZE SCALING — Spearman correlation between per-event peak period and a size
     proxy is POSITIVE with bootstrap 95% CI excluding zero. THIS IS THE ACTUAL
     TEST OF THE MECHANISM; T1/T2 are prerequisites.

  SUPPORTED requires T1 AND T2 AND T3.
  Any peak within 20-28 h that is NOT also significantly stronger than the control
  is reported as DIURNAL CONFOUND, not as support: tropical convection has a
  strong ~24 h cycle, and the control composite exists to absorb it.
  INCONCLUSIVE if fewer than 10 events yield a usable detrended series.

USAGE
    python eval_spectral_alignment.py \\
        --panels "data/genesis_*_slim_cape.parquet" \\
        --tracks "data/tracks/tracks_2021.parquet,...,tracks_geomval.parquet" \\
        --out-dir results/metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_spiral_genesis import fit_logistic, predict                        # noqa: E402
from geomval_seasons import (load_storm_crops, read_tracks, resolve_files,    # noqa: E402
                             genesis_events, add_strict_labels, haversine_km)

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
BAND_LO_H, BAND_HI_H = 6.0, 36.0     # resolvable band, fixed in advance
DIURNAL_LO, DIURNAL_HI = 20.0, 28.0  # flagged as confound unless it beats control
MIN_EVENTS = 10
DETREND_ORDER = 2


def detrend(lead, val, order=DETREND_ORDER):
    ok = np.isfinite(lead) & np.isfinite(val)
    if ok.sum() < order + 3:
        return None, None
    x, y = lead[ok], val[ok]
    c = np.polyfit(x, y, order)
    return x, y - np.polyval(c, x)


def periodogram(x_h, resid, periods):
    """Lomb-Scargle power at the given periods (robust to gaps)."""
    if x_h is None or len(x_h) < 8:
        return None
    ang = 2.0 * np.pi / periods
    r = resid - np.mean(resid)
    if not np.any(np.abs(r) > 0):
        return None
    p = lombscargle(x_h.astype(float), r.astype(float), ang, normalize=True)
    return p


def boot_ci(v, rng, n_boot):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 3:
        return (float(np.nanmean(v)) if len(v) else np.nan, np.nan, np.nan, len(v))
    bm = [np.nanmean(v[rng.integers(0, len(v), size=len(v))]) for _ in range(n_boot)]
    return (float(np.nanmean(v)), float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)), len(v))


def main() -> int:
    a = parse_args()
    feats = CURATED
    crops = load_storm_crops(a.panels, a.tracks, feats, extra_cols=["cape", "zeta"],
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)
    ev_all = genesis_events(read_tracks(resolve_files(a.tracks)), a.genesis_thresh_kt)
    seasons = sorted({v["season"] for v in crops.values()})

    # season-blocked score so trajectories come from a model that never saw the season
    scored = {}
    for s in seasons:
        tr = [k for k, v in crops.items() if v["season"] != s]
        te = [k for k, v in crops.items() if v["season"] == s]
        if not tr or not te:
            continue
        Xs, ys = [], []
        for k in tr:
            d, _ = add_strict_labels(crops[k]["df"], ev_all, a.genesis_radius, a.pre_h)
            sp = d[d["pregen"] == 1]
            Xs.append(sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float))
            ys.append((sp["gen_pos"] == 1).to_numpy(float))
        X = np.vstack(Xs); y = np.concatenate(ys)
        fin = np.all(np.isfinite(X), axis=1); X, y = X[fin], y[fin]
        if y.min() == y.max():
            continue
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = fit_logistic((X - mu) / sd, y, l2=1.0)
        for k in te:
            d = crops[k]["df"].copy()
            Xa = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            m = np.all(np.isfinite(Xa), axis=1)
            sc = np.full(len(d), np.nan); sc[m] = predict(w, (Xa[m] - mu) / sd)
            d["s"] = sc
            scored[k] = {"df": d, "season": s}

    periods = np.linspace(BAND_LO_H, BAND_HI_H, a.n_periods)
    rows, spec_pre, spec_ctl = [], [], []

    for ev in ev_all.itertuples(index=False):
        cand = [k for k in scored if scored[k]["season"] == ev.season]
        if not cand:
            continue
        d = pd.concat([scored[k]["df"] for k in cand], ignore_index=True)
        d = d[d["pregen"] == 1]
        if d.empty:
            continue
        lead = (pd.Timestamp(ev.t_g) - pd.to_datetime(d["time"])) / pd.Timedelta(hours=1)
        win = (lead >= 0) & (lead <= a.back_h)
        if not win.any():
            continue
        dd = d[win].copy(); dd["lead"] = lead[win]
        dist = haversine_km(pd.to_numeric(dd["lat"], errors="coerce").to_numpy(float),
                            pd.to_numeric(dd["lon"], errors="coerce").to_numpy(float),
                            ev.lat_g, ev.lon_g)
        near = dist <= a.genesis_radius * 111.32
        far = dist > a.control_radius * 111.32
        if near.sum() < a.min_cells or far.sum() < a.min_cells:
            continue

        gp = dd[near].groupby(dd.loc[near, "lead"].round(0))["s"].mean().sort_index()
        gc = dd[far].groupby(dd.loc[far, "lead"].round(0))["s"].mean().sort_index()
        xp, rp = detrend(gp.index.to_numpy(float), gp.to_numpy(float))
        xc, rc = detrend(gc.index.to_numpy(float), gc.to_numpy(float))
        pp = periodogram(xp, rp, periods)
        pc = periodogram(xc, rc, periods)
        if pp is None or pc is None:
            continue

        # size proxy: area of the near-genesis spiral region -> equivalent radius
        n_cells = int(dd.loc[near, ["lat", "lon"]].drop_duplicates().shape[0])
        radius_km = float(np.sqrt(max(n_cells, 1) * (27.75 ** 2) / np.pi))  # 0.25deg cell
        peak_i = int(np.nanargmax(pp))
        rows.append({"event": f"{ev.season}:{ev.name}", "season": int(ev.season),
                     "peak_period_h": float(periods[peak_i]),
                     "peak_power": float(pp[peak_i]),
                     "ctl_power_at_peak": float(pc[peak_i]),
                     "excess_at_peak": float(pp[peak_i] - pc[peak_i]),
                     "n_cells": n_cells, "radius_km": radius_km,
                     "n_samples": int(len(xp))})
        spec_pre.append(pp); spec_ctl.append(pc)

    if len(rows) < MIN_EVENTS:
        print(f"[spectral] INCONCLUSIVE: only {len(rows)} usable events (min {MIN_EVENTS}).")
        return 0

    res = pd.DataFrame(rows)
    Sp = np.vstack(spec_pre); Sc = np.vstack(spec_ctl)
    rng = np.random.default_rng(0)
    print(f"[spectral] {len(res)} events, band {BAND_LO_H:.0f}-{BAND_HI_H:.0f}h, "
          f"{a.n_periods} period bins, detrend order {DETREND_ORDER}")

    # ---- T1: excess power vs control, per period ----
    t1_rows = []
    for j, P in enumerate(periods):
        m, lo, hi, n = boot_ci(Sp[:, j] - Sc[:, j], rng, a.n_boot)
        t1_rows.append({"period_h": float(P), "excess_mean": m,
                        "ci_lo": lo, "ci_hi": hi, "n": n,
                        "significant": bool(np.isfinite(lo) and lo > 0)})
    t1 = pd.DataFrame(t1_rows)
    sig = t1[t1["significant"]]
    T1 = bool(len(sig))
    print(f"\n[T1] excess power vs control: "
          f"{'PASS' if T1 else 'FAIL'} ({len(sig)}/{len(t1)} period bins significant)")
    if T1:
        b = sig.sort_values("excess_mean", ascending=False).iloc[0]
        print(f"     strongest: {b.period_h:.1f}h  excess={b.excess_mean:+.4f} "
              f"[{b.ci_lo:+.4f},{b.ci_hi:+.4f}]")

    # ---- T2: clustering of per-event peak periods ----
    pk = res["peak_period_h"].to_numpy(float)
    obs_sd = float(np.std(pk))
    unif_sd = (BAND_HI_H - BAND_LO_H) / np.sqrt(12.0)
    bsd = [np.std(pk[rng.integers(0, len(pk), size=len(pk))]) for _ in range(a.n_boot)]
    sd_hi = float(np.percentile(bsd, 97.5))
    T2 = bool(sd_hi < unif_sd)
    print(f"\n[T2] peak-period clustering: {'PASS' if T2 else 'FAIL'}  "
          f"observed sd={obs_sd:.2f}h (95% upper {sd_hi:.2f}h) vs uniform {unif_sd:.2f}h")

    # ---- T3: does peak period scale with size? (the actual mechanism test) ----
    rho = spearmanr(res["radius_km"], res["peak_period_h"]).statistic
    brho = []
    for _ in range(a.n_boot):
        i = rng.integers(0, len(res), size=len(res))
        r = spearmanr(res["radius_km"].to_numpy()[i], pk[i]).statistic
        if np.isfinite(r):
            brho.append(r)
    r_lo, r_hi = (float(np.percentile(brho, 2.5)), float(np.percentile(brho, 97.5))) \
        if len(brho) > 20 else (np.nan, np.nan)
    T3 = bool(np.isfinite(r_lo) and r_lo > 0)
    print(f"\n[T3] size scaling (Spearman peak_period vs radius): "
          f"{'PASS' if T3 else 'FAIL'}  rho={rho:+.3f} 95%CI[{r_lo:+.3f},{r_hi:+.3f}]")

    # diurnal check
    diurnal = t1[(t1.period_h >= DIURNAL_LO) & (t1.period_h <= DIURNAL_HI)]
    diurnal_sig = bool(diurnal["significant"].any())
    peaks_in_diurnal = float(((pk >= DIURNAL_LO) & (pk <= DIURNAL_HI)).mean())
    print(f"\n[diurnal] {peaks_in_diurnal:.0%} of per-event peaks fall in "
          f"{DIURNAL_LO:.0f}-{DIURNAL_HI:.0f}h; excess-vs-control significant there: "
          f"{diurnal_sig}")
    if peaks_in_diurnal > 0.5 and not diurnal_sig:
        print("[diurnal] WARNING: peaks cluster near 24h WITHOUT beating the control "
              "-> DIURNAL CONFOUND, not mechanism support.")

    supported = T1 and T2 and T3
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"\n[spectral] VERDICT: V5.0 beat-alignment {verdict}  (T1={T1} T2={T2} T3={T3})")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "spectral_events.csv", index=False)
    t1.to_csv(out / "spectral_excess_by_period.csv", index=False)
    (out / "spectral_alignment.json").write_text(json.dumps(
        {"n_events": int(len(res)), "band_h": [BAND_LO_H, BAND_HI_H],
         "T1_excess_power": T1, "T2_clustering": T2, "T3_size_scaling": T3,
         "spearman_rho": float(rho), "rho_ci": [r_lo, r_hi],
         "peak_sd_h": obs_sd, "uniform_sd_h": float(unif_sd),
         "diurnal_peak_fraction": peaks_in_diurnal,
         "diurnal_excess_significant": diurnal_sig,
         "verdict": verdict,
         "out_of_band_note": "periods > 36h are not resolvable in a 72h window and "
                             "were not tested"}, indent=2, default=str))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    ax1.plot(t1.period_h, Sp.mean(0), "-", color="#c1121f", lw=2, label="pre-genesis")
    ax1.plot(t1.period_h, Sc.mean(0), "--", color="#457b9d", lw=1.6, label="control")
    ax1.axvspan(DIURNAL_LO, DIURNAL_HI, color="grey", alpha=0.15, label="diurnal band")
    ax1.set_xlabel("period (h)"); ax1.set_ylabel("Lomb-Scargle power")
    ax1.set_title(f"Mean spectrum, {len(res)} events"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.scatter(res.radius_km, res.peak_period_h, s=28, color="#2a9d8f")
    ax2.set_xlabel("equivalent radius (km)"); ax2.set_ylabel("peak period (h)")
    ax2.set_title(f"T3 size scaling: rho={rho:+.3f} [{r_lo:+.3f},{r_hi:+.3f}]")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "spectral_alignment.png"); plt.close(fig)
    print(f"[spectral] wrote {out/'spectral_alignment.json'} (+ csvs, png)")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="V5.0 spectral test of the beat-alignment hypothesis.")
    ap.add_argument("--panels", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--genesis-thresh-kt", type=float, default=34.0)
    ap.add_argument("--genesis-radius", type=float, default=3.0)
    ap.add_argument("--control-radius", type=float, default=5.0)
    ap.add_argument("--back-h", type=float, default=72.0)
    ap.add_argument("--n-periods", type=int, default=31)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=2000)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
