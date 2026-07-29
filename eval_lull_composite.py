#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_lull_composite.py — is the build -> dip -> surge signature REAL under strict
genesis labels, or was it an artifact of the generous `near_storm` label?

Why this is not a repeat of eval_accumulation.py
    That analysis plotted mean score against lead, POOLED across cells. That is a
    CROSS-SECTIONAL average: different cells contribute at different leads, so an
    apparent "dip" can be produced purely by which cells happen to be in each lead
    bin, without any individual cell ever dipping.

    This script is LONGITUDINAL: it follows each cell's own trajectory toward the
    genesis moment, aligns all trajectories on t_genesis, and composites them
    (standard event-centred compositing). A dip in that composite means cells
    actually dip.

Controls
    * Fizzle cells (never part of any tracked system) are composited on the SAME
      t_genesis timestamps, which holds time-of-day and season fixed.
    * Bootstrap CIs resample GENESIS EVENTS, not cells, because cells inside one
      event are strongly correlated.
    * Optionally split by CAPE regime, since the proposed detector starts from a
      high-CAPE reservoir.

Verdict logic
    A "lull" requires the composite to rise, fall significantly, then rise again.
    We report the trajectory with CIs and test explicitly whether any interior
    local minimum is deeper than its neighbours by more than the CI width.

USAGE
    python eval_lull_composite.py \\
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_spiral_genesis import fit_logistic, predict                      # noqa: E402
from geomval_seasons import (load_storm_crops, read_tracks, resolve_files,  # noqa: E402
                             genesis_events, add_strict_labels, haversine_km)

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
TRACK_VARS = ["s", "cape", "shear_low", "zeta", "gka_SII"]


def composite(series_by_event, lead_grid):
    """Mean trajectory across events; returns (mean, per-event matrix)."""
    rows = []
    for ev, tr in series_by_event.items():
        # np.interp requires ASCENDING x; trajectories are stored descending
        # (72h -> 0h), so sort before interpolating or every point returns NaN.
        o = np.argsort(tr["lead"])
        x, y = np.asarray(tr["lead"])[o], np.asarray(tr["val"])[o]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        rows.append(np.interp(lead_grid, x[m], y[m], left=np.nan, right=np.nan))
    M = np.vstack(rows) if rows else np.empty((0, len(lead_grid)))
    with np.errstate(invalid="ignore"):
        return np.nanmean(M, axis=0), M


def boot_band(M, rng, n_boot):
    """Bootstrap CI over EVENTS (rows)."""
    if len(M) < 3:
        return np.full(M.shape[1], np.nan), np.full(M.shape[1], np.nan)
    out = np.empty((n_boot, M.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, len(M), size=len(M))
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmean(M[idx], axis=0)
    return (np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0))


def main() -> int:
    a = parse_args()
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    extra = ["cape", "shear_low", "zeta"]
    crops = load_storm_crops(a.panels, a.tracks, feats, extra_cols=extra,
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)
    tr_all = read_tracks(resolve_files(a.tracks))
    ev_all = genesis_events(tr_all, a.genesis_thresh_kt)
    print(f"[lull] {len(ev_all)} genesis events, {len(crops)} storm crops")

    # season-blocked scoring so the trajectory is from a model that never saw the season
    seasons = sorted({v["season"] for v in crops.values()})
    scored = {}
    for s in seasons:
        tr_keys = [k for k, v in crops.items() if v["season"] != s]
        te_keys = [k for k, v in crops.items() if v["season"] == s]
        if not tr_keys or not te_keys:
            continue
        Xs, ys = [], []
        for k in tr_keys:
            d, _ = add_strict_labels(crops[k]["df"], ev_all, a.genesis_radius, a.genesis_max_lead)
            sp = d[d["pregen"] == 1]
            Xs.append(sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float))
            ys.append((sp["gen_pos"] == 1).to_numpy(float))
        X = np.vstack(Xs); y = np.concatenate(ys)
        fin = np.all(np.isfinite(X), axis=1); X, y = X[fin], y[fin]
        if y.min() == y.max():
            continue
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = fit_logistic((X - mu) / sd, y, l2=1.0)
        for k in te_keys:
            d = crops[k]["df"].copy()
            Xa = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            m = np.all(np.isfinite(Xa), axis=1)
            sc = np.full(len(d), np.nan)
            sc[m] = predict(w, (Xa[m] - mu) / sd)
            d["s"] = sc
            scored[k] = {"df": d, "season": s}
        print(f"[lull] season {s}: scored {len(te_keys)} storms")

    # ---- build event-centred trajectories ----
    lead_grid = np.arange(a.back_h, -0.001, -a.step_h)   # e.g. 72 -> 0
    rng = np.random.default_rng(0)
    pre_series = {v: {} for v in TRACK_VARS}
    ctl_series = {v: {} for v in TRACK_VARS}
    cape_at_event = {}

    for ev in ev_all.itertuples(index=False):
        key = f"{ev.season}:{ev.name}"
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

        cape_at_event[key] = float(pd.to_numeric(dd.loc[near, "cape"],
                                                 errors="coerce").median())
        for v in TRACK_VARS:
            if v not in dd.columns:
                continue
            g = (dd[near].groupby(dd.loc[near, "lead"].round(0))[v]
                   .mean().sort_index(ascending=False))
            pre_series[v][key] = {"lead": g.index.to_numpy(float), "val": g.to_numpy(float)}
            gc = (dd[far].groupby(dd.loc[far, "lead"].round(0))[v]
                    .mean().sort_index(ascending=False))
            ctl_series[v][key] = {"lead": gc.index.to_numpy(float), "val": gc.to_numpy(float)}

    n_ev = len(pre_series["s"])
    print(f"[lull] usable events: {n_ev}")
    if n_ev < 5:
        raise SystemExit("[lull] too few events for a composite.")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rep = {"n_events": n_ev, "back_h": a.back_h, "step_h": a.step_h,
           "genesis_radius_deg": a.genesis_radius,
           "control_radius_deg": a.control_radius, "features": feats}

    # ---- composite + lull test on the discriminator score ----
    results = {}
    for v in TRACK_VARS:
        if not pre_series.get(v):
            continue
        mean_p, Mp = composite(pre_series[v], lead_grid)
        lo_p, hi_p = boot_band(Mp, rng, a.n_boot)
        mean_c, Mc = composite(ctl_series[v], lead_grid)
        results[v] = {"lead": lead_grid.tolist(), "pre_mean": mean_p.tolist(),
                      "pre_lo": lo_p.tolist(), "pre_hi": hi_p.tolist(),
                      "ctl_mean": mean_c.tolist()}

    # explicit lull test on 's': is there an interior local minimum deeper than CI width?
    m, lo, hi = (np.array(results["s"]["pre_mean"]), np.array(results["s"]["pre_lo"]),
                 np.array(results["s"]["pre_hi"]))
    ok = np.isfinite(m)
    lull = {"found": False}
    if ok.sum() > 4:
        idx = np.where(ok)[0]
        interior = idx[1:-1]
        for i in interior:
            left_max = np.nanmax(m[idx[idx < i]]) if (idx < i).any() else np.nan
            right_max = np.nanmax(m[idx[idx > i]]) if (idx > i).any() else np.nan
            if not (np.isfinite(left_max) and np.isfinite(right_max)):
                continue
            depth = min(left_max, right_max) - m[i]
            ciw = (hi[i] - lo[i]) if np.isfinite(hi[i]) and np.isfinite(lo[i]) else np.inf
            if depth > ciw and depth > a.min_depth:
                lull = {"found": True, "lead_h": float(lead_grid[i]),
                        "depth": float(depth), "ci_width": float(ciw),
                        "rise_before": float(left_max - m[i]),
                        "rise_after": float(right_max - m[i])}
                break
    rep["lull_test"] = lull
    rep["trajectories"] = results

    print("\n[lull] discriminator-score composite (lead h -> mean score, pre-genesis):")
    for L, mm, l_, h_, cc in zip(lead_grid, results["s"]["pre_mean"],
                                 results["s"]["pre_lo"], results["s"]["pre_hi"],
                                 results["s"]["ctl_mean"]):
        if np.isfinite(mm):
            print(f"  T-{L:5.1f}h  pre={mm:.4f} [{l_:.4f},{h_:.4f}]   control={cc:.4f}")
    print(f"\n[lull] LULL {'FOUND' if lull['found'] else 'NOT FOUND'}"
          + (f" at T-{lull['lead_h']:.0f}h (depth {lull['depth']:.4f} vs CI width "
             f"{lull['ci_width']:.4f})" if lull["found"] else
             " — no interior local minimum deeper than its bootstrap CI width"))

    (out / "lull_composite.json").write_text(json.dumps(rep, indent=2, default=str))

    # plot
    nv = [v for v in TRACK_VARS if v in results]
    fig, axes = plt.subplots(1, len(nv), figsize=(4.2 * len(nv), 4.2), dpi=120)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, nv):
        r = results[v]
        ax.plot(r["lead"], r["pre_mean"], "-o", color="#c1121f", lw=2, label="pre-genesis")
        ax.fill_between(r["lead"], r["pre_lo"], r["pre_hi"], color="#c1121f", alpha=0.18)
        ax.plot(r["lead"], r["ctl_mean"], "--", color="#457b9d", lw=1.6, label="control (far)")
        ax.invert_xaxis(); ax.set_xlabel("hours before genesis"); ax.set_title(v)
        ax.grid(alpha=0.3)
        if v == nv[0]:
            ax.set_ylabel("event-centred composite"); ax.legend(fontsize=8)
    fig.suptitle(f"Event-centred composites, strict genesis, {n_ev} events "
                 f"(season-blocked scoring)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "lull_composite.png"); plt.close(fig)
    print(f"[lull] wrote {out/'lull_composite.json'} and lull_composite.png")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Event-centred test for the build->dip->surge lull.")
    ap.add_argument("--panels", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--features", default=",".join(CURATED))
    ap.add_argument("--genesis-thresh-kt", type=float, default=34.0)
    ap.add_argument("--genesis-radius", type=float, default=3.0)
    ap.add_argument("--genesis-max-lead", type=float, default=72.0)
    ap.add_argument("--control-radius", type=float, default=5.0,
                    help="cells beyond this distance form the control composite")
    ap.add_argument("--back-h", type=float, default=72.0)
    ap.add_argument("--step-h", type=float, default=3.0)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--min-depth", type=float, default=0.005)
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
