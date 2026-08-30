#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_early_warning.py — does a TRAILING ACCUMULATION of the genesis-score beat
the instantaneous score for early warning?

Motivation
    The accumulation analysis showed the discriminator builds ~42 h before a
    storm. So at a decision time t_d that is L hours before arrival, the trailing
    window [t_d-W, t_d] sits right in that build. This tests whether a causal
    (past-only) rolling-mean of the score is a better early-warning trigger than
    the instantaneous score, at fixed warning leads L.

Design (all causal — no future information used at the decision time)
    s(t)          instantaneous curated genesis-score for every cell-hour
    A_W(t)        trailing mean of s over the previous W hours (per cell)
    decision @ L  cells where a real storm arrives in ~L hours
                  (near_storm==1, pregen==1, t_to_storm_min_h in [L-d, L+d])
    negatives     fizzle spirals (pregen==1, near_storm==0)
    metric        AUC(feature; positive@L vs fizzle) for feature in {s, A24, A48}

If accumulation wins, a threshold on the best feature is swept to report an
operating point (detection rate at a fixed false-alarm rate).

USAGE
    python eval_early_warning.py \\
        --gka-grid    data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --pregen-grid data/gse_panel_70m_slim.parquet \\
        --lead-grid   data/grid_slim_start.parquet \\
        --tracks      data/tracks/tracks_subset.parquet \\
        --out-dir     figures/early_warning
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_spiral_genesis import auc_roc, fit_logistic, predict
from eval_accumulation import crop_join

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
LEADS = [6, 12, 24, 36, 48]     # warning leads to test (hours)
WINDOWS = [24, 48]              # trailing accumulation windows (hours)
LEAD_TOL = 3                   # +/- hours around a lead to call a decision cell


def load_storm_crops(args, feats):
    """Per storm: crop the joined grid, keep continuous per-cell hourly series
    (subsample CELLS, not times, to bound memory while preserving rolling)."""
    tr = pd.read_parquet(args.tracks)
    tr["time"] = pd.to_datetime(tr["time"])
    for c in ("lat", "lon"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    rng = np.random.default_rng(0)
    crops = {}
    for sid, g in tr.groupby(tr["storm_id"].astype(str)):
        g = g.dropna(subset=["lat", "lon", "time"]).sort_values("time")
        if len(g) < 3:
            continue
        name = str(g["name"].iloc[0]) if "name" in g.columns else sid
        t0 = g["time"].min() - pd.Timedelta(hours=args.pre_h)
        t1 = g["time"].max() + pd.Timedelta(hours=6)
        pad = args.pad_deg
        bbox = (g["lat"].min() - pad, g["lat"].max() + pad,
                g["lon"].min() - pad, g["lon"].max() + pad)
        m = crop_join(args, t0, t1, bbox, feats)
        if m.empty:
            continue
        # cap number of distinct cells to bound memory (keep full time series each)
        cells = m[["ilat", "ilon"]].drop_duplicates()
        if len(cells) > args.max_cells:
            cells = cells.iloc[rng.choice(len(cells), args.max_cells, replace=False)]
            m = m.merge(cells, on=["ilat", "ilon"], how="inner")
        key = name
        i = 1
        while key in crops:
            i += 1; key = f"{name}#{i}"
        crops[key] = m
        print(f"[ew] {key}: rows={len(m):,} cells={m[['ilat','ilon']].drop_duplicates().shape[0]:,}")
    if not crops:
        raise SystemExit("[ew] no storm crops.")
    return crops


def fit_curated(crops, feats):
    Xt, Xf = [], []
    for m in crops.values():
        sp = m[m["pregen"] == 1]
        Xt.append(sp[sp["near_storm"] == 1][feats].to_numpy(float))
        Xf.append(sp[sp["near_storm"] == 0][feats].to_numpy(float))
    Xt = np.vstack(Xt); Xf = np.vstack(Xf)
    X = np.vstack([Xt, Xf]); y = np.concatenate([np.ones(len(Xt)), np.zeros(len(Xf))])
    fin = np.all(np.isfinite(X), axis=1); X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=1.0)
    return w, mu, sd


def add_scores(m, feats, w, mu, sd):
    X = m[feats].to_numpy(float)
    fin = np.all(np.isfinite(X), axis=1)
    s = np.full(len(m), np.nan)
    s[fin] = predict(w, (X[fin] - mu) / sd)
    m = m.assign(s=s).sort_values(["ilat", "ilon", "time"])
    gb = m.groupby(["ilat", "ilon"], sort=False)["s"]
    for W in WINDOWS:
        # trailing (past+current) mean over W hourly steps; causal
        m[f"A{W}"] = gb.rolling(W, min_periods=max(3, W // 4)).mean().reset_index(level=[0, 1], drop=True)
    return m


def main():
    args = parse_args()
    feats = CURATED
    crops = load_storm_crops(args, feats)
    w, mu, sd = fit_curated(crops, feats)
    print(f"[ew] curated coefficients: " +
          ", ".join(f"{f}={c:+.3f}" for f, c in zip(feats, w[1:])))

    feat_cols = ["s"] + [f"A{W}" for W in WINDOWS]
    pos = {L: {c: [] for c in feat_cols} for L in LEADS}
    neg = {c: [] for c in feat_cols}
    rng = np.random.default_rng(1)
    for name, m in crops.items():
        m = add_scores(m, feats, w, mu, sd)
        sp = m[m["pregen"] == 1]
        # negatives: fizzle spirals (lead-independent), subsample per storm
        fiz = sp[sp["near_storm"] == 0]
        if len(fiz) > args.max_neg:
            fiz = fiz.iloc[rng.choice(len(fiz), args.max_neg, replace=False)]
        for c in feat_cols:
            neg[c].append(fiz[c].to_numpy(float))
        # positives per lead: real-storm cells with t_to_storm ~ L
        tight = sp[sp["near_storm"] == 1]
        tt = pd.to_numeric(tight["t_to_storm_min_h"], errors="coerce").to_numpy(float)
        for L in LEADS:
            mask = (tt >= L - LEAD_TOL) & (tt <= L + LEAD_TOL)
            for c in feat_cols:
                pos[L][c].append(tight[c].to_numpy(float)[mask])

    negv = {c: np.concatenate(neg[c]) for c in feat_cols}
    rows = []
    for L in LEADS:
        row = {"lead_h": L, "n_pos": int(sum(len(a) for a in pos[L]["s"]))}
        for c in feat_cols:
            p = np.concatenate(pos[L][c]) if pos[L][c] else np.array([])
            row[f"auc_{c}"] = auc_roc(p[np.isfinite(p)], negv[c][np.isfinite(negv[c])]) \
                if len(p) else float("nan")
        rows.append(row)
    res = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "early_warning.csv", index=False)
    print("\n[ew] AUC by warning lead — instantaneous (s) vs trailing accumulation (A24/A48):")
    print(res.to_string(index=False))

    # operating point: at each lead, pick the best feature; sweep threshold on
    # pooled pos@L vs fizzle to report detection rate at a fixed false-alarm rate.
    fa_target = args.false_alarm
    trig = []
    for L in LEADS:
        best_c = max(feat_cols, key=lambda c: (res.loc[res.lead_h == L, f"auc_{c}"].iloc[0]
                                               if np.isfinite(res.loc[res.lead_h == L, f"auc_{c}"].iloc[0]) else 0))
        p = np.concatenate(pos[L][best_c]); p = p[np.isfinite(p)]
        n = negv[best_c][np.isfinite(negv[best_c])]
        thr = np.quantile(n, 1 - fa_target)      # threshold giving fa_target false alarms
        det = float(np.mean(p >= thr)) if len(p) else float("nan")
        trig.append({"lead_h": L, "best_feature": best_c, "threshold": float(thr),
                     "detection_rate": det, "false_alarm_rate": fa_target})
    trig = pd.DataFrame(trig)
    print(f"\n[ew] operating point at false-alarm={fa_target:.0%} (detection rate = fraction of "
          f"storms-in-{{lead}}h flagged):")
    print(trig.to_string(index=False))

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    for c, lbl, mk in [("s", "instantaneous", "-o"), ("A24", "trailing 24h", "-s"),
                       ("A48", "trailing 48h", "-^")]:
        ax1.plot(res["lead_h"], res[f"auc_{c}"], mk, label=lbl, lw=2)
    ax1.axhline(0.5, color="grey", ls="--", lw=1)
    ax1.invert_xaxis()
    ax1.set_xlabel("warning lead (hours before storm)"); ax1.set_ylabel("AUC vs fizzle")
    ax1.set_title("Early-warning skill: accumulation vs instantaneous")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(trig["lead_h"], trig["detection_rate"], "-o", color="#2a9d8f", lw=2)
    ax2.invert_xaxis(); ax2.set_ylim(0, 1)
    ax2.set_xlabel("warning lead (hours)"); ax2.set_ylabel(f"detection rate @ {fa_target:.0%} false alarm")
    ax2.set_title("Trigger operating point (best feature per lead)")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "early_warning.png"); plt.close(fig)
    print(f"[ew] wrote {out/'early_warning.png'}")

    (out / "early_warning.json").write_text(json.dumps({
        "curated_features": feats,
        "coefficients": dict(zip(["bias"] + feats, [float(x) for x in w])),
        "auc_by_lead": res.to_dict(orient="records"),
        "operating_point": trig.to_dict(orient="records"),
        "windows_h": WINDOWS, "lead_tol_h": LEAD_TOL,
    }, indent=2, default=str))
    print(f"[ew] wrote {out/'early_warning.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Trailing-accumulation early-warning trigger vs instantaneous score.")
    ap.add_argument("--gka-grid", required=True)
    ap.add_argument("--pregen-grid", required=True)
    ap.add_argument("--lead-grid", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=4000, help="max distinct cells per storm (full series each)")
    ap.add_argument("--max-neg", type=int, default=40000)
    ap.add_argument("--false-alarm", type=float, default=0.10)
    ap.add_argument("--out-dir", default="figures/early_warning")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
