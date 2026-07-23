#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_accumulation.py — is there a temporal accumulation pattern before genesis?

For the spiral-conditioned discriminator (the curated tighten-vs-fizzle model),
this tracks the genesis-score as a function of lead time (t_to_storm_min_h) for
cells that DO tighten into a storm — pooled and per storm — to test whether the
signal builds over an accumulation window and then drops as the storm arrives,
and whether that shape is consistent across storms.

Two views:
  (1) mean genesis-score vs lead      (the accumulation curve; per-storm overlays)
  (2) AUC(tighten@lead vs fizzle) vs lead   (skill as a function of lead)

USAGE
    python eval_accumulation.py \\
        --gka-grid    data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --pregen-grid data/gse_panel_70m_slim.parquet \\
        --lead-grid   data/grid_slim_start.parquet \\
        --tracks      data/tracks/tracks_subset.parquet \\
        --out-dir     figures/accumulation
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

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
LEAD_EDGES = [0, 3, 6, 9, 12, 18, 24, 36, 48, 72, 96, 120]


def crop_join(args, t0, t1, bbox, feats):
    latS, latN, lonW, lonE = bbox
    F = [("time", ">=", t0), ("time", "<", t1)]
    want = list(dict.fromkeys(["time", "ilat", "ilon", "lat", "lon"] + feats))
    gka = pd.read_parquet(args.gka_grid, columns=want, filters=F)
    if gka.empty:
        return gka
    pg = pd.read_parquet(args.pregen_grid, columns=["time", "ilat", "ilon", "pregen"], filters=F)
    ns = pd.read_parquet(args.lead_grid,
                         columns=["time", "lat", "lon", "near_storm", "t_to_storm_min_h"], filters=F)
    for d in (gka, pg, ns):
        d["time"] = pd.to_datetime(d["time"])
    for d in (gka, ns):
        d["lat"] = d["lat"].round(2); d["lon"] = d["lon"].round(2)
    m = gka.merge(pg, on=["time", "ilat", "ilon"], how="inner") \
           .merge(ns, on=["time", "lat", "lon"], how="inner")
    return m[(m["lat"].between(latS, latN)) & (m["lon"].between(lonW, lonE))]


def collect(args, feats):
    tr = pd.read_parquet(args.tracks)
    tr["time"] = pd.to_datetime(tr["time"])
    for c in ("lat", "lon"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    rng = np.random.default_rng(0)
    per_tighten, fizzle_pool = {}, []
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
        sp = m[m["pregen"] == 1]
        tight = sp[sp["near_storm"] == 1]
        fiz = sp[sp["near_storm"] == 0]
        if len(tight) < args.min_pos:
            continue
        key = name
        i = 1
        while key in per_tighten:
            i += 1; key = f"{name}#{i}"
        cols = feats + ["t_to_storm_min_h"]
        per_tighten[key] = tight[cols].copy()
        if len(fiz) > args.max_fizzle:
            fiz = fiz.iloc[rng.choice(len(fiz), args.max_fizzle, replace=False)]
        fizzle_pool.append(fiz[feats].copy())
        print(f"[accum] {key}: tighten={len(tight):,} fizzle={len(fiz):,}")
    if not per_tighten:
        raise SystemExit("[accum] no storms collected.")
    return per_tighten, pd.concat(fizzle_pool, ignore_index=True)


def fit_curated(per_tighten, fizzle, feats):
    """Fit the curated logistic on all-tighten vs fizzle (standardised)."""
    Xt = np.vstack([d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                    for d in per_tighten.values()])
    Xf = fizzle[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = np.vstack([Xt, Xf]); y = np.concatenate([np.ones(len(Xt)), np.zeros(len(Xf))])
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=1.0)
    return w, mu, sd


def score(df, feats, w, mu, sd):
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fin = np.all(np.isfinite(X), axis=1)
    p = np.full(len(df), np.nan)
    p[fin] = predict(w, (X[fin] - mu) / sd)
    return p


def main():
    args = parse_args()
    feats = CURATED
    per_tighten, fizzle = collect(args, feats)
    w, mu, sd = fit_curated(per_tighten, fizzle, feats)
    fiz_score = score(fizzle, feats, w, mu, sd)
    fiz_score = fiz_score[np.isfinite(fiz_score)]

    centres = [(lo + hi) / 2 for lo, hi in zip(LEAD_EDGES[:-1], LEAD_EDGES[1:])]

    # per-storm accumulation curves (mean score vs lead) + pooled AUC vs lead
    per_curves = {}
    pooled = {c: [] for c in centres}
    for name, df in per_tighten.items():
        s = score(df, feats, w, mu, sd)
        lead = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce").to_numpy(float)
        curve = []
        for lo, hi, c in zip(LEAD_EDGES[:-1], LEAD_EDGES[1:], centres):
            mask = np.isfinite(s) & (lead >= lo) & (lead < hi)
            curve.append(np.nanmean(s[mask]) if mask.sum() >= 30 else np.nan)
            if mask.sum() >= 30:
                pooled[c].append(s[mask])
        per_curves[name] = curve

    # pooled AUC vs lead (tighten@bin vs fizzle background)
    rows = []
    for c in centres:
        pos = np.concatenate(pooled[c]) if pooled[c] else np.array([])
        rows.append({"lead_mid_h": c, "n": int(len(pos)),
                     "mean_score": float(np.nanmean(pos)) if len(pos) else np.nan,
                     "auc_vs_fizzle": auc_roc(pos, fiz_score) if len(pos) else np.nan})
    res = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "accumulation.csv", index=False)
    print("\n[accum] pooled accumulation (lead decreases toward storm):")
    print(res.to_string(index=False))

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for name, curve in per_curves.items():
        ax1.plot(centres, curve, "-", alpha=0.45, lw=1.2, label=name)
    ax1.plot(res["lead_mid_h"], res["mean_score"], "-o", color="black", lw=2.6, label="POOLED")
    ax1.invert_xaxis()
    ax1.set_xlabel("lead time to storm (hours)"); ax1.set_ylabel("mean genesis-score (curated model)")
    ax1.set_title("Accumulation: score vs lead (per storm)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, ncol=2)

    ax2.plot(res["lead_mid_h"], res["auc_vs_fizzle"], "-o", color="#e8734c", lw=2)
    ax2.axhline(0.5, color="grey", ls="--", lw=1)
    ax2.invert_xaxis()
    ax2.set_xlabel("lead time to storm (hours)"); ax2.set_ylabel("AUC (tighten@lead vs fizzle)")
    ax2.set_title("Skill vs lead (pooled)")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "accumulation.png"); plt.close(fig)
    print(f"[accum] wrote {out/'accumulation.png'}")

    (out / "accumulation.json").write_text(json.dumps({
        "curated_features": feats,
        "coefficients": dict(zip(["bias"] + feats, [float(x) for x in w])),
        "pooled": res.to_dict(orient="records"),
        "per_storm_curves": {k: [None if (v is None or not np.isfinite(v)) else float(v) for v in c]
                             for k, c in per_curves.items()},
        "lead_centres_h": centres,
    }, indent=2, default=str))
    print(f"[accum] wrote {out/'accumulation.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Temporal accumulation of the genesis discriminator vs lead time.")
    ap.add_argument("--gka-grid", required=True)
    ap.add_argument("--pregen-grid", required=True)
    ap.add_argument("--lead-grid", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--min-pos", type=int, default=500)
    ap.add_argument("--max-fizzle", type=int, default=40000)
    ap.add_argument("--out-dir", default="figures/accumulation")
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
