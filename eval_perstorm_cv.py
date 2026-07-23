#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_perstorm_cv.py — leave-one-storm-out cross-validation of the curated
spiral-genesis discriminator.

Is the ~0.73 held-out AUC real, or is it leaning on one big storm? This assigns
each spiral cell to a storm (by that storm's time window + bounding box), then
for every storm trains the curated logistic on ALL OTHER storms and tests on the
held-out one. Consistent per-storm AUCs => robust; one dominant storm => fragile.

Population (per storm, conditioned on structure)
    spiral  := pregen == 1  within the storm's window/box
    tighten := spiral AND near_storm == 1     [positive]
    fizzle  := spiral AND near_storm == 0     [negative]

USAGE
    python eval_perstorm_cv.py \\
        --gka-grid    data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --pregen-grid data/gse_panel_70m_slim.parquet \\
        --lead-grid   data/grid_slim_start.parquet \\
        --tracks      data/tracks/tracks_subset.parquet \\
        --out-dir     figures/perstorm_cv
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

# reuse the numpy-only metric + model from the sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_spiral_genesis import auc_roc, fit_logistic, predict

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]


def crop_join(args, t0, t1, bbox, feats):
    latS, latN, lonW, lonE = bbox
    F = [("time", ">=", t0), ("time", "<", t1)]
    want = list(dict.fromkeys(["time", "ilat", "ilon", "lat", "lon"] + feats))
    gka = pd.read_parquet(args.gka_grid, columns=want, filters=F)
    if gka.empty:
        return gka
    pg = pd.read_parquet(args.pregen_grid, columns=["time", "ilat", "ilon", "pregen"], filters=F)
    ns = pd.read_parquet(args.lead_grid, columns=["time", "lat", "lon", "near_storm"], filters=F)
    for d in (gka, pg, ns):
        d["time"] = pd.to_datetime(d["time"])
    for d in (gka, ns):
        d["lat"] = d["lat"].round(2); d["lon"] = d["lon"].round(2)
    m = gka.merge(pg, on=["time", "ilat", "ilon"], how="inner") \
           .merge(ns, on=["time", "lat", "lon"], how="inner")
    m = m[(m["lat"].between(latS, latN)) & (m["lon"].between(lonW, lonE))]
    return m


def collect_per_storm(args, feats):
    tr = pd.read_parquet(args.tracks)
    tr["time"] = pd.to_datetime(tr["time"])
    for c in ("lat", "lon"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    rng = np.random.default_rng(0)
    per = {}
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
        pos = sp[sp["near_storm"] == 1]
        neg = sp[sp["near_storm"] == 0]
        if len(pos) < args.min_pos or len(neg) < args.min_pos:
            print(f"[perstorm] {name}: too few (pos={len(pos)}, neg={len(neg)}) — skip", file=sys.stderr)
            continue

        def sub(df):
            if len(df) > args.max_per_storm:
                df = df.iloc[rng.choice(len(df), args.max_per_storm, replace=False)]
            return df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)

        Xp, Xn = sub(pos), sub(neg)
        key = f"{name}"
        # disambiguate duplicate names (e.g. two UNNAMED)
        i = 1
        while key in per:
            i += 1; key = f"{name}#{i}"
        per[key] = {"Xp": Xp, "Xn": Xn}
        print(f"[perstorm] {key}: pos={len(Xp):,} neg={len(Xn):,}")
    return per


def stack(store, names, feats):
    Xp = np.vstack([store[n]["Xp"] for n in names]) if names else np.empty((0, len(feats)))
    Xn = np.vstack([store[n]["Xn"] for n in names]) if names else np.empty((0, len(feats)))
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    return X, y


def main():
    args = parse_args()
    feats = CURATED
    per = collect_per_storm(args, feats)
    storms = list(per.keys())
    if len(storms) < 3:
        raise SystemExit("[perstorm] need >= 3 storms with enough data.")

    rows = []
    for held in storms:
        train_names = [s for s in storms if s != held]
        Xtr, ytr = stack(per, train_names, feats)
        Xte, yte = stack(per, [held], feats)
        fin_tr = np.all(np.isfinite(Xtr), axis=1)
        fin_te = np.all(np.isfinite(Xte), axis=1)
        Xtr, ytr = Xtr[fin_tr], ytr[fin_tr]
        Xte, yte = Xte[fin_te], yte[fin_te]
        if yte.min() == yte.max():
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = fit_logistic((Xtr - mu) / sd, ytr, l2=args.l2)
        p = predict(w, (Xte - mu) / sd)
        a = auc_roc(p[yte == 1], p[yte == 0])
        # best single feature on held storm, orientation-free
        singles = {}
        for j, f in enumerate(feats):
            av = auc_roc(Xte[yte == 1, j], Xte[yte == 0, j])
            singles[f] = max(av, 1 - av)
        rows.append({"storm": held, "n_pos": int((yte == 1).sum()),
                     "n_neg": int((yte == 0).sum()), "loso_auc": a,
                     "best_single": max(singles, key=singles.get),
                     "best_single_auc": max(singles.values())})
        print(f"[perstorm] holdout {held:14s} LOSO-AUC={a:.4f}  "
              f"(best single {max(singles,key=singles.get)}={max(singles.values()):.3f})")

    res = pd.DataFrame(rows).sort_values("loso_auc", ascending=False)
    aucs = res["loso_auc"].to_numpy()
    summary = {
        "features": feats,
        "n_storms": len(res),
        "loso_auc_mean": float(np.nanmean(aucs)),
        "loso_auc_std": float(np.nanstd(aucs)),
        "loso_auc_min": float(np.nanmin(aucs)),
        "loso_auc_max": float(np.nanmax(aucs)),
        "per_storm": res.to_dict(orient="records"),
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "perstorm_cv.csv", index=False)
    (out / "perstorm_cv.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[perstorm] LOSO-AUC mean={summary['loso_auc_mean']:.4f} "
          f"std={summary['loso_auc_std']:.4f} "
          f"range=[{summary['loso_auc_min']:.3f},{summary['loso_auc_max']:.3f}]  n={len(res)}")
    print(res.to_string(index=False))

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    o = res.sort_values("loso_auc")
    ax.barh(o["storm"], o["loso_auc"], color="#4c9be8")
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="no skill")
    ax.axvline(summary["loso_auc_mean"], color="#e8734c", lw=2,
               label=f"mean={summary['loso_auc_mean']:.3f}")
    ax.set_xlabel("leave-one-storm-out test AUC (curated model)")
    ax.set_title("Per-storm robustness (tighten vs fizzle, spiral-conditioned)")
    ax.legend(fontsize=9); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(out / "perstorm_cv.png"); plt.close(fig)
    print(f"[perstorm] wrote {out/'perstorm_cv.png'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Leave-one-storm-out CV of the curated spiral-genesis model.")
    ap.add_argument("--gka-grid", required=True)
    ap.add_argument("--pregen-grid", required=True)
    ap.add_argument("--lead-grid", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pre-h", type=float, default=72.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--min-pos", type=int, default=500)
    ap.add_argument("--max-per-storm", type=int, default=40000)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out-dir", default="figures/perstorm_cv")
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
