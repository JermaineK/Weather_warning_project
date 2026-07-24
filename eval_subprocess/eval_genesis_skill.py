#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_genesis_skill.py — pipeline-facing LOSO validation of the genesis trigger.

Single-file version of the geom-val battery (see eval_early_warning_loso.py):
for each storm in the tracks file, crops the labelled GKA grid around that
storm (predicate pushdown on time, bbox in-memory), fits the curated ridge
logistic on the OTHER storms' spiral cells, scores the held-out storm
(instantaneous + trailing acc24/acc48), and reports:

  * per-storm held-out tighten-vs-fizzle AUC (the discriminator check)
  * per-lead AUC for instantaneous vs accumulation (the trigger check)
  * storm-block bootstrap 95% CIs

Outputs: <out-dir>/<run-name>_genesis_skill.{json,csv,png}

USAGE (via eval manager)
    python eval_manager.py genesis-skill \\
        --labelled data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --tracks data/tracks/tracks_subset.parquet \\
        --run-name geomval_demo --out-dir results/metrics
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
sys.path.insert(0, str(HERE.parent))  # repo root

from eval_spiral_genesis import auc_roc, fit_logistic, predict  # noqa: E402

DEFAULT_FEATURES = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
WINDOWS = [24, 48]
FEAT_OUT = ["s", "A24", "A48"]


def load_storm_crops(args, feats):
    tr = pd.read_parquet(args.tracks) if str(args.tracks).endswith((".parquet", ".pq")) \
        else pd.read_csv(args.tracks, low_memory=False)
    tr["time"] = pd.to_datetime(tr["time"])
    for c in ("lat", "lon"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    cols = list(dict.fromkeys(
        ["time", "lat", "lon", "ilat", "ilon", "pregen", "near_storm",
         "t_to_storm_min_h"] + feats))
    rng = np.random.default_rng(0)
    crops = {}
    for sid, g in tr.groupby(tr["storm_id"].astype(str)):
        g = g.dropna(subset=["lat", "lon", "time"]).sort_values("time")
        if len(g) < 3:
            continue
        name = str(g["name"].iloc[0]) if "name" in g.columns else sid
        t0 = g["time"].min() - pd.Timedelta(hours=args.pre_h)
        t1 = g["time"].max() + pd.Timedelta(hours=6)
        m = pd.read_parquet(args.labelled, columns=cols,
                            filters=[("time", ">=", t0), ("time", "<", t1)])
        if m.empty:
            continue
        pad = args.pad_deg
        m = m[(m["lat"].between(g["lat"].min() - pad, g["lat"].max() + pad))
              & (m["lon"].between(g["lon"].min() - pad, g["lon"].max() + pad))]
        if m.empty:
            continue
        cells = m[["ilat", "ilon"]].drop_duplicates()
        if len(cells) > args.max_cells:
            cells = cells.iloc[rng.choice(len(cells), args.max_cells, replace=False)]
            m = m.merge(cells, on=["ilat", "ilon"], how="inner")
        key = name
        i = 1
        while key in crops:
            i += 1
            key = f"{name}#{i}"
        crops[key] = m
        print(f"[genesis-skill] {key}: rows={len(m):,}")
    if len(crops) < 3:
        raise SystemExit(f"[genesis-skill] need >= 3 storms with data, got {len(crops)}.")
    return crops


def fit_on(crops, names, feats, l2):
    Xs, ys = [], []
    for n in names:
        sp = crops[n][crops[n]["pregen"] == 1]
        X = sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        y = (sp["near_storm"].to_numpy(float) == 1).astype(float)
        Xs.append(X); ys.append(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=l2)
    return w, mu, sd


def score_crop(m, feats, w, mu, sd):
    X = m[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fin = np.all(np.isfinite(X), axis=1)
    s = np.full(len(m), np.nan)
    s[fin] = predict(w, (X[fin] - mu) / sd)
    m = m.assign(s=s).sort_values(["ilat", "ilon", "time"])
    gb = m.groupby(["ilat", "ilon"], sort=False)["s"]
    for W in WINDOWS:
        m[f"A{W}"] = (gb.rolling(W, min_periods=max(3, W // 4)).mean()
                        .reset_index(level=[0, 1], drop=True))
    return m


def main() -> int:
    args = parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    leads = [int(x) for x in args.leads.split(",")]
    crops = load_storm_crops(args, feats)
    storms = list(crops.keys())
    rng = np.random.default_rng(1)

    overall, per_lead = {}, {s: {} for s in storms}
    for s in storms:
        w, mu, sd = fit_on(crops, [n for n in storms if n != s], feats, args.l2)
        m = score_crop(crops[s].copy(), feats, w, mu, sd)
        sp = m[m["pregen"] == 1]
        tight, fiz = sp[sp["near_storm"] == 1], sp[sp["near_storm"] == 0]
        if len(fiz) > args.max_neg:
            fiz = fiz.iloc[rng.choice(len(fiz), args.max_neg, replace=False)]
        overall[s] = auc_roc(tight["s"].to_numpy(float), fiz["s"].to_numpy(float)) \
            if len(tight) and len(fiz) else float("nan")
        tt = pd.to_numeric(tight["t_to_storm_min_h"], errors="coerce").to_numpy(float)
        for L in leads:
            mask = (tt >= L - args.lead_tol) & (tt <= L + args.lead_tol)
            if mask.sum() < args.min_bin or not len(fiz):
                continue
            per_lead[s][L] = {c: auc_roc(tight[c].to_numpy(float)[mask],
                                         fiz[c].to_numpy(float))
                              for c in FEAT_OUT}
        print(f"[genesis-skill] holdout {s:14s} overall AUC={overall[s]:.3f} "
              f"leads={sorted(per_lead[s])}")

    # aggregate with storm-block bootstrap
    rows = []
    for L in leads:
        for c in FEAT_OUT:
            vals = np.array([per_lead[s][L][c] for s in storms if L in per_lead[s]], float)
            if not len(vals):
                continue
            point = float(np.nanmean(vals))
            if len(vals) >= 2:
                bm = [np.nanmean(vals[rng.integers(0, len(vals), size=len(vals))])
                      for _ in range(args.n_boot)]
                lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
            else:
                lo = hi = float("nan")
            rows.append({"lead_h": L, "feature": c, "auc_mean": point,
                         "ci_lo": lo, "ci_hi": hi, "n_storms": int(len(vals))})
    res = pd.DataFrame(rows)

    ov = np.array([v for v in overall.values() if np.isfinite(v)])
    summary = {
        "features": feats,
        "n_storms": len(storms),
        "overall_loso_auc_mean": float(np.nanmean(ov)) if len(ov) else float("nan"),
        "overall_loso_auc_by_storm": {k: (None if not np.isfinite(v) else float(v))
                                      for k, v in overall.items()},
        "per_lead": res.to_dict(orient="records"),
    }

    if args.out:  # canonical output path (pipeline contract); csv/png are siblings
        base = str(Path(args.out).with_suffix(""))
        Path(base).parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"{args.run_name}_genesis_skill"
    Path(f"{base}.json").write_text(json.dumps(summary, indent=2, default=str))
    res.to_csv(f"{base}.csv", index=False)
    print(f"[genesis-skill] overall LOSO AUC mean={summary['overall_loso_auc_mean']:.3f} "
          f"over {len(storms)} storms")
    print(f"[genesis-skill] wrote {base}.json / .csv")

    if not res.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        colors = {"s": "#4c72b0", "A24": "#dd8452", "A48": "#55a868"}
        labels = {"s": "instantaneous", "A24": "trailing 24h", "A48": "trailing 48h"}
        for c in FEAT_OUT:
            sub = res[res.feature == c].sort_values("lead_h")
            if sub.empty:
                continue
            ax.plot(sub.lead_h, sub.auc_mean, "-o", color=colors[c], label=labels[c], lw=2)
            ax.fill_between(sub.lead_h, sub.ci_lo, sub.ci_hi, color=colors[c], alpha=0.18)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("warning lead (hours before storm)")
        ax.set_ylabel("held-out AUC vs fizzle (LOSO)")
        ax.set_title(f"{args.run_name}: genesis trigger skill (per-storm LOSO, 95% CI)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{base}.png"); plt.close(fig)
        print(f"[genesis-skill] wrote {base}.png")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="LOSO validation of the spiral-conditioned genesis trigger.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--out", default=None,
                    help="Canonical JSON output path (overrides run-name/out-dir naming).")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--leads", default="6,12,24,36,48")
    ap.add_argument("--lead-tol", type=int, default=3)
    ap.add_argument("--min-bin", type=int, default=30)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--max-neg", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--l2", type=float, default=1.0)
    # orchestrator compatibility (ignored)
    ap.add_argument("--chunk-rows", type=int, default=0)
    ap.add_argument("--parquet-rows", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
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
