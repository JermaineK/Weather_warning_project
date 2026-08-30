#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_precursor_backtrace.py — do the ULTRA-RELIABLE cells look special 48-72h EARLIER?

The stacked-gate analysis showed >99% precision is reachable, but ~100% of those
hits sit at lead < 6h: it is nowcasting, not early warning. This script asks the
follow-up that actually matters:

    Take the cells that eventually become near-certain hits. Rewind each one to
    48-72h before that moment (the pre-storm accumulation peak found earlier).
    At THAT time, were they already separable from background?

If yes, the separating attributes are a genuine early-warning signature -- they
identify, two to three days ahead, the cells destined to become certain.

Method
    1. certain cells := top --certain-quantile of the combined rank
       (geometry score x shear_low) among spiral cells; record (cell, t_certain)
       as the FIRST time each cell qualifies.
    2. precursor sample := the same cell at t_certain - [back_lo, back_hi] hours.
    3. background := spiral cells that NEVER become certain, sampled at the same
       timestamps (so climatology/diurnal effects cancel).
    4. Report per-attribute AUC (precursor vs background) at that lookback, plus
       a multivariate LOSO-free logistic for a combined precursor score.

USAGE
    python eval_precursor_backtrace.py \\
        --panel data/gse_panel_70m.parquet \\
        --start 2025-02-12 --end 2025-03-02 \\
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
from eval_spiral_genesis import auc_roc, fit_logistic, predict  # noqa: E402

ATTRS = ["shear_low", "shear_deep", "S_shear", "E_energy", "G_struct",
         "gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio",
         "gka_SAI", "zeta", "msl_d3h", "SFI", "sph_vdr_std"]
LOAD_COLS = ["time", "ilat", "ilon", "lat", "lon", "pregen", "near_storm",
             "t_to_storm_min_h"] + ATTRS


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.panel, columns=list(dict.fromkeys(LOAD_COLS)),
                         filters=[("time", ">=", pd.Timestamp(args.start)),
                                  ("time", "<", pd.Timestamp(args.end))])
    df["time"] = pd.to_datetime(df["time"])
    sp = df[df["pregen"] == 1].copy()
    print(f"[backtrace] window rows={len(df):,}  spiral={len(sp):,}")

    # ---- 1. define "certain" cells (nowcast-reliable regime) ----
    gq = pd.to_numeric(sp["gka_shear_quench"], errors="coerce").rank(pct=True)
    geom_rank = 1.0 - gq                      # low shear_quench favours tightening
    shear_rank = pd.to_numeric(sp["shear_low"], errors="coerce").rank(pct=True)
    sp["combo"] = (geom_rank + shear_rank) / 2.0
    cut = sp["combo"].quantile(args.certain_quantile)
    sp["is_certain"] = (sp["combo"] >= cut).fillna(False)
    cert = sp[sp["is_certain"]]
    prec = float((cert["near_storm"] == 1).mean()) if len(cert) else float("nan")
    print(f"[backtrace] certain cells: {len(cert):,} "
          f"(top {1-args.certain_quantile:.1%}), precision={prec:.1%}")

    # first time each cell becomes certain
    first = (cert.groupby(["ilat", "ilon"], as_index=False)["time"].min()
                 .rename(columns={"time": "t_certain"}))
    print(f"[backtrace] distinct certain cells: {len(first):,}")

    certain_cells = set(map(tuple, first[["ilat", "ilon"]].to_numpy()))
    all_cells = sp[["ilat", "ilon"]].drop_duplicates()
    bg_cells = [c for c in map(tuple, all_cells.to_numpy()) if c not in certain_cells]
    print(f"[backtrace] background cells (never certain): {len(bg_cells):,}")

    # ---- 2. precursor sample: same cell, back_lo..back_hi hours earlier ----
    sp_idx = sp.set_index(["ilat", "ilon", "time"]).sort_index()
    pre_rows = []
    for ilat, ilon, tc in first.itertuples(index=False):
        lo = tc - pd.Timedelta(hours=args.back_hi)
        hi = tc - pd.Timedelta(hours=args.back_lo)
        try:
            sub = sp_idx.loc[(ilat, ilon)]
        except KeyError:
            continue
        sel = sub[(sub.index >= lo) & (sub.index <= hi)]
        if len(sel):
            r = sel.reset_index().assign(ilat=ilat, ilon=ilon, t_certain=tc)
            pre_rows.append(r)
    if not pre_rows:
        raise SystemExit("[backtrace] no precursor rows found; widen --start or the lookback.")
    pre = pd.concat(pre_rows, ignore_index=True)
    # exclude any precursor row that is ALREADY certain (keep it a true precursor)
    if "combo" in pre.columns:
        pre = pre[pre["combo"] < cut]
    print(f"[backtrace] precursor rows ({args.back_lo}-{args.back_hi}h before certainty): {len(pre):,}")

    # ---- 3. background at the SAME timestamps ----
    times_needed = pd.Index(pre["time"].unique())
    bg_set = set(bg_cells)
    bg = sp[sp["time"].isin(times_needed)].copy()
    key = list(map(tuple, bg[["ilat", "ilon"]].to_numpy()))
    bg = bg[[k in bg_set for k in key]]
    rng = np.random.default_rng(0)
    if len(bg) > args.max_bg:
        bg = bg.iloc[rng.choice(len(bg), args.max_bg, replace=False)]
    print(f"[backtrace] background rows at matched timestamps: {len(bg):,}")
    if len(bg) < 100:
        raise SystemExit("[backtrace] too few background rows for a comparison.")

    # ---- 4. per-attribute separation at the lookback ----
    rows = []
    for c in ATTRS:
        a = pd.to_numeric(pre[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(bg[c], errors="coerce").to_numpy(float)
        auc = auc_roc(a, b)
        rows.append({"attribute": c, "auc": auc,
                     "auc_orientation_free": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                     "precursor_median": float(np.nanmedian(a)),
                     "background_median": float(np.nanmedian(b))})
    res = pd.DataFrame(rows).sort_values("auc_orientation_free", ascending=False)

    # combined precursor score (simple logistic, in-sample -> upper bound)
    X = np.vstack([pre[ATTRS].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                   bg[ATTRS].apply(pd.to_numeric, errors="coerce").to_numpy(float)])
    y = np.concatenate([np.ones(len(pre)), np.zeros(len(bg))])
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=1.0)
    p = predict(w, (X - mu) / sd)
    auc_multi = auc_roc(p[y == 1], p[y == 0])

    print(f"\n[backtrace] separation of future-certain cells from background, "
          f"{args.back_lo}-{args.back_hi}h BEFORE they become certain:")
    print(res.to_string(index=False))
    print(f"\n[backtrace] combined logistic (in-sample upper bound) AUC={auc_multi:.3f}")
    print("[backtrace] coefficients: " +
          ", ".join(f"{a}={c:+.3f}" for a, c in zip(ATTRS, w[1:])))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "precursor_backtrace.csv", index=False)
    (out / "precursor_backtrace.json").write_text(json.dumps({
        "certain_quantile": args.certain_quantile,
        "certain_precision": prec,
        "n_certain_cells": int(len(first)),
        "lookback_h": [args.back_lo, args.back_hi],
        "n_precursor_rows": int(len(pre)), "n_background_rows": int(len(bg)),
        "attributes": res.to_dict("records"),
        "combined_auc_insample": float(auc_multi),
        "coefficients": dict(zip(ATTRS, [float(x) for x in w[1:]])),
    }, indent=2, default=str))

    top = res.head(8)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    ax.barh(top["attribute"][::-1], top["auc_orientation_free"][::-1], color="#4c9be8")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.axvline(auc_multi, color="#e8734c", lw=2, label=f"combined={auc_multi:.3f}")
    ax.set_xlabel("AUC (future-certain vs background), orientation-free")
    ax.set_title(f"Precursor signature {args.back_lo}-{args.back_hi}h before certainty")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "precursor_backtrace.png"); plt.close(fig)
    print(f"[backtrace] wrote {out/'precursor_backtrace.csv'} / .json / .png")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Back-trace ultra-reliable cells to their pre-storm window.")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--certain-quantile", type=float, default=0.999)
    ap.add_argument("--back-lo", type=float, default=48.0)
    ap.add_argument("--back-hi", type=float, default=72.0)
    ap.add_argument("--max-bg", type=int, default=300000)
    ap.add_argument("--out-dir", default="results/metrics")
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
