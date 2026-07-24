#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gse_stack.py — can stacking GEOMETRY + SHEAR + ENERGY + TIME-PROFILE gates
buy very high reliability at reduced coverage?

Hypothesis under test (GSE integration):
    A cell with the right geometry, favourable shear, sufficient energy AND the
    right temporal profile should be a far more reliable genesis indicator than
    any single factor, even if the number of firing cells drops sharply.

Method (all spiral-conditioned: pregen == 1)
    geometry : curated ridge-logistic score (gka_shear_quench/msl_nd/SII/knee_ratio)
    shear    : shear_low level  (+ optional trailing shear tendency)
    energy   : E_energy (GSE proxy; NOTE no true CAPE exists in this dataset)
    time     : causal trailing accumulation of the geometry score (acc48) and
               the build/sustained shape indicators
    Gates are applied as quantile thresholds; precision/recall is reported for
    each cumulative stack, with LEAVE-ONE-STORM-OUT fitting of the geometry
    model so the reported precision is not in-sample.

Outputs: <out-dir>/gse_stack.{csv,json,png}

USAGE
    python eval_gse_stack.py \\
        --panel data/gse_panel_70m.parquet \\
        --tracks data/tracks/tracks_geomval.parquet \\
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

GEOM_FEATS = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
BASE_COLS = ["time", "lat", "lon", "ilat", "ilon", "S_shear", "E_energy",
             "shear_low", "shear_deep", "pregen", "near_storm", "t_to_storm_min_h"]


def load_storm_crops(args):
    tr = pd.read_parquet(args.tracks) if str(args.tracks).endswith((".parquet", ".pq")) \
        else pd.read_csv(args.tracks, low_memory=False)
    tr["time"] = pd.to_datetime(tr["time"])
    for c in ("lat", "lon"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    cols = list(dict.fromkeys(BASE_COLS + GEOM_FEATS))
    rng = np.random.default_rng(0)
    crops = {}
    for sid, g in tr.groupby(tr["storm_id"].astype(str)):
        g = g.dropna(subset=["lat", "lon", "time"]).sort_values("time")
        if len(g) < 3:
            continue
        name = str(g["name"].iloc[0]) if "name" in g.columns else sid
        t0 = g["time"].min() - pd.Timedelta(hours=args.pre_h)
        t1 = g["time"].max() + pd.Timedelta(hours=6)
        m = pd.read_parquet(args.panel, columns=cols,
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
        print(f"[gse-stack] {key}: rows={len(m):,}")
    if len(crops) < 3:
        raise SystemExit("[gse-stack] need >= 3 storms.")
    return crops


def add_derived(m):
    """Geometry accumulation + shear tendency, all causal (per cell)."""
    m = m.sort_values(["ilat", "ilon", "time"])
    gb = m.groupby(["ilat", "ilon"], sort=False)
    s = gb["geom"]
    m["acc24"] = s.rolling(24, min_periods=6).mean().reset_index(level=[0, 1], drop=True)
    m["acc48"] = s.rolling(48, min_periods=12).mean().reset_index(level=[0, 1], drop=True)
    m["build"] = m["acc24"] - m["acc48"]
    sl = gb["shear_low"]
    m["dshear_1h"] = sl.diff()
    m["shear_up_frac6"] = (gb["dshear_1h"].rolling(6, min_periods=3)
                             .apply(lambda a: (a > 0).mean(), raw=True)
                             .reset_index(level=[0, 1], drop=True))
    return m


def fit_geom(crops, names):
    Xs, ys = [], []
    for n in names:
        sp = crops[n][crops[n]["pregen"] == 1]
        Xs.append(sp[GEOM_FEATS].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        ys.append((sp["near_storm"].to_numpy(float) == 1).astype(float))
    X = np.vstack(Xs); y = np.concatenate(ys)
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return fit_logistic((X - mu) / sd, y, l2=1.0), mu, sd


def main() -> int:
    args = parse_args()
    crops = load_storm_crops(args)
    storms = list(crops.keys())

    # LOSO scoring: each storm's geometry score comes from a model that never saw it
    scored = []
    for s in storms:
        w, mu, sd = fit_geom(crops, [n for n in storms if n != s])
        m = crops[s].copy()
        X = m[GEOM_FEATS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        fin = np.all(np.isfinite(X), axis=1)
        gscore = np.full(len(m), np.nan)
        gscore[fin] = predict(w, (X[fin] - mu) / sd)
        m["geom"] = gscore
        m = add_derived(m)
        m["storm"] = s
        scored.append(m[m["pregen"] == 1])
        print(f"[gse-stack] scored {s} (LOSO)")
    sp = pd.concat(scored, ignore_index=True)
    y = (sp["near_storm"].to_numpy(float) == 1).astype(int)
    print(f"\n[gse-stack] spiral rows={len(sp):,}  tighten={y.sum():,} "
          f"base precision={y.mean():.1%}")

    q = args.gate_quantile
    def thr(col):
        v = pd.to_numeric(sp[col], errors="coerce")
        return float(v.quantile(q))

    gates = {
        "geometry":  (pd.to_numeric(sp["geom"], errors="coerce") >= thr("geom")),
        "shear_low": (pd.to_numeric(sp["shear_low"], errors="coerce") >= thr("shear_low")),
        "energy":    (pd.to_numeric(sp["E_energy"], errors="coerce") >= thr("E_energy")),
        "time_acc":  (pd.to_numeric(sp["acc48"], errors="coerce") >= thr("acc48")),
        "shear_up":  (pd.to_numeric(sp["shear_up_frac6"], errors="coerce") >= 0.5),
    }

    rows = []
    # individual gates
    for name, gmask in gates.items():
        gm = gmask.fillna(False).to_numpy()
        rows.append({"stack": name, "kind": "single", "n_fired": int(gm.sum()),
                     "coverage": float(gm.mean()),
                     "precision": float(y[gm].mean()) if gm.any() else np.nan,
                     "recall": float(y[gm].sum() / max(y.sum(), 1))})
    # cumulative stack in the hypothesis order
    order = ["geometry", "shear_low", "energy", "time_acc", "shear_up"]
    cum = np.ones(len(sp), bool)
    for name in order:
        cum = cum & gates[name].fillna(False).to_numpy()
        rows.append({"stack": " + ".join(order[:order.index(name) + 1]),
                     "kind": "cumulative", "n_fired": int(cum.sum()),
                     "coverage": float(cum.mean()),
                     "precision": float(y[cum].mean()) if cum.any() else np.nan,
                     "recall": float(y[cum].sum() / max(y.sum(), 1))})
    res = pd.DataFrame(rows)

    # precision/coverage sweep on the best combined score (geom x shear_low rank)
    gr = pd.to_numeric(sp["geom"], errors="coerce").rank(pct=True)
    sr = pd.to_numeric(sp["shear_low"], errors="coerce").rank(pct=True)
    ar = pd.to_numeric(sp["acc48"], errors="coerce").rank(pct=True)
    combo = (gr + sr + ar) / 3.0
    sweep = []
    for qq in [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]:
        m = (combo >= combo.quantile(qq)).fillna(False).to_numpy()
        sweep.append({"quantile": qq, "n_fired": int(m.sum()),
                      "coverage": float(m.mean()),
                      "precision": float(y[m].mean()) if m.any() else np.nan,
                      "recall": float(y[m].sum() / max(y.sum(), 1))})
    sweep = pd.DataFrame(sweep)

    print("\n[gse-stack] individual + cumulative gates "
          f"(gate quantile={q}):")
    print(res.to_string(index=False))
    print("\n[gse-stack] combined-rank sweep (geom+shear_low+acc48):")
    print(sweep.to_string(index=False))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "gse_stack.csv", index=False)
    sweep.to_csv(out / "gse_stack_sweep.csv", index=False)
    (out / "gse_stack.json").write_text(json.dumps({
        "base_precision": float(y.mean()), "n_spiral": int(len(sp)),
        "gate_quantile": q, "gates": res.to_dict("records"),
        "sweep": sweep.to_dict("records"), "storms": storms,
        "note": "LOSO geometry scoring; E_energy is a GSE proxy, not true CAPE",
    }, indent=2, default=str))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    cum_rows = res[res.kind == "cumulative"]
    a1.plot(range(len(cum_rows)), cum_rows.precision, "-o", color="#2a9d8f", label="precision")
    a1.plot(range(len(cum_rows)), cum_rows.recall, "-s", color="#e76f51", label="recall")
    a1.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"base rate {y.mean():.2f}")
    a1.set_xticks(range(len(cum_rows)))
    a1.set_xticklabels([f"+{o}" for o in order], rotation=30, ha="right", fontsize=8)
    a1.set_title("Cumulative gate stack"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.plot(sweep.coverage, sweep.precision, "-o", color="#4c72b0")
    a2.axhline(y.mean(), color="grey", ls="--", lw=1)
    a2.set_xscale("log"); a2.set_xlabel("coverage (fraction of spiral cells firing)")
    a2.set_ylabel("precision"); a2.set_title("Precision vs coverage (combined rank)")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "gse_stack.png"); plt.close(fig)
    print(f"\n[gse-stack] wrote {out/'gse_stack.csv'} / _sweep.csv / .json / .png")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Stacked GSE + geometry + time-profile reliability test.")
    ap.add_argument("--panel", required=True, help="gse_panel_70m.parquet (G/S/E + gka + labels).")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--gate-quantile", type=float, default=0.75)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
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
