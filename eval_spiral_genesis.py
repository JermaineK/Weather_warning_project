#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_spiral_genesis.py — the FAIR genesis test: within the spiral-like
population, do the geometry signals separate spirals that TIGHTEN into a real
storm from spirals that FIZZLE?

Rationale
    Comparing storm cells to calm ocean is easy and inflates skill. The honest
    question for a geometric-pumping model is whether, *conditioned on spiral
    structure*, the developing systems score in a different band than the
    non-developing ones. Even a modest but consistent band separation is a
    working discriminator.

Population & labels (all conditioned on the structured population)
    spiral   := pregen == 1                       (genesis-favorable / structured)
    tighten  := spiral AND near_storm == 1        (structure that is part of a real,
                                                   IBTrACS-tracked storm)          [positive]
    fizzle   := spiral AND near_storm == 0        (structured spiral that never
                                                   organised into a tracked storm) [negative]

Keys: gse_panel(pregen) joins on (time,ilat,ilon); grid_slim_start(near_storm)
joins on (time,lat,lon). The gka grid carries BOTH key systems and the signals,
so it is the join backbone. Data are streamed in weekly chunks (predicate
pushdown on time) and subsampled per chunk to bound memory.

Outputs (to --out-dir)
    spiral_genesis.csv      per-signal AUC + score-band percentiles per class
    spiral_genesis.json     summary incl. refined multivariate model (train/test)
    spiral_genesis_bands.png violin of the best signal + AUC bars

USAGE
    python eval_spiral_genesis.py \\
        --gka-grid    data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --pregen-grid data/gse_panel_70m_slim.parquet \\
        --lead-grid   data/grid_slim_start.parquet \\
        --out-dir     figures/spiral_genesis
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


# ---------------- metrics / model (numpy only) ----------------

def auc_roc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    tie = np.zeros(len(cnt)); np.add.at(tie, inv, ranks); tie /= cnt
    ranks = tie[inv]
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def fit_logistic(X, y, l2=1.0, iters=60):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        R = np.eye(X.shape[1]) * l2; R[0, 0] = 0.0
        H = X.T @ (X * W[:, None]) + R
        g = X.T @ (y - p) - R @ w
        try:
            w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return w


def predict(w, X):
    X = np.c_[np.ones(len(X)), X]
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


# ---------------- streaming join ----------------

def week_chunks(start, end, days):
    t = pd.Timestamp(start); end = pd.Timestamp(end)
    while t < end:
        yield t, min(t + pd.Timedelta(days=days), end)
        t += pd.Timedelta(days=days)


# base columns needed to construct the theory-driven phi feature on the fly
PHI_BASE = ["gka_relax", "gka_A_overlap", "gka_kappa"]


def load_join(args, t0, t1, signals, extra_cols=()):
    F = [("time", ">=", t0), ("time", "<", t1)]
    want = ["time", "ilat", "ilon", "lat", "lon"] + list(signals) + list(extra_cols)
    want = list(dict.fromkeys(want))  # dedupe, preserve order
    gka = pd.read_parquet(args.gka_grid, columns=want, filters=F)
    if gka.empty:
        return gka
    pg = pd.read_parquet(args.pregen_grid, columns=["time", "ilat", "ilon", "pregen"], filters=F)
    ns = pd.read_parquet(args.lead_grid,
                         columns=["time", "lat", "lon", "near_storm", "t_to_storm_min_h"], filters=F)
    for d in (gka, pg, ns):
        d["time"] = pd.to_datetime(d["time"])
    gka["lat"] = gka["lat"].round(2); gka["lon"] = gka["lon"].round(2)
    ns["lat"] = ns["lat"].round(2); ns["lon"] = ns["lon"].round(2)
    m = gka.merge(pg, on=["time", "ilat", "ilon"], how="inner") \
           .merge(ns, on=["time", "lat", "lon"], how="inner")
    return m


def build_phi(m: pd.DataFrame) -> pd.DataFrame:
    """gka_phi = (1/(|relax|+eps)) * agree / (V_zeta+eps), robust-scaled (median/MAD).

    V_zeta = 7-step centered rolling variance of gka_kappa (=zeta) per (ilat,ilon).
    Mirrors compute_gka_features.add_gka_features but built from precomputed
    GKA columns (relax=gka_relax, agree=gka_A_overlap). Scaled per chunk.
    """
    m = m.sort_values(["ilat", "ilon", "time"])
    vz = (m.groupby(["ilat", "ilon"], sort=False)["gka_kappa"]
            .rolling(7, center=True, min_periods=3).var()
            .reset_index(level=[0, 1], drop=True))
    vzv = np.where(np.isfinite(vz.to_numpy(float)), vz.to_numpy(float), 0.0)
    relax = pd.to_numeric(m["gka_relax"], errors="coerce").to_numpy(float)
    agree = pd.to_numeric(m["gka_A_overlap"], errors="coerce").to_numpy(float)
    phi_raw = (1.0 / (np.abs(relax) + 1e-6)) * agree / (vzv + 1e-6)
    med = np.nanmedian(phi_raw)
    mad = np.nanmean(np.abs(phi_raw - med)) + 1e-6
    m["gka_phi"] = (phi_raw - med) / mad
    return m


def collect(args, signals):
    rng = np.random.default_rng(0)
    keep = list(signals) + (["gka_phi"] if args.build_phi else [])
    extra = PHI_BASE if args.build_phi else ()
    frames = []
    for t0, t1 in week_chunks(args.start, args.end, args.chunk_days):
        m = load_join(args, t0, t1, signals, extra_cols=extra)
        if m.empty:
            continue
        if args.build_phi:
            m = build_phi(m)
        spiral = m[m["pregen"] == 1]
        if spiral.empty:
            continue
        tighten = spiral[spiral["near_storm"] == 1]
        fizzle = spiral[spiral["near_storm"] == 0]

        def sub(df, label):
            if len(df) > args.max_per_chunk:
                df = df.iloc[rng.choice(len(df), args.max_per_chunk, replace=False)]
            out = df[keep + ["time"]].copy()
            out["y"] = label
            return out

        frames.append(sub(tighten, 1))
        frames.append(sub(fizzle, 0))
        print(f"[spiral] {t0:%Y-%m-%d}..{t1:%Y-%m-%d}  merged={len(m):,} "
              f"spiral={len(spiral):,} tighten={len(tighten):,} fizzle={len(fizzle):,}")
    if not frames:
        raise SystemExit("[spiral] no data collected.")
    return pd.concat(frames, ignore_index=True)


# ---------------- main ----------------

def main():
    args = parse_args()
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    df = collect(args, signals)
    if args.build_phi and "gka_phi" not in signals:
        signals = signals + ["gka_phi"]
    y = df["y"].to_numpy(float)
    print(f"\n[spiral] pooled tighten={int(y.sum()):,}  fizzle={int((1-y).sum()):,}")

    pct = [10, 25, 50, 75, 90]
    rows = []
    for s in signals:
        v = pd.to_numeric(df[s], errors="coerce").to_numpy(float)
        pos, neg = v[y == 1], v[y == 0]
        row = {"signal": s, "auc_tighten_vs_fizzle": auc_roc(pos, neg)}
        for p in pct:
            row[f"tighten_p{p}"] = float(np.nanpercentile(pos, p))
            row[f"fizzle_p{p}"] = float(np.nanpercentile(neg, p))
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("auc_tighten_vs_fizzle", ascending=False)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "spiral_genesis.csv", index=False)
    print("\n[spiral] per-signal AUC (tighten vs fizzle) and score bands:")
    print(res[["signal", "auc_tighten_vs_fizzle"] +
              [f"fizzle_p50" ] + [f"tighten_p50"]].to_string(index=False))

    # ---- refined multivariate model: train Feb-Mar, test Apr ----
    split = pd.Timestamp(args.train_end)
    tr = df["time"] <= split
    te = df["time"] > split
    Xall = df[signals].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.all(np.isfinite(Xall), axis=1)
    tr &= finite; te &= finite
    refined = {}
    if tr.sum() > 50 and te.sum() > 50 and y[te].min() != y[te].max():
        mu = Xall[tr].mean(0); sd = Xall[tr].std(0) + 1e-9
        Xtr = (Xall[tr] - mu) / sd; Xte = (Xall[te] - mu) / sd
        w = fit_logistic(Xtr, y[tr], l2=args.l2)
        p_te = predict(w, Xte)
        auc_multi = auc_roc(p_te[y[te] == 1], p_te[y[te] == 0])
        # best single signal on the SAME test rows, for a fair comparison
        singles = {s: auc_roc(Xall[te][y[te] == 1, i], Xall[te][y[te] == 0, i])
                   for i, s in enumerate(signals)}
        singles = {s: (a if a >= 0.5 else 1 - a) for s, a in singles.items()}  # orientation-free
        best_single = max(singles, key=singles.get)
        refined = {
            "train_end": args.train_end,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "auc_multivariate_test": auc_multi,
            "best_single_signal": best_single,
            "best_single_auc_test": singles[best_single],
            "coefficients": dict(zip(["bias"] + signals, [float(x) for x in w])),
            "single_aucs_test": {k: float(v) for k, v in singles.items()},
        }
        print(f"\n[spiral] refined logistic  test-AUC={auc_multi:.4f}  "
              f"(best single '{best_single}'={singles[best_single]:.4f})  "
              f"train={int(tr.sum()):,} test={int(te.sum()):,}")
    else:
        print("[spiral] insufficient train/test split for refined model.", file=sys.stderr)

    summary = {
        "n_tighten": int(y.sum()), "n_fizzle": int((1 - y).sum()),
        "per_signal": res.to_dict(orient="records"),
        "refined_model": refined,
    }
    (out_dir / "spiral_genesis.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[spiral] wrote {out_dir/'spiral_genesis.json'}")

    # ---- plot: violin of best signal + AUC bars ----
    best = res.iloc[0]["signal"]
    vb = pd.to_numeric(df[best], errors="coerce").to_numpy(float)
    fig, (axv, axb) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    data = [vb[(y == 0) & np.isfinite(vb)], vb[(y == 1) & np.isfinite(vb)]]
    parts = axv.violinplot(data, showmedians=True)
    axv.set_xticks([1, 2]); axv.set_xticklabels(["fizzle\n(spiral, no storm)", "tighten\n(spiral -> storm)"])
    axv.set_ylabel(best)
    axv.set_title(f"Score bands: {best}  (AUC={res.iloc[0]['auc_tighten_vs_fizzle']:.3f})")
    # orientation-free AUC so anti-oriented features (e.g. shear_quench, msl_nd)
    # show their true discriminative power rather than <0.5
    res = res.assign(auc_ofree=res["auc_tighten_vs_fizzle"].apply(lambda a: max(a, 1 - a)))
    order = res.sort_values("auc_ofree")
    axb.barh(order["signal"], order["auc_ofree"], color="#4c9be8")
    axb.axvline(0.5, color="grey", ls="--", lw=1)
    if refined:
        axb.axvline(refined["auc_multivariate_test"], color="#e8734c", ls="-", lw=2,
                    label=f"refined multi (test={refined['auc_multivariate_test']:.3f})")
        axb.legend(fontsize=8)
    axb.set_xlabel("AUC (tighten vs fizzle, orientation-free)"); axb.set_title("Spiral-conditioned skill")
    fig.tight_layout(); fig.savefig(out_dir / "spiral_genesis_bands.png"); plt.close(fig)
    print(f"[spiral] wrote {out_dir/'spiral_genesis_bands.png'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Spiral-conditioned genesis discrimination (tighten vs fizzle).")
    ap.add_argument("--gka-grid", required=True)
    ap.add_argument("--pregen-grid", required=True)
    ap.add_argument("--lead-grid", required=True)
    ap.add_argument("--signals", default="gka_SII,gka_SAI,gka_score,gka_kappa,gka_chirality")
    ap.add_argument("--build-phi", action="store_true",
                    help="Construct theory-driven gka_phi on the fly from gka_relax/gka_A_overlap/"
                         "rolling-var(gka_kappa) and add it to the signal set.")
    ap.add_argument("--start", default="2025-02-01")
    ap.add_argument("--end", default="2025-04-30")
    ap.add_argument("--train-end", default="2025-03-31")
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--max-per-chunk", type=int, default=50000)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out-dir", default="figures/spiral_genesis")
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
