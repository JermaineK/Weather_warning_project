#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multiseason_battery.py — re-run the genesis validation battery across seasons.

Runs, with LEAVE-ONE-SEASON-OUT fitting (a model never sees the season it is
scored on, so neither the storm nor its large-scale environment leaks):

  A. Discriminator skill   per-storm tighten-vs-fizzle AUC
  B. Early-warning trigger instantaneous vs trailing acc24/acc48 by lead,
                           plus the paired A48-minus-instantaneous advantage
  C. Shape filter          does "building OR sustained-high" survive held-out?
  D. Precursor backtrace   are future-certain cells separable 48-72h earlier?

Each is reported with a storm-block bootstrap 95% CI and an explicit count of
contributing storms, so thin bins stay visible rather than hidden.

USAGE
    python eval_multiseason_battery.py \\
        --panels "data/genesis_*_slim.parquet" \\
        --tracks "data/tracks/tracks_2021.parquet,data/tracks/tracks_2022.parquet,data/tracks/tracks_2023.parquet,data/tracks/tracks_2024.parquet,data/tracks/tracks_geomval.parquet" \\
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
from geomval_seasons import load_storm_crops, train_keys_for     # noqa: E402

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
WINDOWS = [24, 48]
FEAT_OUT = ["s", "A24", "A48"]
LEADS = [6, 12, 24, 36, 48]
LEAD_TOL = 3
BACK_LO, BACK_HI = 48.0, 72.0
PRECURSOR_ATTRS = ["shear_low", "gka_msl_nd", "zeta", "gka_knee_ratio", "SFI",
                   "sph_vdr_std", "gka_SII", "gka_shear_quench", "E_energy_proxy"]


def boot_ci(vals: np.ndarray, rng, n_boot: int):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) < 2:
        return (float(np.nanmean(vals)) if len(vals) else np.nan, np.nan, np.nan, len(vals))
    bm = [np.nanmean(vals[rng.integers(0, len(vals), size=len(vals))]) for _ in range(n_boot)]
    return (float(np.nanmean(vals)), float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)), len(vals))


def fit_on(crops, keys, feats, l2=1.0):
    Xs, ys = [], []
    for k in keys:
        d = crops[k]["df"]
        sp = d[d["pregen"] == 1]
        Xs.append(sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        ys.append((sp["near_storm"].to_numpy(float) == 1).astype(float))
    X = np.vstack(Xs); y = np.concatenate(ys)
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    if y.min() == y.max():
        return None
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return fit_logistic((X - mu) / sd, y, l2=l2), mu, sd


def score_df(m, feats, w, mu, sd):
    X = m[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fin = np.all(np.isfinite(X), axis=1)
    s = np.full(len(m), np.nan)
    s[fin] = predict(w, (X[fin] - mu) / sd)
    m = m.assign(s=s).sort_values(["ilat", "ilon", "time"])
    gb = m.groupby(["ilat", "ilon"], sort=False)["s"]
    for W in WINDOWS:
        m[f"A{W}"] = (gb.rolling(W, min_periods=max(3, W // 4)).mean()
                        .reset_index(level=[0, 1], drop=True))
    m["build"] = m["A24"] - m["A48"]
    return m


def main() -> int:
    a = parse_args()
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    extra = ["shear_low", "zeta", "SFI", "sph_vdr_std", "thermo_shear",
             "pdrop_nd", "t2m_anom_local", "gka_SAI"]
    crops = load_storm_crops(a.panels, a.tracks, feats, extra_cols=extra,
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)
    keys = list(crops)
    seasons = sorted({crops[k]["season"] for k in keys})
    rng = np.random.default_rng(1)
    print(f"\n[battery] {len(keys)} storms / {len(seasons)} seasons, cv={a.cv}, feats={feats}")

    overall, per_lead, shape_rows, pre_rows, bg_rows = {}, {}, [], [], []
    for k in keys:
        tk = train_keys_for(crops, k, cv=a.cv)
        fit = fit_on(crops, tk, feats, a.l2)
        if fit is None:
            continue
        w, mu, sd = fit
        m = score_df(crops[k]["df"].copy(), feats, w, mu, sd)
        sp = m[m["pregen"] == 1]
        tight, fiz = sp[sp["near_storm"] == 1], sp[sp["near_storm"] == 0]
        if len(fiz) > a.max_neg:
            fiz = fiz.iloc[rng.choice(len(fiz), a.max_neg, replace=False)]
        if not len(tight) or not len(fiz):
            continue

        # A. discriminator
        overall[k] = auc_roc(tight["s"].to_numpy(float), fiz["s"].to_numpy(float))

        # B. early warning by lead
        tt = pd.to_numeric(tight["t_to_storm_min_h"], errors="coerce").to_numpy(float)
        per_lead[k] = {}
        for L in LEADS:
            msk = (tt >= L - LEAD_TOL) & (tt <= L + LEAD_TOL)
            if msk.sum() < a.min_bin:
                continue
            per_lead[k][L] = {c: auc_roc(tight[c].to_numpy(float)[msk],
                                         fiz[c].to_numpy(float)) for c in FEAT_OUT}

        # C. shape filter on fired alerts (threshold = 90th pct of fizzle A48)
        thr = float(np.nanquantile(fiz["A48"].to_numpy(float), 0.90))
        fired = sp[pd.to_numeric(sp["A48"], errors="coerce") >= thr]
        if len(fired) > 50:
            yf = (fired["near_storm"].to_numpy(float) == 1).astype(int)
            sustained = float(np.nanquantile(fired["A48"].to_numpy(float), 0.60))
            keep = ((fired["build"] > 0) |
                    (fired["A48"] >= sustained)).fillna(False).to_numpy()
            shape_rows.append({
                "storm": k, "season": crops[k]["season"],
                "precision_all": float(yf.mean()),
                "precision_filtered": float(yf[keep].mean()) if keep.any() else np.nan,
                "kept_frac": float(keep.mean()),
                "recall_kept": float(yf[keep].sum() / max(yf.sum(), 1)),
            })

        # D. precursor backtrace: cells that become top-decile certain
        cutoff = float(np.nanquantile(sp["s"].to_numpy(float), a.certain_quantile))
        cert = sp[sp["s"] >= cutoff]
        if len(cert) > 20:
            firsts = cert.groupby(["ilat", "ilon"], as_index=False)["time"].min()
            cert_cells = set(map(tuple, firsts[["ilat", "ilon"]].to_numpy()))
            idx = sp.set_index(["ilat", "ilon", "time"]).sort_index()
            for ilat, ilon, tc in firsts.itertuples(index=False):
                lo, hi = tc - pd.Timedelta(hours=BACK_HI), tc - pd.Timedelta(hours=BACK_LO)
                try:
                    sub = idx.loc[(ilat, ilon)]
                except KeyError:
                    continue
                sel = sub[(sub.index >= lo) & (sub.index <= hi)]
                sel = sel[sel["s"] < cutoff]
                if len(sel):
                    pre_rows.append(sel.assign(storm=k))
            bgm = sp[[tuple(x) not in cert_cells
                      for x in sp[["ilat", "ilon"]].to_numpy()]]
            if len(bgm) > a.max_neg:
                bgm = bgm.iloc[rng.choice(len(bgm), a.max_neg, replace=False)]
            bg_rows.append(bgm.assign(storm=k))
        print(f"[battery] {k:22s} AUC={overall[k]:.3f} leads={sorted(per_lead[k])}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"n_storms": len(overall), "seasons": seasons, "cv": a.cv,
                 "features": feats}

    # ---- A ----
    ov = np.array(list(overall.values()), float)
    mean, lo, hi, n = boot_ci(ov, rng, a.n_boot)
    rep["discriminator"] = {"auc_mean": mean, "ci": [lo, hi], "n_storms": n,
                            "by_storm": {k: float(v) for k, v in overall.items()}}
    print(f"\n[A] discriminator LOSO AUC = {mean:.3f}  95%CI[{lo:.3f},{hi:.3f}]  n={n} storms")

    # ---- B ----
    rows = []
    for L in LEADS:
        for c in FEAT_OUT:
            vals = np.array([per_lead[k][L][c] for k in per_lead if L in per_lead[k]], float)
            m_, l_, h_, n_ = boot_ci(vals, rng, a.n_boot)
            rows.append({"lead_h": L, "feature": c, "auc_mean": m_,
                         "ci_lo": l_, "ci_hi": h_, "n_storms": n_})
    lead_df = pd.DataFrame(rows)
    lead_df.to_csv(out / "multiseason_leads.csv", index=False)
    print("\n[B] early-warning skill by lead (per-storm mean, 95% CI, n storms):")
    for L in LEADS:
        sub = lead_df[lead_df.lead_h == L]
        line = f"  {L:2d}h: "
        for c in FEAT_OUT:
            r = sub[sub.feature == c].iloc[0]
            line += f"{c}={r.auc_mean:.3f}[{r.ci_lo:.3f},{r.ci_hi:.3f}] "
        line += f" n={int(sub.iloc[0].n_storms)}"
        print(line)
    adv = []
    for L in LEADS:
        d = np.array([per_lead[k][L]["A48"] - per_lead[k][L]["s"]
                      for k in per_lead if L in per_lead[k]], float)
        m_, l_, h_, n_ = boot_ci(d, rng, a.n_boot)
        beats = bool(np.isfinite(l_) and l_ > 0)
        adv.append({"lead_h": L, "delta_auc": m_, "ci_lo": l_, "ci_hi": h_,
                    "n_storms": n_, "A48_beats_instant": beats})
        print(f"  A48-inst @{L:2d}h: {m_:+.3f} [{l_:+.3f},{h_:+.3f}] n={n_} "
              f"{'A48 WINS' if beats else 'n.s.'}")
    rep["early_warning"] = {"by_lead": lead_df.to_dict("records"), "advantage": adv}

    # ---- C ----
    if shape_rows:
        sh = pd.DataFrame(shape_rows)
        sh.to_csv(out / "multiseason_shape_filter.csv", index=False)
        d = (sh["precision_filtered"] - sh["precision_all"]).to_numpy(float)
        m_, l_, h_, n_ = boot_ci(d, rng, a.n_boot)
        rep["shape_filter"] = {"precision_gain_mean": m_, "ci": [l_, h_], "n_storms": n_,
                               "mean_kept_frac": float(sh["kept_frac"].mean()),
                               "mean_recall_kept": float(sh["recall_kept"].mean())}
        print(f"\n[C] shape filter precision gain = {m_:+.3f} 95%CI[{l_:+.3f},{h_:+.3f}] "
              f"n={n_} storms; keeps {sh['kept_frac'].mean():.1%} of alerts, "
              f"{sh['recall_kept'].mean():.1%} of true positives")

    # ---- D ----
    if pre_rows and bg_rows:
        pre = pd.concat(pre_rows, ignore_index=True)
        bg = pd.concat(bg_rows, ignore_index=True)
        attrs = [c for c in PRECURSOR_ATTRS if c in pre.columns and c in bg.columns]
        prows = []
        for c in attrs:
            au = auc_roc(pd.to_numeric(pre[c], errors="coerce").to_numpy(float),
                         pd.to_numeric(bg[c], errors="coerce").to_numpy(float))
            prows.append({"attribute": c, "auc": au,
                          "auc_orientation_free": max(au, 1 - au) if np.isfinite(au) else np.nan})
        pdf = pd.DataFrame(prows).sort_values("auc_orientation_free", ascending=False)
        pdf.to_csv(out / "multiseason_precursor.csv", index=False)
        rep["precursor"] = {"lookback_h": [BACK_LO, BACK_HI],
                            "n_precursor": int(len(pre)), "n_background": int(len(bg)),
                            "attributes": pdf.to_dict("records")}
        print(f"\n[D] precursor separation {BACK_LO:.0f}-{BACK_HI:.0f}h before certainty "
              f"(n_pre={len(pre):,}, n_bg={len(bg):,}):")
        print(pdf.head(8).to_string(index=False))

    (out / "multiseason_battery.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n[battery] wrote {out/'multiseason_battery.json'} (+ csvs)")

    # plot
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
    colors = {"s": "#4c72b0", "A24": "#dd8452", "A48": "#55a868"}
    labels = {"s": "instantaneous", "A24": "trailing 24h", "A48": "trailing 48h"}
    for c in FEAT_OUT:
        sub = lead_df[lead_df.feature == c].sort_values("lead_h")
        ax.plot(sub.lead_h, sub.auc_mean, "-o", color=colors[c], label=labels[c], lw=2)
        ax.fill_between(sub.lead_h, sub.ci_lo, sub.ci_hi, color=colors[c], alpha=0.18)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.invert_xaxis()
    for L in LEADS:
        n_ = int(lead_df[(lead_df.lead_h == L) & (lead_df.feature == "s")].iloc[0].n_storms)
        ax.annotate(f"n={n_}", (L, 0.505), fontsize=7, ha="center", color="grey")
    ax.set_xlabel("warning lead (hours before storm)")
    ax.set_ylabel(f"held-out AUC (leave-one-{a.cv}-out)")
    ax.set_title(f"Multi-season genesis trigger: {len(overall)} storms, "
                 f"{len(seasons)} seasons")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "multiseason_leads.png"); plt.close(fig)
    print(f"[battery] wrote {out/'multiseason_leads.png'}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-season genesis validation battery.")
    ap.add_argument("--panels", required=True, help="glob/comma-list of per-season slim panels")
    ap.add_argument("--tracks", required=True, help="glob/comma-list of per-season track files")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--features", default=",".join(CURATED))
    ap.add_argument("--cv", choices=["season", "storm"], default="season")
    ap.add_argument("--certain-quantile", type=float, default=0.99)
    ap.add_argument("--min-bin", type=int, default=30)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
    ap.add_argument("--max-neg", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--l2", type=float, default=1.0)
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
