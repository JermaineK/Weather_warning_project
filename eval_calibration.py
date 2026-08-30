#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_calibration.py — is the genesis discriminator's probability USABLE, not just rank-correct?

Everything measured so far is discrimination (AUC: does it rank correctly).
Nothing measured calibration (does a score of 0.7 mean a 70% chance). For any
operational reading, calibration is the binding constraint: with a rare event,
a high AUC routinely coexists with unusable precision at every threshold.

Produces, on leave-one-season-out held-out predictions:

  * Brier score and Brier SKILL score vs climatology (the base rate)
  * MURPHY DECOMPOSITION  BS = reliability - resolution + uncertainty
      reliability (lower better) = calibration error
      resolution  (higher better) = discrimination, i.e. how far bins depart
                                    from the base rate
  * Reliability diagram (observed frequency vs forecast probability, with the
    sharpness histogram, since a flat forecast is trivially well-calibrated)
  * Operating points: precision / recall / F1 / frequency-of-alerts by threshold
  * The same, split BY LEAD, because calibration at 6h says nothing about 48h
  * Optional isotonic recalibration, fitted on OTHER seasons and applied to the
    held-out season (never fitted on the data it is scored against)

USAGE
    python eval_calibration.py \\
        --panels "data/genesis_*_slim.parquet" \\
        --tracks "data/tracks/tracks_2021.parquet,...,data/tracks/tracks_geomval.parquet" \\
        --label strict_genesis --out-dir results/metrics
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
from eval_spiral_genesis import auc_roc, fit_logistic, predict          # noqa: E402
from geomval_seasons import (load_storm_crops, read_tracks, resolve_files,  # noqa: E402
                             genesis_events, add_strict_labels)

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
LEAD_BANDS = [(0, 12), (12, 24), (24, 48), (48, 72)]


# ---------------- verification metrics ----------------

def brier(y, p):
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p):
    base = float(np.mean(y))
    bs_ref = float(np.mean((base - y) ** 2))
    return (1.0 - brier(y, p) / bs_ref) if bs_ref > 0 else np.nan


def murphy_decomposition(y, p, n_bins=10):
    """BS = reliability - resolution + uncertainty (all on the same scale)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    obar = float(np.mean(y))
    N = len(y)
    rel = res = 0.0
    rows = []
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            rows.append({"bin_lo": edges[k], "bin_hi": edges[k + 1], "n": 0,
                         "mean_forecast": np.nan, "observed_freq": np.nan})
            continue
        pk = float(np.mean(p[m]))
        ok = float(np.mean(y[m]))
        rel += nk / N * (pk - ok) ** 2
        res += nk / N * (ok - obar) ** 2
        rows.append({"bin_lo": float(edges[k]), "bin_hi": float(edges[k + 1]),
                     "n": nk, "mean_forecast": pk, "observed_freq": ok})
    unc = obar * (1 - obar)
    return {"reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc), "base_rate": obar}, pd.DataFrame(rows)


def isotonic_fit(p, y):
    """Minimal PAVA isotonic regression (no sklearn)."""
    o = np.argsort(p)
    ps, ys = p[o], y[o].astype(float)
    v = ys.copy()
    w = np.ones_like(v)
    # pool adjacent violators
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1] + 1e-12:
            new_w = w[i] + w[i + 1]
            new_v = (v[i] * w[i] + v[i + 1] * w[i + 1]) / new_w
            v[i] = new_v; w[i] = new_w
            v = np.delete(v, i + 1); w = np.delete(w, i + 1)
            ps = np.delete(ps, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    return ps, v


def isotonic_apply(knots_x, knots_y, p):
    return np.interp(p, knots_x, knots_y, left=knots_y[0], right=knots_y[-1])


def operating_points(y, p, thresholds):
    rows = []
    n = len(y)
    for t in thresholds:
        fire = p >= t
        nf = int(fire.sum())
        tp = int((fire & (y == 1)).sum())
        prec = tp / nf if nf else np.nan
        rec = tp / max(int(y.sum()), 1)
        f1 = (2 * prec * rec / (prec + rec)) if (nf and prec + rec > 0) else np.nan
        rows.append({"threshold": float(t), "alert_rate": nf / n, "n_alerts": nf,
                     "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


# ---------------- main ----------------

def main() -> int:
    a = parse_args()
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    crops = load_storm_crops(a.panels, a.tracks, feats,
                             extra_cols=["shear_low", "zeta"],
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)

    if a.label == "strict_genesis":
        ev = genesis_events(read_tracks(resolve_files(a.tracks)), a.genesis_thresh_kt)
        kept = {}
        for k, rec in crops.items():
            df, _ = add_strict_labels(rec["df"], ev, a.genesis_radius, a.genesis_max_lead)
            sp = df[df["pregen"] == 1]
            pos = sp[sp["gen_pos"] == 1]
            neg = sp[(sp["gen_pos"] == 0) & (sp["near_storm"] == 0)]
            if len(pos) < a.min_bin or len(neg) < a.min_bin:
                continue
            d = pd.concat([pos, neg], ignore_index=True)
            d["y"] = (d["gen_pos"] == 1).astype(int)
            d["lead_col"] = d["gen_lead_h"]
            kept[k] = {**rec, "df": d}
        crops = kept
    else:
        for k, rec in crops.items():
            d = rec["df"]
            d["y"] = (d["near_storm"] == 1).astype(int)
            d["lead_col"] = d["t_to_storm_min_h"]

    seasons = sorted({v["season"] for v in crops.values()})
    print(f"[calib] {len(crops)} storms / {len(seasons)} seasons, label={a.label}")

    # leave-one-season-out predictions; calibrator also fitted out-of-season
    P, Y, LEAD, SEASON, PCAL = [], [], [], [], []
    for s in seasons:
        tr_keys = [k for k, v in crops.items() if v["season"] != s]
        te_keys = [k for k, v in crops.items() if v["season"] == s]
        if not tr_keys or not te_keys:
            continue
        Xtr = np.vstack([crops[k]["df"][feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                         for k in tr_keys])
        ytr = np.concatenate([crops[k]["df"]["y"].to_numpy(float) for k in tr_keys])
        fin = np.all(np.isfinite(Xtr), axis=1)
        Xtr, ytr = Xtr[fin], ytr[fin]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = fit_logistic((Xtr - mu) / sd, ytr, l2=a.l2)
        ptr = predict(w, (Xtr - mu) / sd)
        kx, ky = isotonic_fit(ptr, ytr)          # calibrator from TRAIN seasons only

        for k in te_keys:
            d = crops[k]["df"]
            X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            m = np.all(np.isfinite(X), axis=1)
            if not m.any():
                continue
            p = predict(w, (X[m] - mu) / sd)
            P.append(p); Y.append(d["y"].to_numpy(int)[m])
            LEAD.append(pd.to_numeric(d["lead_col"], errors="coerce").to_numpy(float)[m])
            SEASON.append(np.full(m.sum(), s))
            PCAL.append(isotonic_apply(kx, ky, p))
        print(f"[calib] season {s}: trained on {len(tr_keys)} storms, scored {len(te_keys)}")

    p = np.concatenate(P); y = np.concatenate(Y)
    lead = np.concatenate(LEAD); season = np.concatenate(SEASON)
    pcal = np.concatenate(PCAL)
    print(f"[calib] pooled held-out rows={len(y):,}  base rate={y.mean():.4f}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"label": a.label, "n_rows": int(len(y)), "base_rate": float(y.mean()),
                 "seasons": [int(s) for s in seasons], "features": feats}

    for tag, pp in (("raw", p), ("isotonic", pcal)):
        dec, rel_df = murphy_decomposition(y, pp, a.n_bins)
        rep[tag] = {"auc": auc_roc(pp[y == 1], pp[y == 0]),
                    "brier": brier(y, pp), "brier_skill": brier_skill(y, pp), **dec}
        rel_df.to_csv(out / f"calibration_reliability_{tag}.csv", index=False)
        r = rep[tag]
        print(f"\n[calib] {tag.upper()}")
        print(f"  AUC              {r['auc']:.4f}")
        print(f"  Brier            {r['brier']:.5f}")
        print(f"  Brier skill      {r['brier_skill']:+.4f}   (vs climatology)")
        print(f"  reliability      {r['reliability']:.5f}   (lower = better calibrated)")
        print(f"  resolution       {r['resolution']:.5f}   (higher = more informative)")
        print(f"  uncertainty      {r['uncertainty']:.5f}   (base rate variance)")

    # operating points
    thr = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19),
                                    np.nanquantile(p, [0.5, 0.75, 0.9, 0.95, 0.99])]))
    ops = operating_points(y, p, thr)
    ops.to_csv(out / "calibration_operating_points.csv", index=False)
    print("\n[calib] operating points (raw score):")
    show = ops[ops.n_alerts > 0]
    for _, r in show.iterrows():
        if r.threshold in (0.05, 0.25, 0.5, 0.75, 0.95) or abs(r.f1 - show.f1.max()) < 1e-12:
            star = "  <- best F1" if abs(r.f1 - show.f1.max()) < 1e-12 else ""
            print(f"  thr={r.threshold:.2f}  alert_rate={r.alert_rate:6.2%}  "
                  f"precision={r.precision:6.2%}  recall={r.recall:6.2%}  F1={r.f1:.3f}{star}")

    # by lead band
    lead_rows = []
    for lo, hi in LEAD_BANDS:
        m = (lead >= lo) & (lead < hi) | ((y == 0) & np.isnan(lead))
        mm = ((lead >= lo) & (lead < hi)) | (y == 0)
        if mm.sum() < 100 or y[mm].sum() < 20:
            continue
        dec, _ = murphy_decomposition(y[mm], p[mm], a.n_bins)
        lead_rows.append({"lead_lo": lo, "lead_hi": hi, "n": int(mm.sum()),
                          "base_rate": dec["base_rate"],
                          "auc": auc_roc(p[mm][y[mm] == 1], p[mm][y[mm] == 0]),
                          "brier": brier(y[mm], p[mm]),
                          "brier_skill": brier_skill(y[mm], p[mm]),
                          "reliability": dec["reliability"], "resolution": dec["resolution"]})
    lead_df = pd.DataFrame(lead_rows)
    if not lead_df.empty:
        lead_df.to_csv(out / "calibration_by_lead.csv", index=False)
        print("\n[calib] by lead band (positives in band vs all fizzle):")
        print(lead_df.round(4).to_string(index=False))
        rep["by_lead"] = lead_df.to_dict("records")

    (out / "calibration.json").write_text(json.dumps(rep, indent=2, default=str))

    # ---- plots: reliability diagram + sharpness, PR curve ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for tag, pp, col in (("raw", p, "#4c72b0"), ("isotonic", pcal, "#dd8452")):
        _, rdf = murphy_decomposition(y, pp, a.n_bins)
        ok = rdf["n"] > 0
        ax1.plot(rdf.loc[ok, "mean_forecast"], rdf.loc[ok, "observed_freq"],
                 "-o", color=col, label=f"{tag} (BSS={rep[tag]['brier_skill']:+.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax1.axhline(y.mean(), color="grey", ls=":", lw=1, label=f"base rate {y.mean():.3f}")
    ax1.set_xlabel("forecast probability"); ax1.set_ylabel("observed frequency")
    ax1.set_title("Reliability diagram (leave-one-season-out)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.hist(p, bins=a.n_bins, range=(0, 1), alpha=0.15, color="grey")
    ax1b.set_yscale("log"); ax1b.set_ylabel("count (sharpness)", fontsize=8)

    ax2.plot(ops["recall"], ops["precision"], "-o", color="#2a9d8f")
    ax2.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"base rate {y.mean():.3f}")
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision")
    ax2.set_title("Precision-recall (raw score)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "calibration.png"); plt.close(fig)
    print(f"\n[calib] wrote {out/'calibration.json'} (+ csvs, calibration.png)")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Calibration / forecast verification of the genesis discriminator.")
    ap.add_argument("--panels", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--features", default=",".join(CURATED))
    ap.add_argument("--label", choices=["near_storm", "strict_genesis"], default="strict_genesis")
    ap.add_argument("--genesis-thresh-kt", type=float, default=34.0)
    ap.add_argument("--genesis-radius", type=float, default=3.0)
    ap.add_argument("--genesis-max-lead", type=float, default=72.0)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-bin", type=int, default=30)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
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
