#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesis_trigger.py — spiral-conditioned genesis alerts from the labelled GKA grid.

Pipeline-facing implementation of the validated early-warning trigger
(see eval_early_warning_loso.py / PR "V3.2 composite Phi threshold test"):

  1. Fit the curated ridge-logistic discriminator (tighten vs fizzle, conditioned
     on pregen==1 spiral cells) on the configured TRAIN window.
  2. Score every cell-hour: prob_genesis (instantaneous) + causal trailing
     accumulations acc24/acc48 per (ilat,ilon).
  3. Threshold the trigger feature (default prob_genesis, the instantaneous
     score) at a false-alarm quantile computed on train-window fizzle spirals.
     Trailing accumulations acc24/acc48 are still emitted as diagnostics, but
     strict-genesis multi-season validation found they give no skill advantage
     at any lead, so they are NOT the default trigger.
  4. Emit alert cells (pregen==1 rows) in the repo's standard alerts schema:
     time, lat, lon, prob_genesis, alert_genesis (+ acc24/acc48/lead passthrough)
     so seeds.from_alerts / starts / outcomes consume them unchanged.

Single input file: the labelled+ID'd GKA grid (contains features AND labels).
Reads in ilat stripes (predicate pushdown) so the full-period rolling stays
memory-safe on the 70M-row grid.

USAGE (via alerts_logic manager)
    python alerts_logic_manager.py genesis-trigger \\
        --labelled data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --train-end 2025-03-31 \\
        --out results/alerts/alerts_geomval_final.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for eval_spiral_genesis imports

from eval_spiral_genesis import auc_roc, fit_logistic, predict  # noqa: E402

DEFAULT_FEATURES = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
LABEL_COLS = ["pregen", "near_storm", "t_to_storm_min_h"]
KEY_COLS = ["time", "lat", "lon", "ilat", "ilon"]


def _time_filters(args):
    F = []
    if args.start:
        F.append(("time", ">=", pd.Timestamp(args.start)))
    if args.end:
        F.append(("time", "<", pd.Timestamp(args.end)))
    return F or None


def _area_mask(df, area: str | None):
    if not area:
        return df
    latN, lonW, latS, lonE = [float(x) for x in area.split(",")]
    return df[(df["lat"] <= latN) & (df["lat"] >= latS)
              & (df["lon"] >= lonW) & (df["lon"] <= lonE)]


def _ilat_stripes(path: str, n_stripes: int):
    """Split the ilat range into contiguous stripes using parquet stats."""
    pf = pq.ParquetFile(path)
    names = pf.schema.names
    if "ilat" not in names:
        raise SystemExit("[genesis-trigger] input has no 'ilat' column; need the *_id file.")
    ci = names.index("ilat")
    lo = hi = None
    for rg in range(pf.num_row_groups):
        st = pf.metadata.row_group(rg).column(ci).statistics
        if st is None:
            continue
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    if lo is None:
        raise SystemExit("[genesis-trigger] no ilat statistics in parquet metadata.")
    edges = np.linspace(lo, hi + 1, n_stripes + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _load_cape(nc_glob: str, lat_lo: float, lat_hi: float) -> pd.DataFrame | None:
    """Tidy (time,lat,lon,cape) for a latitude band, read from ERA5 CAPE netCDFs.

    Loaded per ilat stripe so the full-domain CAPE field is never held at once.
    Returns None if nothing matches, so the caller can fail loudly rather than
    silently dropping the gate.
    """
    import glob as _glob
    import xarray as xr
    paths = sorted(_glob.glob(nc_glob, recursive=True))
    if not paths:
        return None
    frames = []
    for pth in paths:
        ds = xr.open_dataset(pth)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        latn = "latitude" if "latitude" in ds.coords else "lat"
        lonn = "longitude" if "longitude" in ds.coords else "lon"
        if "cape" not in ds.data_vars:
            ds.close(); continue
        lat_vals = ds[latn].values
        keep = (lat_vals >= lat_lo - 0.13) & (lat_vals <= lat_hi + 0.13)
        if not keep.any():
            ds.close(); continue
        sub = ds[["cape"]].isel({latn: np.where(keep)[0]})
        df = sub.to_dataframe().reset_index()
        ds.close()
        df = df.rename(columns={tname: "time", latn: "lat", lonn: "lon"})
        df["time"] = pd.to_datetime(df["time"])
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").round(2)
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").round(2)
        frames.append(df[["time", "lat", "lon", "cape"]])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["time", "lat", "lon"])


def fit_trigger_model(args, feats):
    """Fit the curated model on train-window spiral rows (tighten vs fizzle)."""
    F = [("time", "<=", pd.Timestamp(args.train_end))]
    if args.start:
        F.append(("time", ">=", pd.Timestamp(args.start)))
    cols = list(dict.fromkeys(KEY_COLS + feats + ["pregen", "near_storm"]))
    df = pd.read_parquet(args.labelled, columns=cols, filters=F)
    df = _area_mask(df, args.area)
    sp = df[df["pregen"] == 1]
    if sp.empty:
        raise SystemExit("[genesis-trigger] no pregen==1 rows in train window.")
    rng = np.random.default_rng(args.seed)
    if len(sp) > args.max_train_rows:
        sp = sp.iloc[rng.choice(len(sp), args.max_train_rows, replace=False)]
    X = sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = (sp["near_storm"].to_numpy(float) == 1).astype(float)
    fin = np.all(np.isfinite(X), axis=1)
    X, y = X[fin], y[fin]
    if y.min() == y.max():
        raise SystemExit("[genesis-trigger] train window has a single class; widen the window.")
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=args.l2)
    p = predict(w, (X - mu) / sd)
    auc = auc_roc(p[y == 1], p[y == 0])
    print(f"[genesis-trigger] train fit: n={len(y):,} pos={int(y.sum()):,} "
          f"in-sample tighten-vs-fizzle AUC={auc:.3f}")
    return w, mu, sd, {"n_train": int(len(y)), "n_pos": int(y.sum()),
                       "train_auc_insample": float(auc)}


def score_stripes(args, feats, w, mu, sd):
    """Pass 2: score all cells stripe-by-stripe; return pregen==1 rows with
    prob/acc columns, plus the train-window fizzle trigger values for
    thresholding."""
    tF = _time_filters(args)
    out_frames, thr_samples = [], []
    train_end = pd.Timestamp(args.train_end)
    cols = list(dict.fromkeys(KEY_COLS + feats + LABEL_COLS))
    stripes = _ilat_stripes(args.labelled, args.stripes)
    for k, (a, b) in enumerate(stripes, 1):
        F = [("ilat", ">=", a), ("ilat", "<", b)] + (tF or [])
        m = pd.read_parquet(args.labelled, columns=cols, filters=F)
        if m.empty:
            continue
        m = _area_mask(m, args.area)
        if m.empty:
            continue
        if args.cape_nc_glob:
            m["lat"] = pd.to_numeric(m["lat"], errors="coerce").round(2)
            m["lon"] = pd.to_numeric(m["lon"], errors="coerce").round(2)
            cp = _load_cape(args.cape_nc_glob, float(m["lat"].min()), float(m["lat"].max()))
            if cp is None:
                raise SystemExit(f"[genesis-trigger] --cape-nc-glob matched no usable "
                                 f"CAPE files: {args.cape_nc_glob}")
            before = len(m)
            m = m.merge(cp, on=["time", "lat", "lon"], how="left")
            if len(m) != before:
                raise SystemExit("[genesis-trigger] CAPE join changed row count; "
                                 "duplicate keys in the CAPE source.")
        X = m[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        fin = np.all(np.isfinite(X), axis=1)
        s = np.full(len(m), np.nan)
        s[fin] = predict(w, (X[fin] - mu) / sd)
        m = m.assign(prob_genesis=s.astype("float32"))
        m = m.sort_values(["ilat", "ilon", "time"])
        gb = m.groupby(["ilat", "ilon"], sort=False)["prob_genesis"]
        for W in (24, 48):
            m[f"acc{W}"] = (gb.rolling(W, min_periods=max(3, W // 4)).mean()
                              .reset_index(level=[0, 1], drop=True).astype("float32"))
        sp = m[m["pregen"] == 1].copy()
        if not sp.empty:
            fiz_tr = sp[(sp["near_storm"] == 0) & (sp["time"] <= train_end)]
            v = pd.to_numeric(fiz_tr[args.trigger_feature], errors="coerce").dropna()
            if len(v):
                thr_samples.append(v.to_numpy(float))
            keep = list(dict.fromkeys(
                KEY_COLS + ["prob_genesis", "acc24", "acc48"] + LABEL_COLS
                + (["cape"] if "cape" in sp.columns else [])))
            out_frames.append(sp[keep])
        print(f"[genesis-trigger] stripe {k}/{len(stripes)} ilat[{a},{b}) "
              f"rows={len(m):,} spiral={len(sp):,}")
        del m
    if not out_frames:
        raise SystemExit("[genesis-trigger] no spiral rows found in window/area.")
    return pd.concat(out_frames, ignore_index=True), np.concatenate(thr_samples)


def main() -> int:
    args = parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]

    if args.coefs_json:
        card = json.loads(Path(args.coefs_json).read_text())
        w = np.array(card["w"], float)
        mu = np.array(card["mu"], float)
        sd = np.array(card["sd"], float)
        feats = card["features"]
        fit_info = {"loaded_from": args.coefs_json}
        print(f"[genesis-trigger] loaded frozen coefficients from {args.coefs_json}")
    else:
        w, mu, sd, fit_info = fit_trigger_model(args, feats)

    alerts, fiz_vals = score_stripes(args, feats, w, mu, sd)
    thr = float(np.quantile(fiz_vals, 1.0 - args.false_alarm))
    trig = pd.to_numeric(alerts[args.trigger_feature], errors="coerce")
    alerts["alert_genesis"] = ((trig >= thr).fillna(False)).astype("int8")

    # ---- CAPE gate (opt-in) ----
    # Cell-level, matched-alert-rate validation (eval_cape_gate_cells.py, 24 storms,
    # season-blocked, strict genesis labels): a PERMISSIVE gate gives a small but
    # robust precision gain (+0.008..+0.017, CI excludes 0 at every alert rate
    # tested, c* = 30th pct of fit-season cape ~ 12 J/kg) at no lead cost.
    # An AGGRESSIVE gate HURTS (c* = 70th pct ~ 370 J/kg: -0.06, CI excludes 0).
    # Off by default because it needs a CAPE source the base pipeline does not build.
    cape_info = {"cape_gate": bool(args.cape_min is not None or args.cape_min_quantile is not None)}
    if cape_info["cape_gate"]:
        if "cape" not in alerts.columns:
            raise SystemExit("[genesis-trigger] CAPE gate requested but no 'cape' column; "
                             "pass --cape-nc-glob or a labelled grid containing cape.")
        cv = pd.to_numeric(alerts["cape"], errors="coerce")
        if args.cape_min is not None:
            c_star = float(args.cape_min)
        else:
            train_c = cv[alerts["time"] <= pd.Timestamp(args.train_end)].dropna()
            c_star = float(train_c.quantile(args.cape_min_quantile))
        n_before = int(alerts["alert_genesis"].sum())
        alerts["alert_genesis"] = ((alerts["alert_genesis"] == 1)
                                   & (cv >= c_star).fillna(False)).astype("int8")
        n_after = int(alerts["alert_genesis"].sum())
        cape_info.update({"cape_c_star": c_star,
                          "cape_min_quantile": args.cape_min_quantile,
                          "alerts_before_cape_gate": n_before,
                          "alerts_after_cape_gate": n_after,
                          "cape_coverage": float(cv.notna().mean())})
        print(f"[genesis-trigger] CAPE gate cape>={c_star:.1f} J/kg: "
              f"{n_before:,} -> {n_after:,} alerts (coverage {cv.notna().mean():.1%})")

    # shape diagnostics: is the signal building / still surging at alert time?
    alerts["build"] = (alerts["acc24"] - alerts["acc48"]).astype("float32")
    alerts["surge"] = (alerts["prob_genesis"] - alerts["acc24"]).astype("float32")

    # optional shape filter (see analysis in PR): false positives are typically
    # FLAT and NOT SUSTAINED; genuine long-lead alerts sit in the dip phase of
    # the build->dip->surge cycle, so the safe rule is the disjunction
    # "building OR sustained-high", never "building" alone.
    shape_info = {"shape_filter": args.shape_filter}
    if args.shape_filter != "none":
        fired = alerts["alert_genesis"] == 1
        sustained_thr = float(pd.to_numeric(
            alerts.loc[fired, "acc48"], errors="coerce").quantile(args.sustained_quantile)) \
            if fired.any() else float("nan")
        if args.shape_filter == "build":
            keep = alerts["build"] > 0
        else:  # build-or-sustained
            keep = (alerts["build"] > 0) | (alerts["acc48"] >= sustained_thr)
        n_before = int(fired.sum())
        alerts["alert_genesis"] = (fired & keep).astype("int8")
        n_after = int(alerts["alert_genesis"].sum())
        shape_info.update({"sustained_quantile": args.sustained_quantile,
                           "sustained_threshold": sustained_thr,
                           "alerts_before_filter": n_before,
                           "alerts_after_filter": n_after})
        print(f"[genesis-trigger] shape filter '{args.shape_filter}': "
              f"{n_before:,} -> {n_after:,} alerts")

    n_alert = int(alerts["alert_genesis"].sum())
    hit = alerts[(alerts["alert_genesis"] == 1)]
    hit_rate = float((hit["near_storm"] == 1).mean()) if len(hit) else float("nan")
    print(f"[genesis-trigger] threshold {args.trigger_feature}>={thr:.4f} "
          f"(false-alarm target {args.false_alarm:.0%} on train fizzle)")
    print(f"[genesis-trigger] alert cells: {n_alert:,}/{len(alerts):,} spiral rows "
          f"({n_alert/max(len(alerts),1):.1%}); near-storm fraction of alerts: {hit_rate:.1%}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(outp, index=False)
    print(f"[genesis-trigger] wrote {outp}  rows={len(alerts):,}")

    card = {
        "features": feats,
        "w": [float(x) for x in w], "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
        "trigger_feature": args.trigger_feature,
        "threshold": thr, "false_alarm_target": args.false_alarm,
        "train_end": args.train_end, "start": args.start, "end": args.end,
        "area": args.area, "l2": args.l2,
        "n_alert_cells": n_alert, "n_spiral_rows": int(len(alerts)),
        "alert_near_storm_fraction": hit_rate,
        **shape_info,
        **cape_info,
        **fit_info,
    }
    card_path = Path(args.model_card or (str(outp) + ".model_card.json"))
    card_path.write_text(json.dumps(card, indent=2))
    print(f"[genesis-trigger] wrote {card_path}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Spiral-conditioned genesis trigger -> standard alerts file.")
    ap.add_argument("--labelled", required=True, help="Labelled+ID'd GKA grid parquet (features + pregen/near_storm).")
    ap.add_argument("--out", required=True, help="Output alerts parquet.")
    ap.add_argument("--model-card", default=None, help="Model card JSON path (default: <out>.model_card.json).")
    ap.add_argument("--coefs-json", default=None, help="Frozen model card to reuse instead of refitting.")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--train-end", default="2025-03-31")
    ap.add_argument("--start", default=None, help="Window start (default: whole file).")
    ap.add_argument("--end", default=None, help="Window end (default: whole file).")
    ap.add_argument("--area", default=None, help='latN,lonW,latS,lonE crop (matches pipeline defaults.area).')
    ap.add_argument("--trigger-feature", default="prob_genesis",
                    choices=["prob_genesis", "acc24", "acc48"],
                    help="Score that fires an alert. Default prob_genesis (instantaneous): "
                         "strict-genesis multi-season validation found trailing accumulation "
                         "gives NO advantage at any lead (24h dAUC -0.037 CI[-0.078,+0.006]).")
    ap.add_argument("--false-alarm", type=float, default=0.10)
    ap.add_argument("--shape-filter", default="none",
                    choices=["none", "build", "build-or-sustained"],
                    help="Post-threshold false-positive filter. NOT RECOMMENDED with the default "
                         "instantaneous trigger: under strict-genesis labels its precision gain is "
                         "+0.015 CI[-0.000,+0.029] (n.s.). It was only significant when alerts were "
                         "fired on acc48, which is itself unsupported.")
    ap.add_argument("--cape-nc-glob", default=None,
                    help="Glob of ERA5 CAPE netCDFs (e.g. 'data_era5/extracted/2025/**/"
                         "era5_2025*_cape.nc'); joined per ilat stripe on (time,lat,lon).")
    ap.add_argument("--cape-min", type=float, default=None,
                    help="Absolute CAPE gate in J/kg. Validated permissive setting ~12 "
                         "(+0.008..+0.017 precision, CI excludes 0). Aggressive gates hurt.")
    ap.add_argument("--cape-min-quantile", type=float, default=None,
                    help="CAPE gate as a quantile of train-window cape (0.30 validated).")
    ap.add_argument("--sustained-quantile", type=float, default=0.60,
                    help="acc48 quantile (among fired alerts) for the sustained-high backstop.")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-rows", type=int, default=2_000_000)
    ap.add_argument("--stripes", type=int, default=12, help="ilat stripes for memory-safe rolling.")
    # orchestrator compatibility (ignored)
    ap.add_argument("--chunk-rows", type=int, default=0)
    ap.add_argument("--parquet-rows", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
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
